"""Identity-safe inputs and reproducible object/spatial holdouts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit


def exact_ids(values):
    """Reject fractional/unsafe floating identities instead of rounding them."""
    result = []
    for value in values:
        if isinstance(value, (float, np.floating)) and abs(value) >= 2**53:
            raise ValueError("Float-packed identity exceeds the exact integer range")
        number = Decimal(str(value))
        if (
            not number.is_finite()
            or number != number.to_integral_value()
            or number < 0
            or number > np.iinfo(np.int64).max
        ):
            raise ValueError(f"Invalid integer identity: {value!r}")
        result.append(int(number))
    return np.asarray(result, dtype=np.int64)


def id_table(path, column="label_id"):
    frame = pd.read_csv(path, sep="\t", dtype={column: str})
    frame[column] = exact_ids(frame[column])
    if frame[column].duplicated().any():
        raise ValueError(f"Duplicate {column}: {path}")
    return frame.set_index(column)


def embedding_table(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as f:
            ids, features = exact_ids(f["label_ids"]), f["features"].copy()
    elif path.suffix == ".npy":
        a = np.load(path, mmap_mode="r", allow_pickle=False)
        if a.ndim != 2 or a.shape[1] < 2:
            raise ValueError("Legacy NPY must contain cell ID in column zero")
        ids, features = exact_ids(a[:, 0]), np.array(a[:, 1:])
    else:
        raise ValueError("Use label_ids/features NPZ or legacy ID-first NPY")
    if (
        features.ndim != 2
        or len(features) != len(ids)
        or len(np.unique(ids)) != len(ids)
        or not np.isfinite(features).all()
    ):
        raise ValueError(f"Invalid feature matrix: {path}")
    return pd.DataFrame(features, index=pd.Index(ids, name="cell_id"))


def map_expression(expression, mapping):
    if (
        not expression.index.is_unique
        or not mapping.index.is_unique
        or not mapping.nucleus_id.is_unique
    ):
        raise ValueError("Expression and mapping must have unique one-to-one identities")
    ids = mapping.index[mapping.nucleus_id.isin(expression.index)]
    frame = expression.loc[mapping.loc[ids, "nucleus_id"].to_numpy()].copy()
    frame.index = pd.Index(ids, name="cell_id")
    return frame


@dataclass
class PairedData:
    cells: np.ndarray
    nuclei: np.ndarray
    expression: np.ndarray
    morphology: dict
    genes: np.ndarray
    xyz: np.ndarray
    pairs: np.ndarray
    labels: np.ndarray
    measurements: pd.DataFrame
    audit: pd.DataFrame
    unmatched_nuclei: np.ndarray


def load_data(config):
    paths = config["inputs"]
    expression_n = id_table(paths["expression"])
    if not np.isin(expression_n.to_numpy(), [0, 1]).all():
        raise ValueError("Expected observed binary calls; do not fill missing calls with zero")
    mapping = pd.read_csv(paths["mapping"], sep="\t", dtype=str)
    mapping["label_id"] = exact_ids(mapping.label_id)
    mapping["nucleus_id"] = exact_ids(mapping.nucleus_id)
    if mapping.label_id.duplicated().any():
        raise ValueError("Duplicate cell IDs in cell/nucleus bridge")
    mapping = mapping.set_index("label_id").rename_axis("cell_id")
    mapping = mapping.loc[mapping.nucleus_id.ne(0)]
    expression = map_expression(expression_n, mapping)
    morphs = {name: embedding_table(path) for name, path in paths["morphology"].items()}
    ids = expression.index
    for frame in morphs.values():
        ids = ids.union(frame.index)
    ids = ids.sort_values()
    audit = pd.DataFrame(index=ids.rename("cell_id"))
    audit["nucleus_id"] = mapping.nucleus_id.reindex(ids).astype("Int64")
    for name, frame in {"expression": expression, **morphs}.items():
        audit[f"has_{name}"] = ids.isin(frame.index)
    audit["positive_calls"] = expression.sum(axis=1).reindex(ids)
    audit["eligible"] = audit.filter(like="has_").all(axis=1) & audit.positive_calls.gt(0)
    eligible = ids[audit.eligible]
    limit = config["analysis"].get("max_cells")
    if limit and len(eligible) > limit:
        eligible = np.sort(
            np.random.default_rng(config["analysis"]["seed"]).choice(
                eligible.to_numpy(), size=limit, replace=False
            )
        )
    cells = np.asarray(eligible, dtype=np.int64)
    audit["selected"] = ids.isin(cells)
    audit["reason"] = np.select(
        [audit.selected, audit.eligible, audit.positive_calls.eq(0)],
        ["selected", "eligible_outside_random_cpu_sample", "all_zero_expression"],
        default="missing_modality",
    )
    nuclei = mapping.loc[cells, "nucleus_id"].to_numpy(np.int64)
    anchors = id_table(paths["nucleus_table"]).reindex(nuclei)
    xyz = anchors[["anchor_x", "anchor_y", "anchor_z"]].to_numpy(float)
    if not np.isfinite(xyz).all():
        raise ValueError("Missing nucleus coordinates; repair before spatial validation")
    sym = pd.read_csv(paths["bilateral"], sep="\t", dtype=str)
    sym["label_id"] = exact_ids(sym.label_id)
    sym = sym.loc[sym.label_id.ne(0)]
    if sym.label_id.duplicated().any():
        raise ValueError("Duplicate nonbackground bilateral IDs")
    indices = sym.set_index("label_id").pair_index.reindex(cells)
    pairs = np.array(
        [
            f"pair:{exact_ids([pair])[0]}"
            if pd.notna(pair) and Decimal(pair) > 0
            else f"cell:{cell}"
            for cell, pair in zip(cells, indices)
        ]
    )
    labels = (
        id_table(paths["manual_labels"])["cell_type"]
        .reindex(cells)
        .fillna("unannotated")
        .to_numpy(str)
    )
    # Bounded table-derived proxies; they are not mesh elongation or surface measures.
    cell_table = id_table(paths["cell_table"]).reindex(cells)
    measurements = pd.DataFrame({"cell_id": cells, "nucleus_id": nuclei})
    for name, frame in [("cell", cell_table), ("nucleus", anchors)]:
        sizes = np.column_stack(
            [frame[f"bb_max_{axis}"] - frame[f"bb_min_{axis}"] for axis in "xyz"]
        )
        measurements[f"{name}_log1p_voxel_count"] = np.log1p(frame.n_pixels.to_numpy(float))
        measurements[f"{name}_bbox_aspect_ratio"] = np.divide(
            sizes.max(axis=1),
            sizes.min(axis=1),
            out=np.full(len(cells), np.nan),
            where=sizes.min(axis=1) > 0,
        )
    return PairedData(
        cells=cells,
        nuclei=nuclei,
        expression=expression.loc[cells].to_numpy(float),
        morphology={name: frame.loc[cells].to_numpy(float) for name, frame in morphs.items()},
        genes=expression.columns.to_numpy(str),
        xyz=xyz,
        pairs=pairs,
        labels=labels,
        measurements=measurements,
        audit=audit.reset_index(),
        unmatched_nuclei=expression_n.index.difference(mapping.nucleus_id).to_numpy(),
    )


def make_splits(data, settings):
    """One fixed holdout per scheme; spatial training purges crossed bilateral pairs."""
    seed, fraction = settings["seed"], settings["test_fraction"]
    if not 0 < fraction < 0.5:
        raise ValueError("test_fraction must be between zero and 0.5")
    regions = KMeans(
        n_clusters=min(settings["spatial_regions"], len(data.cells)), n_init=10, random_state=seed
    ).fit_predict(data.xyz)
    splits = {}
    for scheme in settings["schemes"]:
        if scheme not in {"random_pair_grouped", "spatial"}:
            raise ValueError(f"Unknown holdout scheme: {scheme}")
        groups = data.pairs if scheme == "random_pair_grouped" else regions
        tr, te = next(
            GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(
                data.expression, groups=groups
            )
        )
        purged = tr[np.isin(data.pairs[tr], data.pairs[te])]
        tr = tr[~np.isin(data.pairs[tr], data.pairs[te])]
        if len(tr) < 50 or len(te) < 20:
            raise ValueError("Insufficient cells after split/pair purge; use a larger cohort")
        assert not set(data.pairs[tr]) & set(data.pairs[te])
        if scheme == "spatial":
            assert not set(regions[tr]) & set(regions[te])
        splits[scheme] = (tr, te, purged)
    return splits, regions
