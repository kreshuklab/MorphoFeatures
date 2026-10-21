"""Explicit object identity and numeric relation tables used by spatial probes."""

import pickle
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd

from morphofeatures.data.contracts import EmbeddingTable
from morphofeatures.workflow_validation import FieldError


def integer_ids(values, name="ID"):
    result = []
    for value in values:
        try:
            number = Decimal(str(value))
        except InvalidOperation as error:
            raise ValueError(f"{name} contains a nonnumeric ID: {value!r}") from error
        if (
            not number.is_finite()
            or number != number.to_integral_value()
            or not 0 <= number <= np.iinfo(np.int64).max
        ):
            raise ValueError(f"{name} must contain nonnegative exact int64 IDs; got {value!r}")
        result.append(int(number))
    return np.asarray(result, dtype=np.int64)


def read_table(path):
    return pd.read_csv(path, sep="," if Path(path).suffix == ".csv" else "\t", dtype=str)


def cell_identity(table, settings):
    """Return cell-indexed vectors and a native-ID bridge, without silent guessing."""
    namespace = settings.get("id_namespace", "cell")
    native = np.asarray(table.label_ids, dtype=np.int64)
    if namespace == "cell":
        bridge = pd.DataFrame({"label_id": native, "native_id": native})
    elif namespace == "nucleus":
        path = settings.get("id_mapping")
        if not path:
            raise FieldError(
                "Nucleus IDs require a cell-to-nucleus ID mapping table; numeric overlap is not correspondence.",
                "id_mapping",
                "id_namespace",
            )
        mapping = read_table(path)
        cell = settings.get("cell_column", "label_id")
        nucleus = settings.get("nucleus_column", "nucleus_id")
        if cell not in mapping or nucleus not in mapping:
            raise FieldError(
                f"Mapping needs columns {cell!r} and {nucleus!r}.",
                "id_mapping",
                "cell_column",
                "nucleus_column",
            )
        bridge = pd.DataFrame(
            {"label_id": integer_ids(mapping[cell]), "native_id": integer_ids(mapping[nucleus])}
        )
        bridge = bridge[(bridge.label_id > 0) & (bridge.native_id > 0)]
        if bridge.label_id.duplicated().any() or bridge.native_id.duplicated().any():
            raise FieldError(
                "The positive cell/nucleus mapping must be one-to-one. Resolve multinucleate or duplicate entries explicitly.",
                "id_mapping",
            )
        missing = sorted(set(native) - set(bridge.native_id))
        if missing:
            raise FieldError(
                f"Mapping has no cell ID for {len(missing)} embedded nuclei (examples: {missing[:8]}). Select the matching mapping or cohort.",
                "id_mapping",
            )
        bridge = bridge.set_index("native_id").loc[native].reset_index()
        bridge["label_id"] = bridge.label_id.astype(np.int64)
    else:
        raise FieldError("Choose cell or nucleus as the embedding ID namespace.", "id_namespace")
    if np.any(bridge.label_id.to_numpy() <= 0):
        raise FieldError("Background ID 0 cannot represent a cell.", "embedding")
    return EmbeddingTable(bridge.label_id.to_numpy(), table.features), bridge


class _NumericUnpickler(pickle.Unpickler):
    """Historical NumPy candidate dictionaries only; no general pickle execution."""

    def find_class(self, module, name):
        try:
            from numpy._core import multiarray
        except ImportError:  # NumPy 1.x historical pickles.
            from numpy.core import multiarray

        allowed = {("numpy", "dtype"): np.dtype, ("numpy", "ndarray"): np.ndarray}
        for package in ("numpy.core.multiarray", "numpy._core.multiarray"):
            allowed[(package, "_reconstruct")] = multiarray._reconstruct
            allowed[(package, "scalar")] = multiarray.scalar
        if (module, name) not in allowed:
            raise ValueError(
                "Only numeric historical candidate dictionaries are supported; use a TSV edge table."
            )
        return allowed[(module, name)]


def read_relations(path, *, bilateral=False):
    """Read directed candidate pairs or deduplicated undirected spatial edges."""
    path = Path(path)
    if path.suffix == ".pkl":
        if not bilateral:
            raise FieldError(
                "Spatial adjacency must be a source_id/target_id TSV/CSV, not a bilateral pickle.",
                "adjacency_table",
            )
        if path.stat().st_size > 64 * 1024**2:
            raise FieldError(
                "Candidate pickle exceeds 64 MiB; supply a TSV/CSV edge table.", "candidates"
            )
        with path.open("rb") as stream:
            values = _NumericUnpickler(stream).load()
        if not isinstance(values, dict):
            raise ValueError("Candidates must be an ID-to-ID-list dictionary")
        pairs = [
            (int(integer_ids([k])[0]), int(v)) for k, vs in values.items() for v in integer_ids(vs)
        ]
        frame = pd.DataFrame(pairs, columns=["source_id", "target_id"])
    else:
        frame = read_table(path)
        required = {"source_id", "target_id"}
        if not required <= set(frame):
            raise FieldError(
                "Relation table needs source_id and target_id columns (one edge per row).",
                "candidates" if bilateral else "adjacency_table",
            )
        frame = pd.DataFrame({k: integer_ids(frame[k], k) for k in ["source_id", "target_id"]})
    if (frame == 0).any().any() or (frame.source_id == frame.target_id).any():
        raise ValueError("Relations cannot contain background ID 0 or self-pairs")
    pairs = frame.to_numpy(dtype=np.int64).reshape(-1, 2)
    if not bilateral:
        pairs = np.sort(pairs, axis=1)
    return np.unique(pairs, axis=0)
