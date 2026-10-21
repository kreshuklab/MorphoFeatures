"""Native MoBIE projects: portable ROI images or published PlatyBrowser references.

No viewer SDK or server is required. Images retain instance IDs; saved analysis
columns are joined by exact IDs, never by row order. All volume work is chunked.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import shutil
import sqlite3
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from morphofeatures.analysis.annotations import read_id_table
from morphofeatures.analysis.mesh_inspection import mesh_source_defaults, object_id_mapping
from morphofeatures.analysis.object_inspection import input_config_for_result
from morphofeatures.artifacts import write_json_atomic
from morphofeatures.configuration_editor import load_document
from morphofeatures.data.preprocessing import _index_objects, triple
from morphofeatures.data.volumes import open_volume

SCHEMA = "morphofeatures.mobie_export.v1"
SPATIAL_COLUMNS = [p + a for p in ("anchor_", "bb_min_", "bb_max_") for a in "xyz"]
SOURCE_FIELDS = {
    "resample_raw",
    *(
        k + suffix
        for k in ("raw", "segmentation")
        for suffix in (
            "",
            "_key",
            "_axes",
            "_channel",
            "_timepoint",
            "_spacing_zyx",
            "_origin_zyx",
            "_unit",
        )
    ),
    "spacing_zyx",
    "origin_zyx",
    "unit",
    "roi",
    "remote_options",
    "id_mapping",
    "platybrowser",
}


def _result(path):
    path = Path(path).expanduser().resolve()
    metadata = json.loads(path.read_text())
    if metadata.get("schema") == "morphofeatures.joint_analysis.v1":
        raise ValueError(
            "Choose a per-dataset analysis.json from this joint result, with that dataset's raw/segmentation source. Joint IDs are internal and cannot identify objects in a single segmentation."
        )
    if metadata.get("schema") not in {
        "morphofeatures.analysis.v1",
        "morphofeatures.comparison.v1",
        "morphofeatures.bilateral.v1",
    }:
        raise ValueError(
            "MoBIE export needs a saved analysis.json, comparison.json or bilateral.json"
        )
    return path, metadata


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Saved result paths must stay inside their analysis directory")
    return path


def export_source_defaults(config):
    """Recover original spatial data, not the stack of isolated training crops."""
    data = config.get("data", {})
    if data.get("source") in {"physical_volume", "grouped_crops"}:
        data = {**data["volume"], "resample_raw": True}
    settings = {k: deepcopy(v) for k, v in data.items() if k in SOURCE_FIELDS}
    prep = data.get("preprocessing")
    if prep and Path(prep).is_file():
        settings.update(
            {
                k: v
                for k, v in json.loads(Path(prep).read_text())["settings"].items()
                if k in SOURCE_FIELDS
            }
        )
    settings.update({k: v for k, v in mesh_source_defaults(config).items() if k in SOURCE_FIELDS})
    if config.get("inspection", {}).get("platybrowser"):
        settings["platybrowser"] = deepcopy(config["inspection"]["platybrowser"])
    return settings


def resolve_mobie_export(settings, *, require_analysis=True):
    settings = deepcopy(settings)
    metadata = {}
    if settings.get("analysis"):
        _, metadata = _result(settings["analysis"])
    elif require_analysis:
        raise ValueError("Choose a saved analysis/comparison result or link an earlier analysis")
    if settings.get("input_config"):
        config = load_document(settings["input_config"])
    else:
        name = next(iter(metadata.get("representations", {})), None)
        config = input_config_for_result(metadata, name)
    defaults = export_source_defaults(config)
    settings = {**defaults, **settings}
    if config:
        settings["source_config"] = config
    mode = settings.setdefault("mode", "portable")
    if mode not in {"portable", "platybrowser"}:
        raise ValueError("MoBIE export mode must be portable or platybrowser")
    name = settings.setdefault("dataset_name", "morphofeatures")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Dataset name must use letters, digits, underscores, dots or hyphens")
    for key, default in (("max_voxels", 256**3), ("max_artifact_gb", 4.0)):
        value = settings.setdefault(key, default)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    limits = np.asarray(settings.get("contrast_limits", [0, 255]), dtype=float)
    if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
        raise ValueError("Raw contrast limits need two finite, increasing numbers")
    if mode == "platybrowser":
        from morphofeatures.analysis.platybrowser import resolve_platybrowser

        if not settings.get("platybrowser"):
            raise ValueError("Choose a PlatyBrowser project, revision, raw and segmentation source")
        resolved = resolve_platybrowser({"inspection": {"platybrowser": settings["platybrowser"]}})
        settings["platybrowser"] = resolved["inspection"]["platybrowser"]
        settings["published_config"] = resolved
        if not settings.get("id_mapping"):
            settings["id_mapping"] = resolved["data"].get("id_mapping")
    else:
        if not settings.get("segmentation"):
            raise ValueError(
                "Choose an original instance segmentation volume for the MoBIE project"
            )
        roi = np.asarray(settings.get("roi"))
        if (
            roi.shape != (2, 3)
            or not np.issubdtype(roi.dtype, np.number)
            or not np.isfinite(roi).all()
            or np.any(roi != np.floor(roi))
            or np.any(roi[0] < 0)
            or np.any(roi[1] <= roi[0])
        ):
            raise ValueError("Portable export requires ROI start/stop in integer Z, Y, X voxels")
        if int(np.prod(roi[1] - roi[0], dtype=object)) > settings["max_voxels"]:
            raise ValueError(
                "Export ROI exceeds max_voxels; reduce the region or raise the reviewed limit"
            )
        settings["roi"] = roi.astype(np.int64).tolist()
        for key, default, positive in (
            ("spacing_zyx", [1, 1, 1], True),
            ("origin_zyx", [0, 0, 0], False),
        ):
            settings[key] = triple(settings.get(key, default), key, positive=positive).tolist()
        if settings.get("unit", "voxel") not in {"voxel", "nm", "um"}:
            raise ValueError("Use voxel, nm or um as the export coordinate unit")
        block = triple(settings.get("block_shape", [32, 32, 32]), "block_shape", integer=True)
        if np.any(block > 128):
            raise ValueError("Export blocks must be at most 128 voxels per side")
        settings["block_shape"] = block.tolist()
        _validate_spatial_sources(settings)
    if settings.get("id_mapping"):
        object_id_mapping(settings)
    for path in settings.get("attachments", []):
        if not Path(path).is_file():
            raise ValueError(f"Additional artifact is not an existing file: {path}")
    return settings


def _validate_spatial_sources(settings):
    """Metadata-only review: alignment and bounds before allocating output files."""
    with ExitStack() as stack:
        seg = stack.enter_context(_volume(settings, "segmentation"))
        if np.any(np.asarray(settings["roi"])[1] > seg.shape):
            raise ValueError("Export ROI extends outside the segmentation")
        if not np.issubdtype(seg.dtype, np.integer):
            raise ValueError("MoBIE needs integer instance IDs, not a foreground probability mask")
        for kind in ("raw", "segmentation"):
            if not settings.get(kind):
                continue
            for key in ("spacing_zyx", "origin_zyx"):
                if kind == "raw" and settings.get("resample_raw"):
                    continue
                if not np.allclose(settings.get(kind + "_" + key, settings[key]), settings[key]):
                    raise ValueError(
                        "Raw/segmentation spatial metadata differ; resample to one grid first"
                    )
            if settings.get(kind + "_unit", settings.get("unit", "voxel")) != settings.get(
                "unit", "voxel"
            ):
                raise ValueError("Raw/segmentation coordinate units differ")
        if settings.get("raw"):
            raw = stack.enter_context(_volume(settings, "raw"))
            if raw.shape != seg.shape:
                raise ValueError(
                    "Raw and segmentation must share a spatial grid for portable export"
                )
            if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
                raw.dtype, np.complexfloating
            ):
                raise ValueError("Raw images must contain real numeric intensities")


@contextmanager
def _volume(settings, kind):
    with open_volume(
        settings[kind],
        settings.get(kind + "_key"),
        settings.get(kind + "_axes", "zyx"),
        settings.get(kind + "_channel", 0),
        timepoint=settings.get(kind + "_timepoint", 0),
        remote_options=settings.get("remote_options"),
    ) as volume:
        if kind != "raw" or not settings.get("resample_raw"):
            yield volume
            return
        from morphofeatures.data.dino_views import sample_volume

        spacing, origin = np.array(settings["spacing_zyx"]), np.array(settings["origin_zyx"])
        native_spacing = np.array(settings.get("raw_spacing_zyx", spacing))
        native_origin = np.array(settings.get("raw_origin_zyx", origin))
        lower, upper = np.array(settings["roi"])
        if np.any(origin + (lower - 0.5) * spacing < native_origin - native_spacing / 2) or np.any(
            origin + (upper - 0.5) * spacing
            > native_origin + (np.array(volume.shape) - 0.5) * native_spacing
        ):
            raise ValueError("The raw image does not cover the physical export ROI")
        with _volume(settings, "segmentation") as seg:
            shape = seg.shape

        class ResampledRaw:
            dtype = np.dtype("float32")

            def __init__(self):
                self.shape = shape

            def read(self, slices):
                dimensions = tuple(s.stop - s.start for s in slices)
                indices = np.indices(dimensions).reshape(3, -1).T + [s.start for s in slices]
                points = origin + indices * spacing
                sigma = np.maximum(0, (spacing / native_spacing - 1) / 2)
                return sample_volume(
                    volume, points, native_spacing, native_origin, sigma=sigma
                ).reshape(dimensions)

        yield ResampledRaw()


def _blocks(shape, chunks):
    for start in itertools.product(*(range(0, int(s), int(c)) for s, c in zip(shape, chunks))):
        yield tuple(slice(a, min(a + int(c), int(s))) for a, c, s in zip(start, chunks, shape))


def _ome_array(path, name, shape, dtype, chunks, spacing, origin, unit):
    import zarr
    from numcodecs import GZip

    kwargs = {"zarr_format": 2} if int(zarr.__version__.split(".")[0]) >= 3 else {}
    group = zarr.open_group(str(path), mode="w", **kwargs)
    axes = [{"name": a, "type": "space"} for a in "zyx"]
    if unit != "voxel":
        for axis in axes:
            axis["unit"] = {"um": "micrometer", "nm": "nanometer"}[unit]
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": name,
            "axes": axes,
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": spacing.tolist()},
                        {"type": "translation", "translation": origin.tolist()},
                    ],
                }
            ],
        }
    ]
    create = group.create_array if kwargs else group.create_dataset
    chunk_keys = (
        {"chunk_key_encoding": {"name": "v2", "configuration": {"separator": "/"}}}
        if kwargs
        else {"dimension_separator": "/"}
    )
    array = create(
        "0",
        shape=tuple(map(int, shape)),
        dtype=dtype,
        chunks=tuple(map(int, np.minimum(chunks, shape))),
        compressor=GZip(level=1),
        **chunk_keys,
    )
    return array


def _portable(settings, root, progress):
    """Copy only the reviewed region; index the copied labels for exact coverage."""
    start, stop = np.asarray(settings["roi"], dtype=np.int64)
    shape, chunks = stop - start, np.asarray(settings["block_shape"])
    spacing, origin = np.asarray(settings["spacing_zyx"]), np.asarray(settings["origin_zyx"])
    unit = settings.get("unit", "voxel")
    images = root / "images"
    images.mkdir()
    sources = {}
    with ExitStack() as stack:
        seg = stack.enter_context(_volume(settings, "segmentation"))
        if np.any(stop > seg.shape):
            raise ValueError("Export ROI extends outside the segmentation")
        if not np.issubdtype(seg.dtype, np.integer):
            raise ValueError("MoBIE needs integer instance IDs, not a foreground probability mask")
        volumes = {"objects": seg}
        if settings.get("raw"):
            raw = stack.enter_context(_volume(settings, "raw"))
            if raw.shape != seg.shape:
                raise ValueError(
                    "Raw and segmentation must share a spatial grid for portable export"
                )
            if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
                raw.dtype, np.complexfloating
            ):
                raise ValueError("Raw images must contain real numeric intensities")
            for key, default in (("spacing_zyx", spacing), ("origin_zyx", origin)):
                for kind in ("raw", "segmentation"):
                    if kind == "raw" and settings.get("resample_raw"):
                        continue
                    if not np.allclose(settings.get(kind + "_" + key, default), default):
                        raise ValueError(
                            "Raw/segmentation spatial metadata differ; resample to one grid first"
                        )
            if any(settings.get(k + "_unit", unit) != unit for k in ("raw", "segmentation")):
                raise ValueError("Raw/segmentation coordinate units differ")
            volumes["raw"] = raw
        for name, volume in volumes.items():
            # BDV/MoBIE renders raw data as real-valued pixels; float32 handles
            # uncommon source integer widths without integer contrast overflow.
            dtype = np.int64 if name == "objects" else np.float32
            output = _ome_array(
                images / f"{name}.ome.zarr",
                name,
                shape,
                dtype,
                chunks,
                spacing,
                origin + start * spacing,
                unit,
            )
            for count, slices in enumerate(_blocks(shape, chunks), 1):
                absolute = tuple(
                    slice(s.start + int(a), s.stop + int(a)) for s, a in zip(slices, start)
                )
                block = volume.read(absolute)
                if name == "objects" and (
                    int(block.min()) < 0 or int(block.max()) > np.iinfo(np.int64).max
                ):
                    raise ValueError("Instance IDs must be nonnegative int64; zero is background")
                output[slices] = block.astype(dtype)
                if progress:
                    progress(phase="copy", source=name, blocks=count)
            sources[name] = {
                "segmentation" if name == "objects" else "image": {
                    "imageData": {"ome.zarr": {"relativePath": f"images/{name}.ome.zarr"}}
                }
            }
    # Scan exported pixels, so bounds and coverage describe exactly what is shared.
    with tempfile.TemporaryDirectory(prefix="mobie-index-") as tmp:
        with (
            sqlite3.connect(str(Path(tmp) / "objects.sqlite")) as db,
            open_volume(images / "objects.ome.zarr", "0") as seg,
        ):
            _index_objects(seg, np.zeros(3, dtype=int), shape, chunks, db, progress)
            rows = db.execute("SELECT * FROM objects ORDER BY id").fetchall()
    if not rows:
        raise ValueError("No nonzero segmentation objects occur in the export region")
    records = []
    for label_id, n, z0, y0, x0, z1, y1, x1, sz, sy, sx in rows:
        lo, hi = np.array([z0, y0, x0]), np.array([z1, y1, x1])
        record = {
            "label_id": label_id,
            "voxel_count": n,
            "touches_export_boundary": bool(np.any(lo == 0) or np.any(hi == shape)),
        }
        for prefix, coordinate in (
            ("anchor_", np.array([sz, sy, sx]) / n),
            ("bb_min_", lo),
            ("bb_max_", hi - 1),
        ):
            physical = origin + (start + coordinate) * spacing
            record.update({prefix + axis: float(value) for axis, value in zip("zyx", physical)})
        records.append(record)
    table = pd.DataFrame(records)
    return (
        sources,
        table,
        {
            "mode": "portable",
            "roi_zyx": settings["roi"],
            "shape_zyx": shape.tolist(),
            "unit": unit,
            "spacing_zyx": spacing.tolist(),
            "origin_zyx": (origin + start * spacing).tolist(),
            "boundary_objects": int(table.touches_export_boundary.sum()),
            "coverage_basis": "actual exported segmentation voxels",
            "raw_resampled_to_segmentation_grid": bool(settings.get("resample_raw")),
        },
    )


def _published(settings, root):
    from morphofeatures.analysis.platybrowser import _linked_file
    from morphofeatures.data.remote_n5 import HttpCache

    config = settings["published_config"]
    provenance = config["inspection"]["mesh"]["remote_provenance"]
    cache = HttpCache(settings["platybrowser"])
    sources = {}
    (root / "images").mkdir()
    for name, kind in (("raw", "image"), ("objects", "segmentation")):
        source = provenance["raw" if name == "raw" else "segmentation"]
        _, content = _linked_file(cache, source["xml_url"])
        xml = ET.fromstring(content)
        xml.find("./SequenceDescription/ViewSetups/ViewSetup/name").text = name
        ET.ElementTree(xml).write(
            root / "images" / f"{name}.xml", encoding="utf-8", xml_declaration=True
        )
        sources[name] = {kind: {"imageData": {"bdv.n5.s3": {"relativePath": f"images/{name}.xml"}}}}
    table = read_id_table(config["data"]["object_table"])
    if not set(SPATIAL_COLUMNS) <= set(table):
        raise ValueError(
            "Published segmentation table lacks physical anchors/bounds required by MoBIE"
        )
    coordinates = table[SPATIAL_COLUMNS].to_numpy(float)
    if not np.isfinite(coordinates).all():
        raise ValueError("Published anchors/bounds contain invalid physical coordinates")
    return (
        sources,
        table[table.label_id != 0].copy(),
        {
            "mode": "platybrowser",
            "published_source": provenance,
            "coverage_basis": "published object catalogue (no whole-volume voxel scan)",
        },
    )


def _analysis_tables(path, metadata, settings=None):
    if metadata.get("schema") == "morphofeatures.bilateral.v1":
        from morphofeatures.analysis.bilateral_export import bilateral_entries

        yield from bilateral_entries(path, metadata, settings or {})
        return
    entries = metadata.get("representations") or {"Embedding": metadata}
    for name, entry in entries.items():
        folder = _inside(path.parent, entry.get("directory", "."))
        frame = read_id_table(folder / "coordinates.tsv")
        if (folder / "object_labels.tsv").is_file():
            labels = read_id_table(folder / "object_labels.tsv")
            frame = labels.merge(
                frame.drop(columns=[c for c in labels if c != "label_id" and c in frame]),
                on="label_id",
                how="left",
                validate="one_to_one",
            )
        if (frame.label_id == 0).any():
            raise ValueError(
                "Embedding label ID zero is background and cannot be exported as an object"
            )
        yield str(name), folder, frame.drop(columns="representation", errors="ignore")


def _view(
    column="label_id", *, raw=True, contrast=(0, 255), axes=None, table=None, title="Objects"
):
    displays = []
    if raw:
        displays.append(
            {
                "imageDisplay": {
                    "name": "Raw",
                    "sources": ["raw"],
                    "color": "white",
                    "contrastLimits": list(contrast),
                    "opacity": 1.0,
                }
            }
        )
    display = {
        "name": "Objects",
        "sources": ["objects"],
        "opacity": 0.55,
        "lut": "glasbey",
        "colorByColumn": column,
        "randomColorSeed": 42,
        "showTable": True,
        "showSelectedSegmentsIn3d": False,
        "showScatterPlot": bool(axes),
    }
    if axes:
        display["scatterPlotAxes"] = axes
    if table:
        display["additionalTables"] = [table]
    displays.append({"segmentationDisplay": display})
    return {
        "uiSelectionGroup": "MorphoFeatures",
        "isExclusive": True,
        "description": title,
        "sourceDisplays": displays,
    }


def _add_analyses(settings, path, metadata, default, directory, views, progress):
    mapping = object_id_mapping(settings)
    spatial_ids = set(map(int, default.label_id))
    reports = []
    contrast = settings.get("contrast_limits", [0, 255])
    raw = settings["mode"] == "platybrowser" or bool(settings.get("raw"))
    for index, (name, folder, frame) in enumerate(_analysis_tables(path, metadata, settings)):
        embedding_ids = frame.label_id.tolist()
        segments = [
            mapping.get(int(v)) if settings.get("id_mapping") else int(v) for v in embedding_ids
        ]
        matched = np.asarray([v is not None and v in spatial_ids for v in segments])
        excluded = pd.DataFrame(
            {
                "embedding_label_id": embedding_ids,
                "segmentation_id": pd.array(segments, dtype="Int64"),
                "reason": [
                    "unmapped ID" if v is None else "absent from exported segmentation/catalogue"
                    for v in segments
                ],
            }
        )[~matched]
        prefix = f"mf{index}"
        excluded.to_csv(
            directory.parent.parent / "misc" / f"{prefix}-excluded-ids.tsv", sep="\t", index=False
        )
        if not matched.any():
            raise ValueError(
                f"No IDs from {name} match the exported segmentation; check ROI and cell/nucleus ID mapping"
            )
        if len(excluded) and settings.get("require_all_ids", False):
            raise ValueError(
                f"{name}: {len(excluded)} embedding IDs are missing; see excluded-ID table or choose a larger ROI/correct mapping"
            )
        selected = frame.loc[matched].copy()
        selected.insert(0, "embedding_label_id", selected.pop("label_id").to_numpy())
        selected.insert(
            0,
            "label_id",
            np.array([s for s, keep in zip(segments, matched) if keep], dtype=np.int64),
        )
        if selected.label_id.duplicated().any():
            raise ValueError("Multiple embedding objects map to the same segmentation instance")
        is_bilateral = metadata.get("schema") == "morphofeatures.bilateral.v1"
        if is_bilateral:
            selected["best_partner_segmentation_id"] = pd.array(
                [
                    None
                    if pd.isna(v)
                    else mapping.get(int(v))
                    if settings.get("id_mapping")
                    else int(v)
                    for v in selected.best_partner_id
                ],
                dtype="Int64",
            )
            from morphofeatures.analysis.bilateral_export import choose_pairs

            pair_frame = selected.rename(
                columns={"label_id": "segmentation_id", "embedding_label_id": "label_id"}
            )
            pairs = choose_pairs(pair_frame, settings)
        columns = {}
        for i, col in enumerate(selected.columns):
            if col == "label_id":
                continue
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(col)) or "column"
            candidate = f"{prefix}_{safe}"
            if candidate in columns.values() or candidate in default:
                candidate += f"_{i}"
            columns[col] = candidate
        selected = selected.rename(columns=columns)
        # MoBIE loads additional tables by object ID. Spatially present objects
        # outside the embedding subset remain visible and explicitly unlabelled.
        merged = default[["label_id"]].merge(
            selected, on="label_id", how="left", validate="one_to_one"
        )
        id_col = columns["embedding_label_id"]
        # Avoid pandas left-join promotion through float64 for large IDs.
        exact_ids = dict(zip(selected.label_id.map(int), selected[id_col].map(int)))
        merged[id_col] = pd.array([exact_ids.get(int(v)) for v in merged.label_id], dtype="Int64")
        for key in ("best_partner_id", "native_id", "best_partner_segmentation_id"):
            if key in columns:
                lookup = dict(zip(selected.label_id.map(int), selected[columns[key]]))
                merged[columns[key]] = pd.array(
                    [lookup.get(int(v)) for v in merged.label_id], dtype="Int64"
                )
        for category in ("known_label", "predicted_label", "cluster"):
            col = columns.get(category)
            if col:
                merged[col] = merged[col].astype("string").fillna("unlabeled")
        table_name = f"{prefix}.tsv"
        merged.to_csv(directory / table_name, sep="\t", index=False, na_rep="NaN")
        if is_bilateral:
            rank_view = _view(
                columns["rank"],
                raw=raw,
                contrast=contrast,
                table=table_name,
                title=f"{name}: bilateral candidate retrieval rank (lower is better); not verified symmetry.",
            )
            display = rank_view["sourceDisplays"][-1]["segmentationDisplay"]
            display.update(
                lut="viridis",
                valueLimits=[
                    1,
                    max(2, float(frame["rank"].max())) if frame["rank"].notna().any() else 2,
                ],
            )
            views[f"{prefix}-bilateral-rank"] = rank_view
            for row in pairs.itertuples():
                if (
                    pd.isna(row.best_partner_segmentation_id)
                    or int(row.best_partner_segmentation_id) not in spatial_ids
                ):
                    continue
                view = _view(
                    "label_id",
                    raw=raw,
                    contrast=contrast,
                    table=table_name,
                    title=f"{name}: query cell {row.label_id}, retrieved candidate {row.best_partner_id}, rank {row.rank}. Candidate correspondence is a hypothesis.",
                )
                view["sourceDisplays"][-1]["segmentationDisplay"].update(
                    selectedSegmentIds=[
                        f"objects;0;{int(row.segmentation_id)}",
                        f"objects;0;{int(row.best_partner_segmentation_id)}",
                    ],
                    showSelectedSegmentsIn3d=True,
                    opacityNotSelected=0.05,
                )
                views[f"{prefix}-pair-{int(row.label_id)}"] = view
        for category in ("cluster", "known_label", "predicted_label"):
            if category not in columns:
                continue
            for method in ("spatial", "umap", "pca"):
                axes = [columns.get(method + "_1"), columns.get(method + "_2")]
                if method != "spatial" and not all(axes):
                    continue
                key = f"{prefix}-{category}-{method}"
                views[key] = _view(
                    columns[category],
                    raw=raw,
                    contrast=contrast,
                    axes=axes if method != "spatial" else None,
                    table=table_name,
                    title=f"{name}: {category.replace('_', ' ')} / {method}. Predictions are estimates; clusters are exploratory groups.",
                )
        if settings.get("include_feature_table", False):
            from morphofeatures.data.io import load_embeddings

            features = load_embeddings(folder / "embeddings.npz").sorted()
            lookup = {int(v): i for i, v in enumerate(features.label_ids)}
            rows = [lookup[int(v)] for v in selected[id_col]]
            feature_frame = pd.DataFrame(
                features.features[rows],
                columns=[f"{prefix}_feature_{i:04d}" for i in range(features.features.shape[1])],
            )
            feature_frame.insert(0, "label_id", selected.label_id.to_numpy())
            feature_name = f"{prefix}-features.tsv"
            feature_frame.to_csv(directory / feature_name, sep="\t", index=False)
        report = {
            "name": name,
            "table": table_name,
            "columns": columns,
            "embedding_objects": len(frame),
            "matched_objects": int(matched.sum()),
            "excluded_objects": len(excluded),
            "excluded_ids": f"misc/{prefix}-excluded-ids.tsv",
        }
        reports.append(report)
        if progress:
            progress(phase="tables", **{k: v for k, v in report.items() if k not in {"columns"}})
    return reports


def _copy_evidence(settings, path, root):
    """Copy saved evidence with a declared byte limit; do not chase arbitrary links."""
    files = [(path, root / "misc" / "analysis-report.json")]
    if settings.get("id_mapping"):
        mapping = Path(settings["id_mapping"])
        files.append((mapping, root / "misc" / ("id-mapping" + mapping.suffix)))
    if settings.get("include_artifacts", True):
        for source in sorted(path.parent.rglob("*")):
            if source.is_symlink():
                continue
            if source.is_file() and source.suffix.lower() not in {".zip", ".tmp", ".pt", ".pth"}:
                if source.resolve().is_relative_to(root.resolve()):
                    continue
                files.append((source, root / "misc" / "analysis" / source.relative_to(path.parent)))
    for index, source in enumerate(settings.get("attachments", [])):
        source = Path(source)
        files.append((source, root / "misc" / "attachments" / f"{index:02d}-{source.name}"))
    total = sum(p.stat().st_size for p, _ in files)
    if total > float(settings["max_artifact_gb"]) * 1024**3:
        raise ValueError(
            "Saved evidence exceeds max_artifact_gb; increase the limit or disable copying artifacts"
        )
    copied = []
    for source, target in files:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                digest.update(block)
        copied.append(
            {
                "path": target.relative_to(root).as_posix(),
                "bytes": target.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return copied


def export_mobie(settings, destination, progress=None):
    """Create a fresh project and return an export report for the Results browser."""
    from morphofeatures.config import repository_root
    from morphofeatures.experiments import collect_provenance

    settings = resolve_mobie_export(settings)
    path, metadata = _result(settings["analysis"])
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    project = destination / "project"
    if project.exists() or (destination / "mobie_export.json").exists():
        raise FileExistsError("MoBIE export already exists; choose a new run directory")
    # The analysis tree is archived below; placing the export inside it would
    # recurse and could copy unrelated exports from previous attempts.
    if destination.is_relative_to(path.parent):
        raise ValueError("Choose an export directory outside the saved analysis directory")
    project.mkdir()
    dataset = project / settings["dataset_name"]
    directory = dataset / "tables" / "objects"
    directory.mkdir(parents=True)
    (dataset / "misc").mkdir()
    sources, default, spatial = (
        _published(settings, dataset)
        if settings["mode"] == "platybrowser"
        else _portable(settings, dataset, progress)
    )
    default.to_csv(directory / "default.tsv", sep="\t", index=False)
    sources["objects"]["segmentation"]["tableData"] = {"tsv": {"relativePath": "tables/objects"}}
    views = {
        "default": _view(raw="raw" in sources, contrast=settings.get("contrast_limits", [0, 255]))
    }
    reports = _add_analyses(settings, path, metadata, default, directory, views, progress)
    evidence = _copy_evidence(settings, path, dataset)
    write_json_atomic(dataset / "misc" / "export-settings.json", settings)
    write_json_atomic(
        dataset / "dataset.json",
        {
            "is2D": False,
            "description": settings.get(
                "description", "MorphoFeatures embeddings and object labels"
            ),
            "sources": sources,
            "views": views,
        },
    )
    write_json_atomic(
        project / "project.json",
        {
            "datasets": [settings["dataset_name"]],
            "defaultDataset": settings["dataset_name"],
            "specVersion": "0.3.0",
            "description": settings.get("description", "MorphoFeatures export"),
            "references": ["https://mobie.github.io/specs/mobie_spec.html"],
        },
    )
    is_bilateral = metadata.get("schema") == "morphofeatures.bilateral.v1"
    interpretation = (
        "Bilateral ranks measure how close the best supplied candidate is in feature space. Pair views select query and candidate objects in 3D; candidates are hypotheses, not verified symmetry or anatomical connections."
        if is_bilateral
        else "Known labels, classifier predictions and embedding clusters are separate views. ID agreement alone does not prove spatial or biological correspondence."
    )
    report = {
        "schema": SCHEMA,
        "project": "project",
        "dataset": settings["dataset_name"],
        "segmentation_objects": len(default),
        "spatial": spatial,
        "representations": reports,
        "evidence": evidence,
        "exporter": collect_provenance(repository_root()),
        "views": list(views),
        "interpretation": interpretation,
    }
    write_json_atomic(dataset / "misc" / "export-manifest.json", report)
    summary = "\n".join(
        f"- {r['name']}: {r['matched_objects']} matched; {r['excluded_objects']} excluded (see {r['excluded_ids']})."
        for r in reports
    )
    (project / "README.txt").write_text(
        "MorphoFeatures → MoBIE\n\n"
        "Open this folder in Fiji using MoBIE's Open MoBIE Project (Expert Mode) command.\n"
        f"Project location: folder containing project.json. Dataset: {settings['dataset_name']}.\n"
        f"Image data: {'Remote (S3); internet required' if settings['mode'] == 'platybrowser' else 'Local (OME-Zarr); all exported image pixels included'}.\n"
        "Choose a view in the MorphoFeatures menu; mf0, mf1, ... identify the representations below.\n"
        + (
            "Choose mfN-bilateral-rank to color objects by retrieval rank (lower is better).\n"
            "Choose mfN-pair-ID to select that query and its best candidate with 3D display enabled.\n"
            "The table retains cell IDs and mapped segmentation IDs. A cell can occur in several pairs.\n\n"
            if is_bilateral
            else "UMAP/PCA views use saved coordinates; no model or clustering is rerun. Select a table row\n"
            "or scatter point to locate its object. Enable the viewer's 3D display for selected objects.\n\n"
        )
        + summary
        + "\n\n"
        + interpretation
        + "\nExcluded IDs have no spatial overlay.\n"
        + "Portable ROI boundaries may cut objects; see touches_export_boundary in default.tsv.\n"
        "Copy the whole project folder, not only its JSON or tables. misc contains saved evidence and provenance;\n"
        "MoBIE does not display model checkpoints, reports or NPZ feature files directly.\n"
        "Documentation: https://mobie.github.io/tutorials/installation.html and MorphoFeatures docs/analysis_and_mobie.md\n",
        encoding="utf-8",
    )
    if settings.get("make_zip", True):
        with zipfile.ZipFile(
            destination / "project.zip", "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as archive:
            for file in sorted(project.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(destination))
        report["archive"] = "project.zip"
    write_json_atomic(destination / "mobie_export.json", report)
    return destination / "mobie_export.json"
