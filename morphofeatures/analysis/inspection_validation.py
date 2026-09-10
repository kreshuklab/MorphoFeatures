"""Cancellable, cached identity checks over the actual inspection data."""

import csv
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from functools import lru_cache
from pathlib import Path

import numpy as np

from morphofeatures.analysis.annotations import read_id_table
from morphofeatures.analysis.mesh_inspection import object_id_mapping
from morphofeatures.artifacts import write_json_atomic
from morphofeatures.data.crop_storage import load_crop_array, open_crop_array
from morphofeatures.data.preprocessing import _index_objects, triple
from morphofeatures.data.remote_n5 import is_remote_url
from morphofeatures.data.volumes import open_volume


def inspection_check_key(embedding_ids, config, settings):
    """Key by IDs, source metadata, paths and an explicit in-place edit version."""
    paths = []
    for section in (config.get("data", {}), settings):
        for value in section.values():
            if not isinstance(value, str) or not value or is_remote_url(value):
                continue
            path = Path(value).expanduser()
            if path.exists():
                stat = path.stat()
                paths.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    if settings.get("segmentation") and not is_remote_url(settings["segmentation"]):
        dataset = Path(settings["segmentation"]) / (settings.get("segmentation_key") or "")
        for filename in ("attributes.json", "zarr.json", ".zarray"):
            metadata = dataset / filename
            if metadata.is_file():
                paths.append((str(metadata), metadata.stat().st_mtime_ns))
    payload = json.dumps([config.get("data", {}), settings, paths], sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode())
    digest.update(np.sort(np.asarray(embedding_ids, dtype=np.int64)).tobytes())
    return digest.hexdigest()


def compare_object_ids(embedding_ids, source_ids, mapping=None):
    embedded = set(map(int, embedding_ids))
    # Instance scans exclude background zero; per-object row/file IDs may include 0.
    source = set(map(int, source_ids))
    mapping = {value: value for value in embedded} if mapping is None else mapping
    missing = sorted(
        value for value in embedded if value not in mapping or mapping[value] not in source
    )
    expected = {mapping[value] for value in embedded if value in mapping}
    extra = sorted(source - expected)
    return {
        "embedded_objects": len(embedded),
        "source_objects": len(source),
        "matched_objects": len(embedded) - len(missing),
        "missing_embedding_ids": missing,
        "extra_source_ids": extra,
        "count_equal": len(embedded) == len(source),
        "state": "mismatch" if missing else ("subset" if extra else "match"),
        "interpretation": "Exact numeric ID comparison. Equal IDs/counts do not prove spatial alignment or biological identity. Extra source objects can be expected when only a subset was embedded.",
    }


class CheckCancelled(Exception):
    pass


def check_inspection_ids(embedding_ids, config, settings, directory, *, progress=None, cancel=None):
    """Read masks in bounded blocks. Never count gray levels as cell identities."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def tick(**values):
        if cancel is not None and cancel.is_set():
            raise CheckCancelled
        if progress:
            progress(**values)

    tick(phase="Opening inspection data", blocks=0, total_blocks=1)
    source = settings.get("source", "segmentation")
    report = {"schema": "morphofeatures.inspection_check.v1", "source": source}
    if source == "segmentation" and is_remote_url(settings.get("segmentation")):
        from morphofeatures.analysis.platybrowser import validate_index_reference

        validate_index_reference(settings)
        if settings.get("label_kind", "instances") != "instances":
            return {
                **report,
                "state": "not_comparable",
                "interpretation": "Foreground scores have no object IDs; select an instance segmentation",
            }
        if not settings.get("object_index"):
            raise ValueError(
                "Remote ID checks require an object table; full remote scans are not started automatically"
            )
        tick(phase="Checking the published object table")
        index = read_id_table(settings["object_index"])
        mapping = object_id_mapping(settings) if settings.get("id_mapping") else None
        source_ids = index.label_id
        if settings.get("index_ids", "segmentation") != "segmentation":
            if "segmentation_id" not in index:
                raise ValueError("Remote object bounds must identify the stored segmentation IDs")
            source_ids = read_id_table(settings["object_index"], "segmentation_id").segmentation_id
            if mapping is None:
                mapping = dict(zip(index.label_id.map(int), source_ids.map(int)))
        with open_volume(
            settings["segmentation"],
            settings.get("segmentation_key"),
            settings.get("segmentation_axes", "zyx"),
            settings.get("segmentation_channel", 0),
            remote_options=settings.get("remote_options"),
        ) as volume:
            if not np.issubdtype(volume.dtype, np.integer):
                raise ValueError("Instance segmentation must have an integer dtype")
            lower = index[["bbox_min_" + a for a in "zyx"]].to_numpy()
            upper = index[["bbox_max_" + a for a in "zyx"]].to_numpy()
            if (
                not np.isfinite(lower).all()
                or not np.isfinite(upper).all()
                or np.any(lower != np.floor(lower))
                or np.any(upper != np.floor(upper))
                or np.any(lower < 0)
                or np.any(upper > volume.shape)
                or np.any(upper <= lower)
            ):
                raise ValueError("Object-table bounds are invalid for the remote segmentation grid")
            report.update(shape_zyx=list(volume.shape), dtype=str(volume.dtype))
        report.update(compare_object_ids(embedding_ids, source_ids, mapping))
        report.update(
            validation_basis="object_table",
            voxel_scan=False,
            object_index=str(settings["object_index"]),
            index_ids=settings.get("index_ids", "segmentation"),
            id_mapping=settings.get("id_mapping"),
            published_source=settings.get("remote_provenance"),
            interpretation="IDs checked against the object table, not a full voxel scan. Selected labels are verified when meshes load; matching table entries alone do not prove voxel presence or spatial correspondence.",
        )
        tick(phase="Completed", blocks=1, total_blocks=1)
        return report
    if source == "segmentation":
        kind = settings.get("label_kind", "instances")
        with open_volume(
            settings["segmentation"],
            settings.get("segmentation_key"),
            settings.get("segmentation_axes", "zyx"),
            settings.get("segmentation_channel", 0),
        ) as volume:
            report.update(
                shape_zyx=list(volume.shape),
                dtype=str(volume.dtype),
                segmentation=str(settings["segmentation"]),
                dataset_key=settings.get("segmentation_key"),
            )
            if (
                kind != "instances"
                or volume.dtype == np.dtype(bool)
                or not np.issubdtype(volume.dtype, np.integer)
            ):
                sample = volume.read(tuple(slice(0, min(16, n)) for n in volume.shape))
                return {
                    **report,
                    "state": "not_comparable",
                    "embedded_objects": len(set(embedding_ids)),
                    "source_objects": None,
                    "sample_range": [float(sample.min()), float(sample.max())],
                    "interpretation": "Foreground/binary/score volumes have no per-object IDs. Use an integer instance segmentation, or one prepared binary crop per row with matching data.label_ids. Connected-component numbers would not establish correspondence.",
                }
            chunks = getattr(volume.dataset, "chunks", None)
            spatial_chunks = (
                [chunks[volume.axes.index(a)] for a in "zyx"] if chunks else [64, 64, 64]
            )
            blocks = triple(
                settings.get("check_block_shape", np.minimum(spatial_chunks, 128)),
                "check_block_shape",
                integer=True,
            )
            if np.any(blocks > 256):
                raise ValueError("ID-check blocks may not exceed 256 voxels per side")

            class CheckedVolume:
                def read(self, slices):
                    tick(phase="Scanning instance labels")
                    return volume.read(slices)

            with tempfile.TemporaryDirectory(prefix="scan-", dir=directory) as temporary:
                # sqlite3's transaction context does not close its file handle.
                # Close explicitly before TemporaryDirectory cleanup on NFS.
                with closing(sqlite3.connect(str(Path(temporary) / "index.sqlite"))) as database:
                    _index_objects(
                        CheckedVolume(),
                        np.zeros(3, dtype=int),
                        np.asarray(volume.shape),
                        blocks,
                        database,
                        tick,
                    )
                    ids = [
                        int(row[0])
                        for row in database.execute("SELECT id FROM objects ORDER BY id")
                    ]
                    index_path = Path(temporary) / "objects.tsv"
                    with index_path.open("w") as stream:
                        writer = csv.writer(stream, delimiter="\t")
                        writer.writerow(
                            ["label_id", "voxel_count"]
                            + [f"bbox_{edge}_{axis}" for edge in ("min", "max") for axis in "zyx"]
                        )
                        writer.writerows(
                            database.execute(
                                "SELECT id,n,z0,y0,x0,z1,y1,x1 FROM objects ORDER BY id"
                            )
                        )
                    tick(phase="Comparing object IDs")
                    os.replace(index_path, directory / "objects.tsv")
            report["object_index"] = str((directory / "objects.tsv").resolve())
            report["index_ids"] = "segmentation"
            if len(ids) == 1 and ids[0] in {1, 255} and len(set(embedding_ids)) > 1:
                return {
                    **report,
                    "state": "not_comparable",
                    "embedded_objects": len(set(embedding_ids)),
                    "source_objects": None,
                    "interpretation": "This volume is binary-like (one foreground value, 1 or 255) but multiple objects were embedded. Foreground values cannot distinguish those objects; a matching instance segmentation is needed.",
                }
        mapping = object_id_mapping(settings) if settings.get("id_mapping") else None
        if settings.get("id_mapping"):
            report["id_mapping"] = str(settings["id_mapping"])
        if settings.get("object_index") and not mapping:
            index = read_id_table(settings["object_index"])
            if "segmentation_id" in index:
                mapped = read_id_table(settings["object_index"], "segmentation_id")
                mapping = dict(zip(index.label_id.map(int), mapped.segmentation_id.map(int)))
        report.update(compare_object_ids(embedding_ids, ids, mapping))
    elif source == "prepared":
        data = config.get("data", {})
        ids = load_crop_array(data, "label_ids")
        if (
            ids.ndim != 1
            or not np.issubdtype(ids.dtype, np.integer)
            or len(np.unique(ids)) != len(ids)
        ):
            raise ValueError("Prepared masks require unique integer data.label_ids")
        with open_crop_array(data, "loss_masks") as masks:
            if (
                masks.ndim not in (4, 5)
                or masks.shape[0] != len(ids)
                or (masks.ndim == 5 and masks.shape[1] != 1)
            ):
                raise ValueError(
                    "Prepared masks must have shape (N,Z,Y,X) or (N,1,Z,Y,X), with one data.label_ids entry per row. A whole-volume foreground file is not this format"
                )
            report["mask_shape"] = list(masks.shape)
        report.update(compare_object_ids(embedding_ids, ids))
        report["pixel_check"] = (
            "Array shape and ID vector checked; mask pixels are not exhaustively scanned. Selected meshes validate their foreground when loaded."
        )
    elif source == "files":
        if settings.get("mesh_table"):
            table_path = Path(settings["mesh_table"])
            table = read_id_table(table_path)
            if "mesh_path" not in table:
                raise ValueError("Mesh table requires label_id and mesh_path")
            ids = [
                int(row.label_id)
                for row in table.itertuples()
                if (table_path.parent / str(row.mesh_path)).is_file()
            ]
        else:
            directory_path = Path(settings["mesh_directory"])
            ids = [
                int(path.stem)
                for path in directory_path.iterdir()
                if path.is_file()
                and path.suffix.lower() in {".ply", ".obj", ".glb", ".stl"}
                and re.fullmatch(r"\d+", path.stem)
            ]
        report.update(compare_object_ids(embedding_ids, ids))
    else:
        raise ValueError("Unsupported inspection source")
    tick(phase="Completed", blocks=1, total_blocks=1)
    return report


class InspectionCheck:
    def __init__(self, key):
        self.key = key
        self.cancel = threading.Event()
        self.lock = threading.Lock()
        self.status = {"phase": "Queued", "blocks": 0, "total_blocks": 1}
        self.future = None

    def update(self, **values):
        with self.lock:
            self.status.update(values)

    def snapshot(self):
        with self.lock:
            return dict(self.status)


@lru_cache(maxsize=1)
def _executor():
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="inspection-check")


_active_checks = {}
_active_lock = threading.Lock()


def start_inspection_check(embedding_ids, config, settings, root, *, force=False):
    """Return immediately; workers never call Streamlit or touch source data."""
    key = inspection_check_key(embedding_ids, config, settings)
    task = InspectionCheck(key)
    directory = Path(root) / key

    def run():
        path = directory / "check.json"
        if path.is_file() and not force:
            try:
                report = json.loads(path.read_text())
                if isinstance(report, dict) and report.get("state") in {
                    "match",
                    "subset",
                    "mismatch",
                    "not_comparable",
                }:
                    if not report.get("object_index") or Path(report["object_index"]).is_file():
                        return {**report, "cached": True}
            except (OSError, ValueError, TypeError):
                pass  # An incomplete or stale cache can be rebuilt from the source.
        try:
            report = check_inspection_ids(
                embedding_ids, config, settings, directory, progress=task.update, cancel=task.cancel
            )
            report["cache_key"] = key
            write_json_atomic(path, report)
            return report
        except CheckCancelled:
            return {"state": "cancelled"}
        except Exception as error:
            return {"state": "error", "error": str(error)}

    identity = (str(Path(root).resolve()), key)
    with _active_lock:
        existing = _active_checks.get(identity)
        if existing is not None and not existing.future.done():
            return existing
        task.future = _executor().submit(run)
        # Retain only running jobs; completed results live in the on-disk cache.
        for old in [item for item, value in _active_checks.items() if value.future.done()]:
            del _active_checks[old]
        _active_checks[identity] = task
    return task
