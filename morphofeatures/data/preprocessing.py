"""Stream aligned raw/instance volumes into ID-indexed, masked MAE crops.

Volume scans use bounded tiles and a disk-backed object index. Extraction
allocates one fixed-size crop at a time; outputs use memmaps or chunked containers.
"""

from __future__ import annotations

import csv
import itertools
import sqlite3
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage

from morphofeatures.artifacts import write_json_atomic
from morphofeatures.data.crop_storage import crop_storage_backend
from morphofeatures.data.crops import _normalize_crop
from morphofeatures.data.volumes import open_volume
from morphofeatures.representations import fingerprint_file


def triple(value, name, *, integer=False, positive=True):
    values = np.asarray(value)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"{name} requires three finite ZYX values")
    if positive and np.any(values <= 0):
        raise ValueError(f"{name} values must be positive")
    if integer and (not np.equal(values, np.floor(values)).all()):
        raise ValueError(f"{name} values must be integers")
    return values.astype(int if integer else float)


def validate_preprocessing(settings):
    crop_storage_backend(settings.get("output_format", "npy"))
    for key in ("raw", "segmentation"):
        if not Path(settings.get(key, "")).exists() or not settings.get(key):
            raise ValueError(f"{key} container does not exist")
    triple(settings.get("crop_shape", [32, 32, 32]), "crop_shape", integer=True)
    triple(settings.get("block_shape", [64, 64, 64]), "block_shape", integer=True)
    spacing = triple(settings.get("spacing_zyx", []), "spacing_zyx")
    origin = triple(settings.get("origin_zyx", [0, 0, 0]), "origin_zyx", positive=False)
    if settings.get("unit") not in {"voxel", "nm", "um", "micrometer"}:
        raise ValueError("Specify coordinate unit: voxel, nm, um, or micrometer")
    for kind in ("raw", "segmentation"):
        if not np.allclose(
            triple(settings.get(kind + "_spacing_zyx", spacing), kind + "_spacing_zyx"), spacing
        ):
            raise ValueError(
                "Raw and segmentation voxel spacing differ; resample onto one grid before extraction"
            )
        if not np.allclose(
            triple(
                settings.get(kind + "_origin_zyx", origin), kind + "_origin_zyx", positive=False
            ),
            origin,
        ):
            raise ValueError("Raw and segmentation origins differ; align volumes before extraction")
    if int(settings.get("min_voxels", 1)) < 1 or int(settings.get("max_objects", 0)) < 0:
        raise ValueError("min_voxels must be positive and max_objects nonnegative")
    if settings.get("oversized", "skip") not in {"skip", "clip"}:
        raise ValueError("oversized must be skip or clip")
    if settings.get("boundary", "pad") not in {"pad", "skip"}:
        raise ValueError("boundary must be pad or skip")
    if settings.get("roi_boundary", "skip") not in {"skip", "allow"}:
        raise ValueError("roi_boundary must be skip or allow")
    if settings.get("center", "bbox") not in {"bbox", "centroid"}:
        raise ValueError("center must be bbox or centroid")
    if not np.isfinite(float(settings.get("background", 0))):
        raise ValueError("background must be finite")
    with (
        open_volume(
            settings["raw"],
            settings.get("raw_key"),
            settings.get("raw_axes", "zyx"),
            settings.get("raw_channel", 0),
        ) as raw,
        open_volume(
            settings["segmentation"],
            settings.get("segmentation_key"),
            settings.get("segmentation_axes", "zyx"),
            settings.get("segmentation_channel", 0),
        ) as segmentation,
    ):
        if raw.shape != segmentation.shape:
            raise ValueError(
                f"Raw/segmentation spatial shapes do not match: {raw.shape} versus {segmentation.shape}"
            )
        if not np.issubdtype(segmentation.dtype, np.integer):
            raise ValueError("Instance segmentation requires an integer dataset")
        roi = settings.get("roi")
        if roi is not None:
            if not isinstance(roi, (list, tuple)) or len(roi) != 2:
                raise ValueError("roi requires [start_zyx, stop_zyx]")
            lower = triple(roi[0], "roi start", integer=True, positive=False)
            upper = triple(roi[1], "roi stop", integer=True)
            if np.any(lower < 0) or np.any(upper > raw.shape) or np.any(upper <= lower):
                raise ValueError("ROI must be within the volume and use half-open ZYX voxel bounds")


def _index_objects(segmentation, roi_start, roi_stop, block_shape, db, progress):
    db.execute(
        "CREATE TABLE objects (id INTEGER PRIMARY KEY, n INTEGER, z0 INTEGER, y0 INTEGER, x0 INTEGER, z1 INTEGER, y1 INTEGER, x1 INTEGER, sz REAL, sy REAL, sx REAL)"
    )
    update = """INSERT INTO objects VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
        n=n+excluded.n,z0=min(z0,excluded.z0),y0=min(y0,excluded.y0),x0=min(x0,excluded.x0),
        z1=max(z1,excluded.z1),y1=max(y1,excluded.y1),x1=max(x1,excluded.x1),
        sz=sz+excluded.sz,sy=sy+excluded.sy,sx=sx+excluded.sx"""
    total = int(np.prod(np.ceil((roi_stop - roi_start) / block_shape)))
    for count, origin in enumerate(
        itertools.product(
            *(range(a, b, step) for a, b, step in zip(roi_start, roi_stop, block_shape))
        ),
        1,
    ):
        origin = np.asarray(origin)
        stop = np.minimum(origin + block_shape, roi_stop)
        labels = segmentation.read(tuple(slice(int(a), int(b)) for a, b in zip(origin, stop)))
        if (
            not np.issubdtype(labels.dtype, np.integer)
            or labels.min() < 0
            or int(labels.max()) > np.iinfo(np.int64).max
        ):
            raise ValueError(
                "Segmentation IDs must be nonnegative integers representable as int64; zero is background"
            )
        ids, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        local = inverse.reshape(labels.shape).astype(np.int32) + 1
        objects = ndimage.find_objects(local)
        rows = []
        for index, (label_id, box) in enumerate(zip(ids, objects)):
            if not label_id:
                continue
            lower = np.array([s.start for s in box])
            upper = np.array([s.stop for s in box])
            positions = np.nonzero(local[box] == index + 1)
            sums = (
                np.array([p.sum(dtype=np.float64) for p in positions])
                + (origin + lower) * counts[index]
            )
            rows.append(
                (
                    int(label_id),
                    int(counts[index]),
                    *(origin + lower).tolist(),
                    *(origin + upper).tolist(),
                    *sums.tolist(),
                )
            )
        db.executemany(update, rows)
        db.commit()
        if progress:
            progress(phase="scan", blocks=count, total_blocks=total)


def _save_crops(staging, destination, count, shape, settings):
    """Consolidate temporary crops one row at a time into the selected storage."""
    output_format = settings.get("output_format", "npy")
    backend = crop_storage_backend(output_format)
    dimensions = (count, *(int(s) for s in shape))
    specifications = {
        "crops": ("crops", np.float32, dimensions),
        "loss_masks": ("masks", np.uint8 if output_format == "n5" else bool, dimensions),
        "label_ids": ("label_ids", np.int64, (count,)),
    }
    data, arrays = {}, {}
    with ExitStack() as stack:
        store = None
        if backend is not None:
            path = destination / f"crops.{output_format}"
            store = stack.enter_context(backend.File(str(path), "x"))
            store.attrs.update(
                {
                    "schema": "morphofeatures.masked_crops.v1",
                    "unit": settings["unit"],
                    "spacing_zyx": list(settings["spacing_zyx"]),
                    "origin_zyx": list(settings.get("origin_zyx", [0, 0, 0])),
                    "normalization": settings.get("normalization", "dtype"),
                    "background": float(settings.get("background", 0)),
                }
            )
        for field, (key, dtype, dims) in specifications.items():
            if store is None:
                path = destination / f"{key}.npy"
                array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=dims)
                stack.callback(array._mmap.close)
                stack.callback(array.flush)
            else:
                chunks = (
                    (1, *(min(s, 64) for s in dims[1:])) if len(dims) > 1 else (min(count, 4096),)
                )
                array = store.create_dataset(
                    key, shape=dims, dtype=dtype, chunks=chunks, compression="gzip"
                )
                array.attrs["axes"] = "nzyx" if len(dims) == 4 else "n"
                data[field + "_key"] = key
            data[field] = str(path)
            arrays[field] = array
        # Write each compressed ID chunk once instead of recompressing it for
        # every object. Pixel memory remains bounded by a single crop.
        for first in range(0, count, 4096):
            stop = min(first + 4096, count)
            label_ids = np.empty(stop - first, dtype=np.int64)
            for index in range(first, stop):
                temporary = staging / f"{index}.npz"
                with np.load(temporary, allow_pickle=False) as patch:
                    arrays["crops"][index] = patch["crop"]
                    arrays["loss_masks"][index] = patch["mask"]
                    label_ids[index - first] = patch["label_id"]
                temporary.unlink()
            arrays["label_ids"][first:stop] = label_ids
    staging.rmdir()
    return data


def preprocess(settings, destination, progress=None):
    validate_preprocessing(settings)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "objects.sqlite").exists() or (destination / "preprocessing.json").exists():
        raise FileExistsError(
            "Preprocessing destination already contains a run; choose a new directory"
        )
    staging = destination / ".patches"
    staging.mkdir(exist_ok=False)
    shape = triple(settings.get("crop_shape", [32, 32, 32]), "crop_shape", integer=True)
    spacing = triple(settings["spacing_zyx"], "spacing_zyx")
    origin = triple(settings.get("origin_zyx", [0, 0, 0]), "origin_zyx", positive=False)
    summary = {"processed": 0, "skipped": 0, "failed": 0, "not_selected": 0}
    fields = ["label_id", "status", "reason", "row", "voxel_count", "truncated", "padded"]
    fields += [
        f"{name}_{axis}"
        for name in (
            "bbox_min",
            "bbox_max",
            "centroid_voxel",
            "crop_start_voxel",
            "crop_center_coordinate",
        )
        for axis in "zyx"
    ]
    with (
        open_volume(
            settings["raw"],
            settings.get("raw_key"),
            settings.get("raw_axes", "zyx"),
            settings.get("raw_channel", 0),
        ) as raw,
        open_volume(
            settings["segmentation"],
            settings.get("segmentation_key"),
            settings.get("segmentation_axes", "zyx"),
            settings.get("segmentation_channel", 0),
        ) as segmentation,
        sqlite3.connect(destination / "objects.sqlite") as db,
        (destination / "objects.tsv").open("w") as manifest,
    ):
        if raw.shape != segmentation.shape:
            raise ValueError(
                f"Raw/segmentation spatial shapes do not match: {raw.shape} versus {segmentation.shape}"
            )
        roi = settings.get("roi") or [[0, 0, 0], list(raw.shape)]
        start = triple(roi[0], "roi start", integer=True, positive=False)
        stop = triple(roi[1], "roi stop", integer=True)
        if np.any(start < 0) or np.any(stop > raw.shape) or np.any(stop <= start):
            raise ValueError(
                "ROI must be nonempty, within the volume, and use half-open ZYX voxel bounds"
            )
        _index_objects(
            segmentation,
            start,
            stop,
            triple(settings.get("block_shape", [64, 64, 64]), "block_shape", integer=True),
            db,
            progress,
        )
        writer = csv.DictWriter(manifest, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        selection = set(int(v) for v in settings.get("object_ids", []) or [])
        for row in db.execute("SELECT * FROM objects ORDER BY id"):
            label_id, count = row[:2]
            lower, upper = np.array(row[2:5]), np.array(row[5:8])
            centroid = np.array(row[8:11]) / count
            item = {"label_id": label_id, "voxel_count": count, "status": "skipped", "reason": ""}
            for name, vector in (
                ("bbox_min", lower),
                ("bbox_max", upper),
                ("centroid_voxel", centroid),
            ):
                item.update({f"{name}_{axis}": v for axis, v in zip("zyx", vector)})
            reason = None
            if selection and label_id not in selection:
                reason = "not selected"
            elif settings.get("max_objects", 0) and summary["processed"] >= settings["max_objects"]:
                reason = "object limit"
            elif count < settings.get("min_voxels", 1):
                reason = "below min_voxels"
            touches_roi = np.any(lower == start) or np.any(upper == stop)
            if not reason and touches_roi and settings.get("roi_boundary", "skip") == "skip":
                reason = "touches ROI boundary; extent may be incomplete"
            center = (
                np.floor(centroid).astype(int)
                if settings.get("center", "bbox") == "centroid"
                else (lower + upper) // 2
            )
            crop_start = center - shape // 2
            crop_stop = crop_start + shape
            clipped = np.any(lower < crop_start) or np.any(upper > crop_stop)
            padded = np.any(crop_start < start) or np.any(crop_stop > stop)
            if not reason and clipped and settings.get("oversized", "skip") == "skip":
                reason = "object does not fit crop; increase crop_shape or choose oversized=clip"
            if not reason and padded and settings.get("boundary", "pad") == "skip":
                reason = "crop extends beyond ROI boundary"
            item.update(truncated=bool(clipped or touches_roi), padded=bool(padded))
            item.update({f"crop_start_voxel_{axis}": int(v) for axis, v in zip("zyx", crop_start)})
            item.update(
                {
                    f"crop_center_coordinate_{axis}": float(v)
                    for axis, v in zip("zyx", origin + center * spacing)
                }
            )
            if reason:
                item["reason"] = reason
                summary[
                    "not_selected" if reason in {"not selected", "object limit"} else "skipped"
                ] += 1
            else:
                try:
                    read_start, read_stop = (
                        np.maximum(crop_start, start),
                        np.minimum(crop_stop, stop),
                    )
                    source = tuple(slice(int(a), int(b)) for a, b in zip(read_start, read_stop))
                    target = tuple(
                        slice(int(a), int(b))
                        for a, b in zip(read_start - crop_start, read_stop - crop_start)
                    )
                    intensity = np.zeros(tuple(shape), dtype=raw.dtype)
                    mask = np.zeros(tuple(shape), dtype=bool)
                    intensity[target] = raw.read(source)
                    mask[target] = segmentation.read(source) == label_id
                    if not mask.any() or not np.isfinite(intensity).all():
                        raise ValueError("empty crop mask or nonfinite raw values")
                    patch = _normalize_crop(intensity, mask, settings.get("normalization", "dtype"))
                    patch[~mask] = float(settings.get("background", 0))
                    np.savez(
                        staging / f"{summary['processed']}.npz",
                        crop=patch,
                        mask=mask,
                        label_id=np.int64(label_id),
                    )
                    item.update(status="processed", row=summary["processed"])
                    summary["processed"] += 1
                except (ValueError, OSError) as error:
                    item.update(status="failed", reason=str(error))
                    summary["failed"] += 1
            writer.writerow(item)
            if progress:
                progress(phase="extract", label_id=label_id, **summary)
        if selection:
            present = {
                int(row[0])
                for row in db.execute("SELECT id FROM objects")
                if int(row[0]) in selection
            }
            for label_id in sorted(selection - present):
                writer.writerow(
                    {"label_id": label_id, "status": "skipped", "reason": "ID absent from ROI"}
                )
                summary["skipped"] += 1
    metadata = {
        "schema": "morphofeatures.preprocessing.v1",
        "settings": settings,
        "summary": summary,
        "raw": fingerprint_file(settings["raw"]),
        "segmentation": fingerprint_file(settings["segmentation"]),
        "axes": "zyx",
        "unit": settings["unit"],
        "spacing_zyx": spacing.tolist(),
        "origin_zyx": origin.tolist(),
        "coordinate_convention": "integer voxel indices refer to voxel centers; physical = origin + index * spacing; bounding boxes are half-open",
        "mask": "exact instance equality, independent of raw zero intensity",
        "selection_scope": "ROI",
        "crop_shape": shape.tolist(),
        "output_format": settings.get("output_format", "npy"),
    }
    write_json_atomic(destination / "preprocessing.json", metadata)
    if not summary["processed"]:
        raise ValueError(
            f"No objects processed; see {destination / 'objects.tsv'} for skipped/failed reasons"
        )
    data = _save_crops(staging, destination, summary["processed"], shape, settings)
    metadata["arrays"] = data
    write_json_atomic(destination / "preprocessing.json", metadata)
    config = {
        "seed": 42,
        "device": "auto",
        "data": {
            "source": "masked_crops",
            **data,
            "preprocessing": str(destination / "preprocessing.json"),
        },
        "mae": {
            "architecture_version": "position-aware-3d-v2",
            "input_shape": shape.tolist(),
            "patch_size": [4 if int(s) % 4 == 0 else 1 for s in shape],
            "embedding_dim": 128,
            "encoder_heads": 4,
        },
        "training": {"epochs": 2, "batch_size": 2, "learning_rate": 0.001},
        "inference": {"batch_size": 4},
    }
    path = destination / "mae_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path
