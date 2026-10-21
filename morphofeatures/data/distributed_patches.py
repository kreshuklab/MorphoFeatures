"""Reusable physical fine patches, grouped by exact native object ID.

``layout: distributed_patches`` in a preprocessing stage selects this path.
``patches.sampling=physical_resampled`` (default) uses ``tile_size_um`` as physical
width and ``crop_shape`` as the saved ZYX grid. ``native_voxels`` instead cuts
``crop_shape`` raw voxels directly, with no raw interpolation or antialias filter.
A non-overlapping lattice is anchored at the foreground centroid,
bounds center, or bounds minimum. Eligibility integrates foreground volume over
each tile; bounded farthest-point selection spreads patches across the object.
This is sparse texture sampling, not a whole-cell or exhaustive-voxel encoding.

Raw images are interpolated with antialias filtering when downsampled; instance
IDs use nearest-neighbor sampling. All patches of an object share its capped,
deterministic 1st/99th-percentile normalization. Background is zero with an
explicit mask, so valid zero-valued foreground is never mistaken for background.
The generated data_config.yaml is for frozen DINO, not the single-crop MAE.
See docs/workspace_workflows.md for controls, array contracts and limitations.
"""

from __future__ import annotations

import csv
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import yaml

from morphofeatures.artifacts import write_json_atomic
from morphofeatures.data.crop_storage import crop_storage_backend, open_crop_array
from morphofeatures.data.dino_views import (
    PhysicalObjects,
    distributed_indices,
    eligible_tiles,
    index_source,
    inspection_config,
    physical_tiles,
    sample_volume,
    source_identity,
    tile_grid_origin,
    validate_source,
)
from morphofeatures.data.object_eligibility import ObjectEligibilityError
from morphofeatures.data.preprocessing import _save_crops, triple
from morphofeatures.workflow_validation import FieldError, at_fields

PATCH_VERSION = "distributed-physical-patches-v1"
PATCH_DEFAULTS = {
    "sampling": "physical_resampled",
    "tile_size_um": 1.0,
    "max_tiles": 64,
    "exhaustive": False,
    "min_foreground_fraction": 0.5,
    "tile_anchor": "centroid",
    "percentiles": [1.0, 99.0],
    "max_candidate_tiles": 250000,
}


def validate_patch_preparation(settings):
    """Read metadata only; raise errors at the responsible Prepare data fields."""
    crop_storage_backend(settings.get("output_format", "npy"))
    options = {**PATCH_DEFAULTS, **settings.get("patches", {})}
    native = options["sampling"] == "native_voxels"
    if options["sampling"] not in {"physical_resampled", "native_voxels"}:
        raise FieldError("Choose physical_resampled or native_voxels", "patches.sampling")
    if not native:
        with at_fields("patches.tile_size_um"):
            width = np.broadcast_to(np.asarray(options["tile_size_um"], dtype=float), (3,))
            triple(width, "Patch width (µm)")
            if not np.allclose(width, width[0]):
                raise ValueError(
                    "Use a cubic physical patch: one width in µm, or three equal widths"
                )
    with at_fields("crop_shape"):
        shape = triple(settings.get("crop_shape", [32] * 3), "Saved patch shape", integer=True)
        if native and (np.any(shape > 128) or np.any(shape < 1)):
            raise ValueError("Use 1–128 native raw voxels along each patch axis")
        if not native and (not np.all(shape == shape[0]) or np.prod(shape, dtype=object) > 128**3):
            raise ValueError("Use a cubic saved grid of at most 128³ voxels for fine DINO patches")
    if options["tile_anchor"] not in {"centroid", "bbox_center", "bbox_min"}:
        raise FieldError("Choose centroid, bbox_center or bbox_min", "patches.tile_anchor")
    fraction = options["min_foreground_fraction"]
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise FieldError(
            "Enter a fraction from 0 to 1; 0.5 means at least half foreground",
            "patches.min_foreground_fraction",
        )
    for key, value, minimum, maximum in (
        ("patches.max_tiles", options["max_tiles"], 1, 1000000),
        ("patches.max_candidate_tiles", options["max_candidate_tiles"], 1, 1000000),
        ("max_object_voxels", settings.get("max_object_voxels", 256**3), 1, 1000000000),
        ("max_scan_voxels", settings.get("max_scan_voxels", 128**3), 1, 256**3),
        ("min_voxels", settings.get("min_voxels", 1), 1, 2**63 - 1),
        ("max_objects", settings.get("max_objects", 0), 0, 2**63 - 1),
        ("seed", settings.get("seed", 42), 0, 2**32 - 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise FieldError(f"Use an integer between {minimum} and {maximum}", key)
    if not isinstance(options["exhaustive"], bool):
        raise FieldError("Use true or false", "patches.exhaustive")
    with at_fields("patches.percentiles"):
        low, high = options["percentiles"]
        if not 0 <= low < high <= 100:
            raise ValueError("Percentiles must increase within 0–100, for example [1, 99]")
    source = validate_source(settings)
    if native:
        # Ignore a previously entered physical width; serialize the width actually used.
        options["tile_size_um"] = (shape * np.asarray(source["raw_spacing_zyx"])).tolist()
    if source.get("object_ids") == []:
        source.pop("object_ids")
    if source["segmentation"].startswith(("http:", "https:")) and not source.get("object_index"):
        roi = np.asarray(source["roi"], dtype=np.int64)
        if np.prod(roi[1] - roi[0], dtype=object) > settings.get("max_scan_voxels", 128**3):
            raise FieldError(
                "Remote ROI exceeds the scan budget. Reduce the segmentation ROI or choose a coarser segmentation level.",
                "roi",
                "max_scan_voxels",
            )
    return source, options, shape


def patch_regions(reader, obj, options, shape):
    """Physical tiles, or a lattice snapped to exact raw voxel edges for native cuts."""
    if options.get("sampling", "physical_resampled") != "native_voxels":
        return physical_tiles(obj, options)
    spacing = np.asarray(reader.source["raw_spacing_zyx"])
    origin = np.asarray(reader.source["raw_origin_zyx"])
    lower, upper = obj["bounds"]
    width = shape * spacing
    lower, center = tile_grid_origin(lower, upper, width, obj["points"], options["tile_anchor"])
    # Tie-breaking is reproducible. A corner-anchored grid starts at/below the bounds;
    # centered grids snap the intended anchor to the nearest native voxel edge.
    index = (lower - origin) / spacing + 0.5
    start = np.floor(index + 1e-8) if center is None else np.floor(index + 0.5)
    lower = origin + (start - 0.5) * spacing
    lower += np.minimum(0, np.floor((obj["bounds"][0] - lower) / width + 1e-8)) * width
    cells = eligible_tiles(
        obj["points"],
        obj["spacing"],
        lower,
        upper,
        width,
        options["min_foreground_fraction"],
        options["max_candidate_tiles"],
    )
    centers = lower + (cells + 0.5) * width
    chosen = (
        np.arange(len(cells))
        if options["exhaustive"]
        else distributed_indices(centers, options["max_tiles"], center=center)
    )
    return (
        [(lower + cells[i] * width, lower + (cells[i] + 1) * width, None) for i in chosen],
        len(cells),
        lower,
    )


def read_patch(reader, obj, lo, hi, shape):
    if reader.views.get("sampling", "physical_resampled") != "native_voxels":
        return resample_patch(reader, obj, lo, hi, shape)
    return resample_patch(reader, obj, lo, hi, shape, native=True)


def resample_patch(reader, obj, lo, hi, shape, *, native=False):
    """Sample one physical cube, a Z plane at a time; return normalized pixels/mask."""
    source = reader.source
    raw_spacing = np.asarray(source["raw_spacing_zyx"])
    pixel = (hi - lo) / shape
    sigma = np.maximum(0, (pixel / raw_spacing - 1) / 2)
    image, mask = np.empty(tuple(shape), dtype=np.float32), np.empty(tuple(shape), dtype=bool)
    if native:
        # Read exact source samples, without passing raw intensities through interpolation.
        start = np.rint((lo - np.asarray(source["raw_origin_zyx"])) / raw_spacing + 0.5).astype(int)
        stop = start + shape
        read_lo, read_hi = np.maximum(start, 0), np.minimum(stop, reader.raw.shape)
        image.fill(0)
        if np.all(read_hi > read_lo):
            image[
                tuple(slice(int(a), int(b)) for a, b in zip(read_lo - start, read_hi - start))
            ] = reader.raw.read(tuple(slice(int(a), int(b)) for a, b in zip(read_lo, read_hi)))
    yy, xx = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    points = np.empty((yy.size, 3), dtype=np.float64)
    points[:, 1] = lo[1] + (yy.ravel() + 0.5) * pixel[1]
    points[:, 2] = lo[2] + (xx.ravel() + 0.5) * pixel[2]
    for z in range(shape[0]):
        points[:, 0] = lo[0] + (z + 0.5) * pixel[0]
        if not native:
            image[z] = sample_volume(
                reader.raw, points, raw_spacing, np.asarray(source["raw_origin_zyx"]), sigma=sigma
            ).reshape(tuple(shape[1:]))
        mask[z] = (
            sample_volume(
                reader.seg,
                points,
                np.asarray(source["segmentation_spacing_zyx"]),
                np.asarray(source["segmentation_origin_zyx"]),
                label=int(obj["row"]["segmentation_id"]),
            ).reshape(tuple(shape[1:]))
            > 0.5
        )
        if source.get("roi") is not None:
            roi = np.asarray(source["roi"])
            spacing, origin = (
                np.asarray(source["segmentation_spacing_zyx"]),
                np.asarray(source["segmentation_origin_zyx"]),
            )
            inside = np.all(
                (points >= origin + (roi[0] - 0.5) * spacing)
                & (points < origin + (roi[1] - 0.5) * spacing),
                axis=1,
            )
            mask[z] &= inside.reshape(tuple(shape[1:]))
    if not np.isfinite(image).all():
        raise ValueError("Patch contains nonfinite raw intensities")
    low, high = obj["normalization"]
    image = np.clip((image - low) / (high - low), 0, 1)
    image[~mask] = 0
    return image, mask


def prepare_patches(settings, destination, progress=None):
    source, options, shape = validate_patch_preparation(settings)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(
        (destination / name).exists() for name in (".patches", "source.json", "preprocessing.json")
    ):
        raise FileExistsError(
            "Preparation destination already contains a run; choose a new directory"
        )
    staging = destination / ".patches"
    staging.mkdir()
    objects = index_source(source, destination)
    object_ids, offsets, positions, records = [], [0], [], []
    count = 0
    with (
        PhysicalObjects(source, objects, options) as reader,
        (destination / "patches.tsv").open("w", newline="") as stream,
    ):
        fields = [
            "row",
            "label_id",
            "segmentation_id",
            "status",
            "reason",
            "sampled_foreground_fraction",
            "normalization_low",
            "normalization_high",
            "coverage_scope",
        ]
        fields += [
            prefix + axis for prefix in ("lower_um_", "upper_um_", "center_um_") for axis in "zyx"
        ]
        manifest = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        manifest.writeheader()
        for row in objects.to_dict("records"):
            label_id = int(row["label_id"])
            record = {
                "label_id": label_id,
                "eligible_patches": 0,
                "selected_patches": 0,
                "saved_patches": 0,
                "status": "skipped",
                "reason": "",
            }
            first = count
            try:
                obj = reader.prepare(row)
                record["coverage_scope"] = obj.get("coverage_scope", "complete_object")
                if len(obj["points"]) < settings.get("min_voxels", 1):
                    raise ValueError("Object has fewer foreground voxels than min_voxels")
                regions, eligible, _ = patch_regions(reader, obj, options, shape)
                record.update(eligible_patches=eligible, selected_patches=len(regions))
                if not regions:
                    raise ValueError(
                        "No patches meet the foreground cutoff. Lower patches.min_foreground_fraction, reduce patches.tile_size_um or change patches.tile_anchor."
                    )
                for lo, hi, _ in regions:
                    entry = {
                        "label_id": label_id,
                        "segmentation_id": int(row["segmentation_id"]),
                        "status": "saved",
                        "coverage_scope": obj.get("coverage_scope", "complete_object"),
                        "normalization_low": obj["normalization"][0],
                        "normalization_high": obj["normalization"][1],
                    }
                    entry.update(
                        {
                            prefix + axis: float(v)
                            for prefix, values in (
                                ("lower_um_", lo),
                                ("upper_um_", hi),
                                ("center_um_", (lo + hi) / 2),
                            )
                            for axis, v in zip("zyx", values)
                        }
                    )
                    try:
                        image, mask = read_patch(reader, obj, lo, hi, shape)
                        entry["sampled_foreground_fraction"] = float(mask.mean())
                        if not mask.any():
                            raise ValueError(
                                "No foreground survives the saved voxel grid; increase crop_shape or use a smaller physical patch"
                            )
                        np.savez(
                            staging / f"{count}.npz",
                            crop=image,
                            mask=mask,
                            label_id=np.int64(label_id),
                        )
                        entry["row"] = count
                        positions.append((lo + hi) / 2)
                        count += 1
                    except (ValueError, OSError) as error:
                        entry.update(status="excluded", reason=str(error))
                        if source.get("on_object_error", "skip") == "error":
                            manifest.writerow(entry)
                            raise
                    manifest.writerow(entry)
                if count == first:
                    raise ValueError(
                        "All selected patches failed; see patches.tsv for individual reasons"
                    )
                record.update(status="saved", saved_patches=count - first)
                if count - first < len(regions):
                    record["reason"] = "Some selected patches were excluded; see patches.tsv"
                object_ids.append(label_id)
                offsets.append(count)
            except (ValueError, OSError) as error:
                if (
                    isinstance(error, ObjectEligibilityError)
                    and error.record["decision"] == "error"
                ) or (
                    not isinstance(error, ObjectEligibilityError)
                    and source.get("on_object_error", "skip") == "error"
                ):
                    raise
                record["reason"] = str(error)
            records.append(record)
            if progress:
                progress(
                    phase="patches", objects=len(records), total_objects=len(objects), patches=count
                )
    import pandas as pd

    pd.DataFrame(records).to_csv(destination / "coverage.tsv", sep="\t", index=False)
    metadata = {
        "schema": "morphofeatures.distributed_patches.v1",
        "algorithm": "distributed-native-patches-v1"
        if options["sampling"] == "native_voxels"
        else PATCH_VERSION,
        "layout": "distributed_patches",
        "source": source,
        "dataset_identity": source_identity(source),
        "patches": options,
        "crop_shape": shape.tolist(),
        "normalization": "object_foreground_percentile",
        "saved_voxel_size_um": source["raw_spacing_zyx"]
        if options["sampling"] == "native_voxels"
        else (np.broadcast_to(options["tile_size_um"], (3,)) / shape).tolist(),
        "coordinate_convention": "source origins are voxel centers; patch bounds are voxel edges; centers are ZYX µm",
        "summary": {
            "selected_objects": len(objects),
            "saved_objects": len(object_ids),
            "skipped_objects": len(objects) - len(object_ids),
            "saved_patches": count,
        },
        "settings": settings,
    }
    write_json_atomic(destination / "preprocessing.json", metadata)
    if not count:
        raise FieldError(
            f"No distributed patches were saved. Inspect {destination / 'coverage.tsv'} for per-object reasons.",
            "patches",
            "roi",
        )
    data = _save_crops(
        staging,
        destination,
        count,
        shape,
        {
            **settings,
            "unit": "um",
            "spacing_zyx": metadata["saved_voxel_size_um"],
            "origin_zyx": [0, 0, 0],
            "normalization": "object_foreground_percentile",
            "background": 0,
        },
        grouped={
            "label_ids": np.asarray(object_ids, dtype=np.int64),
            "patch_offsets": np.asarray(offsets, dtype=np.int64),
            "positions_um": np.asarray(positions, dtype=np.float64),
        },
    )
    data.update(
        source="grouped_crops",
        preprocessing=str(destination / "preprocessing.json"),
        volume=source,
        patch_recipe={
            "algorithm": "distributed-native-patches-v1"
            if options["sampling"] == "native_voxels"
            else PATCH_VERSION,
            **options,
            "crop_shape": shape.tolist(),
            "saved_voxel_size_um": metadata["saved_voxel_size_um"],
            "normalization": "object_foreground_percentile",
        },
    )
    inspection = inspection_config(source, destination / "objects.tsv")["inspection"]
    inspection["mesh"].update(object_index=str(destination / "objects.tsv"), index_ids="embedding")
    config = {
        "seed": settings.get("seed", 42),
        "device": "auto",
        "data": data,
        "inspection": inspection,
    }
    path = destination / "data_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def preview_patches(settings, *, object_id=None, index_directory=None, progress=None, cancel=None):
    """Preview actual saved voxels without leaving files or allowing an unbounded scan."""
    import tempfile

    source, options, shape = validate_patch_preparation(settings)
    if not source.get("object_index") and index_directory is not None:
        from morphofeatures.data.patch_preview import cached_bounds

        source["object_index"] = str(
            cached_bounds(source, index_directory, progress=progress, cancel=cancel)
        )
    if not source.get("object_index"):
        roi = np.asarray(source.get("roi") or [[0, 0, 0], source["segmentation_shape_zyx"]])
        if np.prod(roi[1] - roi[0], dtype=object) > 128**3:
            raise FieldError(
                "Preview needs an object index or an ROI of at most 128³ segmentation voxels. The worker can scan a larger local selection.",
                "roi",
                "object_index",
            )
    if object_id is not None:
        source.update(object_ids=[int(object_id)], max_objects=0)
    with tempfile.TemporaryDirectory(prefix="morphofeatures-patch-preview-") as folder:
        objects = index_source(source, folder)
        from morphofeatures.data.object_eligibility import assess_objects, eligibility_message

        eligibility, summary = assess_objects(source, objects)
        allowed = eligibility[eligibility.decision.isin(["eligible", "partial"])].label_id
        if not len(allowed):
            raise FieldError(
                eligibility_message(summary).split("See Prepare data")[0]
                + "Prepare data → Fine patch contrast and safety limits → Maximum object bounding-box voxels controls the mask read. Boundary objects need a larger ROI where tissue is available, or a different preview ID.",
                "max_object_voxels",
                "roi",
                "object_ids",
            )
        objects = objects[objects.label_id.isin(allowed)]
        chosen = int(np.random.default_rng(source.get("seed", 42)).integers(len(objects)))
        if progress:
            progress(
                phase="Reading one object's mask and selecting patches", blocks=0, total_blocks=1
            )
        with PhysicalObjects(source, objects, options) as reader:
            obj = reader.prepare(objects.iloc[chosen].to_dict())
            if len(obj["points"]) < source.get("min_voxels", 1):
                raise FieldError(
                    "Preview object has fewer foreground voxels than min_voxels", "min_voxels"
                )
            regions, eligible, _ = patch_regions(reader, obj, options, shape)
            if not regions:
                raise FieldError(
                    "No eligible patches for the previewed object. Lower the foreground cutoff, reduce patch width or change grid placement.",
                    "patches.min_foreground_fraction",
                    "patches.tile_size_um",
                    "patches.tile_anchor",
                )
            previews = []
            for i, (lo, hi, _) in enumerate(regions[:6]):
                if progress:
                    progress(
                        phase="Reading selected raw patches",
                        blocks=i,
                        total_blocks=min(6, len(regions)),
                    )
                previews.append((*read_patch(reader, obj, lo, hi, shape), (lo + hi) / 2))
            if progress:
                progress(phase="Preparing whole-object context", blocks=0, total_blocks=1)
            reader.views = {
                **options,
                "size": 128,
                "coarse_fractions": [0.5],
                "axes": [0, 1, 2],
                "mask": False,
            }
            overview = list(reader.planes(obj, "whole_label"))
            return previews, {
                "label_id": int(obj["row"]["label_id"]),
                "eligible": eligible,
                "selected": len(regions),
                "object_index": source.get("object_index"),
                "sampling": options["sampling"],
                "saved_voxel_size_um": source["raw_spacing_zyx"]
                if options["sampling"] == "native_voxels"
                else (np.asarray(options["tile_size_um"]) / shape).tolist(),
                "coverage_scope": obj["coverage_scope"],
                "eligibility": summary,
                "overview": overview,
            }


def grouped_arrays(stack: ExitStack, data):
    """Validate grouped storage; metadata reads are bounded by rows, not pixels.

    label_ids: unique int64 object IDs; patch_offsets: half-open row ranges of
    length Nobjects+1; patch_label_ids: repeated exact integer ID per crop row;
    positions_um: float64 ZYX physical centers. Each object must own >=1 patch.
    """
    arrays = {}
    for field in (
        "crops",
        "loss_masks",
        "label_ids",
        "patch_offsets",
        "patch_label_ids",
        "positions_um",
    ):
        with at_fields("config.data." + field, "config.data." + field + "_key"):
            if not data.get(field):
                raise ValueError(
                    "Load data_config.yaml from distributed patch preparation; all grouped arrays are required"
                )
            arrays[field] = stack.enter_context(open_crop_array(data, field))
    crops, masks = arrays["crops"], arrays["loss_masks"]
    recipe = data.get("patch_recipe", {})
    native = recipe.get("sampling") == "native_voxels"
    if (
        crops.ndim != 4
        or not len(crops)
        or masks.shape != crops.shape
        or any(n < 1 for n in crops.shape[1:])
        or (not native and len(set(crops.shape[1:])) != 1)
    ):
        raise FieldError(
            "Grouped crops and masks must match (Npatches, Z, Y, X). Physical-resampled patches require cubic dimensions; native patches can be rectangular.",
            "config.data.crops",
            "config.data.loss_masks",
        )
    if native:
        with at_fields("config.data.patch_recipe"):
            spacing = triple(recipe.get("saved_voxel_size_um", []), "Native voxel spacing")
            if (
                not np.isfinite(spacing).all()
                or tuple(recipe.get("crop_shape", [])) != crops.shape[1:]
            ):
                raise ValueError(
                    "Native patch metadata must include positive measured spacing and the actual stored crop_shape"
                )
    ids, offsets, patch_ids = (arrays[k] for k in ("label_ids", "patch_offsets", "patch_label_ids"))
    for key in ("label_ids", "patch_offsets", "patch_label_ids"):
        if arrays[key].ndim != 1 or arrays[key].dtype != np.dtype("int64"):
            raise FieldError(
                "Grouped IDs and offsets require one-dimensional int64 arrays; IDs must never pass through floats",
                "config.data." + key,
            )
    ids, offsets = np.asarray(ids[:]), np.asarray(offsets[:])
    if (
        len(np.unique(ids)) != len(ids)
        or len(offsets) != len(ids) + 1
        or offsets[0] != 0
        or offsets[-1] != len(crops)
        or np.any(np.diff(offsets) <= 0)
        or patch_ids.shape != (len(crops),)
    ):
        raise FieldError(
            "Unique object IDs and patch offsets must describe every crop exactly once in contiguous nonempty groups",
            "config.data.label_ids",
            "config.data.patch_offsets",
        )
    positions = arrays["positions_um"]
    if positions.shape != (len(crops), 3):
        raise FieldError(
            "Patch positions must contain one ZYX center in µm per crop", "config.data.positions_um"
        )
    for first in range(0, len(crops), 4096):
        stop = min(first + 4096, len(crops))
        expected = ids[np.searchsorted(offsets[1:], np.arange(first, stop), side="right")]
        if not np.array_equal(patch_ids[first:stop], expected):
            raise FieldError(
                "Patch IDs disagree with object IDs/offsets; use the arrays from the same preparation run",
                "config.data.patch_label_ids",
                "config.data.patch_offsets",
            )
        if not np.isfinite(positions[first:stop]).all():
            raise FieldError(
                "Patch centers must be finite ZYX µm coordinates", "config.data.positions_um"
            )
    return crops, masks, ids, offsets
