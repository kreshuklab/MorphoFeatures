"""Deterministic, inspectable preparation of masked 3D crops for embedding models.

The functions in this module deliberately stop at crop preparation.  Model-specific
masking (for example MAE patch masking) belongs to the model, while spatial alignment,
segmentation masking, and intensity normalization belong here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import ndimage


def _integer_triple(value: Sequence[int], name: str, allow_zero: bool = False) -> tuple[int, int, int]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain three z, y, x values")
    result = tuple(int(item) for item in value)
    minimum = 0 if allow_zero else 1
    if any(item < minimum for item in result):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must contain three {qualifier} z, y, x values")
    return result


def _resolution_triple(value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) != 3 or any(float(item) <= 0 for item in value):
        raise ValueError("resolution_zyx must contain three positive z, y, x values")
    return tuple(float(item) for item in value)


def _validate_label_volume(labels: np.ndarray) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 3:
        raise ValueError("A segmentation must be a three-dimensional z, y, x array")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or not np.allclose(array, np.rint(array)):
            raise ValueError("Segmentation labels must be finite integer values")
        array = np.rint(array).astype(np.int64)
    if np.any(array < 0):
        raise ValueError("Segmentation labels must be non-negative; zero is background")
    return array


@dataclass(frozen=True)
class MappedVolumeROI:
    """An intensity ROI and aligned final-label ROI in z, y, x order."""

    raw: np.ndarray
    labels: np.ndarray
    origin_zyx: tuple[int, int, int]
    resolution_zyx: tuple[float, float, float]
    source: Mapping[str, object]

    def __post_init__(self) -> None:
        if np.asarray(self.raw).shape != np.asarray(self.labels).shape:
            raise ValueError("Raw and label ROIs must have identical shapes")
        if np.asarray(self.raw).ndim != 3:
            raise ValueError("Raw and label ROIs must be three-dimensional")
        _validate_label_volume(self.labels)
        _integer_triple(self.origin_zyx, "origin_zyx", allow_zero=True)
        _resolution_triple(self.resolution_zyx)


@dataclass(frozen=True)
class MaskedCropBatch:
    """Training crops plus the arrays and table needed for visual quality control."""

    crops: np.ndarray
    label_ids: np.ndarray
    masks: np.ndarray
    raw_crops: np.ndarray
    manifest: pd.DataFrame

    def __post_init__(self) -> None:
        crops = np.asarray(self.crops)
        ids = np.asarray(self.label_ids)
        if crops.ndim != 4:
            raise ValueError("crops must have shape (n, z, y, x)")
        if ids.ndim != 1 or len(ids) != len(crops):
            raise ValueError("label_ids must contain one ID per crop")
        if not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids):
            raise ValueError("label_ids must be unique integers")
        if np.asarray(self.masks).shape != crops.shape or np.asarray(self.raw_crops).shape != crops.shape:
            raise ValueError("masks and raw_crops must have the same shape as crops")
        if not np.all(np.isfinite(crops)):
            raise ValueError("Prepared crops contain NaN or infinite values")
        if len(self.manifest) != len(crops):
            raise ValueError("manifest must contain one row per crop")


def map_fragment_labels(fragment_labels: np.ndarray, fragment_to_segment: np.ndarray) -> np.ndarray:
    """Map watershed/Paintera fragment IDs to final segmentation IDs."""

    fragments = _validate_label_volume(fragment_labels)
    mapping = np.asarray(fragment_to_segment)
    if mapping.ndim != 1 or not np.issubdtype(mapping.dtype, np.integer):
        raise ValueError("fragment_to_segment must be a one-dimensional integer array")
    if fragments.size and int(fragments.max()) >= len(mapping):
        raise ValueError("Fragment ID exceeds the fragment-to-segment mapping length")
    return mapping[fragments]


def load_mapped_n5_roi(
    container: Path,
    raw_dataset: str,
    fragment_dataset: str,
    mapping_dataset: str,
    roi_start_zyx: Sequence[int],
    roi_shape_zyx: Sequence[int],
    resolution_zyx: Sequence[float],
) -> MappedVolumeROI:
    """Read one bounded N5 ROI and map fragment IDs to final cell labels.

    This is useful for Paintera-style data where an efficient multiscale fragment
    volume is accompanied by a one-dimensional fragment-to-segment assignment.
    The function never reads the full raw or fragment volume.  It currently reads
    the mapping array once; for the Platynereis example this is about 146 MB.
    """

    try:
        import z5py
    except ImportError as error:  # pragma: no cover - depends on conda-only z5py
        raise RuntimeError("Mapped N5 input requires z5py (normally installed with conda)") from error

    origin = _integer_triple(roi_start_zyx, "roi_start_zyx", allow_zero=True)
    shape = _integer_triple(roi_shape_zyx, "roi_shape_zyx")
    resolution = _resolution_triple(resolution_zyx)
    source_path = Path(container).expanduser().resolve()
    store = z5py.File(str(source_path), "r")
    raw_source = store[raw_dataset]
    fragment_source = store[fragment_dataset]
    if tuple(raw_source.shape) != tuple(fragment_source.shape):
        raise ValueError("Raw and fragment datasets must have identical shapes at the chosen scale")
    stop = tuple(start + size for start, size in zip(origin, shape))
    if any(end > bound for end, bound in zip(stop, raw_source.shape)):
        raise ValueError("Requested ROI exceeds the selected N5 dataset")
    selection = tuple(slice(start, end) for start, end in zip(origin, stop))
    raw = np.asarray(raw_source[selection])
    fragments = np.asarray(fragment_source[selection])
    mapping = np.asarray(store[mapping_dataset][:])
    labels = map_fragment_labels(fragments, mapping)
    source = {
        "container": str(source_path),
        "raw_dataset": raw_dataset,
        "fragment_dataset": fragment_dataset,
        "mapping_dataset": mapping_dataset,
        "full_shape_zyx": [int(item) for item in raw_source.shape],
    }
    return MappedVolumeROI(raw, labels, origin, resolution, source)


def describe_labels(labels: np.ndarray) -> pd.DataFrame:
    """Return voxel counts, bounds, centroids, and ROI-border status by label ID."""

    segmentation = _validate_label_volume(labels)
    columns = [
        "label_id",
        "voxel_count_roi",
        "touches_roi_border",
        "center_z",
        "center_y",
        "center_x",
        "bb_min_z",
        "bb_min_y",
        "bb_min_x",
        "bb_max_z",
        "bb_max_y",
        "bb_max_x",
    ]
    ids, counts = np.unique(segmentation, return_counts=True)
    count_by_id = dict(zip(ids.tolist(), counts.tolist()))
    objects = ndimage.find_objects(segmentation)
    rows = []
    for label_id in ids:
        label_id = int(label_id)
        if label_id == 0:
            continue
        bounds = objects[label_id - 1]
        if bounds is None:
            continue
        local = np.argwhere(segmentation[bounds] == label_id)
        starts = np.array([axis.start for axis in bounds], dtype=np.int64)
        center = local.mean(axis=0) + starts
        minima = starts
        maxima = np.array([axis.stop for axis in bounds], dtype=np.int64)
        touches_border = bool(np.any(minima == 0) or np.any(maxima == np.asarray(segmentation.shape)))
        rows.append(
            {
                "label_id": label_id,
                "voxel_count_roi": int(count_by_id[label_id]),
                "touches_roi_border": touches_border,
                "center_z": float(center[0]),
                "center_y": float(center[1]),
                "center_x": float(center[2]),
                "bb_min_z": int(minima[0]),
                "bb_min_y": int(minima[1]),
                "bb_min_x": int(minima[2]),
                "bb_max_z": int(maxima[0]),
                "bb_max_y": int(maxima[1]),
                "bb_max_x": int(maxima[2]),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("label_id", ignore_index=True)


def _normalize_crop(raw_crop: np.ndarray, cell_mask: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(raw_crop)
    if mode == "dtype":
        if np.issubdtype(values.dtype, np.integer):
            limits = np.iinfo(values.dtype)
            normalized = (values.astype(np.float32) - limits.min) / float(limits.max - limits.min)
        else:
            normalized = values.astype(np.float32)
            if normalized.size and (normalized.min() < 0 or normalized.max() > 1):
                raise ValueError("Floating-point dtype normalization expects intensities in [0, 1]")
    elif mode == "percentile":
        foreground = values[cell_mask]
        if not foreground.size:
            raise ValueError("Cannot normalize an empty cell mask")
        low, high = np.percentile(foreground.astype(np.float32), (1.0, 99.0))
        normalized = (values.astype(np.float32) - low) / max(float(high - low), np.finfo(np.float32).eps)
        normalized = np.clip(normalized, 0.0, 1.0)
    elif mode == "none":
        normalized = values.astype(np.float32)
    else:
        raise ValueError("normalization must be 'dtype', 'percentile', or 'none'")
    return np.asarray(normalized, dtype=np.float32)


def extract_masked_cell_crops(
    raw: np.ndarray,
    labels: np.ndarray,
    crop_shape_zyx: Sequence[int],
    *,
    label_ids: Optional[Sequence[int]] = None,
    min_cell_voxels: int = 1,
    min_crop_coverage: float = 0.0,
    require_complete_in_roi: bool = True,
    max_cells: Optional[int] = None,
    seed: int = 42,
    normalization: str = "dtype",
) -> MaskedCropBatch:
    """Extract fixed, centroid-aligned raw crops and zero intensities outside each cell."""

    intensities = np.asarray(raw)
    segmentation = _validate_label_volume(labels)
    if intensities.ndim != 3 or intensities.shape != segmentation.shape:
        raise ValueError("raw and labels must be aligned three-dimensional arrays")
    if not np.all(np.isfinite(intensities)):
        raise ValueError("Raw volume contains NaN or infinite values")
    crop_shape = np.asarray(_integer_triple(crop_shape_zyx, "crop_shape_zyx"), dtype=np.int64)
    if min_cell_voxels <= 0:
        raise ValueError("min_cell_voxels must be positive")
    if max_cells is not None and int(max_cells) <= 0:
        raise ValueError("max_cells must be positive when provided")
    if not 0.0 <= min_crop_coverage <= 1.0:
        raise ValueError("min_crop_coverage must be between zero and one")

    table = describe_labels(segmentation)
    keep = table["voxel_count_roi"] >= int(min_cell_voxels)
    if require_complete_in_roi:
        keep &= ~table["touches_roi_border"]
    if label_ids is not None:
        requested = np.asarray(label_ids)
        if requested.ndim != 1 or not np.all(np.isfinite(requested)) or not np.allclose(requested, np.rint(requested)):
            raise ValueError("label_ids must contain finite integer values")
        keep &= table["label_id"].isin(np.rint(requested).astype(np.int64))
    candidates = table.loc[keep].copy()

    records, crops, masks, raw_crops = [], [], [], []
    before = crop_shape // 2
    for row in candidates.itertuples(index=False):
        center = np.rint([row.center_z, row.center_y, row.center_x]).astype(np.int64)
        start = center - before
        stop = start + crop_shape
        if np.any(start < 0) or np.any(stop > np.asarray(segmentation.shape)):
            continue
        selection = tuple(slice(int(left), int(right)) for left, right in zip(start, stop))
        label_crop = segmentation[selection]
        mask = label_crop == int(row.label_id)
        coverage = float(mask.sum()) / float(row.voxel_count_roi)
        if coverage < min_crop_coverage:
            continue
        raw_crop = intensities[selection]
        normalized = _normalize_crop(raw_crop, mask, normalization)
        crops.append(normalized * mask)
        masks.append(mask)
        raw_crops.append(normalized)
        record = row._asdict()
        record.update(
            {
                "crop_start_z": int(start[0]),
                "crop_start_y": int(start[1]),
                "crop_start_x": int(start[2]),
                "crop_coverage": coverage,
                "mask_fraction": float(mask.mean()),
            }
        )
        records.append(record)

    if not records:
        raise ValueError("No labels satisfy the crop selection and coverage requirements")
    order = np.arange(len(records))
    if max_cells is not None and len(order) > int(max_cells):
        order = np.sort(np.random.default_rng(seed).choice(order, size=int(max_cells), replace=False))
    manifest = pd.DataFrame([records[index] for index in order]).reset_index(drop=True)
    return MaskedCropBatch(
        np.stack([crops[index] for index in order]).astype(np.float32, copy=False),
        manifest["label_id"].to_numpy(dtype=np.int64),
        np.stack([masks[index] for index in order]),
        np.stack([raw_crops[index] for index in order]).astype(np.float32, copy=False),
        manifest,
    )


def save_crop_batch(
    batch: MaskedCropBatch,
    output_directory: Path,
    provenance: Optional[Mapping[str, object]] = None,
) -> Mapping[str, Path]:
    """Save model inputs and QC companions beneath a configured output directory."""

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "crops": destination / "crops.npy",
        "label_ids": destination / "label_ids.npy",
        "masks": destination / "cell_masks.npy",
        "raw_crops": destination / "raw_crops.npy",
        "manifest": destination / "crop_manifest.tsv",
        "provenance": destination / "preprocessing.json",
    }
    np.save(paths["crops"], batch.crops)
    np.save(paths["label_ids"], batch.label_ids)
    np.save(paths["masks"], batch.masks)
    np.save(paths["raw_crops"], batch.raw_crops)
    batch.manifest.to_csv(paths["manifest"], sep="\t", index=False)
    document = dict(provenance or {})
    document.update(
        {
            "n_crops": int(len(batch.label_ids)),
            "crop_shape_zyx": [int(item) for item in batch.crops.shape[-3:]],
            "label_ids": [int(item) for item in batch.label_ids],
            "intensity_range": [float(batch.crops.min()), float(batch.crops.max())],
        }
    )
    paths["provenance"].write_text(
        json.dumps(document, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return paths
