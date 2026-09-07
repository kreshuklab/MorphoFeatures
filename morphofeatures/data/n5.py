"""Read-only N5 discovery and lazy masked-patch loading.

The legacy Platynereis nucleus data are stored as one N5 dataset of masked
``(z, y, x)`` intensity patches and a second dataset whose rows are
``(label_id, z, y, x)``.  This module keeps those storage concerns separate
from the MAE implementation and never walks into N5 chunk directories while
discovering metadata.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:  # Torch is an optional dependency of the package.
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - exercised in minimal installations
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


@dataclass(frozen=True)
class N5DatasetMetadata:
    """Metadata available without reading any N5 data chunk."""

    container: str
    key: str
    shape: tuple[int, ...]
    dtype: str
    chunks: tuple[int, ...]
    compression: Mapping[str, Any]
    storage_dimensions: tuple[int, ...]
    storage_block_size: tuple[int, ...]
    axes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["shape"] = list(self.shape)
        value["chunks"] = list(self.chunks)
        value["storage_dimensions"] = list(self.storage_dimensions)
        value["storage_block_size"] = list(self.storage_block_size)
        return value


@dataclass(frozen=True)
class PatchIndex:
    """Validated relationship between a patch store and its biological IDs."""

    labels: np.ndarray
    positions_zyx: np.ndarray
    unique_label_ids: np.ndarray
    first_indices: np.ndarray
    counts: np.ndarray

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels)
        positions = np.asarray(self.positions_zyx)
        unique = np.asarray(self.unique_label_ids)
        if labels.ndim != 1 or positions.shape != (len(labels), 3):
            raise ValueError("Patch positions must contain rows of label_id, z, y, x")
        if not np.issubdtype(labels.dtype, np.integer) or np.any(labels <= 0):
            raise ValueError("Patch label IDs must be positive integers; zero is background")
        if not np.issubdtype(positions.dtype, np.integer) or np.any(positions < 0):
            raise ValueError("Patch z, y, x coordinates must be non-negative integers")
        if unique.ndim != 1 or len(unique) != len(np.unique(unique)):
            raise ValueError("Biological label IDs must be a unique one-dimensional array")


@dataclass(frozen=True)
class PatchQC:
    patch_index: int
    label_id: int
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    foreground_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_attributes(path: Path) -> Mapping[str, Any]:
    attributes = path / "attributes.json"
    if not attributes.is_file():
        return {}
    with attributes.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"N5 attributes are not a mapping: {attributes}")
    return value


def discover_n5_metadata(container: Path, axes_by_key: Mapping[str, str] | None = None):
    """Discover datasets without descending into their chunk directories.

    N5 stores dimensions in Java order, while z5py exposes them to NumPy in
    reverse order.  ``shape`` and ``chunks`` below use the z5py/NumPy order;
    both storage-order values are retained to make the conversion explicit.
    """

    root = Path(container).expanduser().resolve()
    if not root.is_dir() or not (root / "attributes.json").is_file():
        raise ValueError(f"Not an N5 container with attributes.json: {root}")
    axes_by_key = dict(axes_by_key or {})
    discovered = []

    def visit(directory: Path, key: str) -> None:
        attributes = _read_attributes(directory)
        if "dimensions" in attributes and "dataType" in attributes:
            dimensions = tuple(int(item) for item in attributes["dimensions"])
            blocks = tuple(int(item) for item in attributes.get("blockSize", ()))
            axes = axes_by_key.get(key)
            shape = tuple(reversed(dimensions))
            chunks = tuple(reversed(blocks))
            if axes is not None and len(axes) != len(shape):
                raise ValueError(f"Axes {axes!r} do not match {key!r} rank {len(shape)}")
            discovered.append(
                N5DatasetMetadata(
                    container=str(root),
                    key=key,
                    shape=shape,
                    dtype=str(attributes["dataType"]),
                    chunks=chunks,
                    compression=dict(attributes.get("compression", {})),
                    storage_dimensions=dimensions,
                    storage_block_size=blocks,
                    axes=axes,
                )
            )
            return
        for child in sorted(directory.iterdir()):
            if child.is_dir() and (child / "attributes.json").is_file():
                visit(child, f"{key}/{child.name}".strip("/"))

    visit(root, "")
    return discovered


def write_n5_inventory(
    containers: Iterable[Path], destination: Path, axes_by_key: Mapping[str, str] | None = None
) -> Path:
    """Write compact JSON metadata for one or more containers."""

    rows = []
    for container in containers:
        rows.extend(item.to_dict() for item in discover_n5_metadata(container, axes_by_key))
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(output))
    return output


def load_patch_index(
    positions_container: Path,
    positions_key: str = "positions",
    ids_key: str = "ids",
    *,
    expected_patch_count: int | None = None,
) -> PatchIndex:
    """Load the compact patch index, not the intensity patches themselves."""

    try:
        import z5py
    except ImportError as error:  # pragma: no cover - depends on optional z5py
        raise RuntimeError("N5 input requires z5py (normally installed with conda)") from error
    store = z5py.File(str(Path(positions_container).expanduser().resolve()), "r")
    if positions_key not in store or ids_key not in store:
        raise ValueError(f"N5 index requires datasets {positions_key!r} and {ids_key!r}")
    positions = np.asarray(store[positions_key][:])
    supplied_ids = np.asarray(store[ids_key][:])
    if positions.ndim != 2 or positions.shape[1] != 4:
        raise ValueError("positions dataset must have shape (n_patches, 4)")
    if expected_patch_count is not None and len(positions) != int(expected_patch_count):
        raise ValueError("Patch and position datasets have different row counts")
    if not np.issubdtype(positions.dtype, np.integer):
        raise ValueError("positions must contain integer label IDs and coordinates")
    labels = positions[:, 0].astype(np.int64, copy=False)
    coordinates = positions[:, 1:].astype(np.int64, copy=False)
    unique, first, counts = np.unique(labels, return_index=True, return_counts=True)
    if np.any(labels[1:] < labels[:-1]):
        raise ValueError("Patch rows must be grouped in increasing label_id order")
    if not np.array_equal(unique, supplied_ids):
        raise ValueError("ids dataset does not exactly match unique position label IDs")
    return PatchIndex(labels, coordinates, unique, first, counts)


def deterministic_label_split(
    label_ids: Sequence[int],
    fractions: Sequence[float] = (0.8, 0.1, 0.1),
    *,
    seed: int = 42,
    max_labels: int | None = None,
) -> dict[str, np.ndarray]:
    """Split biological IDs before selecting any repeated patch rows."""

    ids = np.asarray(label_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("label_ids must be a one-dimensional integer array")
    if np.any(ids <= 0) or len(np.unique(ids)) != len(ids):
        raise ValueError("label_ids must contain unique positive values")
    split = np.asarray(fractions, dtype=float)
    if split.shape != (3,) or np.any(split < 0) or not np.isclose(split.sum(), 1.0):
        raise ValueError("split fractions must be three non-negative values summing to one")
    rng = np.random.default_rng(int(seed))
    selected = ids.copy()
    if max_labels is not None:
        if int(max_labels) < 3:
            raise ValueError("max_labels must allow at least train, validation, and test IDs")
        if len(selected) > int(max_labels):
            selected = np.sort(rng.choice(selected, size=int(max_labels), replace=False))
    order = rng.permutation(selected)
    raw_counts = split * len(order)
    counts = np.floor(raw_counts).astype(int)
    for index in np.argsort(raw_counts - counts)[::-1][: len(order) - counts.sum()]:
        counts[index] += 1
    if len(order) >= 3 and np.any((split > 0) & (counts == 0)):
        raise ValueError("Selected label count is too small for all non-empty split fractions")
    first = int(counts[0])
    second = first + int(counts[1])
    return {
        "train": np.sort(order[:first]),
        "validation": np.sort(order[first:second]),
        "test": np.sort(order[second:]),
    }


def select_patch_indices(
    index: PatchIndex,
    label_ids: Sequence[int],
    *,
    patches_per_label: int | None,
    seed: int,
) -> np.ndarray:
    """Choose patch rows deterministically while retaining every parent ID."""

    rng = np.random.default_rng(int(seed))
    selected = []
    for label_id in np.asarray(label_ids, dtype=np.int64):
        location = int(np.searchsorted(index.unique_label_ids, label_id))
        if location >= len(index.unique_label_ids) or index.unique_label_ids[location] != label_id:
            raise ValueError(f"Unknown biological label_id {label_id}")
        available = np.arange(
            index.first_indices[location],
            index.first_indices[location] + index.counts[location],
            dtype=np.int64,
        )
        if patches_per_label is not None and len(available) > int(patches_per_label):
            available = np.sort(rng.choice(available, size=int(patches_per_label), replace=False))
        selected.append(available)
    return np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)


def preprocess_masked_patch(
    patch: np.ndarray,
    *,
    normalization: str = "dtype",
    mask_mode: str = "nonzero",
) -> tuple[np.ndarray, np.ndarray | None, PatchQC]:
    """Normalize one already-masked patch while preserving zero background."""

    values = np.asarray(patch)
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("Each patch must be a finite three-dimensional z, y, x array")
    foreground = values != 0
    if mask_mode not in {"nonzero", "all"}:
        raise ValueError("mask_mode must be 'nonzero' or 'all'")
    output = values.astype(np.float32)
    if normalization == "dtype":
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError("dtype normalization requires an integer patch dtype")
        limits = np.iinfo(values.dtype)
        output = (output - limits.min) / float(limits.max - limits.min)
    elif normalization == "foreground_percentile":
        if foreground.any():
            low, high = np.percentile(output[foreground], (1.0, 99.0))
            output = np.clip((output - low) / max(float(high - low), 1e-6), 0.0, 1.0)
            output[~foreground] = 0.0
    elif normalization == "foreground_zscore":
        if foreground.any():
            mean = float(output[foreground].mean())
            std = float(output[foreground].std())
            output = (output - mean) / max(std, 1e-6)
            output[~foreground] = 0.0
    elif normalization != "none":
        raise ValueError(
            "normalization must be dtype, foreground_percentile, foreground_zscore, or none"
        )
    qc = PatchQC(
        patch_index=-1,
        label_id=-1,
        minimum=float(values.min()),
        maximum=float(values.max()),
        mean=float(values.mean()),
        standard_deviation=float(values.std()),
        foreground_fraction=float(foreground.mean()),
    )
    loss_mask = foreground.astype(np.float32) if mask_mode == "nonzero" else None
    return output.astype(np.float32, copy=False), loss_mask, qc


class N5MaskedPatchDataset(Dataset):
    """Worker-safe lazy reader with bounded deterministic invalid-patch replacement."""

    def __init__(
        self,
        patches_container: Path,
        patches_key: str,
        patch_index: PatchIndex,
        selected_indices: Sequence[int],
        *,
        normalization: str = "dtype",
        mask_mode: str = "nonzero",
        min_foreground_fraction: float = 0.0,
        replacement_attempts: int = 16,
        augment_flip: bool = False,
        augment_rot90: bool = False,
        resample_label_ids: Sequence[int] | None = None,
        patches_per_label: int | None = None,
        seed: int = 42,
        mode: str = "train",
    ):
        if torch is None:  # pragma: no cover - optional dependency guard
            raise RuntimeError("N5 MAE datasets require morphofeatures[modern-training]")
        self.container = str(Path(patches_container).expanduser().resolve())
        self.key = str(patches_key)
        self.index = patch_index
        self.selected_indices = np.asarray(selected_indices, dtype=np.int64)
        if self.selected_indices.ndim != 1 or np.any(self.selected_indices < 0):
            raise ValueError("selected_indices must be a non-negative one-dimensional array")
        if len(self.selected_indices) and self.selected_indices.max() >= len(self.index.labels):
            raise ValueError("selected_indices exceed the patch position table")
        if normalization not in {"dtype", "foreground_percentile", "foreground_zscore", "none"}:
            raise ValueError("Unsupported patch normalization")
        if mask_mode not in {"nonzero", "all"}:
            raise ValueError("Unsupported mask_mode")
        if not 0 <= float(min_foreground_fraction) <= 1:
            raise ValueError("min_foreground_fraction must be between zero and one")
        if int(replacement_attempts) < 0:
            raise ValueError("replacement_attempts must be non-negative")
        if mode not in {"train", "encode", "inspect"}:
            raise ValueError("mode must be train, encode, or inspect")
        self.normalization = normalization
        self.mask_mode = mask_mode
        self.min_foreground_fraction = float(min_foreground_fraction)
        self.replacement_attempts = int(replacement_attempts)
        self.augment_flip = bool(augment_flip)
        self.augment_rot90 = bool(augment_rot90)
        self.resample_label_ids = (
            None if resample_label_ids is None else np.asarray(resample_label_ids, dtype=np.int64)
        )
        self.patches_per_label = None if patches_per_label is None else int(patches_per_label)
        if (self.resample_label_ids is None) != (self.patches_per_label is None):
            raise ValueError("resample_label_ids and patches_per_label must be configured together")
        self.seed = int(seed)
        self.epoch = 0
        self.mode = mode
        self._owner_pid = None
        self._store = None
        self._patches = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update({"_owner_pid": None, "_store": None, "_patches": None})
        return state

    def _dataset(self):
        pid = os.getpid()
        if self._patches is None or self._owner_pid != pid:
            try:
                import z5py
            except ImportError as error:  # pragma: no cover
                raise RuntimeError("N5 input requires z5py") from error
            self._store = z5py.File(self.container, "r")
            if self.key not in self._store:
                raise ValueError(f"Patch dataset {self.key!r} is missing from {self.container}")
            self._patches = self._store[self.key]
            if len(self._patches) != len(self.index.labels):
                raise ValueError("Patch and position datasets have different row counts")
            self._owner_pid = pid
        return self._patches

    def __len__(self) -> int:
        return len(self.selected_indices)

    def set_epoch(self, epoch: int) -> None:
        """Select new same-parent patches and augmentations reproducibly for one epoch."""

        self.epoch = int(epoch)
        if self.resample_label_ids is not None:
            self.selected_indices = select_patch_indices(
                self.index,
                self.resample_label_ids,
                patches_per_label=self.patches_per_label,
                seed=self.seed + 1_000_003 * self.epoch,
            )

    def _replacement_candidates(self, initial: int):
        label_id = int(self.index.labels[initial])
        location = int(np.searchsorted(self.index.unique_label_ids, label_id))
        start = int(self.index.first_indices[location])
        count = int(self.index.counts[location])
        offset = initial - start
        yield initial
        if count <= 1:
            return
        stride = max(1, count // max(1, self.replacement_attempts))
        for attempt in range(1, self.replacement_attempts + 1):
            yield start + ((offset + attempt * stride) % count)

    def read_sample(self, item: int):
        initial = int(self.selected_indices[int(item)])
        dataset = self._dataset()
        last = None
        for patch_index in self._replacement_candidates(initial):
            patch = np.asarray(dataset[int(patch_index)])
            processed, loss_mask, qc = preprocess_masked_patch(
                patch, normalization=self.normalization, mask_mode=self.mask_mode
            )
            qc = PatchQC(
                patch_index=int(patch_index),
                label_id=int(self.index.labels[patch_index]),
                minimum=qc.minimum,
                maximum=qc.maximum,
                mean=qc.mean,
                standard_deviation=qc.standard_deviation,
                foreground_fraction=qc.foreground_fraction,
            )
            last = (processed, loss_mask, qc)
            if qc.foreground_fraction >= self.min_foreground_fraction:
                return last
        assert last is not None
        raise ValueError(
            f"No patch for label_id {last[2].label_id} met minimum foreground fraction "
            f"{self.min_foreground_fraction} after {self.replacement_attempts + 1} bounded reads"
        )

    def _augment(self, crop: np.ndarray, mask: np.ndarray | None, item: int):
        rng = np.random.default_rng(self.seed + int(item) + 1_000_003 * self.epoch)
        if self.augment_flip:
            for axis in range(3):
                if rng.random() < 0.5:
                    crop = np.flip(crop, axis=axis)
                    if mask is not None:
                        mask = np.flip(mask, axis=axis)
        if self.augment_rot90:
            axes = ((0, 1), (0, 2), (1, 2))[int(rng.integers(0, 3))]
            turns = int(rng.integers(0, 4))
            crop = np.rot90(crop, turns, axes=axes)
            if mask is not None:
                mask = np.rot90(mask, turns, axes=axes)
        return np.ascontiguousarray(crop), None if mask is None else np.ascontiguousarray(mask)

    def __getitem__(self, item: int):
        crop, loss_mask, qc = self.read_sample(item)
        if self.mode == "train":
            crop, loss_mask = self._augment(crop, loss_mask, item)
        crop_tensor = torch.from_numpy(crop[None])
        if self.mode == "encode":
            return torch.tensor(qc.label_id, dtype=torch.int64), crop_tensor
        if self.mode == "inspect":
            return qc, crop, loss_mask
        if loss_mask is None:
            return crop_tensor
        return crop_tensor, torch.from_numpy(loss_mask[None])


class N5GroupedPatchDataset(Dataset):
    """Lazy one-row-per-parent dataset for the verified nucleus MAE sampling unit.

    A sample contains the spatially central ``group_size`` stored texture patches
    belonging to one ``label_id``. Short groups are padded and marked invalid;
    no intensity patch is duplicated to fill a group.
    """

    def __init__(
        self,
        patches_container: Path,
        patches_key: str,
        patch_index: PatchIndex,
        label_ids: Sequence[int],
        *,
        group_size: int,
        position_stride_zyx: Sequence[float] = (8.0, 8.0, 8.0),
        normalization: str = "dtype",
        mask_mode: str = "nonzero",
        min_foreground_fraction: float = 0.0,
        mode: str = "train",
    ):
        if torch is None:  # pragma: no cover - optional dependency guard
            raise RuntimeError("N5 MAE datasets require morphofeatures[modern-training]")
        self.container = str(Path(patches_container).expanduser().resolve())
        self.key = str(patches_key)
        self.index = patch_index
        self.label_ids = np.asarray(label_ids, dtype=np.int64)
        if self.label_ids.ndim != 1 or np.any(self.label_ids <= 0):
            raise ValueError("label_ids must be a one-dimensional array of positive IDs")
        if len(np.unique(self.label_ids)) != len(self.label_ids):
            raise ValueError("label_ids must be unique")
        unknown = np.setdiff1d(self.label_ids, self.index.unique_label_ids)
        if len(unknown):
            raise ValueError(f"Unknown biological label IDs: {unknown[:5].tolist()}")
        self.group_size = int(group_size)
        if self.group_size < 2:
            raise ValueError("group_size must be at least two so one patch can be hidden")
        stride = np.asarray(position_stride_zyx, dtype=np.float32)
        if stride.shape != (3,) or np.any(~np.isfinite(stride)) or np.any(stride <= 0):
            raise ValueError("position_stride_zyx must contain three finite positive values")
        self.position_stride_zyx = stride
        if normalization not in {"dtype", "foreground_percentile", "foreground_zscore", "none"}:
            raise ValueError("Unsupported patch normalization")
        if mask_mode not in {"nonzero", "all"}:
            raise ValueError("Unsupported mask_mode")
        if not 0 <= float(min_foreground_fraction) <= 1:
            raise ValueError("min_foreground_fraction must be between zero and one")
        if mode not in {"train", "encode", "inspect"}:
            raise ValueError("mode must be train, encode, or inspect")
        self.normalization = normalization
        self.mask_mode = mask_mode
        self.min_foreground_fraction = float(min_foreground_fraction)
        self.mode = mode
        self.epoch = 0
        self._owner_pid = None
        self._store = None
        self._patches = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update({"_owner_pid": None, "_store": None, "_patches": None})
        return state

    def _dataset(self):
        pid = os.getpid()
        if self._patches is None or self._owner_pid != pid:
            try:
                import z5py
            except ImportError as error:  # pragma: no cover
                raise RuntimeError("N5 input requires z5py") from error
            self._store = z5py.File(self.container, "r")
            if self.key not in self._store:
                raise ValueError(f"Patch dataset {self.key!r} is missing from {self.container}")
            self._patches = self._store[self.key]
            if len(self._patches) != len(self.index.labels):
                raise ValueError("Patch and position datasets have different row counts")
            self._owner_pid = pid
        return self._patches

    def __len__(self) -> int:
        return len(self.label_ids)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _candidate_indices(self, label_id: int):
        location = int(np.searchsorted(self.index.unique_label_ids, label_id))
        start = int(self.index.first_indices[location])
        count = int(self.index.counts[location])
        indices = np.arange(start, start + count, dtype=np.int64)
        coordinates = self.index.positions_zyx[indices].astype(np.float32)
        # The original nucleus COM table is not required at runtime. Median
        # patch position is a stable, robust center defined by the indexed data.
        center = np.median(coordinates, axis=0)
        relative = (coordinates - center) / self.position_stride_zyx
        distance = np.square(relative).sum(axis=1)
        order = np.lexsort((indices, distance))
        return indices[order], relative[order]

    def read_group(self, item: int):
        label_id = int(self.label_ids[int(item)])
        candidates, relative_positions = self._candidate_indices(label_id)
        dataset = self._dataset()
        patches = []
        positions = []
        patch_indices = []
        for patch_index, relative in zip(candidates, relative_positions):
            raw = np.asarray(dataset[int(patch_index)])
            processed, _, qc = preprocess_masked_patch(
                raw,
                normalization=self.normalization,
                mask_mode=self.mask_mode,
            )
            if qc.foreground_fraction < self.min_foreground_fraction:
                continue
            patches.append(processed[None])
            positions.append(relative)
            patch_indices.append(int(patch_index))
            if len(patches) == self.group_size:
                break
        if len(patches) < 2:
            raise ValueError(
                f"label_id {label_id} has only {len(patches)} usable patches; at least two "
                "are required for grouped masking"
            )
        patch_shape = tuple(patches[0].shape[1:])
        output_patches = np.zeros((self.group_size, 1) + patch_shape, dtype=np.float32)
        output_positions = np.zeros((self.group_size, 3), dtype=np.float32)
        valid = np.zeros(self.group_size, dtype=bool)
        count = len(patches)
        output_patches[:count] = np.stack(patches)
        output_positions[:count] = np.stack(positions)
        valid[:count] = True
        return {
            "label_id": label_id,
            "patches": output_patches,
            "positions_zyx": output_positions,
            "valid": valid,
            "patch_indices": np.asarray(patch_indices, dtype=np.int64),
        }

    def read_sample(self, item: int):
        """Return one central usable patch for bounded legacy-style QC helpers."""

        label_id = int(self.label_ids[int(item)])
        candidates, _ = self._candidate_indices(label_id)
        dataset = self._dataset()
        for patch_index in candidates:
            processed, loss_mask, qc = preprocess_masked_patch(
                np.asarray(dataset[int(patch_index)]),
                normalization=self.normalization,
                mask_mode=self.mask_mode,
            )
            if qc.foreground_fraction >= self.min_foreground_fraction:
                return processed, loss_mask, PatchQC(
                    patch_index=int(patch_index),
                    label_id=label_id,
                    minimum=qc.minimum,
                    maximum=qc.maximum,
                    mean=qc.mean,
                    standard_deviation=qc.standard_deviation,
                    foreground_fraction=qc.foreground_fraction,
                )
        raise ValueError(f"label_id {label_id} has no usable patches")

    def __getitem__(self, item: int):
        sample = self.read_group(item)
        if self.mode == "inspect":
            return sample
        patches = torch.from_numpy(sample["patches"])
        positions = torch.from_numpy(sample["positions_zyx"])
        valid = torch.from_numpy(sample["valid"])
        if self.mode == "encode":
            return (
                torch.tensor(sample["label_id"], dtype=torch.int64),
                patches,
                positions,
                valid,
            )
        return patches, positions, valid
