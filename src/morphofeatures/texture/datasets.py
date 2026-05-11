"""PyTorch datasets for texture-based MorphoFeatures encoders."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data.dataset import Dataset
except ImportError:  # pragma: no cover - exercised only without torch installed.
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[misc, assignment]


ArrayLike = Any
Transform = Callable[[ArrayLike], ArrayLike]


def _clone_view(data: ArrayLike) -> ArrayLike:
    """Clone torch tensors or copy NumPy-like arrays for augmentation views."""

    if hasattr(data, "clone"):
        return data.clone()
    return np.array(data, copy=True)


def _rescale_nearest(mask: np.ndarray, scale: int) -> np.ndarray:
    """Upsample a binary mask with nearest-neighbor interpolation."""

    try:
        from skimage.transform import rescale
    except ImportError as exc:
        raise ImportError("Text patch nucleus masking requires scikit-image.") from exc

    try:
        return rescale(mask, scale, channel_axis=None, order=0, preserve_range=True)
    except TypeError:
        return rescale(mask, scale, multichannel=False, order=0, preserve_range=True)


class CellDataset(Dataset):  # type: ignore[misc]
    """Base dataset for reading cell-centered raw intensity crops.

    Args:
        cell_nucleus_tables: Pair of cell and nucleus metadata tables.
        nucleus_by_cell: Mapping from cell label ID to nucleus label ID.
        cell_data: Cell segmentation volume.
        nucleus_data: Nucleus segmentation volume.
        raw_data: Raw intensity volume.
        predict: Whether the dataset is used for inference.
        indices: Optional subset of cell IDs.
        transforms: Transform applied to target/inference data.
        transforms_sim: Transform applied to contrastive input views.
        size_cut: Maximum crop side length around the nucleus center.
        resolution_um: Physical voxel resolution used by metadata tables.
        high_resolution_shape: Reference high-resolution volume shape.
    """

    def __init__(
        self,
        cell_nucleus_tables: tuple[pd.DataFrame, pd.DataFrame] | list[pd.DataFrame],
        nucleus_by_cell: dict[int, int],
        cell_data: ArrayLike,
        nucleus_data: ArrayLike,
        raw_data: ArrayLike,
        predict: bool = False,
        indices: Sequence[int] | np.ndarray | None = None,
        transforms: Transform | None = None,
        transforms_sim: Transform | None = None,
        size_cut: int = 200,
        resolution_um: Sequence[float] = (0.025, 0.01, 0.01),
        high_resolution_shape: Sequence[int] = (11416, 25916, 27499),
    ) -> None:
        """Initialize shared cell crop metadata and volumes."""

        self.resolution_um = np.asarray(resolution_um, dtype=float)
        self.high_resolution_shape = np.asarray(high_resolution_shape, dtype=float)
        self.cell_table = cell_nucleus_tables[0]
        self.nucleus_table = cell_nucleus_tables[1]
        self.nucleus_by_cell = nucleus_by_cell
        self.cell_data = cell_data
        self.nucleus_data = nucleus_data
        self.raw_data = raw_data
        self.predict = predict
        self.transforms = transforms
        self.transforms_sim = transforms_sim
        self.size_cut = size_cut

        if indices is None:
            indices = list(nucleus_by_cell.keys())
        if not isinstance(indices, (list, tuple, np.ndarray)):
            raise TypeError(f"indices must be list, tuple, or numpy array, got {type(indices).__name__}")

        self.indices = np.asarray(indices).astype(int)
        self.resolution_scale = self.get_resolution_scale()
        self.cell_reference_boxes = self.get_bounding_boxes(self.cell_table)

    def __len__(self) -> int:
        """Return the number of indexed cells."""

        return int(len(self.indices))

    def transform(self, data: ArrayLike, transforms: Transform | None = None) -> ArrayLike:
        """Apply a transform if one is configured."""

        if transforms is None:
            return data
        return transforms(data)

    def get_bounding_boxes(self, table: pd.DataFrame) -> list[tuple[slice, slice, slice] | list[Any]]:
        """Convert table bounding boxes from physical units to volume slices."""

        boxes = [
            [
                slice(
                    int(np.rint(row[f"bb_min_{axis}"] / self.resolution_um[index])),
                    int(np.rint(row[f"bb_max_{axis}"] / self.resolution_um[index])),
                )
                for index, axis in enumerate(("z", "y", "x"))
            ]
            for _, row in table.iterrows()
        ]
        updated = [self.update_bounding_box(box) for box in boxes]
        return [[]] + updated

    def get_resolution_scale(self) -> np.ndarray:
        """Return scale between current cell volume and reference high-resolution shape."""

        return np.asarray(self.cell_data.shape, dtype=float) / self.high_resolution_shape

    def update_bounding_box(self, bounding_box: Sequence[slice]) -> tuple[slice, slice, slice]:
        """Scale a high-resolution bounding box into the active cell volume."""

        updated = []
        for axis, old_slice in enumerate(bounding_box):
            axis_scale = self.resolution_scale[axis]
            updated.append(
                slice(
                    int(np.floor(old_slice.start * axis_scale)),
                    int(np.ceil(old_slice.stop * axis_scale)),
                )
            )
        return tuple(updated)  # type: ignore[return-value]

    def center_of_mass(self, label_id: int, cell: bool = False) -> list[int]:
        """Return the metadata anchor point in current-volume pixel coordinates."""

        table = self.cell_table if cell else self.nucleus_table
        id_data = table[table["label_id"] == label_id]
        center_um = [id_data[f"anchor_{axis}"].values.item() for axis in ("z", "y", "x")]
        return [
            int(np.rint(value / self.resolution_um[index] * self.resolution_scale[index]))
            for index, value in enumerate(center_um)
        ]

    def cut_to_size(self, cell_id: int) -> tuple[slice, slice, slice]:
        """Intersect the cell bounding box with a nucleus-centered fixed-size crop."""

        bounding_box = self.cell_reference_boxes[cell_id]
        if not bounding_box:
            raise IndexError(f"Missing bounding box for cell ID {cell_id}.")
        nucleus_id = self.nucleus_by_cell[cell_id]
        nucleus_center = [int(value) for value in self.center_of_mass(nucleus_id)]
        half_size = int(self.size_cut / 2)
        cut_box = [slice(max(0, axis_center - half_size), axis_center + half_size) for axis_center in nucleus_center]
        return tuple(
            slice(max(cell_slice.start, cut_slice.start), min(cut_slice.stop, cell_slice.stop))
            for cell_slice, cut_slice in zip(bounding_box, cut_box, strict=True)
        )  # type: ignore[return-value]


class RawAutoencoderContrastiveCellDataset(CellDataset):
    """Contrastive dataset that returns full cell or nucleus raw crops."""

    def __init__(self, *args: Any, remove_nucleus: bool = False, only_nucleus: bool = False, **kwargs: Any) -> None:
        """Initialize full-crop contrastive dataset options."""

        self.remove_nucleus = remove_nucleus or bool(kwargs.pop("remove_nucl", False))
        self.only_nucleus = only_nucleus or bool(kwargs.pop("only_nucl", False))
        self.dilate_mask = bool(kwargs.pop("dilate_mask", False))
        super().__init__(*args, **kwargs)

    def get_data_stack(self, cell_id: int) -> np.ndarray:
        """Extract a masked raw crop for one cell."""

        cell_box = self.cut_to_size(cell_id)
        cell_mask = self.cell_data[cell_box] == cell_id
        raw_mask = self.raw_data[cell_box] * cell_mask
        if self.remove_nucleus or self.only_nucleus:
            nucleus_id = self.nucleus_by_cell[cell_id]
            nucleus_mask = self.nucleus_data[cell_box] == nucleus_id
            raw_mask = raw_mask * (nucleus_mask if self.only_nucleus else np.invert(nucleus_mask))
        return raw_mask

    def __getitem__(self, index: int) -> Any:
        """Return one prediction crop or a pair of contrastive views."""

        cell_id = int(self.indices[index])
        data_stack = self.get_data_stack(cell_id)

        if self.predict:
            return self.transform(data_stack, self.transforms)
        if torch is None:
            raise ImportError("Training texture datasets requires torch.")

        targets = [self.transform(data_stack.copy(), self.transforms) for _ in range(2)]
        inputs = [self.transform(_clone_view(target), self.transforms_sim) for target in targets]
        return torch.stack(inputs), torch.stack(targets)


class TextPatchContrastiveCellDataset(CellDataset):
    """Contrastive dataset that returns fixed-radius raw texture patches."""

    def __init__(
        self,
        *args: Any,
        cell_hr_vol: ArrayLike,
        radius: int,
        crops_file: str | None = None,
        remove_nucleus: bool = False,
        only_nucleus: bool = False,
        take_every: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize patch positions and patch extraction options."""

        self.high_resolution_cell_data = cell_hr_vol
        self.radius = radius
        self.remove_nucleus = remove_nucleus or bool(kwargs.pop("remove_nucl", False))
        self.only_nucleus = only_nucleus or bool(kwargs.pop("only_nucl", False))
        self.take_every = take_every

        if crops_file is not None:
            try:
                import z5py
            except ImportError as exc:
                raise ImportError("Reading texture crop positions requires z5py.") from exc
            positions_file = z5py.File(crops_file)
            self.positions = positions_file["positions"]
            self.all_ids = positions_file["ids"][:]
        else:
            self.positions = None
            self.all_ids = None

        super().__init__(*args, **kwargs)
        self.bounding_box_scale = int(np.rint(self.raw_data.shape[0] / self.cell_data.shape[0]))

    def __len__(self) -> int:
        """Return the number of sampled patch positions."""

        if self.positions is None:
            return 0
        return int(self.positions.shape[0] / self.take_every)

    def get_data_stack(self, cell_id: int, location: Sequence[int]) -> np.ndarray:
        """Extract a masked high-resolution raw patch for one cell."""

        random_box = [slice(value - self.radius, value + self.radius) for value in location]
        cell_box = self.cut_to_size(cell_id)
        crop_box = [
            slice(cell_slice.start + patch_slice.start, cell_slice.start + patch_slice.stop)
            for cell_slice, patch_slice in zip(cell_box, random_box, strict=True)
        ]
        high_resolution_box = tuple(
            slice(crop_slice.start * self.bounding_box_scale, crop_slice.stop * self.bounding_box_scale)
            for crop_slice in crop_box
        )

        high_resolution_crop = self.raw_data[high_resolution_box] * (
            self.high_resolution_cell_data[high_resolution_box] == cell_id
        )
        nucleus_crop = self.nucleus_data[tuple(crop_box)] == self.nucleus_by_cell[cell_id]
        if self.remove_nucleus and np.any(nucleus_crop):
            high_resolution_crop = high_resolution_crop * (1 - _rescale_nearest(nucleus_crop, self.bounding_box_scale))
        if self.only_nucleus:
            upsampled_nucleus = _rescale_nearest(nucleus_crop, self.bounding_box_scale)
            high_resolution_crop = self.raw_data[high_resolution_box] * upsampled_nucleus
        return high_resolution_crop

    def __getitem__(self, index: int) -> Any:
        """Return one prediction patch or a pair of contrastive patch views."""

        if self.positions is None:
            raise ValueError("TextPatchContrastiveCellDataset requires crop positions.")
        position = self.positions[index * self.take_every]
        cell_id = int(position[0])
        patch_location = position[1:]

        data_stack = self.get_data_stack(cell_id, patch_location)
        if self.predict:
            return self.transform(data_stack, self.transforms)
        if torch is None:
            raise ImportError("Training texture datasets requires torch.")

        targets = [self.transform(data_stack.copy(), self.transforms) for _ in range(2)]
        inputs = [self.transform(_clone_view(target), self.transforms_sim) for target in targets]
        return torch.stack(inputs), torch.stack(targets)


RawAEContrCellDataset = RawAutoencoderContrastiveCellDataset
TextPatchContrCellDataset = TextPatchContrastiveCellDataset
