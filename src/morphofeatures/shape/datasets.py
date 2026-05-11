"""Datasets for point-cloud shape models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[misc, assignment]


Transform = Callable[[np.ndarray], np.ndarray]


def _channel_first(array: np.ndarray) -> np.ndarray:
    """Convert point/feature arrays from ``(N, C)`` to ``(C, N)`` when needed."""

    if array.ndim == 2 and (array.shape[0] > array.shape[1] or array.shape[-1] in {3, 6}):
        return array.T
    return array


class ShapePointCloudDataset(Dataset):  # type: ignore[misc]
    """Dataset backed by arrays of point clouds and per-point features."""

    def __init__(
        self,
        points: np.ndarray,
        features: np.ndarray | None = None,
        ids: np.ndarray | None = None,
        transform: Transform | None = None,
        contrastive: bool = False,
    ) -> None:
        """Initialize a point-cloud dataset from in-memory arrays."""

        self.points = np.asarray(points, dtype=np.float32)
        self.features_from_points = features is None
        self.features = np.asarray(features if features is not None else points, dtype=np.float32)
        if self.points.shape[0] != self.features.shape[0]:
            raise ValueError("points and features must have the same number of samples.")
        self.ids = np.asarray(ids if ids is not None else np.arange(self.points.shape[0]), dtype=np.int64)
        if self.ids.shape[0] != self.points.shape[0]:
            raise ValueError("ids must have one value per sample.")
        self.transform = transform
        self.contrastive = contrastive

    def __len__(self) -> int:
        """Return the number of point clouds."""

        return int(self.points.shape[0])

    def _view(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return one transformed point/feature view."""

        points = self.points[index].copy()
        features = self.features[index].copy()
        if self.transform is not None:
            points = self.transform(points)
            if self.features_from_points:
                features = points
        return points, features

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one sample, duplicating views for contrastive training."""

        if torch is None:
            raise ImportError("ShapePointCloudDataset requires torch when indexed.")

        if self.contrastive:
            views = [self._view(index), self._view(index)]
            points = torch.as_tensor(np.stack([_channel_first(view[0]) for view in views]), dtype=torch.float32)
            features = torch.as_tensor(np.stack([_channel_first(view[1]) for view in views]), dtype=torch.float32)
            ids = torch.as_tensor(np.repeat(self.ids[index], 2), dtype=torch.long)
        else:
            points_array, features_array = self._view(index)
            points = torch.as_tensor(_channel_first(points_array), dtype=torch.float32)
            features = torch.as_tensor(_channel_first(features_array), dtype=torch.float32)
            ids = torch.as_tensor(self.ids[index], dtype=torch.long)
        return {"id": ids, "points": points, "features": features}


def load_shape_arrays(data_config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load point, feature, and ID arrays from a shape data config."""

    if "npz_path" in data_config:
        loaded = np.load(data_config["npz_path"])
        points = loaded["points"]
        features = loaded["features"] if "features" in loaded else None
        ids = loaded["ids"] if "ids" in loaded else None
        return points, features, ids

    if "points_path" not in data_config:
        raise KeyError("Shape data config requires 'npz_path' or 'points_path'.")

    points = np.load(data_config["points_path"])
    features = np.load(data_config["features_path"]) if data_config.get("features_path") else None
    ids = np.load(data_config["ids_path"]) if data_config.get("ids_path") else None
    return points, features, ids
