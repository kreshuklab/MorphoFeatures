"""Dataloader factories for shape models."""

from __future__ import annotations

from typing import Any

import numpy as np

from morphofeatures.data.splits import train_val_split
from morphofeatures.shape.datasets import ShapePointCloudDataset, load_shape_arrays


def _collate_shape(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate shape samples and flatten contrastive views."""

    try:
        import torch
    except ImportError as exc:
        raise ImportError("Shape dataloaders require torch.") from exc

    ids = torch.stack([item["id"] for item in batch])
    points = torch.stack([item["points"] for item in batch])
    features = torch.stack([item["features"] for item in batch])
    if points.ndim == 4:
        batch_size, views, n_points, dims = points.shape
        points = points.reshape(batch_size * views, n_points, dims)
        features = features.reshape(batch_size * views, n_points, features.shape[-1])
        ids = ids.reshape(batch_size * views)
    return {"id": ids, "points": points, "features": features}


def _make_loader(dataset: Any, loader_config: dict[str, Any]) -> Any:
    """Instantiate a torch DataLoader."""

    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("Shape dataloaders require torch.") from exc

    return DataLoader(dataset, collate_fn=_collate_shape, **loader_config)


def get_train_val_loaders(data_config: dict[str, Any], loader_config: dict[str, Any]) -> dict[str, Any]:
    """Build train and validation loaders for contrastive shape training."""

    points, features, ids = load_shape_arrays(data_config)
    if ids is None:
        ids = np.arange(points.shape[0])

    validation_fraction = float(data_config.get("validation_fraction", data_config.get("split", 0.2)))
    train_ids, val_ids = train_val_split(ids, validation_fraction=validation_fraction, seed=data_config.get("seed"))
    id_to_index = {int(label_id): index for index, label_id in enumerate(ids)}
    train_indices = np.asarray([id_to_index[int(label_id)] for label_id in train_ids])
    val_indices = np.asarray([id_to_index[int(label_id)] for label_id in val_ids])

    train_dataset = ShapePointCloudDataset(
        points[train_indices],
        None if features is None else features[train_indices],
        ids[train_indices],
        contrastive=True,
    )
    val_dataset = ShapePointCloudDataset(
        points[val_indices],
        None if features is None else features[val_indices],
        ids[val_indices],
        contrastive=True,
    )
    train_loader_config = dict(loader_config)
    val_loader_config = dict(loader_config)
    val_loader_config["shuffle"] = False
    return {
        "train": _make_loader(train_dataset, train_loader_config),
        "val": _make_loader(val_dataset, val_loader_config),
    }


def get_simple_loader(data_config: dict[str, Any], loader_config: dict[str, Any]) -> Any:
    """Build a non-contrastive loader for shape inference."""

    points, features, ids = load_shape_arrays(data_config)
    dataset = ShapePointCloudDataset(points, features, ids, contrastive=False)
    return _make_loader(dataset, dict(loader_config))
