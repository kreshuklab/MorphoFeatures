"""Point-cloud loaders replacing the missing legacy shape data module."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset


SUPPORTED_SUFFIXES = {".npy", ".npz", ".off", ".ply", ".obj"}


def _normalize(points):
    centered = points - points.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(centered, axis=1).max()
    return centered if scale == 0 else centered / scale


def _random_rotation(rng):
    matrix = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)


class PointCloudDataset(Dataset):
    def __init__(self, paths: Sequence[Path], label_ids: Sequence[int], num_points=1024,
                 contrastive=False, augment=True, seed=42):
        self.paths = [Path(path) for path in paths]
        self.label_ids = np.asarray(label_ids, dtype=np.int64)
        self.num_points, self.contrastive, self.augment, self.seed = int(num_points), bool(contrastive), bool(augment), int(seed)
        if len(self.paths) != len(self.label_ids):
            raise ValueError("paths and label_ids must have identical lengths")

    def __len__(self):
        return len(self.paths)

    def _load(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        if path.suffix.lower() == ".npz":
            archive = np.load(path)
            points = np.asarray(archive["points"])
            normals = np.asarray(archive["normals"]) if "normals" in archive else np.zeros_like(points)
        elif path.suffix.lower() == ".npy":
            matrix = np.load(path)
            points, normals = np.asarray(matrix[:, :3]), np.asarray(matrix[:, 3:6]) if matrix.shape[1] >= 6 else np.zeros_like(matrix[:, :3])
        else:
            try:
                import trimesh
            except ImportError as error:
                raise RuntimeError("Mesh inputs require morphofeatures[legacy-training]") from error
            mesh = trimesh.load_mesh(path, process=False)
            points, face_indices = trimesh.sample.sample_surface(mesh, self.num_points)
            normals = mesh.face_normals[face_indices]
        return points.astype(np.float32), normals.astype(np.float32)

    def _view(self, points, normals, rng):
        indices = rng.choice(len(points), size=self.num_points, replace=len(points) < self.num_points)
        points, normals = _normalize(points[indices]), normals[indices]
        if self.augment:
            rotation = _random_rotation(rng)
            points, normals = points @ rotation.T, normals @ rotation.T
            points *= rng.uniform(0.9, 1.1, size=3)
        return points.T.astype(np.float32), np.concatenate((points, normals), axis=1).T.astype(np.float32)

    def __getitem__(self, index):
        points, normals = self._load(self.paths[index])
        rng = np.random.default_rng(self.seed + index + np.random.randint(0, 2**16))
        if self.contrastive:
            views = [self._view(points, normals, rng) for _ in range(2)]
            points_out = torch.from_numpy(np.stack([view[0] for view in views]))
            features_out = torch.from_numpy(np.stack([view[1] for view in views]))
        else:
            view = self._view(points, normals, rng)
            points_out, features_out = torch.from_numpy(view[0]), torch.from_numpy(view[1])
        return {"id": torch.tensor(self.label_ids[index], dtype=torch.long),
                "points": points_out, "features": features_out}


def _discover(config: Dict) -> Tuple[List[Path], np.ndarray]:
    if config.get("manifest"):
        manifest = Path(config["manifest"])
        frame = pd.read_csv(manifest, sep="\t")
        if not {"label_id", "path"}.issubset(frame.columns):
            raise ValueError("Shape manifest requires label_id and path columns")
        paths = [Path(value) if Path(value).is_absolute() else manifest.resolve().parent / value for value in frame["path"]]
        return paths, frame["label_id"].to_numpy(dtype=np.int64)
    root = Path(config["root"])
    paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in SUPPORTED_SUFFIXES)
    return paths, np.asarray([int(path.stem) for path in paths], dtype=np.int64)


def _collate_contrastive(batch):
    return {"id": torch.stack([item["id"] for item in batch]).repeat_interleave(2),
            "points": torch.cat([item["points"] for item in batch]),
            "features": torch.cat([item["features"] for item in batch])}


def get_train_val_loaders(dataset_config: Dict, loader_config: Dict):
    paths, ids = _discover(dataset_config)
    dataset = PointCloudDataset(paths, ids, num_points=dataset_config.get("num_points", 1024),
                                contrastive=True, seed=dataset_config.get("seed", 42))
    order = np.random.default_rng(dataset.seed).permutation(len(dataset))
    validation_size = max(1, int(round(len(dataset) * float(dataset_config.get("validation_fraction", 0.2)))))
    if len(order) - validation_size < 1:
        raise ValueError("Shape training requires at least two point clouds")
    train_config, validation_config = dict(loader_config), dict(loader_config)
    train_config.setdefault("shuffle", True)
    validation_config["shuffle"] = False
    return {"train": DataLoader(Subset(dataset, order[validation_size:]), collate_fn=_collate_contrastive, **train_config),
            "val": DataLoader(Subset(dataset, order[:validation_size]), collate_fn=_collate_contrastive, **validation_config)}


def get_simple_loader(dataset_config: Dict, loader_config: Dict):
    paths, ids = _discover(dataset_config)
    dataset = PointCloudDataset(paths, ids, num_points=dataset_config.get("num_points", 1024),
                                contrastive=False, augment=False, seed=dataset_config.get("seed", 42))
    config = dict(loader_config)
    config["shuffle"] = False
    return DataLoader(dataset, **config)
