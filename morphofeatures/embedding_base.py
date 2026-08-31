"""Common method interface for legacy and modern embedding workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from morphofeatures.analysis.classification import cross_validate_logistic
from morphofeatures.data.io import export_embeddings


def aggregate_patch_embeddings(label_ids, patch_features) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(label_ids, dtype=np.int64)
    features = np.asarray(patch_features)
    if len(ids) != len(features):
        raise ValueError("Patch ids and features must have the same row count")
    unique_ids = np.unique(ids)
    aggregated = np.vstack([features[ids == label_id].mean(axis=0) for label_id in unique_ids])
    return unique_ids, aggregated


class EmbeddingMethod(ABC):
    @abstractmethod
    def train(self, train_loader, validation_loader=None) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def encode_cells(self, loader) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def aggregate_patches(label_ids, patch_features):
        return aggregate_patch_embeddings(label_ids, patch_features)

    @staticmethod
    def export_embeddings(path: Path, label_ids, features) -> Path:
        return export_embeddings(path, label_ids, features)

    @staticmethod
    def evaluate_embeddings(features, labels, class_names: Iterable[str], seed: int = 42):
        standardized = StandardScaler().fit_transform(features)
        return cross_validate_logistic(standardized, labels, tuple(class_names), seed=seed)
