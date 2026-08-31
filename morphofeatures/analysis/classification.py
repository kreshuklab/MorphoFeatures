"""Deterministic classification utilities for legacy and new embeddings."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from morphofeatures.data.contracts import EmbeddingTable


LEGACY_CELL_TYPES = (
    "epithelial",
    "neuron",
    "midgut",
    "muscle",
    "secretory",
    "ciliated",
    "dark",
)


@dataclass(frozen=True)
class ClassificationResult:
    scores: np.ndarray
    confusion: np.ndarray
    class_names: Tuple[str, ...]

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean(self.scores))

    @property
    def std_accuracy(self) -> float:
        return float(np.std(self.scores))


def load_class_labels(
    path: Path, skip_types: Optional[Iterable[str]] = None
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    frame = pd.read_csv(path, sep="\t")
    required = {"label_id", "cell_type"}
    if not required.issubset(frame.columns):
        raise ValueError("Classification table requires label_id and cell_type columns")
    skipped = set(skip_types or ())
    frame = frame[~frame["cell_type"].isin(skipped)].copy()
    present = set(frame["cell_type"].astype(str))
    ordered = [name for name in LEGACY_CELL_TYPES if name in present and name not in skipped]
    ordered.extend(sorted(present.difference(ordered)))
    label_map = {name: index for index, name in enumerate(ordered)}
    labels = frame["cell_type"].map(label_map).to_numpy(dtype=np.int64)
    ids = frame["label_id"].to_numpy(dtype=np.int64)
    return ids, labels, tuple(ordered)


def select_labeled_embeddings(
    embeddings: EmbeddingTable, label_ids: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    positions = {int(label_id): index for index, label_id in enumerate(embeddings.label_ids)}
    missing = [int(label_id) for label_id in label_ids if int(label_id) not in positions]
    if missing:
        raise ValueError("{} labeled cells are missing from embeddings".format(len(missing)))
    rows = np.asarray([positions[int(label_id)] for label_id in label_ids], dtype=np.int64)
    return embeddings.features[rows], labels


def cross_validate_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    folds: int = 5,
    seed: int = 42,
    max_iter: int = 2000,
) -> ClassificationResult:
    _, counts = np.unique(labels, return_counts=True)
    if counts.size < 2:
        raise ValueError("At least two classes are required")
    n_splits = min(int(folds), int(counts.min()))
    if n_splits < 2:
        raise ValueError("Each class must have at least two examples")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores: List[float] = []
    matrices: List[np.ndarray] = []
    all_labels = np.arange(len(class_names))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for train_indices, test_indices in splitter.split(features, labels):
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=max_iter, random_state=seed)
            model.fit(features[train_indices], labels[train_indices])
            predictions = model.predict(features[test_indices])
            scores.append(float(model.score(features[test_indices], labels[test_indices])))
            matrices.append(confusion_matrix(labels[test_indices], predictions, labels=all_labels))
    return ClassificationResult(np.asarray(scores), np.sum(matrices, axis=0), tuple(class_names))


def fit_predict_probabilities(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    all_features: np.ndarray,
    seed: int = 42,
    max_iter: int = 2000,
) -> np.ndarray:
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=max_iter, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(train_features, train_labels)
    return model.predict_proba(all_features)
