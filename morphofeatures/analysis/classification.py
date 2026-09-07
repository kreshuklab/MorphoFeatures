"""Deterministic classification utilities for legacy and new embeddings."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from morphofeatures.data.contracts import EmbeddingTable
from morphofeatures.data.io import load_embeddings

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
    class_names: tuple[str, ...]
    predictions: np.ndarray | None = None
    labels: np.ndarray | None = None
    label_ids: np.ndarray | None = None

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean(self.scores))

    @property
    def std_accuracy(self) -> float:
        return float(np.std(self.scores))

    @property
    def per_class_recall(self) -> np.ndarray:
        denominator = self.confusion.sum(axis=1)
        return np.divide(
            np.diag(self.confusion),
            denominator,
            out=np.full(len(denominator), np.nan, dtype=float),
            where=denominator > 0,
        )


def load_class_labels(
    path: Path, skip_types: Iterable[str] | None = None
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
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
) -> tuple[np.ndarray, np.ndarray]:
    positions = {int(label_id): index for index, label_id in enumerate(embeddings.label_ids)}
    missing = [int(label_id) for label_id in label_ids if int(label_id) not in positions]
    if missing:
        raise ValueError(f"{len(missing)} labeled cells are missing from embeddings")
    rows = np.asarray([positions[int(label_id)] for label_id in label_ids], dtype=np.int64)
    return embeddings.features[rows], labels


def cross_validate_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    folds: int = 5,
    seed: int = 42,
    max_iter: int = 2000,
    c: float = 1.0,
    class_weight: str | None = None,
) -> ClassificationResult:
    return cross_validate_shallow_classifier(
        features,
        labels,
        class_names,
        model="logistic",
        folds=folds,
        seed=seed,
        max_iter=max_iter,
        c=c,
        class_weight=class_weight,
    )


def cross_validate_shallow_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    *,
    model: str = "logistic",
    folds: int = 5,
    seed: int = 42,
    max_iter: int = 2000,
    c: float = 1.0,
    class_weight: str | None = None,
    hidden_dimensions: Sequence[int] = (64,),
    label_ids: np.ndarray | None = None,
) -> ClassificationResult:
    """Evaluate a linear probe or one-hidden-layer MLP without scaling leakage."""

    features = np.asarray(features)
    labels = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("features must be 2D with one row per label")
    if not np.all(np.isfinite(features)):
        raise ValueError("Classifier features contain NaN or infinite values")
    _, counts = np.unique(labels, return_counts=True)
    if counts.size < 2:
        raise ValueError("At least two classes are required")
    n_splits = min(int(folds), int(counts.min()))
    if n_splits < 2:
        raise ValueError("Each class must have at least two examples")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores: list[float] = []
    matrices: list[np.ndarray] = []
    predictions = np.full(len(labels), -1, dtype=np.int64)
    all_labels = np.arange(len(class_names))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for train_indices, test_indices in splitter.split(features, labels):
            if model == "logistic":
                estimator = LogisticRegression(
                    C=float(c),
                    solver="lbfgs",
                    max_iter=int(max_iter),
                    random_state=seed,
                    class_weight=class_weight,
                )
            elif model == "mlp":
                hidden = tuple(int(value) for value in hidden_dimensions)
                if not hidden or any(value <= 0 for value in hidden):
                    raise ValueError("hidden_dimensions must contain positive integers")
                estimator = MLPClassifier(
                    hidden_layer_sizes=hidden,
                    activation="relu",
                    alpha=1e-4,
                    max_iter=int(max_iter),
                    random_state=seed,
                )
            else:
                raise ValueError("model must be logistic or mlp")
            pipeline = make_pipeline(StandardScaler(), estimator)
            pipeline.fit(features[train_indices], labels[train_indices])
            fold_predictions = pipeline.predict(features[test_indices])
            predictions[test_indices] = fold_predictions
            scores.append(float(np.mean(fold_predictions == labels[test_indices])))
            matrices.append(
                confusion_matrix(labels[test_indices], fold_predictions, labels=all_labels)
            )
    return ClassificationResult(
        np.asarray(scores),
        np.sum(matrices, axis=0),
        tuple(class_names),
        predictions,
        labels.copy(),
        None if label_ids is None else np.asarray(label_ids, dtype=np.int64).copy(),
    )


def evaluate_embedding_classifier(
    embedding_path: Path,
    labels_path: Path,
    *,
    output_dir: Path | None = None,
    model: str = "logistic",
    folds: int = 5,
    seed: int = 42,
    max_iter: int = 2000,
    c: float = 1.0,
    class_weight: str | None = "balanced",
    hidden_dimensions: Sequence[int] = (64,),
    minimum_class_count: int = 2,
    skip_types: Iterable[str] | None = None,
) -> ClassificationResult:
    """Join annotations by ``label_id``, run deterministic CV, and save interpretable tables."""

    embeddings = load_embeddings(Path(embedding_path))
    annotation_ids, labels, class_names = load_class_labels(Path(labels_path), skip_types)
    positions = {int(label_id): index for index, label_id in enumerate(embeddings.label_ids)}
    keep = np.asarray([int(label_id) in positions for label_id in annotation_ids], dtype=bool)
    matched_ids = annotation_ids[keep]
    matched_labels = labels[keep]
    if not len(matched_ids):
        raise ValueError("No annotation label_id values occur in the embedding table")
    rows = np.asarray([positions[int(label_id)] for label_id in matched_ids], dtype=np.int64)
    features = embeddings.features[rows]
    present, counts = np.unique(matched_labels, return_counts=True)
    retained = present[counts >= max(2, int(minimum_class_count))]
    retained_mask = np.isin(matched_labels, retained)
    matched_ids = matched_ids[retained_mask]
    matched_labels = matched_labels[retained_mask]
    features = features[retained_mask]
    if len(retained) < 2:
        raise ValueError("At least two matched cell types need two or more examples")
    remap = {int(old): new for new, old in enumerate(retained)}
    remapped = np.asarray([remap[int(value)] for value in matched_labels], dtype=np.int64)
    retained_names = tuple(class_names[int(value)] for value in retained)
    result = cross_validate_shallow_classifier(
        features,
        remapped,
        retained_names,
        model=model,
        folds=folds,
        seed=seed,
        max_iter=max_iter,
        c=c,
        class_weight=class_weight if model == "logistic" else None,
        hidden_dimensions=hidden_dimensions,
        label_ids=matched_ids,
    )
    if output_dir is not None:
        import json

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result.confusion, index=result.class_names, columns=result.class_names).to_csv(
            output / "confusion_matrix.tsv", sep="\t", index_label="true_cell_type"
        )
        pd.DataFrame(
            {
                "label_id": result.label_ids,
                "true_cell_type": [result.class_names[value] for value in result.labels],
                "predicted_cell_type": [
                    result.class_names[value] for value in result.predictions
                ],
            }
        ).to_csv(output / "cross_validated_predictions.tsv", sep="\t", index=False)
        summary = {
            "model": model,
            "embedding": str(Path(embedding_path).resolve()),
            "labels": str(Path(labels_path).resolve()),
            "n_matched": int(len(result.labels)),
            "class_names": list(result.class_names),
            "fold_accuracies": result.scores.tolist(),
            "mean_accuracy": result.mean_accuracy,
            "std_accuracy": result.std_accuracy,
            "per_class_recall": {
                name: float(value) for name, value in zip(result.class_names, result.per_class_recall)
            },
            "caveat": (
                "Cross-validation measures association and predictive separability; it does not "
                "establish a biological mechanism or independent-animal generalization."
            ),
        }
        (output / "classifier_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


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
