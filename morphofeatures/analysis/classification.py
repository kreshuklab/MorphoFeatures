"""Deterministic classification utilities for legacy and new embeddings."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

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


CLASSIFIER_NAMES = {
    "logistic": "Linear classifier (logistic regression)",
    "linear": "Linear classifier (logistic regression)",
    "mlp": "MLP neural network",
    "knn": "K-nearest neighbors",
}


class UnsupervisedUMAP(TransformerMixin, BaseEstimator):
    """Training-only UMAP with bounded exact-neighbor transforms and numeric persistence.

    A normal sklearn Pipeline passes y to its transformers. UMAP would then use
    the class labels to build a supervised map, so deliberately ignore y here.
    Exact transform neighbors avoid serializing executable ANN search objects.
    """

    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, n_epochs=200, seed=42):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_epochs = n_epochs
        self.seed = seed

    def _model(self, count):
        from umap import UMAP

        return UMAP(
            n_components=self.n_components,
            n_neighbors=min(self.n_neighbors, count - 1),
            min_dist=self.min_dist,
            n_epochs=self.n_epochs,
            random_state=self.seed,
            transform_seed=self.seed,
            init="random",
            n_jobs=1,
            metric="euclidean",
        )

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32, order="C")
        if len(X) < 4:
            raise ValueError(
                "Classifier UMAP requires at least four objects in every training fold"
            )
        self.reducer_ = self._model(len(X)).fit(X)  # Never pass y to UMAP.
        # Use this same transform implementation before and after saving a reference.
        self.reducer_._small_data = True
        self.reducer_._input_distance_func = "euclidean"
        if hasattr(self.reducer_, "_knn_search_index"):
            del self.reducer_._knn_search_index
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "reducer_")
        X = np.asarray(X, dtype=np.float32, order="C")
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError(
                "Classifier UMAP inputs must be finite and match the fitted feature dimensions"
            )
        raw = self.reducer_._raw_data
        if X.shape == raw.shape and np.array_equal(X, raw):
            return self.reducer_.embedding_.copy()
        if not len(X):
            return np.empty((0, self.n_components), dtype=np.float32)
        batch = max(1, min(128, 2_000_000 // len(raw)))
        result = np.concatenate(
            [self.reducer_.transform(X[i : i + batch]) for i in range(0, len(X), batch)]
        )
        if not np.isfinite(result).all():
            raise ValueError(
                "Classifier UMAP produced nonfinite coordinates; revise the classifier projection settings"
            )
        return result

    def numeric_state(self):
        import umap

        return (
            {
                "version": umap.__version__,
                "a": float(self.reducer_._a),
                "b": float(self.reducer_._b),
                "initial_alpha": float(self.reducer_._initial_alpha),
            },
            self.reducer_._raw_data,
            self.reducer_.embedding_,
        )

    def restore_numeric(self, state, raw, embedding):
        import joblib
        import umap

        if state["version"] != umap.__version__:
            raise ValueError(
                f"This reference UMAP was saved with umap-learn {state['version']}; use that version to reuse its fitted transform"
            )
        raw = np.asarray(raw, dtype=np.float32, order="C")
        self.reducer_ = self._model(len(raw))
        # Attributes used by UMAP's euclidean, dense, exact-neighbor transform path.
        self.reducer_._raw_data = raw
        self.reducer_.embedding_ = np.asarray(embedding, dtype=np.float32)
        self.reducer_._input_hash = joblib.hash(raw)
        self.reducer_._n_neighbors = min(self.n_neighbors, len(raw) - 1)
        self.reducer_._small_data = True
        self.reducer_._sparse_data = False
        self.reducer_._input_distance_func = "euclidean"
        self.reducer_._metric_kwds = {}
        self.reducer_._disconnection_distance = np.inf
        self.reducer_._a, self.reducer_._b = state["a"], state["b"]
        self.reducer_._initial_alpha = state["initial_alpha"]
        self.reducer_.tqdm_kwds = {}
        self.n_features_in_ = raw.shape[1]
        return self


def classifier_space(settings):
    return settings.get("classifier_space") or (
        "pca" if settings.get("evaluation_pca") else "embeddings"
    )


def classifier_preprocessing(
    *,
    normalization="standardize",
    evaluation_pca=None,
    classifier_space=None,
    classifier_dimensions=None,
    classifier_standardize=False,
    classifier_umap_neighbors=15,
    classifier_umap_min_dist=0.1,
    classifier_umap_epochs=200,
    seed=42,
):
    from sklearn.decomposition import PCA

    steps = []
    if normalization == "standardize":
        steps.append(StandardScaler())
    elif normalization == "l2":
        steps.append(Normalizer())
    elif normalization != "none":
        raise ValueError("normalization must be standardize, l2, or none")
    space = classifier_space or ("pca" if evaluation_pca else "embeddings")
    if space not in {"embeddings", "pca", "umap"}:
        raise ValueError("classifier_space must be embeddings, pca or umap")
    if space != "embeddings":
        dimensions = (
            classifier_dimensions
            if classifier_dimensions is not None
            else evaluation_pca
            if space == "pca" and evaluation_pca
            else 2
        )
        if isinstance(dimensions, bool) or int(dimensions) != dimensions or int(dimensions) < 1:
            raise ValueError("Classifier dimensions must be a positive integer")
        if space == "pca":
            steps.append(PCA(n_components=int(dimensions), random_state=int(seed)))
        else:
            if (
                int(classifier_umap_neighbors) < 2
                or int(classifier_umap_neighbors) != classifier_umap_neighbors
            ):
                raise ValueError("Classifier UMAP neighbors must be an integer of at least 2")
            if not 0 <= float(classifier_umap_min_dist) <= 1:
                raise ValueError("Classifier UMAP minimum distance must be between 0 and 1")
            if (
                int(classifier_umap_epochs) < 3
                or int(classifier_umap_epochs) != classifier_umap_epochs
            ):
                raise ValueError("Classifier UMAP epochs must be an integer of at least 3")
            steps.append(
                UnsupervisedUMAP(
                    int(dimensions),
                    int(classifier_umap_neighbors),
                    float(classifier_umap_min_dist),
                    int(classifier_umap_epochs),
                    int(seed),
                )
            )
    if classifier_standardize:
        steps.append(StandardScaler())
    return steps


CLASSIFIER_TRANSFORM_KEYS = (
    "classifier_space", "classifier_dimensions", "classifier_standardize",
    "classifier_umap_neighbors", "classifier_umap_min_dist", "classifier_umap_epochs",
)


def classifier_pipeline(
    model="logistic",
    *,
    normalization="standardize",
    evaluation_pca=None,
    seed=42,
    max_iter=2000,
    c=1.0,
    class_weight="balanced",
    hidden_dimensions=(64,),
    knn_k=5,
    classifier_space=None,
    classifier_dimensions=None,
    classifier_standardize=False,
    classifier_umap_neighbors=15,
    classifier_umap_min_dist=0.1,
    classifier_umap_epochs=200,
):
    """The shared Tools/workflow estimator; fit preprocessing on training rows only."""
    from sklearn.neighbors import KNeighborsClassifier

    steps = classifier_preprocessing(
        normalization=normalization,
        evaluation_pca=evaluation_pca,
        seed=seed,
        classifier_space=classifier_space,
        classifier_dimensions=classifier_dimensions,
        classifier_standardize=classifier_standardize,
        classifier_umap_neighbors=classifier_umap_neighbors,
        classifier_umap_min_dist=classifier_umap_min_dist,
        classifier_umap_epochs=classifier_umap_epochs,
    )
    if model in {"logistic", "linear"}:
        estimator = LogisticRegression(
            C=float(c),
            solver="lbfgs",
            max_iter=int(max_iter),
            random_state=int(seed),
            class_weight=class_weight,
        )
    elif model == "mlp":
        hidden = tuple(int(v) for v in hidden_dimensions)
        if not hidden or min(hidden) < 1:
            raise ValueError("hidden_dimensions must contain positive integers")
        estimator = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            alpha=1e-4,
            max_iter=int(max_iter),
            random_state=int(seed),
        )
    elif model == "knn":
        if int(knn_k) < 1:
            raise ValueError("knn_k must be positive")
        estimator = KNeighborsClassifier(n_neighbors=int(knn_k))
    else:
        raise ValueError("model must be logistic, mlp, or knn")
    return make_pipeline(*steps, estimator)


def configured_classifier(model, settings, *, train_count):
    return classifier_pipeline(
        model,
        normalization=settings.get("normalization", "standardize"),
        evaluation_pca=settings.get("evaluation_pca"),
        seed=settings.get("seed", 42),
        max_iter=settings.get("max_iter", 2000),
        c=settings.get("linear_c", 1.0),
        class_weight=settings.get("class_weight", "balanced"),
        hidden_dimensions=settings.get("hidden_dimensions", [64]),
        knn_k=min(int(settings.get("knn_k", 5)), train_count),
        **{k: settings[k] for k in CLASSIFIER_TRANSFORM_KEYS if k in settings},
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
    from morphofeatures.analysis.annotations import read_annotations

    frame = read_annotations(path, {"label_column": "cell_type"})
    frame = frame[frame.known_label.notna()].copy()
    frame["cell_type"] = frame.known_label
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
    classifier_settings: dict | None = None,
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
            pipeline = classifier_pipeline(
                model,
                seed=seed,
                max_iter=max_iter,
                c=c,
                class_weight=class_weight,
                hidden_dimensions=hidden_dimensions,
                **{
                    k: v
                    for k, v in (classifier_settings or {}).items()
                    if k in (*CLASSIFIER_TRANSFORM_KEYS, "normalization", "evaluation_pca")
                },
            )
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
    classifier_settings: dict | None = None,
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
    order = np.argsort(matched_ids)
    matched_ids, matched_labels, features = (
        matched_ids[order],
        matched_labels[order],
        features[order],
    )
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
        class_weight=class_weight if model in {"logistic", "linear"} else None,
        hidden_dimensions=hidden_dimensions,
        label_ids=matched_ids,
        classifier_settings=classifier_settings,
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
                "predicted_cell_type": [result.class_names[value] for value in result.predictions],
            }
        ).to_csv(output / "cross_validated_predictions.tsv", sep="\t", index=False)
        summary = {
            "model": model,
            "classifier_settings": {"normalization": "standardize", **(classifier_settings or {})},
            "embedding": str(Path(embedding_path).resolve()),
            "labels": str(Path(labels_path).resolve()),
            "n_matched": int(len(result.labels)),
            "class_names": list(result.class_names),
            "fold_accuracies": result.scores.tolist(),
            "mean_accuracy": result.mean_accuracy,
            "std_accuracy": result.std_accuracy,
            "per_class_recall": {
                name: float(value)
                for name, value in zip(result.class_names, result.per_class_recall)
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
