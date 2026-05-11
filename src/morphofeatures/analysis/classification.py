"""Cell-type classification utilities for MorphoFeatures embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from morphofeatures.data.embeddings import EmbeddingTable, merge_embedding_tables, read_embedding_table

CELL_TYPES: tuple[str, ...] = ("epithelial", "neuron", "midgut", "muscle", "secretory", "ciliated", "dark")


def reorder(embedding: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort an embedding matrix by corresponding IDs."""

    table = EmbeddingTable(ids=indices, features=embedding)
    sorted_table = read_sorted_table(table)
    return sorted_table.features, sorted_table.ids


def read_sorted_table(table: EmbeddingTable) -> EmbeddingTable:
    """Return an already-loaded embedding table sorted by label ID."""

    order = np.argsort(table.ids)
    return EmbeddingTable(ids=table.ids[order], features=table.features[order])


def get_embed(embed_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one embedding file and return ``(features, ids)``."""

    table = read_embedding_table(embed_path)
    return table.features, table.ids


def merge_embeds(embedding_files: Sequence[str | Path], scale: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Merge one or more embedding files by feature columns."""

    table = merge_embedding_tables(embedding_files, scale=scale)
    return table.features, table.ids


def get_labels(label_path: str | Path, skip_types: Sequence[str] | None = None) -> tuple[np.ndarray, tuple[str, ...]]:
    """Load supervised cell-type labels.

    Args:
        label_path: TSV with ``label_id`` and ``cell_type`` columns.
        skip_types: Optional cell types to exclude.

    Returns:
        ``(id_label_pairs, class_names)`` where pairs contain label ID and class index.
    """

    label_frame = pd.read_csv(label_path, sep="\t")
    class_names = tuple(cell_type for cell_type in CELL_TYPES if not skip_types or cell_type not in skip_types)
    if skip_types:
        label_frame = label_frame[~label_frame["cell_type"].isin(skip_types)]
    type_to_label = {cell_type: index for index, cell_type in enumerate(class_names)}
    id_to_label = {
        int(row["label_id"]): type_to_label[row["cell_type"]]
        for _, row in label_frame.iterrows()
        if row["cell_type"] in type_to_label
    }
    return np.asarray(list(id_to_label.items()), dtype=int), class_names


def get_class_embeds(all_encoded: np.ndarray, all_ids: np.ndarray, id_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select embeddings for labeled cells."""

    ids = id_labels[:, 0].astype(int)
    labels = id_labels[:, 1].astype(int)
    id_to_index = {int(label_id): index for index, label_id in enumerate(all_ids)}
    selected_indices = [id_to_index[int(label_id)] for label_id in ids]
    return all_encoded[selected_indices], labels


def train_cv_regr(data: np.ndarray, labels: np.ndarray, class_names: Sequence[str] = CELL_TYPES, n_splits: int = 5) -> dict[str, np.ndarray | float]:
    """Train and evaluate logistic regression with stratified cross-validation."""

    try:
        from sklearn import metrics
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        import warnings
    except ImportError as exc:
        raise ImportError("Classification requires scikit-learn.") from exc

    splitter = StratifiedKFold(n_splits=n_splits)
    scores = []
    confusion_matrices = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for train_idx, test_idx in splitter.split(data, labels):
            logistic_regr = LogisticRegression(C=1, multi_class="auto", solver="lbfgs", max_iter=1000)
            logistic_regr.fit(data[train_idx], labels[train_idx])
            score = logistic_regr.score(data[test_idx], labels[test_idx])
            scores.append(score)
            predictions = logistic_regr.predict(data[test_idx])
            confusion_matrices.append(metrics.confusion_matrix(labels[test_idx], predictions))

    score_array = np.asarray(scores)
    return {
        "mean_accuracy": float(score_array.mean()),
        "std_accuracy": float(score_array.std()),
        "confusion_matrix": np.sum(np.asarray(confusion_matrices), axis=0),
        "class_names": np.asarray(class_names),
    }


def predict_and_save(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    all_features: np.ndarray,
    cell_ids: np.ndarray,
    path_to_save: str | Path,
    class_names: Sequence[str] = CELL_TYPES,
) -> pd.DataFrame:
    """Train logistic regression on labeled cells and save probabilities for all cells."""

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise ImportError("Classification requires scikit-learn.") from exc

    model = LogisticRegression(C=1, multi_class="auto", solver="lbfgs", max_iter=1000)
    model.fit(train_features, train_labels)
    predictions = model.predict_proba(all_features)
    output = np.column_stack((cell_ids[:, np.newaxis], predictions))
    prediction_frame = pd.DataFrame(data=output, columns=["label_id", *class_names])
    prediction_frame.to_csv(path_to_save, index=False, sep="\t")
    return prediction_frame


def main(argv: list[str] | None = None) -> None:
    """Run the logistic-regression classification CLI."""

    parser = argparse.ArgumentParser(description="Train logistic regression to classify embeddings.")
    parser.add_argument("embedding_files", type=Path, nargs="+")
    parser.add_argument("--train-data-file", type=Path, default=Path("analysis/data/class_labels.tsv"))
    parser.add_argument("--pred-path", type=Path, default=None)
    parser.add_argument("--skip-types", type=str, default=None, nargs="*")
    parser.add_argument("--agglomerate", type=int, default=None)
    args = parser.parse_args(argv)

    features, label_ids = merge_embeds(args.embedding_files)
    if args.agglomerate is not None:
        try:
            from sklearn import cluster
        except ImportError as exc:
            raise ImportError("Feature agglomeration requires scikit-learn.") from exc
        features = cluster.FeatureAgglomeration(n_clusters=args.agglomerate).fit_transform(features)

    train_data, class_names = get_labels(args.train_data_file, skip_types=args.skip_types)
    train_features, train_labels = get_class_embeds(features, label_ids, train_data)
    result = train_cv_regr(train_features, train_labels, class_names)
    print(f"Mean: {result['mean_accuracy']:.4f}, std: {result['std_accuracy']:.4f}")
    print(result["confusion_matrix"])
    if args.pred_path:
        predict_and_save(train_features, train_labels, features, label_ids, args.pred_path, class_names)


if __name__ == "__main__":
    main()
