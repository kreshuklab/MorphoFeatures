"""UMAP, Leiden clustering, and cluster evaluation utilities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from morphofeatures.data.embeddings import read_embedding_table

BIG_MERGE_IDS: tuple[int, ...] = (
    19150,
    21994,
    19465,
    27204,
    18324,
    28660,
    12279,
    20136,
    13911,
    18402,
    27925,
    17365,
    28685,
    23288,
    19856,
    25523,
    16801,
    18146,
    22422,
    28158,
    28058,
    30556,
    27641,
    17343,
    18437,
)


@dataclass(slots=True)
class LabeledEmbedding:
    """Embeddings aligned with optional cell-type metadata."""

    features: np.ndarray
    cell_types: np.ndarray
    type_names: list[str]
    ids: np.ndarray


def _standardize(features: np.ndarray) -> np.ndarray:
    """Standardize feature columns without requiring scikit-learn."""

    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (features - mean) / std


def get_data(embedding_file: str | Path, label_file: str | Path, excluded_ids: Sequence[int] = BIG_MERGE_IDS) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load embeddings and labels in the legacy tuple format."""

    labeled = load_labeled_embeddings(embedding_file, label_file, excluded_ids)
    return labeled.features, labeled.cell_types, labeled.type_names, labeled.ids


def load_labeled_embeddings(
    embedding_file: str | Path,
    label_file: str | Path,
    excluded_ids: Sequence[int] = BIG_MERGE_IDS,
) -> LabeledEmbedding:
    """Load and align embeddings with manually curated cell labels."""

    embedding = read_embedding_table(embedding_file)
    label_frame = pd.read_csv(label_file, sep="\t")
    to_select = (~label_frame["label_id"].isin(excluded_ids)) & (label_frame["is_corr"])
    selected_ids = label_frame.loc[to_select, "label_id"].astype(int).to_numpy()
    type_names = list(label_frame["cell_label"].unique())
    label_to_id = {label: index for index, label in enumerate(type_names)}
    cell_types = label_frame.loc[to_select, "cell_label"].map(label_to_id).to_numpy()
    id_to_index = {int(label_id): index for index, label_id in enumerate(embedding.ids)}
    selected_features = embedding.features[[id_to_index[int(label_id)] for label_id in selected_ids]]
    return LabeledEmbedding(_standardize(selected_features), cell_types, type_names, selected_ids)


def get_umap(
    embeddings: np.ndarray,
    neib: int = 15,
    metric: str = "euclidean",
    min_dist: float = 0.0,
    n_components: int = 2,
) -> np.ndarray:
    """Compute a UMAP embedding."""

    try:
        import umap
    except ImportError as exc:
        raise ImportError("UMAP computation requires umap-learn.") from exc

    fit_umap = umap.UMAP(n_neighbors=neib, metric=metric, min_dist=min_dist, n_components=n_components)
    return fit_umap.fit_transform(embeddings)


def leiden_labels_from_umap_graph(embeddings: np.ndarray, n_neighbors: int = 20, resolution: float = 0.004) -> tuple[np.ndarray, np.ndarray]:
    """Fit UMAP for its graph and run Leiden community detection."""

    try:
        import igraph as ig
        import leidenalg
        import networkx as nx
        import umap
    except ImportError as exc:
        raise ImportError("Leiden clustering requires umap-learn, networkx, igraph, and leidenalg.") from exc

    fit_umap = umap.UMAP(n_neighbors=n_neighbors, metric="euclidean", min_dist=0.0, n_components=2)
    umap_embedding = fit_umap.fit_transform(embeddings)
    graph_factory = getattr(nx, "from_scipy_sparse_array", nx.from_scipy_sparse_matrix)
    networkx_graph = graph_factory(fit_umap.graph_)
    graph = ig.Graph.from_networkx(networkx_graph)
    partition = leidenalg.find_partition(graph, leidenalg.CPMVertexPartition, resolution_parameter=resolution)
    return np.asarray(partition.membership), umap_embedding


def show_types_in_clusters(cluster_labels: np.ndarray, cell_types: np.ndarray, type_names: Sequence[str]) -> pd.DataFrame:
    """Return a cluster-by-cell-type count table."""

    clusters = np.unique(cluster_labels)
    type_distribution = np.zeros((int(np.max(clusters)) + 1, len(type_names)), dtype=int)
    for cluster_label in clusters:
        cluster_types = cell_types[cluster_labels == cluster_label]
        unique_types, type_counts = np.unique(cluster_types, return_counts=True)
        type_distribution[int(cluster_label)][unique_types.astype(int)] = type_counts
    return pd.DataFrame(data=type_distribution[:, 1:], columns=type_names[1:])


def analyze(
    cluster_labels: np.ndarray,
    cell_types: np.ndarray,
    cell_ids: np.ndarray,
    neighbors_by_cell: Mapping[int, Sequence[int]],
) -> float:
    """Compute the mean of cell-type homogeneity and bilateral-neighbor score."""

    try:
        from sklearn import metrics
    except ImportError as exc:
        raise ImportError("Cluster analysis requires scikit-learn.") from exc

    labeled_mask = cell_types != 0
    homogeneity = metrics.homogeneity_score(cell_types[labeled_mask], cluster_labels[labeled_mask])
    label_by_id = {int(label_id): label for label_id, label in zip(cell_ids, cluster_labels, strict=True)}
    in_same_cluster = []
    selected_ids = set(int(label_id) for label_id in cell_ids)
    for cell_id in cell_ids:
        label = label_by_id[int(cell_id)]
        if label == -1 or int(cell_id) not in neighbors_by_cell:
            continue
        neighbor_ids = [int(neighbor) for neighbor in neighbors_by_cell[int(cell_id)] if int(neighbor) in selected_ids]
        if neighbor_ids:
            in_same_cluster.append(any(label_by_id[neighbor] == label for neighbor in neighbor_ids))
    bilateral_score = float(np.mean(in_same_cluster)) if in_same_cluster else 0.0
    return float(np.mean([bilateral_score, homogeneity]))


def save_labels(ids: np.ndarray, cluster_labels: np.ndarray, umap_embedding: np.ndarray, out_name: str | Path) -> pd.DataFrame:
    """Save cluster membership and UMAP coordinates in MoBIE-compatible TSV format."""

    labels_frame = pd.DataFrame({"label_id": ids, "cluster": cluster_labels, "bool": np.ones_like(ids)})
    output = labels_frame.pivot(index="label_id", columns="cluster")["bool"].fillna(0)
    output["cluster"] = cluster_labels
    output["umap_1"], output["umap_2"] = umap_embedding[:, 0], umap_embedding[:, 1]
    output.to_csv(out_name, sep="\t")
    return output


def plot_types(embedding: np.ndarray, cell_types: np.ndarray, type_names: Sequence[str]) -> None:
    """Plot a UMAP embedding colored by cell type."""

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise ImportError("Plotting requires matplotlib and seaborn.") from exc

    colors = sns.color_palette("Set1")
    colors = [colors[index] for index in [4, 1, 3, 0, 2, 5, 6]]
    for cell_type in np.unique(cell_types):
        type_data = embedding[cell_types == cell_type]
        if cell_type == 0:
            plt.scatter(type_data[:, 0], type_data[:, 1], color="grey", alpha=0.2, label=type_names[cell_type])
        else:
            plt.scatter(type_data[:, 0], type_data[:, 1], color=colors[cell_type - 1], label=type_names[cell_type])
    plt.legend(loc="upper left", fontsize=15)
    plt.show()


def plot_labels_separately(umap_embedding: np.ndarray, cluster_labels: np.ndarray) -> None:
    """Plot one figure per cluster label."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires matplotlib.") from exc

    for label in np.unique(cluster_labels):
        plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=(0.5, 0.5, 0.5), s=0.1, alpha=0.5)
        plt.scatter(umap_embedding[cluster_labels == label, 0], umap_embedding[cluster_labels == label, 1], c="red", s=10)
        plt.show()


def main(argv: list[str] | None = None) -> None:
    """Run UMAP and Leiden clustering from the command line."""

    import pickle

    parser = argparse.ArgumentParser(description="Plot UMAP and cluster embeddings.")
    parser.add_argument("embedding_file", type=Path)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--plot-labels-separately", action="store_true")
    parser.add_argument("--umap-nn", type=int, default=15)
    parser.add_argument("--clust-nn", type=int, default=20)
    parser.add_argument("--clust-res", type=float, default=0.004)
    parser.add_argument("--label-file", type=Path, default=Path("analysis/data/types_and_intensity_corr.tsv"))
    parser.add_argument("--neighbors-file", type=Path, default=Path("analysis/data/bilateral_neighbors.pkl"))
    args = parser.parse_args(argv)

    with args.neighbors_file.open("rb") as handle:
        neighbors_by_cell = pickle.load(handle)
    embedding, cell_types, type_names, ids = get_data(args.embedding_file, args.label_file)
    umap_embedding = get_umap(embedding, neib=args.umap_nn)
    plot_types(umap_embedding, cell_types, type_names)
    labels, _ = leiden_labels_from_umap_graph(embedding, n_neighbors=args.clust_nn, resolution=args.clust_res)
    print(f"Score: {analyze(labels, cell_types, ids, neighbors_by_cell):.3f}")
    print(show_types_in_clusters(labels, cell_types, type_names))
    if args.plot_labels_separately:
        plot_labels_separately(umap_embedding, labels)
    if args.save_path:
        save_labels(ids, labels, umap_embedding, args.save_path)


if __name__ == "__main__":
    main()
