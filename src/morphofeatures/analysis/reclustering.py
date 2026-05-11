"""Utilities for reclustering one existing cluster label."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from morphofeatures.analysis.clustering import analyze, get_data, save_labels


def recluster_label(
    embeddings: np.ndarray,
    umap_embedding: np.ndarray,
    current_labels: np.ndarray,
    label_to_recluster: int,
    cell_types: np.ndarray,
    cell_ids: np.ndarray,
    neighbors_by_cell: Mapping[int, Sequence[int]],
    resolution: float = 0.05,
    n_neighbors: int = 20,
    plot: bool = True,
) -> np.ndarray:
    """Recluster one label at higher Leiden resolution."""

    try:
        import igraph as ig
        import leidenalg
        import networkx as nx
        import umap
    except ImportError as exc:
        raise ImportError("Reclustering requires umap-learn, networkx, igraph, and leidenalg.") from exc

    selected = current_labels == label_to_recluster
    fit_umap = umap.UMAP(n_neighbors=n_neighbors, metric="euclidean", min_dist=0.0, n_components=2)
    fit_umap.fit_transform(embeddings[selected])
    graph_factory = getattr(nx, "from_scipy_sparse_array", nx.from_scipy_sparse_matrix)
    graph = ig.Graph.from_networkx(graph_factory(fit_umap.graph_))
    partition = leidenalg.find_partition(graph, leidenalg.CPMVertexPartition, resolution_parameter=resolution)
    part_labels = np.asarray(partition.membership) + 1
    updated_labels = np.copy(current_labels)
    updated_labels[selected] = part_labels + np.max(current_labels)

    if plot:
        import matplotlib.pyplot as plt

        plot_labels = np.zeros_like(current_labels)
        plot_labels[selected] = part_labels
        plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=plot_labels, cmap="tab10")
        plt.show()

    print("Before", analyze(current_labels, cell_types, cell_ids, neighbors_by_cell))
    print("After", analyze(updated_labels, cell_types, cell_ids, neighbors_by_cell))
    return updated_labels


def load_cl_labels(cluster_file_name: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load cluster labels and UMAP coordinates from a clustering TSV."""

    labels_matrix = np.asarray(pd.read_csv(cluster_file_name, sep="\t"))
    labels = labels_matrix[:, -3]
    umap_embedding = labels_matrix[:, -2:]
    return labels, umap_embedding


def main(argv: list[str] | None = None) -> None:
    """Run the reclustering CLI."""

    parser = argparse.ArgumentParser(description="Recluster a single cluster label.")
    parser.add_argument("embedding_file", type=Path)
    parser.add_argument("clustering_file", type=Path)
    parser.add_argument("label", type=int)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--clust-nn", type=int, default=20)
    parser.add_argument("--clust-res", type=float, default=0.05)
    parser.add_argument("--label-file", type=Path, default=Path("analysis/data/types_and_intensity_corr.tsv"))
    parser.add_argument("--neighbors-file", type=Path, default=Path("analysis/data/bilateral_neighbors.pkl"))
    args = parser.parse_args(argv)

    with args.neighbors_file.open("rb") as handle:
        neighbors_by_cell = pickle.load(handle)
    embeddings, cell_types, _, ids = get_data(args.embedding_file, args.label_file)
    labels, umap_embedding = load_cl_labels(args.clustering_file)
    new_labels = recluster_label(
        embeddings,
        umap_embedding,
        labels,
        args.label,
        cell_types,
        ids,
        neighbors_by_cell,
        n_neighbors=args.clust_nn,
        resolution=args.clust_res,
    )
    if args.save_path:
        save_labels(ids, new_labels, umap_embedding, args.save_path)


if __name__ == "__main__":
    main()
