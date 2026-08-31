"""UMAP and graph clustering with current-library compatibility."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans


def compute_umap(
    features: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    n_components: int = 2,
    seed: int = 42,
    n_epochs: Optional[int] = None,
) -> np.ndarray:
    try:
        import umap
    except (ImportError, OSError) as error:
        raise RuntimeError("UMAP requires the 'analysis' optional dependency group") from error
    neighbors = min(max(2, int(n_neighbors)), max(2, len(features) - 1))
    reducer = umap.UMAP(
        n_neighbors=neighbors,
        metric="euclidean",
        min_dist=min_dist,
        n_components=n_components,
        random_state=seed,
        n_jobs=1,
        n_epochs=n_epochs,
    )
    return reducer.fit_transform(features)


def scipy_graph_to_networkx(graph: sparse.spmatrix):
    """Bridge NetworkX 2.x and 3.x sparse graph constructor names."""
    try:
        import networkx as nx
    except ImportError as error:
        raise RuntimeError("NetworkX is required for graph conversion") from error
    constructor = getattr(nx, "from_scipy_sparse_array", None)
    if constructor is None:
        constructor = nx.from_scipy_sparse_matrix
    return constructor(graph)


def leiden_from_sparse_graph(graph: sparse.spmatrix, resolution: float = 0.004, seed: int = 42) -> np.ndarray:
    try:
        import igraph as ig
        import leidenalg
    except ImportError as error:
        raise RuntimeError("Leiden clustering requires python-igraph and leidenalg") from error
    coo = sparse.triu(graph, k=1).tocoo()
    edges = list(zip(coo.row.tolist(), coo.col.tolist()))
    network = ig.Graph(n=graph.shape[0], edges=edges, directed=False)
    weights = coo.data.astype(float).tolist()
    partition = leidenalg.find_partition(
        network,
        leidenalg.CPMVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        seed=seed,
    )
    return np.asarray(partition.membership, dtype=np.int64)


def cluster_embeddings(
    features: np.ndarray,
    method: str = "leiden",
    n_neighbors: int = 20,
    resolution: float = 0.004,
    n_clusters: int = 8,
    seed: int = 42,
) -> np.ndarray:
    if method == "kmeans":
        clusters = min(max(2, int(n_clusters)), len(features))
        return KMeans(n_clusters=clusters, random_state=seed, n_init=10).fit_predict(features)
    if method != "leiden":
        raise ValueError("method must be 'leiden' or 'kmeans'")
    try:
        import umap
    except (ImportError, OSError) as error:
        raise RuntimeError("Leiden graph construction requires umap-learn") from error
    neighbors = min(max(2, int(n_neighbors)), max(2, len(features) - 1))
    reducer = umap.UMAP(
        n_neighbors=neighbors,
        metric="euclidean",
        min_dist=0.0,
        n_components=2,
        random_state=seed,
        n_jobs=1,
    )
    reducer.fit(features)
    return leiden_from_sparse_graph(reducer.graph_, resolution=resolution, seed=seed)
