"""UMAP and graph clustering with current-library compatibility."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans

PROJECTION_DEFAULTS = {
    "normalization": "standardize",
    "cluster_method": "kmeans",
    "clusters": 8,
    "neighbors": 15,
    "min_dist": 0.0,
    "umap_epochs": 50,
    "resolution": 0.004,
    "seed": 42,
}


def normalize_features(features, normalization="standardize"):
    from sklearn.preprocessing import StandardScaler, normalize

    if normalization == "standardize":
        return StandardScaler().fit_transform(features)
    if normalization == "l2":
        return normalize(features)
    if normalization == "none":
        return np.asarray(features)
    raise ValueError("normalization must be standardize, l2, or none")


def project_embeddings(ids, features, settings):
    """One projection/clustering implementation for Tools and central workflows."""
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    settings = {**PROJECTION_DEFAULTS, **settings}
    if len(ids) < 3 or features.shape[1] < 2:
        raise ValueError("Analysis requires at least three objects and two features")
    order = np.argsort(ids)
    ids, features = np.asarray(ids)[order], np.asarray(features)[order]
    subset = int(settings.get("subset", 0))
    if subset < 0 or 0 < subset < 3:
        raise ValueError("subset must be zero (all) or at least three objects")
    if subset and subset < len(ids):
        rows = np.sort(
            np.random.default_rng(int(settings["seed"])).choice(len(ids), subset, replace=False)
        )
        ids, features = ids[rows], features[rows]
    seed = int(settings["seed"])
    data = normalize_features(features, settings["normalization"])
    pca = PCA(n_components=2, random_state=seed).fit_transform(data)
    clusters = cluster_embeddings(
        data,
        method=settings["cluster_method"],
        n_clusters=int(settings["clusters"]),
        n_neighbors=int(settings["neighbors"]),
        resolution=float(settings["resolution"]),
        seed=seed,
    )
    frame = pd.DataFrame(
        {"label_id": ids, "cluster": clusters, "pca_1": pca[:, 0], "pca_2": pca[:, 1]}
    )
    if settings.get("umap", True):
        if len(ids) < 4:
            raise ValueError(
                "UMAP requires at least four objects; disable umap for smaller subsets"
            )
        reduced = compute_umap(
            data,
            n_neighbors=int(settings["neighbors"]),
            min_dist=float(settings["min_dist"]),
            seed=seed,
            n_epochs=settings["umap_epochs"],
        )
        frame["umap_1"], frame["umap_2"] = reduced.T
    diagnostics = {
        "clusters_observed": int(len(np.unique(clusters))),
        "projected_objects": len(ids),
        "projection_settings": settings,
        "cluster_space": "normalized embedding features",
    }
    if 1 < len(np.unique(clusters)) < len(ids):
        diagnostics["silhouette"] = float(
            silhouette_score(data, clusters, sample_size=min(2000, len(ids)), random_state=seed)
        )
    return frame, diagnostics


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


def leiden_from_sparse_graph(
    graph: sparse.spmatrix, resolution: float = 0.004, seed: int = 42
) -> np.ndarray:
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
