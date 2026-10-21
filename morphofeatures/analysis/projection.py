"""UMAP and graph clustering with current-library compatibility."""

from __future__ import annotations

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
    "umap_pca_components": None,
    "cluster_pca_components": None,
    "resolution": 0.004,
    "seed": 42,
}

NORMALIZATION_NAMES = {
    "none": "No analysis scaling",
    "l2": "L2 of the complete vector",
    "standardize": "Standardize each feature",
}


def analysis_normalizations(settings, *, default="standardize"):
    """Scaling conditions evaluated on the same IDs/folds, not plot-only variants."""
    from morphofeatures.workflow_validation import FieldError

    values = settings.get("analysis_normalizations")
    if values is None:
        values = [settings.get("normalization", default)]
    if (
        not isinstance(values, list) or not values
        or any(not isinstance(v, str) or v not in NORMALIZATION_NAMES for v in values)
        or len(set(values)) != len(values)
    ):
        raise FieldError(
            "Choose at least one distinct analysis scaling: none (unchanged vectors), "
            "l2 (unit length per complete vector), or standardize (per-feature scaling).",
            "analysis_normalizations",
        )
    if len(values) > 1 and settings.get("use_published_layout"):
        raise FieldError(
            "Disable the saved published layout before comparing scaling methods; "
            "a saved map cannot show the effect of newly selected preprocessing.",
            "analysis_normalizations", "use_published_layout",
        )
    return values


def validate_projection_settings(settings):
    """Validate the optional projection-only PCA before any worker reads inputs."""
    from numbers import Integral

    from morphofeatures.workflow_validation import FieldError

    analysis_normalizations(settings)

    for key, title in (
        ("umap_pca_components", "UMAP"), ("cluster_pca_components", "clustering")
    ):
        count = settings.get(key)
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, Integral) or count < 0
        ):
            raise FieldError(
                f"PCA components before {title} must be a positive whole number (for example 50), "
                "or 0 / null to use the full embedding vectors.", key,
            )
    for key, minimum in (("clusters", 2), ("neighbors", 2), ("seed", 0)):
        value = settings.get(key, PROJECTION_DEFAULTS[key])
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise FieldError(f"{key} must be a whole number of at least {minimum}.", key)
    for key, options in (
        ("normalization", {"standardize", "l2", "none"}),
        ("cluster_method", {"kmeans", "leiden"}),
    ):
        if settings.get(key, PROJECTION_DEFAULTS[key]) not in options:
            raise FieldError(f"Choose {', '.join(sorted(options))} for {key}.", key)
    for key, minimum, maximum in (("min_dist", 0, 1), ("resolution", 0, np.inf)):
        try:
            value = float(settings.get(key, PROJECTION_DEFAULTS[key]))
            valid = np.isfinite(value) and minimum <= value <= maximum
        except (ValueError, TypeError):
            valid = False
        if not valid:
            raise FieldError(f"{key} must be a finite number between {minimum} and {maximum}.", key)
    epochs = settings.get("umap_epochs", 50)
    if epochs is not None and (
        isinstance(epochs, bool) or not isinstance(epochs, Integral) or epochs < 3
    ):
        raise FieldError("UMAP epochs must be null or an integer of at least 3.", "umap_epochs")


def prepare_umap_input(features, pca_components=None, *, seed=42):
    """Optionally reduce already-normalized features, without whitening.

    Fit on the projected cohort (pooled across datasets for shared maps), never
    on the displayed two-component PCA coordinates. Cap to the available
    centered sample rank and feature width; record the actual fit in reports.
    """
    from sklearn.decomposition import PCA

    validate_projection_settings({"umap_pca_components": pca_components})
    features = np.asarray(features)
    details = {
        "enabled": bool(pca_components),
        "requested_components": pca_components,
        "effective_components": None,
        "input_features": int(features.shape[1]),
        "fitted_objects": int(len(features)),
        "explained_variance_ratio_sum": None,
    }
    if not pca_components:
        return features, details
    if len(features) < 2:
        raise ValueError("PCA before UMAP requires at least two objects")
    count = min(int(pca_components), features.shape[1], len(features) - 1)
    reducer = PCA(n_components=count, random_state=int(seed), whiten=False)
    reduced = reducer.fit_transform(features)
    details.update(
        effective_components=count,
        explained_variance_ratio_sum=float(np.nansum(reducer.explained_variance_ratio_)),
    )
    return reduced, details


def umap_input_description(details):
    """Human-readable provenance shared by saved results, Tools and the CLI."""
    if not details.get("enabled"):
        return "UMAP uses the full normalized feature vectors; PCA before UMAP is off."
    count = details["effective_components"]
    requested = details["requested_components"]
    cap = (
        f" (requested {requested}; limited by the object/feature count)"
        if count != requested else ""
    )
    return (
        f"UMAP input: {details['input_features']} features → "
        f"{count} PCA components{cap} → 2D UMAP. "
        f"PCA retains {details['explained_variance_ratio_sum']:.1%} "
        "of variance after normalization, "
        f"fitted on the {details['fitted_objects']} objects in this map."
    )


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
    validate_projection_settings(settings)
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
    cluster_data, cluster_input = prepare_umap_input(
        data, settings["cluster_pca_components"], seed=seed
    )
    clusters = cluster_embeddings(
        cluster_data,
        method=settings["cluster_method"],
        n_clusters=int(settings["clusters"]),
        n_neighbors=int(settings["neighbors"]),
        resolution=float(settings["resolution"]),
        seed=seed,
    )
    frame = pd.DataFrame(
        {"label_id": ids, "cluster": clusters, "pca_1": pca[:, 0], "pca_2": pca[:, 1]}
    )
    umap_input = None
    if settings.get("umap", True):
        if len(ids) < 4:
            raise ValueError(
                "UMAP requires at least four objects; disable umap for smaller subsets"
            )
        umap_data, umap_input = prepare_umap_input(
            data, settings["umap_pca_components"], seed=seed
        )
        reduced = compute_umap(
            umap_data,
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
        "cluster_space": "PCA components of normalized embeddings"
        if cluster_input["enabled"] else "normalized embedding features",
        "cluster_input": cluster_input,
        "umap_input": umap_input,
    }
    if 1 < len(np.unique(clusters)) < len(ids):
        diagnostics["silhouette"] = float(
            silhouette_score(cluster_data, clusters, sample_size=min(2000, len(ids)), random_state=seed)
        )
    return frame, diagnostics


def compute_umap(
    features: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    n_components: int = 2,
    seed: int = 42,
    n_epochs: int | None = None,
    pca_components: int | None = None,
) -> np.ndarray:
    features, _ = prepare_umap_input(features, pca_components, seed=seed)
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
    pca_components: int | None = None,
) -> np.ndarray:
    features, _ = prepare_umap_input(features, pca_components, seed=seed)
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
