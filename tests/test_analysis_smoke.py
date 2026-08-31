import importlib.util

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from morphofeatures.analysis.classification import (
    cross_validate_logistic,
    load_class_labels,
    select_labeled_embeddings,
)
from morphofeatures.analysis.projection import cluster_embeddings, compute_umap
from morphofeatures.config import load_config
from morphofeatures.data.io import load_embeddings


def test_logistic_regression_on_bundled_labels():
    config = load_config()
    table = load_embeddings(config.paths.analysis_data / "morphofeatures_all_cells.npy")
    ids, labels, class_names = load_class_labels(config.paths.analysis_data / "class_labels.tsv")
    features, labels = select_labeled_embeddings(table, ids, labels)
    features = StandardScaler().fit_transform(features[:, :24])
    result = cross_validate_logistic(features, labels, class_names, folds=3, seed=7, max_iter=500)
    assert result.scores.shape == (3,)
    assert result.confusion.sum() == len(labels)
    assert 0.0 <= result.mean_accuracy <= 1.0


@pytest.mark.optional
def test_umap_and_clustering_small_bundled_subset():
    if importlib.util.find_spec("umap") is None:
        pytest.skip("umap-learn is not installed")
    try:
        import umap  # noqa: F401
    except (ImportError, OSError) as error:
        pytest.skip("umap-learn runtime is unavailable: {}".format(error))
    config = load_config()
    table = load_embeddings(config.paths.analysis_data / "morphofeatures_all_cells.npy")
    features = StandardScaler().fit_transform(table.features[:64, :16])
    projection = compute_umap(features, n_neighbors=8, seed=7, n_epochs=20)
    labels = cluster_embeddings(features, method="kmeans", n_clusters=4, seed=7)
    assert projection.shape == (64, 2)
    assert labels.shape == (64,)
    assert len(np.unique(labels)) == 4
