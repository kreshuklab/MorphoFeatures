import numpy as np
import pandas as pd
import pytest

from morphofeatures.data.contracts import BOUNDING_BOX_TABLE, EmbeddingTable, VolumeSpec
from morphofeatures.data.io import load_embeddings
from morphofeatures.data.synthetic import make_synthetic_dataset, save_synthetic_dataset
from morphofeatures.embedding_base import aggregate_patch_embeddings
from morphofeatures.analysis.context import aggregate_neighbors, agglomerate_features


def test_volume_and_table_contracts(tmp_path):
    spec = VolumeSpec(tmp_path / "raw.npy", "data", (0.04, 0.01, 0.01))
    assert spec.coordinate_order == ("z", "y", "x")
    with pytest.raises(ValueError):
        VolumeSpec(tmp_path / "raw.npy", "data", (1, 1, 1), coordinate_order=("x", "y", "z"))
    BOUNDING_BOX_TABLE.validate(
        ["label_id", "bb_min_z", "bb_min_y", "bb_min_x", "bb_max_z", "bb_max_y", "bb_max_x"]
    )


def test_synthetic_fixture_round_trip(tmp_path):
    destination = save_synthetic_dataset(tmp_path, seed=3)
    assert np.load(destination / "raw.npy").shape == (24, 32, 32)
    assert pd.read_csv(destination / "cell_to_nucleus.tsv", sep="\t").shape == (4, 2)
    table = load_embeddings(destination / "embeddings.npy")
    assert table.features.shape == (4, 8)


def test_patch_aggregation_is_label_stable():
    ids = np.array([2, 1, 2, 1])
    features = np.array([[2.0, 4.0], [0.0, 2.0], [4.0, 6.0], [2.0, 4.0]])
    aggregated_ids, aggregated = aggregate_patch_embeddings(ids, features)
    assert np.array_equal(aggregated_ids, [1, 2])
    assert np.allclose(aggregated, [[1, 3], [3, 5]])


def test_embedding_contract_rejects_non_integral_ids():
    with pytest.raises(ValueError):
        EmbeddingTable.from_array(np.array([[1.5, 0.0], [2.0, 1.0]]))


def test_neighbor_aggregation_and_agglomeration():
    table = EmbeddingTable(np.array([1, 2, 3]), np.arange(18, dtype=float).reshape(3, 6))
    context = aggregate_neighbors(table, {1: [2], 2: [1, 3], 3: [2]})
    reduced = agglomerate_features(context, n_features=3)
    assert context.features.shape == (3, 6)
    assert reduced.features.shape == (3, 3)
