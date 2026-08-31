import numpy as np

from morphofeatures.analysis.validation import EXPECTED_ARRAYS, validate_bundled_artifacts
from morphofeatures.config import load_config
from morphofeatures.data.contracts import EmbeddingTable


def test_published_array_shapes_and_id_contract():
    config = load_config()
    results = validate_bundled_artifacts(config.paths.analysis_data)
    assert set(results) == set(EXPECTED_ARRAYS)
    assert all(result.passed for result in results.values())


def test_morphofeatures_have_six_80_dimensional_groups():
    path = load_config().paths.analysis_data / "morphofeatures_all_cells.npy"
    matrix = np.load(path, mmap_mode="r")
    table = EmbeddingTable.from_array(matrix, require_finite=False)
    assert table.features.shape == (11382, 6 * 80)
