import numpy as np

from morphofeatures.shape.utils import get_khop_neighbors, mesh_to_graph, read_off


def test_off_reader_and_mesh_graph(tmp_path):
    off = tmp_path / "tetra.off"
    off.write_text(
        "OFF\n4 4 0\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
        "3 0 1 2\n3 0 1 3\n3 0 2 3\n3 1 2 3\n",
        encoding="ascii",
    )
    vertices, faces = read_off(off)
    graph = mesh_to_graph(faces)
    assert vertices.shape == (4, 3)
    assert faces.shape == (4, 4)
    assert np.array_equal(get_khop_neighbors(graph, 0, 1), [0, 1, 2, 3])
