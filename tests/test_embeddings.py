"""Tests for embedding table utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from _path import add_src_to_path

add_src_to_path()

from morphofeatures.data.embeddings import EmbeddingTable, merge_embedding_tables, read_embedding_table, reorder_by_id


class EmbeddingTests(unittest.TestCase):
    """Validate embedding I/O and alignment."""

    def test_reorder_by_id_sorts_rows(self) -> None:
        """Embeddings are sorted by label ID."""

        table = reorder_by_id(np.array([[2.0], [1.0]]), np.array([20, 10]))
        np.testing.assert_array_equal(table.ids, np.array([10, 20]))
        np.testing.assert_array_equal(table.features[:, 0], np.array([1.0, 2.0]))

    def test_merge_embedding_tables_validates_ids(self) -> None:
        """Multiple embedding files merge by columns after ID sorting."""

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first = tmp_path / "a.npy"
            second = tmp_path / "b.npy"
            np.save(first, np.array([[2, 2.0], [1, 1.0]]))
            np.save(second, np.array([[1, 10.0], [2, 20.0]]))
            merged = merge_embedding_tables([first, second])
        np.testing.assert_array_equal(merged.ids, np.array([1, 2]))
        np.testing.assert_array_equal(merged.features, np.array([[1.0, 10.0], [2.0, 20.0]]))

    def test_embedding_table_as_matrix(self) -> None:
        """The standard matrix format keeps label IDs first."""

        table = EmbeddingTable(ids=np.array([3]), features=np.array([[1.0, 2.0]]))
        np.testing.assert_array_equal(table.as_matrix(), np.array([[3.0, 1.0, 2.0]]))

    def test_read_text_embedding(self) -> None:
        """Plain text embeddings are readable."""

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "emb.np"
            np.savetxt(path, np.array([[2, 2.0], [1, 1.0]]))
            table = read_embedding_table(path)
        np.testing.assert_array_equal(table.ids, np.array([1, 2]))


if __name__ == "__main__":
    unittest.main()
