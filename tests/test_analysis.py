"""Tests for lightweight analysis helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from _path import add_src_to_path

add_src_to_path()

from morphofeatures.analysis.classification import get_class_embeds, get_labels
from morphofeatures.analysis.clustering import save_labels, show_types_in_clusters
from morphofeatures.analysis.bilateral import get_nearest_index


class AnalysisTests(unittest.TestCase):
    """Validate analysis utilities that do not require optional ML packages."""

    def test_get_labels_respects_skipped_types(self) -> None:
        """Skipped classes are excluded from label arrays and class names."""

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.tsv"
            pd.DataFrame(
                {"label_id": [1, 2], "cell_type": ["neuron", "muscle"]}
            ).to_csv(path, sep="\t", index=False)
            labels, class_names = get_labels(path, skip_types=["muscle"])
        self.assertEqual(labels.tolist(), [[1, 1]])
        self.assertNotIn("muscle", class_names)

    def test_get_class_embeds_aligns_by_id(self) -> None:
        """Labeled embeddings are selected by label ID, not row order."""

        features = np.array([[10.0], [20.0], [30.0]])
        ids = np.array([10, 20, 30])
        label_pairs = np.array([[30, 1], [10, 0]])
        selected, labels = get_class_embeds(features, ids, label_pairs)
        np.testing.assert_array_equal(selected[:, 0], np.array([30.0, 10.0]))
        np.testing.assert_array_equal(labels, np.array([1, 0]))

    def test_save_labels_contains_expected_columns(self) -> None:
        """Cluster output contains cluster and UMAP columns."""

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "clusters.tsv"
            frame = save_labels(
                np.array([1, 2]),
                np.array([0, 1]),
                np.array([[0.1, 0.2], [0.3, 0.4]]),
                output,
            )
        self.assertIn("cluster", frame.columns)
        self.assertIn("umap_1", frame.columns)
        self.assertIn("umap_2", frame.columns)

    def test_show_types_in_clusters_counts_types(self) -> None:
        """Type count table has one row per cluster."""

        frame = show_types_in_clusters(np.array([0, 0, 1]), np.array([1, 2, 1]), ["None", "a", "b"])
        self.assertEqual(frame.loc[0, "a"], 1)
        self.assertEqual(frame.loc[0, "b"], 1)
        self.assertEqual(frame.loc[1, "a"], 1)

    def test_get_nearest_index_without_globals(self) -> None:
        """Bilateral nearest-neighbor ranks use explicit side mask and neighbors."""

        nearest = np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]])
        ids = np.array([1, 2, 3])
        neighbors = {1: [2], 2: [1], 3: [2]}
        side_mask = np.array([False, True, False, True])
        ranks = get_nearest_index(nearest, ids, neighbors, side_mask)
        np.testing.assert_array_equal(ranks, np.array([1, 1, 1]))


if __name__ == "__main__":
    unittest.main()
