"""Smoke tests for texture dataset geometry helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from _path import add_src_to_path

add_src_to_path()

from morphofeatures.texture.datasets import CellDataset


class TextureDatasetTests(unittest.TestCase):
    """Validate texture dataset setup with small mocked arrays."""

    def test_cell_dataset_len_and_crop(self) -> None:
        """CellDataset can compute a nucleus-centered crop from mock tables."""

        cell_table = pd.DataFrame(
            {
                "label_id": [1],
                "bb_min_z": [0.0],
                "bb_max_z": [4.0],
                "bb_min_y": [0.0],
                "bb_max_y": [4.0],
                "bb_min_x": [0.0],
                "bb_max_x": [4.0],
                "anchor_z": [2.0],
                "anchor_y": [2.0],
                "anchor_x": [2.0],
            }
        )
        nucleus_table = pd.DataFrame(
            {
                "label_id": [1],
                "bb_min_z": [0.0],
                "bb_max_z": [4.0],
                "bb_min_y": [0.0],
                "bb_max_y": [4.0],
                "bb_min_x": [0.0],
                "bb_max_x": [4.0],
                "anchor_z": [2.0],
                "anchor_y": [2.0],
                "anchor_x": [2.0],
            }
        )
        volume = np.ones((4, 4, 4), dtype=int)
        dataset = CellDataset(
            [cell_table, nucleus_table],
            {1: 1},
            volume,
            volume,
            volume,
            indices=[1],
            size_cut=2,
            resolution_um=(1.0, 1.0, 1.0),
            high_resolution_shape=(4, 4, 4),
        )
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.cut_to_size(1), (slice(1, 3), slice(1, 3), slice(1, 3)))


if __name__ == "__main__":
    unittest.main()
