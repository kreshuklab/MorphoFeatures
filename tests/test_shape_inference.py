"""Smoke tests for shape data loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from _path import add_src_to_path

add_src_to_path()

from morphofeatures.shape.datasets import load_shape_arrays


class ShapeInferenceTests(unittest.TestCase):
    """Validate shape array loading without requiring torch."""

    def test_load_shape_arrays_from_npz(self) -> None:
        """NPZ files provide points, features, and IDs."""

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shape.npz"
            np.savez(
                path,
                points=np.zeros((2, 4, 3), dtype=np.float32),
                features=np.ones((2, 4, 6), dtype=np.float32),
                ids=np.array([10, 20]),
            )
            points, features, ids = load_shape_arrays({"npz_path": path})
        self.assertEqual(points.shape, (2, 4, 3))
        self.assertEqual(features.shape, (2, 4, 6))
        np.testing.assert_array_equal(ids, np.array([10, 20]))


if __name__ == "__main__":
    unittest.main()
