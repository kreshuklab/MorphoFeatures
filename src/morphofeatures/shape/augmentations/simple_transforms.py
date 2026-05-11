"""Simple NumPy point-cloud transforms for shape augmentation."""

from __future__ import annotations

from collections.abc import Callable
import random

import numpy as np


def center(points: np.ndarray) -> np.ndarray:
    """Center points around their mean coordinate."""

    return points - points.mean(axis=-2, keepdims=True)


def normalize(points: np.ndarray) -> np.ndarray:
    """Scale points into a unit-ish coordinate range."""

    max_abs = np.abs(points).max()
    if max_abs == 0:
        return points
    return points * ((1 / max_abs) * 0.9999999)


class RandomCompose:
    """Apply a random subset of transforms in sequence."""

    def __init__(self, *transforms: Callable[[np.ndarray], np.ndarray], num_compositions: int = 2) -> None:
        """Initialize transform list and number of sampled transforms."""

        if not all(callable(transform) for transform in transforms):
            raise TypeError("All transforms must be callable.")
        if len(transforms) < num_compositions:
            raise ValueError("num_compositions cannot exceed the number of transforms.")
        self.transforms = list(transforms)
        self.num_compositions = num_compositions

    def __call__(self, tensors: np.ndarray) -> np.ndarray:
        """Apply sampled transforms to the input array."""

        intermediate = tensors
        for transform in random.sample(self.transforms, self.num_compositions):
            intermediate = transform(intermediate)
        return intermediate


class SymmetryTransform:
    """Randomly reflect coordinates across coordinate axes."""

    def __call__(self, tensor: np.ndarray) -> np.ndarray:
        """Apply random axis flips."""

        axis_mask = np.random.randint(0, 2, 3, dtype=bool)
        transformed = tensor.copy()
        for index, use_axis in enumerate(axis_mask):
            if use_axis:
                transformed[:, index] = np.max(transformed[:, index]) - transformed[:, index]
        return transformed


class AnisotropicScaleTransform:
    """Randomly scale each coordinate axis independently."""

    def __init__(self, low_scale: float = 0.9, high_scale: float = 1.1) -> None:
        """Initialize axis scale range."""

        self.low_scale = low_scale
        self.high_scale = high_scale

    def __call__(self, tensor: np.ndarray) -> np.ndarray:
        """Apply anisotropic random scaling."""

        scale_diff = self.high_scale - self.low_scale
        scales = self.low_scale + np.random.rand(3) * scale_diff
        return tensor * scales


class AxisRotationTransform:
    """Randomly rotate a point cloud around coordinate axes."""

    def __init__(self, x_rot: float, y_rot: float, z_rot: float) -> None:
        """Initialize maximum axis rotation angles."""

        self.rotation_angles = [x_rot, y_rot, z_rot]

    @staticmethod
    def compute_rot_matrix(phi: np.ndarray, shuffle: bool = False) -> np.ndarray:
        """Compute a 3D rotation matrix for Euler-like axis rotations."""

        x_rot = np.array(
            [[1, 0, 0], [0, np.cos(phi[0]), -np.sin(phi[0])], [0, np.sin(phi[0]), np.cos(phi[0])]]
        )
        y_rot = np.array(
            [[np.cos(phi[1]), 0, np.sin(phi[1])], [0, 1, 0], [-np.sin(phi[1]), 0, np.cos(phi[1])]]
        )
        z_rot = np.array(
            [[np.cos(phi[2]), -np.sin(phi[2]), 0], [np.sin(phi[2]), np.cos(phi[2]), 0], [0, 0, 1]]
        )

        matrices = [x_rot, y_rot, z_rot]
        if shuffle:
            random.shuffle(matrices)
        return np.matmul(matrices[2], np.matmul(matrices[1], matrices[0]))

    def __call__(self, tensor: np.ndarray) -> np.ndarray:
        """Apply a random axis rotation."""

        phi = np.zeros(3)
        for index, angle in enumerate(self.rotation_angles):
            if angle > 0:
                phi[index] = float(2 * random.random() * angle - angle) / 180.0
        rotation_matrix = self.compute_rot_matrix(phi, shuffle=True)
        return tensor @ rotation_matrix.T
