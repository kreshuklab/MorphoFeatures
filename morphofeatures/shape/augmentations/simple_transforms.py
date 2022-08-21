import random
import numpy as np


def center(points):
    points = points - points.mean(axis=-2, keepdims=True)
    return points


def normalize(points):
    scale = (1 / np.abs(points).max()) * 0.9999999
    points = points * scale
    return points


class RandomCompose:
    def __init__(self, *transforms, num_compositions=2):
        assert all([callable(t) for t in transforms])
        assert len(transforms) > num_compositions
        self.transforms = list(transforms)
        self.num_compositions = num_compositions

    def __call__(self, tensors):
        transforms = random.sample(self.transforms, 2)
        intermediate = tensors
        for transform in transforms:
            intermediate = transform(intermediate)
        return intermediate


class SymmetryTransform:

    def __call__(self, tensor):
        axis = np.random.randint(0, 2, 3, dtype=bool)
        for i, ax in enumerate(axis):
            if ax:
                tensor[:, i] = np.max(tensor[:, i]) - tensor[:, i]
        return tensor


class AnisotropicScaleTransform:

    def __init__(self, low_scale=0.9, high_scale=1.1):
        self.low_scale = low_scale
        self.high_scale = high_scale

    def __call__(self, tensor):
        scale_diff = self.high_scale - self.low_scale
        scales = self.low_scale + np.random.rand(3) * (scale_diff)
        return tensor * scales


class AxisRotationTransform:

    def __init__(self, x_rot, y_rot, z_rot):
        self.rot_angles = [x_rot, y_rot, z_rot]

    @staticmethod
    def compute_rot_matrix(phi, shuffle=False):
        x = np.array(
            [[1, 0, 0],
             [0, np.cos(phi[0]), -np.sin(phi[0])],
             [0, np.sin(phi[0]), np.cos(phi[0])]]
        )

        y = np.array(
            [[np.cos(phi[1]), 0, np.sin(phi[1])],
             [0, 1, 0],
             [-np.sin(phi[1]), 0, np.cos(phi[1])]]
        )

        z = np.array(
            [[np.cos(phi[2]), -np.sin(phi[2]), 0],
             [np.sin(phi[2]), np.cos(phi[2]), 0],
             [0, 0, 1]]
        )

        mat = [x, y, z]
        if shuffle:
            random.shuffle(mat)
        rotation_matrix = np.matmul(mat[2], np.matmul(mat[1], mat[0]))
        return rotation_matrix

    def __call__(self, tensor):
        phi = np.zeros(3)
        for i, angle in enumerate(self.rot_angles):
            if angle > 0:
                phi[i] = float(2 * random.random() * angle - angle) / 180.
        rot_matrix = self.compute_rot_matrix(phi, shuffle=True)

        return tensor @ rot_matrix.T
