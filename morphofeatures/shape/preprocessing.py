"""Segmentation-to-mesh preprocessing with maintained scientific Python tools."""

from pathlib import Path
from typing import Sequence
import numpy as np


def mask_to_mesh(mask, resolution: Sequence[float] = (1.0, 1.0, 1.0), smoothing_iterations=5, target_faces=5000):
    try:
        import trimesh
        from skimage.measure import marching_cubes
    except ImportError as error:
        raise RuntimeError("Shape preprocessing requires morphofeatures[legacy-training]") from error
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 3 or not np.any(binary):
        raise ValueError("mask must be a non-empty 3D array")
    vertices, faces, _, _ = marching_cubes(binary.astype(np.uint8), level=0.5, spacing=resolution)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    if smoothing_iterations > 0:
        trimesh.smoothing.filter_taubin(mesh, iterations=int(smoothing_iterations))
    if target_faces > 0 and len(mesh.faces) > target_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=int(target_faces))
        except BaseException:
            pass
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def export_label_mesh(segmentation, label_id, output: Path, resolution=(1.0, 1.0, 1.0), **kwargs):
    mesh = mask_to_mesh(segmentation == int(label_id), resolution=resolution, **kwargs)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(destination)
    return destination
