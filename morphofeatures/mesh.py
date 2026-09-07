"""Mesh I/O and bounded sampling on a declared physical coordinate grid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from morphofeatures.artifacts import write_json_atomic
from morphofeatures.data.preprocessing import triple
from morphofeatures.data.volumes import open_volume


@dataclass
class Surface:
    vertices: np.ndarray
    faces: np.ndarray
    object_ids: np.ndarray
    intensity: np.ndarray | None = None


def load_surface(path, object_id=-1):
    try:
        import trimesh
    except ImportError as error:
        raise RuntimeError("Mesh I/O requires morphofeatures[workspace] (trimesh)") from error
    path = Path(path)
    loaded = trimesh.load(str(path), process=False)
    meshes = []
    if isinstance(loaded, trimesh.Scene):
        for node in loaded.graph.nodes_geometry:
            transform, name = loaded.graph[node]
            part = loaded.geometry[name].copy()
            part.apply_transform(transform)
            meshes.append(part)
    else:
        meshes = [loaded]
    vertices, faces, ids = [], [], []
    offset = 0
    for part in meshes:
        if not isinstance(part, trimesh.Trimesh) or not len(part.faces):
            raise ValueError("Input must contain triangle meshes")
        label_ids = getattr(part, "vertex_attributes", {}).get("label_id")
        if label_ids is None:
            ply = part.metadata.get("_ply_raw", {}).get("vertex", {}).get("data")
            if (
                getattr(getattr(ply, "dtype", None), "names", None)
                and "label_id" in ply.dtype.names
            ):
                label_ids = ply["label_id"]
        if label_ids is None:
            label_ids = np.full(
                len(part.vertices), int(part.metadata.get("label_id", object_id)), dtype=np.int64
            )
        ids.append(np.asarray(label_ids, dtype=np.int64))
        vertices.append(np.asarray(part.vertices))
        faces.append(np.asarray(part.faces) + offset)
        offset += len(part.vertices)
    if not vertices:
        raise ValueError("Mesh contains no surfaces")
    result = Surface(np.vstack(vertices), np.vstack(faces), np.concatenate(ids))
    companion = path.with_suffix(path.suffix + ".vertices.tsv")
    if companion.exists():
        frame = pd.read_csv(companion, sep="\t")
        if len(frame) != len(result.vertices) or not np.array_equal(
            frame.vertex_index, np.arange(len(result.vertices))
        ):
            raise ValueError("Mesh vertex companion does not match vertex ordering")
        result.object_ids = frame.label_id.to_numpy(dtype=np.int64)
        if "intensity" in frame:
            result.intensity = frame.intensity.to_numpy()
    return result


def sample_surface(surface, settings):
    spacing = triple(settings.get("spacing_zyx", []), "spacing_zyx")
    origin = triple(settings.get("origin_zyx", [0, 0, 0]), "origin_zyx", positive=False)
    if not settings.get("unit"):
        raise ValueError("Declare the common mesh/raw coordinate unit")
    transform = np.asarray(settings.get("mesh_to_world_xyz", np.eye(4)), dtype=float)
    if (
        transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0, 0, 0, 1])
    ):
        raise ValueError("mesh_to_world_xyz must be a finite affine 4x4 matrix")
    vertices = (np.c_[surface.vertices, np.ones(len(surface.vertices))] @ transform.T)[:, :3]
    indices = (vertices[:, ::-1] - origin) / spacing
    method = settings.get("interpolation", "linear")
    if method not in {"nearest", "linear"}:
        raise ValueError("interpolation must be nearest or linear")
    block_shape = triple(settings.get("block_shape", [64, 64, 64]), "block_shape", integer=True)
    values = np.full(len(vertices), np.nan, dtype=np.float64)
    with open_volume(
        settings["raw"],
        settings.get("raw_key"),
        settings.get("raw_axes", "zyx"),
        settings.get("raw_channel", 0),
    ) as raw:
        valid = (
            np.isfinite(indices).all(1)
            & (indices >= 0).all(1)
            & (indices <= np.asarray(raw.shape) - 1).all(1)
        )
        selected = np.flatnonzero(valid)
        bins = np.floor(indices[selected] / block_shape).astype(int)
        _, group = np.unique(bins, axis=0, return_inverse=True)
        for group_id in np.unique(group):
            rows = selected[group == group_id]
            coordinates = indices[rows]
            lower = np.maximum(np.floor(coordinates.min(0)).astype(int), 0)
            upper = np.minimum(np.ceil(coordinates.max(0)).astype(int) + 2, raw.shape)
            block = raw.read(tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))).astype(
                np.float64
            )
            values[rows] = map_coordinates(
                block,
                (coordinates - lower).T,
                order=0 if method == "nearest" else 1,
                mode="nearest",
                prefilter=False,
            )
    return Surface(vertices, surface.faces.copy(), surface.object_ids.copy(), values)


def export_surface(surface, path, metadata=None):
    import trimesh

    path = Path(path)
    if path.suffix.lower() not in {".ply", ".obj", ".glb"}:
        raise ValueError("Mesh export supports PLY, OBJ, and GLB")
    if path.exists():
        raise FileExistsError(f"Mesh export already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=surface.vertices, faces=surface.faces, process=False)
    if surface.intensity is not None:
        finite = np.isfinite(surface.intensity)
        scaled = np.zeros(len(surface.vertices), dtype=np.uint8)
        if finite.any():
            low, high = surface.intensity[finite].min(), surface.intensity[finite].max()
            scaled[finite] = np.clip(
                255 * (surface.intensity[finite] - low) / max(float(high - low), 1e-8), 0, 255
            ).astype(np.uint8)
        mesh.visual.vertex_colors = np.c_[scaled, scaled, scaled, np.where(finite, 255, 0)].astype(
            np.uint8
        )
        if path.suffix.lower() == ".ply":
            mesh.vertex_attributes["intensity"] = surface.intensity.astype(np.float32)
    mesh.export(str(path))
    frame = pd.DataFrame(
        {
            "vertex_index": np.arange(len(surface.vertices)),
            "label_id": surface.object_ids,
            "x": surface.vertices[:, 0],
            "y": surface.vertices[:, 1],
            "z": surface.vertices[:, 2],
        }
    )
    if surface.intensity is not None:
        frame["intensity"] = surface.intensity
    frame.to_csv(path.with_suffix(path.suffix + ".vertices.tsv"), sep="\t", index=False)
    write_json_atomic(
        path.with_suffix(path.suffix + ".metadata.json"),
        {
            **(metadata or {}),
            "vertices": len(surface.vertices),
            "faces": len(surface.faces),
            "vertex_axes": "xyz",
            "unknown_object_id": -1,
            "intensity_method": "point sample at mesh vertices; nearest voxel or trilinear interpolation; no normal-direction averaging",
            "outside_volume": "NaN intensity and transparent vertex color",
            "object_ids_and_intensities": "companion vertices.tsv is authoritative",
            "unsampled_vertices": int(np.isnan(surface.intensity).sum())
            if surface.intensity is not None
            else None,
        },
    )
    return path
