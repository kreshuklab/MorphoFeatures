"""Mesh utilities used by shape augmentations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np


def read_off(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read vertices and faces from an OFF mesh file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()
        if header != "OFF":
            raise ValueError(f"Expected OFF header, found {header!r}.")
        counts = handle.readline().strip().split()
        while not counts or counts[0].startswith("#"):
            counts = handle.readline().strip().split()
        n_vertices, n_faces = int(counts[0]), int(counts[1])
        vertices = np.array([[float(value) for value in handle.readline().split()] for _ in range(n_vertices)])
        faces = np.array([[int(value) for value in handle.readline().split()] for _ in range(n_faces)])
    return vertices, faces


def mesh_to_graph(faces: np.ndarray) -> Any:
    """Convert triangular faces into a NetworkX graph."""

    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("mesh_to_graph requires networkx.") from exc

    graph = nx.Graph()
    for face in np.asarray(faces):
        vertices = face[1:] if len(face) == 4 else face
        for src, dst in _face_edges(vertices):
            graph.add_edge(int(src), int(dst))
    return graph


def _face_edges(vertices: Iterable[int]) -> list[tuple[int, int]]:
    """Return cyclic edges for one face."""

    values = list(vertices)
    return [(values[index], values[(index + 1) % len(values)]) for index in range(len(values))]


def get_khop_neighbors(graph: Any, node_id: int, k: int) -> np.ndarray:
    """Return all graph nodes within ``k`` hops of ``node_id``."""

    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("get_khop_neighbors requires networkx.") from exc

    lengths = nx.single_source_shortest_path_length(graph, node_id, cutoff=k)
    return np.asarray(list(lengths.keys()), dtype=int)
