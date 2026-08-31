"""Mesh utilities referenced by the original ARAP augmentation code."""

from pathlib import Path
import numpy as np


def mesh_to_graph(faces):
    import networkx as nx
    graph = nx.Graph()
    for face in np.asarray(faces, dtype=np.int64):
        vertices = face[-3:] if len(face) > 3 and face[0] == len(face) - 1 else face
        graph.add_edges_from((int(vertices[i]), int(vertices[(i + 1) % len(vertices)])) for i in range(len(vertices)))
    return graph


def get_khop_neighbors(graph, node_id, k):
    import networkx as nx
    return np.asarray(sorted(nx.single_source_shortest_path_length(graph, int(node_id), cutoff=int(k))), dtype=np.int64)


def read_off(path: Path):
    with Path(path).open("r", encoding="ascii") as stream:
        lines = [line.strip() for line in stream if line.strip() and not line.startswith("#")]
    if not lines or lines[0] != "OFF":
        raise ValueError("{} is not an ASCII OFF file".format(path))
    vertex_count, face_count, _ = (int(value) for value in lines[1].split()[:3])
    vertices = np.asarray([[float(value) for value in line.split()[:3]] for line in lines[2:2 + vertex_count]])
    faces = np.asarray([[int(value) for value in line.split()]
                        for line in lines[2 + vertex_count:2 + vertex_count + face_count]], dtype=np.int64)
    return vertices, faces
