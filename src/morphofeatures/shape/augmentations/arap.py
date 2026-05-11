"""As-rigid-as-possible mesh deformation augmentation."""

from __future__ import annotations

import argparse
import random
import time
from typing import Any

import numpy as np

from morphofeatures.shape.mesh_utils import get_khop_neighbors, mesh_to_graph, read_off


class AsRigidAsPossibleDeformation:
    """Random local ARAP deformation for triangular meshes.

    Args:
        region_size: Upper bound on deformed region size in graph hops.
        num_regions: Upper bound on the number of regions to deform.
        deform_scale_low: Lower deformation scale.
        deform_scale_high: Upper deformation scale.
        deformation: Deformation mode: ``random``, ``constant``, or ``normal``.
        smoothing: Smoothing mode: ``random``, ``True``, or ``False``.
    """

    def __init__(
        self,
        region_size: int,
        num_regions: int,
        deform_scale_low: float = 0.0005,
        deform_scale_high: float = 0.025,
        deformation: str = "random",
        smoothing: str | bool = "random",
    ) -> None:
        """Store ARAP deformation sampling parameters."""

        self.region_size = region_size
        self.num_regions = num_regions
        self.scale_low = deform_scale_low
        self.scale_high = deform_scale_high
        self.deformation = deformation
        self.smoothing = smoothing

    def deform(self, vertices: np.ndarray, faces: np.ndarray, handles: np.ndarray, mode: str, selection: np.ndarray) -> np.ndarray:
        """Create ARAP boundary conditions for selected handles."""

        try:
            import igl
        except ImportError as exc:
            raise ImportError("ARAP deformation requires libigl Python bindings.") from exc

        boundary_conditions = np.zeros((handles.size, vertices.shape[1]))
        if mode == "constant":
            deform_vectors = {
                int(index): random.uniform(self.scale_low, self.scale_high) * np.random.randn(3)
                for index in np.unique(selection)
            }
            deform_vectors[-1] = np.zeros(3)
            for handle_index in range(handles.size):
                boundary_conditions[handle_index] = vertices[handles[handle_index]] + deform_vectors[int(selection[handles[handle_index]])]
        elif mode == "normal":
            normals = igl.per_vertex_normals(vertices, faces)
            factors = {
                int(index): random.choice([-1.0, 1.0]) * random.uniform(self.scale_low, self.scale_high)
                for index in np.unique(selection)
            }
            factors[-1] = 0
            for handle_index in range(handles.size):
                boundary_conditions[handle_index] = (
                    vertices[handles[handle_index]]
                    + factors[int(selection[handles[handle_index]])] * normals[handles[handle_index]]
                )
        else:
            raise ValueError(f"Unsupported deformation mode: {mode}")
        return boundary_conditions

    def smooth(self, vertices: np.ndarray, faces: np.ndarray, deformed_vertices: np.ndarray, handle_dict: dict[int, dict[str, np.ndarray]]) -> np.ndarray:
        """Smooth deformed vertices with biharmonic coordinates."""

        try:
            import igl
        except ImportError as exc:
            raise ImportError("ARAP smoothing requires libigl Python bindings.") from exc

        control_ids = random.sample(list(range(vertices.shape[0])), int(vertices.shape[0] / 3))
        for ids in handle_dict.values():
            control_ids = np.setdiff1d(np.asarray(control_ids), ids["smoothing_ids"]).tolist()
        boundary = np.array(control_ids)
        weights = igl.biharmonic_coordinates(deformed_vertices, faces, [[index] for index in control_ids], k=2)
        return weights @ deformed_vertices[boundary[:]]

    def __call__(self, vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply a random ARAP deformation."""

        try:
            import igl
        except ImportError as exc:
            raise ImportError("ARAP deformation requires libigl Python bindings.") from exc

        graph = mesh_to_graph(faces)
        node_ids = list(graph.nodes())
        selection = -1.0 * np.ones(vertices.shape[0])
        handle_dict: dict[int, dict[str, np.ndarray]] = {}

        for region_index in range(random.randint(1, self.num_regions)):
            if not node_ids:
                break
            node_id = random.sample(node_ids, 1)[0]
            inner_size = random.randint(1, max(1, self.region_size - 1))
            outer_size = inner_size + self.region_size
            inner_ids = get_khop_neighbors(graph, node_id, k=inner_size)
            outer_ids = get_khop_neighbors(graph, node_id, k=outer_size)
            node_ids = list(set(node_ids) - set(outer_ids.tolist()))
            smoothing_ids = np.setdiff1d(
                get_khop_neighbors(graph, node_id, k=inner_size + min(3, inner_size)),
                inner_ids,
            )
            selection[inner_ids] = float(region_index)
            handle_dict[int(node_id)] = {"smoothing_ids": smoothing_ids}

        handles = np.array([[index for index, selected in enumerate(selection) if selected >= 0]]).T
        arap = igl.ARAP(vertices, faces, 3, handles)
        deformation_mode = "constant" if self.deformation == "random" and random.random() > 0.5 else self.deformation
        if deformation_mode == "random":
            deformation_mode = "normal"
        boundary_conditions = self.deform(vertices, faces, handles, deformation_mode, selection)
        deformed_vertices = arap.solve(boundary_conditions, vertices)

        smoothing = random.random() > 0.5 if self.smoothing == "random" else bool(self.smoothing)
        if smoothing:
            deformed_vertices = self.smooth(vertices, faces, deformed_vertices, handle_dict)
        return deformed_vertices, selection


def main(argv: list[str] | None = None) -> None:
    """Preview an ARAP deformation from the command line."""

    try:
        import vedo
    except ImportError as exc:
        raise ImportError("The ARAP preview CLI requires vedo.") from exc

    parser = argparse.ArgumentParser(description="Preview ARAP mesh deformation.")
    parser.add_argument("--path-to-mesh", type=str, required=True)
    parser.add_argument("--region-size", type=int, default=10)
    parser.add_argument("--num-regions", type=int, default=9)
    parser.add_argument("--deform-scale-low", type=float, default=0.005)
    parser.add_argument("--deform-scale-high", type=float, default=0.05)
    parser.add_argument("--deformation", default="random")
    parser.add_argument("--smoothing", type=bool, default=True)
    args = parser.parse_args(argv)

    vertices, faces = read_off(args.path_to_mesh)
    faces = np.asarray(faces)[:, 1:] if faces.shape[1] == 4 else np.asarray(faces)
    transform = AsRigidAsPossibleDeformation(
        region_size=args.region_size,
        num_regions=args.num_regions,
        deform_scale_low=args.deform_scale_low,
        deform_scale_high=args.deform_scale_high,
        deformation=args.deformation,
        smoothing=args.smoothing,
    )
    started = time.time()
    deformed_vertices, selection = transform(np.asarray(vertices), faces)
    print("time", time.time() - started)
    plotter = vedo.Plotter(N=2, axes=0)
    mesh = vedo.Mesh([vertices, faces])
    mesh.cmap("rainbow", selection)
    mesh.addScalarBar3D()
    plotter.show(mesh, at=0, interactive=0)
    deformed_mesh = vedo.Mesh([deformed_vertices, faces])
    deformed_mesh.cmap("rainbow", selection)
    deformed_mesh.addScalarBar3D()
    plotter.show(deformed_mesh, at=1, interactive=1)


if __name__ == "__main__":
    main()
