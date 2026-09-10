"""Bounded, ID-matched cell surfaces for comparison and portable mesh exports."""

import io
import json
import zipfile
from pathlib import Path

import numpy as np

from morphofeatures.analysis.annotations import read_id_table
from morphofeatures.data.crop_storage import load_crop_array, open_crop_array
from morphofeatures.data.preprocessing import triple
from morphofeatures.data.remote_n5 import is_remote_url
from morphofeatures.data.volumes import open_volume

MAX_BATCH_BYTES = 256 * 1024**2


def mesh_source_defaults(config):
    data = config.get("data", {})
    settings = {"source": "prepared" if data.get("loss_masks") else "segmentation"}
    if data.get("masks_container"):
        settings.update(segmentation=data["masks_container"], segmentation_key=data.get("mask_key"))
    prep = data.get("preprocessing")
    if prep and Path(prep).is_file():
        metadata = json.loads(Path(prep).read_text())
        settings.update(
            {
                key: value
                for key, value in metadata["settings"].items()
                if key
                in {
                    "segmentation",
                    "segmentation_key",
                    "segmentation_axes",
                    "segmentation_channel",
                    "spacing_zyx",
                    "origin_zyx",
                    "unit",
                }
            }
        )
        settings["object_index"] = str(Path(prep).parent / "objects.tsv")
    settings.update(config.get("inspection", {}).get("mesh", {}))
    return settings


def object_id_mapping(settings):
    """Read exact one-to-one IDs, including the site's cells_to_nuclei table."""
    if not settings.get("id_mapping"):
        return {}
    path = settings["id_mapping"]
    objects = read_id_table(path)
    column = "segmentation_id" if "segmentation_id" in objects else "nucleus_id"
    if column not in objects:
        raise ValueError("ID mapping requires label_id and segmentation_id (or nucleus_id)")
    segments = read_id_table(path, column, require_unique=False)[column]
    # Zero denotes background / no assigned nucleus in site annotation tables.
    assigned = segments > 0
    if segments[assigned].duplicated().any():
        raise ValueError("ID mapping requires unique positive segmentation IDs")
    return dict(zip(objects.loc[assigned, "label_id"].map(int), segments[assigned].map(int)))


def _surface_from_mask(mask, start, spacing, origin, step):
    from skimage.measure import marching_cubes

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("The selected object has no foreground in this region")
    touches = any(np.take(mask, edge, axis=axis).any() for axis in range(3) for edge in (0, -1))
    # Padding closes the surface at a crop boundary; boundary contact is recorded.
    vertices, faces, _, _ = marching_cubes(
        np.pad(mask, 1).astype(np.uint8), level=0.5, step_size=step, allow_degenerate=False
    )
    vertices = (vertices - 1 + start) * spacing + origin
    return vertices[:, ::-1].copy(), faces.astype(np.int64), bool(touches)


def _load_object_mesh(config, object_id, settings):
    source = settings.get("source", "prepared")
    spacing = triple(settings.get("spacing_zyx", [1, 1, 1]), "spacing_zyx")
    origin = triple(settings.get("origin_zyx", [0, 0, 0]), "origin_zyx", positive=False)
    unit = settings.get("unit", "voxel")
    budget = int(settings.get("max_voxels", 256**3))
    if not 8 <= budget <= 512**3:
        raise ValueError("Mesh voxel budget must be between 8 and 512 cubed")
    step = int(settings.get("sampling_step", 1))
    if not 1 <= step <= 8:
        raise ValueError("Surface sampling step must be between 1 and 8 voxels")
    provenance = {"source": source, "unit": unit, "sampling_step": step}
    if source == "files":
        from morphofeatures.mesh import load_surface

        if settings.get("mesh_table"):
            table_path = Path(settings["mesh_table"]).expanduser()
            table = read_id_table(table_path).set_index("label_id")
            if object_id not in table.index or "mesh_path" not in table:
                raise ValueError("Mesh table needs a mesh_path for this label_id")
            path = (table_path.parent / str(table.loc[object_id, "mesh_path"])).resolve()
        else:
            folder = settings.get("mesh_directory")
            if not folder:
                raise ValueError("Choose a mesh folder or a label_id / mesh_path table")
            matches = [
                Path(folder).expanduser() / f"{object_id}.{ext}"
                for ext in ("ply", "obj", "glb", "stl")
            ]
            matches = [path for path in matches if path.is_file()]
            if len(matches) != 1:
                raise ValueError(
                    "Mesh folder needs exactly one ID.ply, ID.obj, ID.glb or ID.stl per object; use a table to disambiguate"
                )
            path = matches[0]
        if path.stat().st_size > MAX_BATCH_BYTES:
            raise ValueError("Mesh file exceeds the 256 MiB inspection limit")
        surface = load_surface(path, object_id)
        vertices, faces, touches = surface.vertices, surface.faces, False
        provenance.update(path=str(path), coordinate_frame="source mesh XYZ", sampling_step=None)
    elif source == "prepared":
        data = config.get("data", {})
        if not data.get("loss_masks"):
            raise ValueError(
                "Prepared mesh views require explicit data.loss_masks. For grouped intensity patches, use an instance segmentation or existing meshes"
            )
        ids = load_crop_array(data, "label_ids")
        found = np.flatnonzero(ids == object_id)
        if len(found) != 1:
            raise ValueError("Prepared masks need exactly one row for this object ID")
        start = np.zeros(3)
        provenance["coordinate_frame"] = "crop-local XYZ"
        prep = data.get("preprocessing")
        if prep and Path(prep).is_file():
            metadata = json.loads(Path(prep).read_text())
            spacing = triple(metadata.get("spacing_zyx", spacing), "spacing_zyx")
            origin = triple(metadata.get("origin_zyx", origin), "origin_zyx", positive=False)
            provenance["unit"] = metadata.get("unit", unit)
            index = read_id_table(Path(prep).parent / "objects.tsv").set_index("label_id")
            row = index.loc[object_id]
            start = np.array([row[f"crop_start_voxel_{axis}"] for axis in "zyx"])
            provenance["coordinate_frame"] = "world XYZ"
            provenance["preprocessing"] = str(prep)
        with open_crop_array(data, "loss_masks") as array:
            if np.prod(array.shape[-3:], dtype=np.int64) > budget:
                raise ValueError(
                    "Prepared mask exceeds the voxel budget; increase it or use a smaller indexed region"
                )
            indexing = (int(found[0]),) + ((0,) if array.ndim == 5 else ())
            mask = np.array(array[indexing], dtype=bool, copy=True)
        vertices, faces, touches = _surface_from_mask(mask, start, spacing, origin, step)
        provenance.update(mask_path=str(data["loss_masks"]), start_zyx=start.tolist())
    elif source == "segmentation":
        from morphofeatures.analysis.platybrowser import validate_index_reference

        validate_index_reference(settings)
        if settings.get("label_kind", "instances") != "instances":
            raise ValueError(
                "Mesh identity requires instance IDs, not a foreground mask or score map"
            )
        if not settings.get("segmentation"):
            raise ValueError("Provide the instance segmentation path and dataset key")
        if is_remote_url(settings["segmentation"]) and not settings.get("object_index"):
            raise ValueError(
                "Streamed meshes require published or supplied object bounds; a remote volume is never scanned automatically"
            )
        with open_volume(
            settings["segmentation"],
            settings.get("segmentation_key"),
            settings.get("segmentation_axes", "zyx"),
            settings.get("segmentation_channel", 0),
            remote_options=settings.get("remote_options"),
        ) as volume:
            if not np.issubdtype(volume.dtype, np.integer):
                raise ValueError("Mesh extraction requires integer instance labels")
            mapping = object_id_mapping(settings)
            if settings.get("id_mapping") and object_id not in mapping:
                raise ValueError("Object is missing from the embedding-to-segmentation ID mapping")
            segmentation_id = mapping.get(object_id, object_id)
            if settings.get("object_index"):
                index_path = Path(settings["object_index"])
                table = read_id_table(index_path).set_index("label_id")
                lookup_id = (
                    segmentation_id if settings.get("index_ids") == "segmentation" else object_id
                )
                if lookup_id not in table.index:
                    raise ValueError("Object ID is absent from the bounding-box table")
                if "segmentation_id" in table:
                    # Preserve exact uint64-sized decimal strings when IDs differ.
                    table["segmentation_id"] = (
                        read_id_table(index_path, "segmentation_id")
                        .set_index(table.index)
                        .segmentation_id
                    )
                    indexed_id = int(table.loc[lookup_id, "segmentation_id"])
                    if mapping and indexed_id != segmentation_id:
                        raise ValueError("Bounding-box table and ID mapping disagree")
                    segmentation_id = indexed_id
                row = table.loc[lookup_id]
                start = triple(
                    [row[f"bbox_min_{a}"] for a in "zyx"], "bbox_min", positive=False, integer=True
                )
                stop = triple(
                    [row[f"bbox_max_{a}"] for a in "zyx"], "bbox_max", positive=False, integer=True
                )
                if np.any(start < 0) or np.any(stop > volume.shape) or np.any(stop <= start):
                    raise ValueError(
                        "Object bounds must be nonempty and inside the segmentation grid"
                    )
                start, stop = np.maximum(0, start - 1), np.minimum(volume.shape, stop + 1)
            else:
                start, stop = np.zeros(3, dtype=int), np.array(volume.shape)
            if np.prod(stop - start, dtype=np.int64) > budget:
                raise ValueError(
                    "Region exceeds the voxel budget. Run the background object-ID check to build bounds, supply a matching objects.tsv table, or increase the read budget"
                )
            raw = volume.read(tuple(slice(int(a), int(b)) for a, b in zip(start, stop)))
            # Compare in the source integer dtype, preserving IDs above 2**53.
            if not 0 < segmentation_id <= np.iinfo(volume.dtype).max:
                raise ValueError("The object ID must be positive and fit the segmentation dtype")
            mask = raw == np.asarray(segmentation_id, dtype=volume.dtype)
            provenance["selected_label_voxels"] = int(mask.sum())
            vertices, faces, touches = _surface_from_mask(mask, start, spacing, origin, step)
        provenance.update(
            path=str(settings["segmentation"]),
            dataset_key=settings.get("segmentation_key"),
            stored_axes=settings.get("segmentation_axes", "zyx"),
            segmentation_id=str(segmentation_id),
            start_zyx=start.tolist(),
            coordinate_frame="world XYZ",
        )
        if settings.get("id_mapping"):
            provenance["id_mapping"] = str(settings["id_mapping"])
        if settings.get("remote_provenance"):
            provenance["published_source"] = settings["remote_provenance"]
    else:
        raise ValueError("Mesh source must be files, prepared, or segmentation")
    if source != "files":
        provenance.update(
            spacing_zyx=spacing.tolist(),
            origin_zyx=origin.tolist(),
            coordinate_convention="Integer voxel indices are voxel centers; physical = origin + index * spacing",
        )
    vertices, faces = np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Mesh vertices must be finite XYZ coordinates")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError("Mesh must contain valid triangles")
    if len(faces) > 2_000_000:
        raise ValueError(
            "Surface exceeds two million faces; increase surface sampling step or simplify the input mesh"
        )
    return {
        "label_id": object_id,
        "vertices": vertices,
        "faces": faces,
        "boundary_contact": touches,
        **provenance,
    }


def load_mesh_batch(config, object_ids, settings, *, limit=6):
    """Limit work before opening any image/mesh; failures are explicit per object."""
    if not 1 <= int(limit) <= 25:
        raise ValueError("Render limit must be between 1 and 25 objects")
    ids = list(dict.fromkeys(int(value) for value in object_ids))
    meshes, failures, size = [], [], 0
    for object_id in ids[: int(limit)]:
        try:
            mesh = _load_object_mesh(config, object_id, settings)
            size += mesh["vertices"].nbytes + mesh["faces"].nbytes
            if size > MAX_BATCH_BYTES:
                raise ValueError(
                    "Combined meshes exceed 256 MiB; reduce the render limit or surface detail"
                )
            meshes.append(mesh)
        except (ValueError, OSError, KeyError, ImportError) as error:
            failures.append({"label_id": str(object_id), "error": str(error)})
    return {"meshes": meshes, "failures": failures, "omitted_ids": ids[int(limit) :]}


def preview_mesh(mesh, max_faces=12000):
    """Vertex clustering bounds display size; full source geometry remains untouched."""
    if not 100 <= int(max_faces) <= 100_000:
        raise ValueError("Preview face budget must be between 100 and 100000")
    vertices, faces = mesh["vertices"], mesh["faces"]
    if len(faces) <= max_faces:
        return mesh
    extent = float(np.ptp(vertices, axis=0).max())
    for divisions in (256, 128, 64, 32, 16, 8, 4, 2, 1):
        cells = np.floor((vertices - vertices.min(0)) / max(extent / divisions, 1e-12)).astype(
            np.int64
        )
        _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
        reduced = np.column_stack(
            [np.bincount(inverse, weights=vertices[:, a]) / counts for a in range(3)]
        )
        triangles = inverse[faces]
        keep = (
            (triangles[:, 0] != triangles[:, 1])
            & (triangles[:, 0] != triangles[:, 2])
            & (triangles[:, 1] != triangles[:, 2])
        )
        triangles = triangles[keep]
        _, unique = np.unique(np.sort(triangles, axis=1), axis=0, return_index=True)
        triangles = triangles[np.sort(unique)]
        if 0 < len(triangles) <= max_faces:
            return {**mesh, "vertices": reduced, "faces": triangles, "preview_simplified": True}
    raise ValueError("Preview budget is too low for this surface")


def mesh_archive(meshes, *, format="ply"):
    """Export source coordinates plus lossless arrays and an exact-ID manifest."""
    import trimesh

    if format not in {"ply", "obj", "glb"}:
        raise ValueError("Choose PLY, OBJ, or GLB mesh export")
    buffer, manifest = io.BytesIO(), []
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for mesh in meshes:
            object_id = str(mesh["label_id"])
            surface = trimesh.Trimesh(vertices=mesh["vertices"], faces=mesh["faces"], process=False)
            archive.writestr(f"{object_id}.{format}", surface.export(file_type=format))
            arrays = io.BytesIO()
            np.savez_compressed(
                arrays,
                vertices=mesh["vertices"],
                faces=mesh["faces"],
                label_id=np.int64(mesh["label_id"]),
            )
            archive.writestr(f"{object_id}.npz", arrays.getvalue())
            manifest.append(
                {
                    **{
                        key: value
                        for key, value in mesh.items()
                        if key not in {"vertices", "faces"}
                    },
                    "label_id": object_id,
                    "vertices": len(mesh["vertices"]),
                    "faces": len(mesh["faces"]),
                }
            )
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "objects": manifest,
                    "vertex_axes": "xyz",
                    "geometry": "Source geometry before display centering and preview simplification; NPZ preserves float64 vertices and exact IDs. Generated surfaces retain the selected voxel sampling step.",
                },
                indent=2,
            ),
        )
    return buffer.getvalue()


def _display_meshes(meshes):
    radii = [np.ptp(mesh["vertices"], axis=0).max() / 2 for mesh in meshes]
    radius = max(max(radii), 1e-6) * 1.08
    for mesh in meshes:
        center = (mesh["vertices"].min(0) + mesh["vertices"].max(0)) / 2
        yield mesh, mesh["vertices"] - center, radius


def interactive_meshes(meshes, *, azimuth=35, elevation=25):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    columns = min(3, len(meshes))
    rows = (len(meshes) + columns - 1) // columns
    figure = make_subplots(
        rows=rows,
        cols=columns,
        specs=[[{"type": "scene"}] * columns for _ in range(rows)],
        subplot_titles=[f"ID {mesh['label_id']}" for mesh in meshes],
        vertical_spacing=min(0.12, 0.25 / rows),
    )
    azimuth, elevation = np.radians([azimuth, elevation])
    eye = dict(
        x=2 * np.cos(elevation) * np.cos(azimuth),
        y=2 * np.cos(elevation) * np.sin(azimuth),
        z=2 * np.sin(elevation),
    )
    for index, (mesh, vertices, radius) in enumerate(_display_meshes(meshes)):
        faces = mesh["faces"]
        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#6697b5",
                flatshading=False,
                name=str(mesh["label_id"]),
                hovertemplate=f"ID {mesh['label_id']}<extra></extra>",
                showscale=False,
            ),
            row=index // columns + 1,
            col=index % columns + 1,
        )
        scene = "scene" if index == 0 else f"scene{index + 1}"
        axes = {
            name: dict(visible=False, range=[-radius, radius])
            for name in ("xaxis", "yaxis", "zaxis")
        }
        figure.update_layout(
            **{scene: {**axes, "aspectmode": "cube", "camera": {"eye": eye}, "bgcolor": "white"}}
        )
    figure.update_layout(
        height=350 * rows,
        margin=dict(l=5, r=5, t=45, b=5),
        showlegend=False,
        uirevision=str([mesh["label_id"] for mesh in meshes]) + str(eye),
        paper_bgcolor="white",
    )
    return figure


def mesh_figure(meshes, *, azimuth=35, elevation=25):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    columns = min(3, len(meshes))
    rows = (len(meshes) + columns - 1) // columns
    figure = plt.figure(figsize=(3 * columns, 3.2 * rows + 0.5))
    for index, (mesh, vertices, radius) in enumerate(_display_meshes(meshes)):
        axis = figure.add_subplot(rows, columns, index + 1, projection="3d")
        faces = mesh["faces"]
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        light = np.abs(normals @ (np.array([1, -1, 2]) / np.sqrt(6)))
        color = np.array([0.40, 0.59, 0.71])[None] * (0.55 + 0.45 * light[:, None])
        axis.add_collection3d(Poly3DCollection(triangles, facecolors=color, edgecolors="none"))
        axis.set(
            xlim=(-radius, radius),
            ylim=(-radius, radius),
            zlim=(-radius, radius),
            title=f"ID {mesh['label_id']}",
        )
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_axis_off()
    figure.suptitle("Cell / nucleus surface comparison", fontsize=12)
    unit = meshes[0].get("unit", "source units")
    figure.text(
        0.5,
        0.02,
        f"Common spatial scale ({unit}); centered for display. Preview surfaces.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    return figure
