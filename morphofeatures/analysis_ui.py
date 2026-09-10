"""Interactive views over saved analysis artifacts; volume work stays in workers."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from morphofeatures.analysis.object_inspection import input_config_for_result, neighbor_sheet
from morphofeatures.analysis.visualization import (
    figure_bytes,
    interactive_projection,
    projection_figure,
    projection_panels,
    selected_object_ids,
)


def projection_results(frame, metadata, path):
    prefix = "projection:" + str(path)
    methods = (["umap"] if "umap_1" in frame else []) + ["pca"]
    method = st.selectbox("Projection", methods, key=prefix + ":method")
    opacity = st.slider(
        "Unlabeled point opacity",
        0.0,
        1.0,
        float(metadata.get("settings", {}).get("unlabeled_opacity", 0.15)),
        key=prefix + ":opacity",
    )
    size = st.slider("Point size", 2, 12, 5, key=prefix + ":size")
    if "representation" not in frame:
        frame["representation"] = "Embedding"
    st.caption(
        "The panels share coordinates and axis scales. Click or lasso points, then add them to your comparison list; repeat anywhere on the map. Each panel has its own legend. Unlabeled opacity uses known labels in every panel."
    )
    for name in frame.representation.unique():
        subset = frame[frame.representation == name]
        palette = metadata.get("representations", {}).get(name, metadata).get("class_colors")
        if palette:
            subset.attrs["class_colors"] = palette
        st.subheader(str(name))
        chart = interactive_projection(subset, method, unlabeled_opacity=opacity, point_size=size)
        event = st.plotly_chart(
            chart,
            key=prefix + ":chart:" + str(name),
            on_select="rerun",
            selection_mode=("points", "box", "lasso"),
            use_container_width=True,
        )
        selected = selected_object_ids(event, subset.label_id)
        event_key = prefix + ":last:" + str(name)
        if selected and selected != st.session_state.get(event_key):
            st.session_state[event_key] = selected
            st.session_state["picked:" + str(path)] = selected
            st.session_state["picked:" + str(path) + ":" + str(name)] = selected
        basket_key = "comparison:" + str(path) + ":" + str(name)
        if st.button(
            "Add selected points to comparison",
            key=prefix + ":add:" + str(name),
            disabled=not selected,
        ):
            st.session_state[basket_key] = list(
                dict.fromkeys(st.session_state.get(basket_key, []) + selected)
            )
        st.caption(
            f"{len(st.session_state.get(basket_key, []))} objects in this representation's comparison list."
        )
        panel_options = {
            "All panels": None,
            **{title: [column] for column, title in projection_panels(subset)},
        }
        panel_choice = st.selectbox(
            "Graphs to export", list(panel_options), key=prefix + ":panels:" + str(name)
        )
        if st.button("Prepare projection SVG / PNG", key=prefix + ":export:" + str(name)):
            import matplotlib.pyplot as plt

            figure = projection_figure(
                subset,
                str(name),
                methods=[method],
                unlabeled_opacity=opacity,
                point_size=size * 2,
                panels=panel_options[panel_choice],
            )
            st.session_state[prefix + ":figure:" + str(name)] = {
                "settings": (method, opacity, size, panel_choice),
                **{extension: figure_bytes(figure, extension) for extension in ("svg", "png")},
            }
            plt.close(figure)
        exported = st.session_state.get(prefix + ":figure:" + str(name), {})
        if exported.get("settings") == (method, opacity, size, panel_choice):
            for extension in ("svg", "png"):
                st.download_button(
                    "Download projection " + extension.upper(),
                    exported[extension],
                    f"{name}-{method}-{panel_choice.lower().replace(' ', '-')}.{extension}",
                    key=prefix + ":download:" + str(name) + extension,
                )
    return method


def inspect_results(config, embeddings, metadata, frame, path):
    from morphofeatures.configuration_editor import load_document
    from morphofeatures.data.io import load_embeddings
    from morphofeatures.representation_analysis import nearest_neighbors
    from morphofeatures.workflow_ui import continue_from_artifact

    prefix = "inspection:" + str(path)
    name = st.selectbox(
        "Embedding to use in a new workflow", list(embeddings), key=prefix + ":representation"
    )
    embedding = embeddings[name]
    table = load_embeddings(embedding)
    allowed = set(table.label_ids)
    if st.button("Analyze these embeddings in a new workflow"):
        continue_from_artifact(config, embedding)
    mode = st.radio(
        "Comparison mode",
        ["Nearest neighbors", "Chosen objects"],
        horizontal=True,
        key=prefix + ":mode",
    )
    labels = (
        frame[frame.representation == name].copy() if "representation" in frame else frame.copy()
    )
    coordinates = labels
    label_file = Path(embedding).parent / "object_labels.tsv"
    if path.suffix == ".json" and label_file.is_file():
        labels = pd.read_csv(label_file, sep="\t", dtype={"label_id": np.int64})
    if mode == "Chosen objects":
        basket_key = "comparison:" + str(path) + ":" + str(name)
        st.session_state[basket_key] = [
            value for value in st.session_state.get(basket_key, []) if value in allowed
        ]
        # Keep the durable list separate from widget state, which Streamlit drops
        # when the user switches modes or leaves Results.
        basket_widget = "widget:" + basket_key
        st.session_state[basket_widget] = st.session_state[basket_key]
        chosen = st.multiselect(
            "Objects to compare",
            table.label_ids.tolist(),
            format_func=str,
            key=basket_widget,
            on_change=lambda: st.session_state.update(
                {basket_key: st.session_state[basket_widget]}
            ),
            help="Add points from the map or search for any object ID here. The mesh render limit does not shorten this list.",
        )
        st.button(
            "Clear comparison list",
            key=prefix + ":clear",
            on_click=lambda: st.session_state.update({basket_key: []}),
        )
        if not chosen:
            st.info("Add selected plot points to the comparison list, or choose object IDs above.")
            return
        neighbors = pd.DataFrame(
            {"label_id": chosen, "role": [f"Object {i + 1}" for i in range(len(chosen))]}
        )
    else:
        picked = [
            value
            for value in st.session_state.get(
                "picked:" + str(path) + ":" + str(name),
                st.session_state.get("picked:" + str(path), []),
            )
            if value in allowed
        ]
        selected_key = prefix + ":id"
        signature = (name, tuple(picked))
        if picked and signature != st.session_state.get(prefix + ":applied"):
            st.session_state[selected_key] = picked[0]
            st.session_state[prefix + ":applied"] = signature
        if st.session_state.get(selected_key) not in allowed:
            st.session_state[selected_key] = int(table.label_ids[0])
        selected = st.selectbox(
            "Object for nearest-neighbor inspection",
            table.label_ids.tolist(),
            format_func=str,
            key=selected_key,
        )
        if picked:
            st.caption("Selected from plot: " + ", ".join(str(v) for v in picked[:25]))
            if len(picked) > 1:
                selected = st.selectbox(
                    "Inspect one of the selected points",
                    picked,
                    format_func=str,
                    key=prefix + ":selected_group",
                )
        count = st.number_input(
            "Number of neighbors",
            0,
            min(24, len(table.label_ids) - 1),
            min(6, len(table.label_ids) - 1),
            key=prefix + ":count",
        )
        spaces = (
            ["Embedding"]
            + (["UMAP"] if "umap_1" in frame else [])
            + (["PCA"] if "pca_1" in frame else [])
        )
        space = st.selectbox("Find neighbors in", spaces, key=prefix + ":space")
        normalization = metadata.get("settings", {}).get("normalization", "standardize")
        if space == "Embedding":
            neighbors = nearest_neighbors(
                embedding, selected, k=int(count), normalization=normalization
            )
            st.caption(
                f"Euclidean distances in {normalization} embedding features. Points that look close in a 2D projection can be far apart here."
            )
        else:
            coords = coordinates.set_index("label_id")[[space.lower() + "_1", space.lower() + "_2"]]
            if selected not in coords.index:
                st.info(
                    "This object is outside the projected subset; select Embedding distance or a plotted object."
                )
                return
            distances = np.linalg.norm(coords.to_numpy() - coords.loc[selected].to_numpy(), axis=1)
            neighbors = pd.DataFrame({"label_id": coords.index, "distance": distances})
            neighbors = (
                neighbors[neighbors.label_id != selected]
                .sort_values(["distance", "label_id"])
                .head(int(count))
            )
            st.caption(
                f"Euclidean distances in the displayed {space} coordinates; these are projection neighbors."
            )
        neighbors = pd.concat(
            [pd.DataFrame({"label_id": [selected], "distance": [0.0]}), neighbors],
            ignore_index=True,
        )
    comparison = neighbors.merge(
        labels.drop(columns="representation", errors="ignore"), on="label_id", how="left"
    )
    st.dataframe(comparison)
    st.download_button(
        "Download comparison IDs and annotations",
        comparison.to_csv(sep="\t", index=False),
        "comparison-objects.tsv",
        key=prefix + ":objects",
    )
    source_config = input_config_for_result(metadata, name)
    explicit = st.text_input(
        "Input data YAML for object inspection",
        value=metadata.get("settings", {}).get("input_config") or "",
        key=prefix + ":config",
    )
    if explicit:
        source_config = load_document(explicit)
    source_config = _inspection_source_controls(
        source_config, prefix, config.paths.output_root / "remote_cache" / "platybrowser"
    )
    if source_config is None:
        return
    streamed = source_config.get("data", {}).get("source") == "platybrowser"
    if st.session_state.get(prefix + ":streamed_images") != streamed:
        st.session_state.pop(prefix + ":source", None)
        st.session_state[prefix + ":streamed_images"] = streamed
    source = st.selectbox(
        "Object image source",
        ["original"] if streamed else ["prepared", "original"],
        key=prefix + ":source",
        help="Prepared uses the model's crops. Original reads bounded raw-volume views using the preprocessing index, or grouped N5 mean patch positions and the configured QC raw volume. Grouped prepared views show one representative patch per nucleus.",
    )
    max_side = st.number_input("Maximum view side (voxels)", 8, 256, 128, key=prefix + ":side")
    gallery = neighbors.head(25)
    if len(neighbors) > len(gallery):
        st.caption(
            f"The slice table shows the first 25 of {len(neighbors)} chosen objects. The full comparison list is retained."
        )
    signature = json.dumps(
        [str(embedding), gallery.to_dict(orient="list"), source_config, source, max_side],
        sort_keys=True,
    )
    show_label = (
        "Show chosen objects" if mode == "Chosen objects" else "Show selected object and neighbors"
    )
    if st.button(show_label, key=prefix + ":show"):
        if not source_config.get("data"):
            st.error(
                "Load the matching mae_config.yaml or grouped N5 data YAML to locate these objects."
            )
        else:
            try:
                st.session_state[prefix + ":sheet"] = {
                    "signature": signature,
                    **neighbor_sheet(
                        source_config, gallery, labels, source=source, max_side=int(max_side)
                    ),
                }
            except Exception as error:
                st.error(str(error))
    sheet = st.session_state.get(prefix + ":sheet", {})
    if sheet.get("signature") == signature:
        st.image(sheet["png"])
        for extension in ("svg", "png"):
            st.download_button(
                "Download neighbor table " + extension.upper(),
                sheet[extension],
                f"object-comparison.{extension}",
                key=prefix + extension,
            )
        st.download_button(
            "Download neighbor IDs and distances",
            neighbors.to_csv(sep="\t", index=False),
            "object-comparison.tsv",
            key=prefix + ":tsv",
        )
    _mesh_comparison(
        source_config,
        neighbors.label_id.tolist(),
        prefix,
        embedding_ids=table.label_ids,
        check_root=config.paths.output_root / "inspection_checks",
    )
    if path.suffix == ".json":
        _volume_export_start(config, metadata, name, path, source_config)


def _inspection_source_controls(source_config, prefix, cache_directory):
    """Metadata is loaded only on Connect; edits cannot leave stale data active."""
    import yaml

    from morphofeatures.analysis.platybrowser import DEFAULTS, resolve_platybrowser

    prefix += ":remote"
    imported = source_config.get("inspection", {}).get("platybrowser")
    signature = json.dumps(imported, sort_keys=True)
    if st.session_state.get(prefix + ":import") != signature:
        for field in ("location", *DEFAULTS, "cache_directory"):
            st.session_state.pop(prefix + ":" + field, None)
        st.session_state[prefix + ":import"] = signature
    location = st.selectbox(
        "Inspection data location",
        ["Local files", "PlatyBrowser streaming"],
        index=1 if imported is not None else 0,
        key=prefix + ":location",
        help="Stream published EM raw images and nucleus labels around selected objects. The project tables locate each nucleus without downloading the full volume.",
    )
    if location == "Local files":
        return source_config
    options = {**DEFAULTS, "cache_directory": str(cache_directory), **(imported or {})}
    for field, label, help_text in (
        (
            "project_url",
            "PlatyBrowser project URL",
            "GitHub repository address. The revision field below selects the metadata version.",
        ),
        (
            "dataset",
            "Published dataset version",
            "Dataset folder within the project, for example 1.0.1. Use the version matching your embedding IDs.",
        ),
        (
            "revision",
            "Project revision / tag / branch",
            "A commit pins the metadata and tables. Tags or branches are resolved to a commit when connected; downloaded YAML records that commit.",
        ),
    ):
        options[field] = st.text_input(
            label, str(options[field]), key=prefix + ":" + field, help=help_text
        )
    options["object_ids"] = st.selectbox(
        "Embedding IDs refer to",
        ["cell", "nucleus"],
        index=["cell", "nucleus"].index(options["object_ids"]),
        key=prefix + ":object_ids",
        help="Cell IDs use the project's cells_to_nuclei mapping. Nucleus IDs match the published nucleus labels directly.",
    )
    with st.expander("Streaming resolution and cache"):
        for field, label in (
            ("raw_level", "Raw resolution level"),
            ("segmentation_level", "Nucleus resolution level"),
        ):
            options[field] = st.number_input(
                label,
                0,
                12,
                int(options[field]),
                key=prefix + ":" + field,
                help="Zero is the finest level. Larger levels reduce transfer size and detail. Raw level 3 aligns with nucleus level 0 in the published EM grid; coordinates are derived from metadata.",
            )
        options["cache_directory"] = st.text_input(
            "Streaming cache folder", options["cache_directory"], key=prefix + ":cache_directory"
        )
        options["cache_size_mb"] = st.number_input(
            "Maximum cached downloads (MiB)",
            16,
            65536,
            int(options["cache_size_mb"]),
            key=prefix + ":cache_size_mb",
            help="Old download entries are removed as needed. Small resolved object catalogs are kept separately for provenance and reuse.",
        )
        options["timeout_seconds"] = st.number_input(
            "Network timeout (seconds)",
            1,
            120,
            int(options["timeout_seconds"]),
            key=prefix + ":timeout_seconds",
        )
        options["cache_version"] = st.text_input(
            "Cache version",
            str(options["cache_version"]),
            key=prefix + ":cache_version",
            help="Change this value and reconnect to fetch fresh metadata and chunks after a source update.",
        )
    document = {
        "schema": "morphofeatures.inspection.v1",
        "inspection": {
            "platybrowser": options,
            "mesh": {
                key: value
                for key, value in source_config.get("inspection", {}).get("mesh", {}).items()
                if key in {"sampling_step", "read_budget_side"}
            },
            "validation": source_config.get("inspection", {}).get("validation", {"auto": True}),
        },
    }
    requested = json.dumps(document, sort_keys=True)
    if st.button("Connect to PlatyBrowser", key=prefix + ":connect"):
        st.session_state.pop(prefix + ":connected", None)
        try:
            with st.spinner("Loading published metadata and object tables..."):
                resolved = resolve_platybrowser(document)
            st.session_state[prefix + ":connected"] = {"request": requested, "config": resolved}
        except Exception as error:
            st.error(f"Could not connect to PlatyBrowser: {error}")
    connection = st.session_state.get(prefix + ":connected", {})
    resolved = connection.get("config") if connection.get("request") == requested else None
    download = deepcopy(document)
    if resolved:
        download["inspection"]["platybrowser"]["revision"] = resolved["inspection"]["platybrowser"][
            "revision"
        ]
    st.download_button(
        "Download inspection YAML",
        yaml.safe_dump(download, sort_keys=False),
        "inspection-platybrowser.yaml",
        key=prefix + ":download",
    )
    if resolved is None:
        st.info(
            "Connect to load the source tables. Image chunks are fetched when you request a view or render meshes."
        )
        return None
    info = resolved["inspection"]["platybrowser"]
    st.caption(
        f"Connected to dataset {info['dataset']} at revision {info['revision'][:12]}. ID checks use the published table; selected mesh labels are verified when loaded."
    )
    mesh = resolved["inspection"]["mesh"]
    st.caption(
        "Nucleus voxel spacing Z, Y, X (µm): " + ", ".join(f"{v:g}" for v in mesh["spacing_zyx"])
    )
    return resolved


def _mesh_comparison(source_config, object_ids, prefix, *, embedding_ids, check_root):
    from morphofeatures.analysis.mesh_inspection import (
        interactive_meshes,
        load_mesh_batch,
        mesh_archive,
        mesh_figure,
        mesh_source_defaults,
        preview_mesh,
    )

    prefix += ":meshes"
    with st.expander("3D cell / nucleus surfaces"):
        defaults = mesh_source_defaults(source_config)
        defaults_key = json.dumps(defaults, sort_keys=True)
        if st.session_state.get(prefix + ":defaults") != defaults_key:
            # Imported inspection settings must replace stale widget defaults.
            for suffix in (
                "source",
                "segmentation",
                "key",
                "axes",
                "channel",
                "index",
                "mapping",
                "spacing_zyx",
                "origin_zyx",
                "unit",
                "directory",
                "table",
                "kind",
                "step",
                "voxels",
                "file_unit",
                "check_auto",
            ):
                st.session_state.pop(prefix + ":" + suffix, None)
            st.session_state[prefix + ":defaults"] = defaults_key
        mode_options = {
            "Instance segmentation": "segmentation",
            "Prepared foreground masks": "prepared",
            "Existing mesh files": "files",
        }
        mode = st.selectbox(
            "Mesh source",
            list(mode_options),
            index=list(mode_options.values()).index(defaults["source"]),
            key=prefix + ":source",
        )
        settings = {**defaults, "source": mode_options[mode]}
        if settings["source"] == "files":
            settings["mesh_directory"] = st.text_input(
                "Mesh folder (ID.ply / ID.obj / ID.glb / ID.stl)",
                defaults.get("mesh_directory") or "",
                key=prefix + ":directory",
            )
            settings["mesh_table"] = st.text_input(
                "Mesh table (optional: label_id, mesh_path)",
                defaults.get("mesh_table") or "",
                key=prefix + ":table",
            )
            settings["unit"] = st.text_input(
                "Mesh coordinate unit",
                defaults.get("unit", "source units"),
                key=prefix + ":file_unit",
            )
        elif settings["source"] == "segmentation":
            settings["segmentation"] = st.text_input(
                "Mesh instance segmentation",
                defaults.get("segmentation", ""),
                key=prefix + ":segmentation",
            )
            settings["segmentation_key"] = st.text_input(
                "Mesh segmentation dataset key",
                defaults.get("segmentation_key") or "",
                key=prefix + ":key",
            )
            settings["segmentation_axes"] = st.text_input(
                "Mesh segmentation axes",
                defaults.get("segmentation_axes", "zyx"),
                key=prefix + ":axes",
            )
            settings["segmentation_channel"] = st.number_input(
                "Mesh segmentation channel",
                0,
                value=int(defaults.get("segmentation_channel", 0)),
                key=prefix + ":channel",
            )
            settings["object_index"] = st.text_input(
                "Object bounding-box table",
                defaults.get("object_index") or "",
                key=prefix + ":index",
                help="Optional objects.tsv, or CSV/TSV with label_id and bbox_min_z/y/x, bbox_max_z/y/x in voxel coordinates; maxima are exclusive. The background ID check builds an index if this is blank.",
            )
            settings["id_mapping"] = st.text_input(
                "Embedding-to-segmentation ID mapping (optional)",
                defaults.get("id_mapping") or "",
                key=prefix + ":mapping",
                help="CSV/TSV with label_id and segmentation_id (or nucleus_id, as in cells_to_nuclei.tsv). Positive assignments must be unique; zero means unassigned. Required when numbering differs; IDs are never matched by row order.",
            )
            kinds = {
                "Instance IDs": "instances",
                "Binary foreground": "binary",
                "Foreground scores": "foreground_scores",
            }
            selected_kind = st.selectbox(
                "Segmentation contents",
                list(kinds),
                index=list(kinds.values()).index(defaults.get("label_kind", "instances")),
                key=prefix + ":kind",
            )
            settings["label_kind"] = kinds[selected_kind]
            data = source_config.get("data", {})
            if settings["segmentation"] == data.get("foreground_mask_container") and settings[
                "segmentation_key"
            ] == data.get("foreground_mask_key"):
                settings["label_kind"] = data.get("foreground_mask_kind", "foreground_scores")
                st.warning(
                    "This is the configured foreground map, which has no object IDs. Select the instance segmentation for nucleus-specific meshes."
                )
            for field, label, default in (
                ("spacing_zyx", "Mesh voxel spacing Z, Y, X", [1, 1, 1]),
                ("origin_zyx", "Mesh volume origin Z, Y, X", [0, 0, 0]),
            ):
                value = st.text_input(
                    label,
                    ", ".join(map(str, defaults.get(field, default))),
                    key=prefix + ":" + field,
                )
                try:
                    settings[field] = [float(part.strip()) for part in value.split(",")]
                except ValueError:
                    st.error(f"{label} needs three numbers.")
                    return
            settings["unit"] = st.text_input(
                "Mesh spatial unit", defaults.get("unit", "voxel"), key=prefix + ":unit"
            )
        if settings["source"] != "files":
            settings["sampling_step"] = st.number_input(
                "Surface sampling step (voxels)",
                1,
                8,
                int(defaults.get("sampling_step", 1)),
                key=prefix + ":step",
                help="One retains voxel resolution. Larger values produce a coarser extracted surface in both previews and mesh files.",
            )
            voxel_side = st.number_input(
                "Mesh read budget (cube side in voxels)",
                16,
                512,
                int(defaults.get("read_budget_side", 256)),
                key=prefix + ":voxels",
                help="Limits the number of voxels read per object to this value cubed. Regions are rejected if they exceed the budget; cells are not silently clipped.",
            )
            settings["max_voxels"] = int(voxel_side) ** 3
        settings, check_pending, comparable = _inspection_check_controls(
            embedding_ids, source_config, settings, check_root, prefix
        )
        limit = st.number_input(
            "Maximum meshes rendered at once",
            1,
            25,
            6,
            key=prefix + ":limit",
            help="Independent of the number of points in your comparison list. Choose a different batch below to inspect more cells.",
        )
        ids_key = prefix + ":ids"
        if st.session_state.get(prefix + ":available") != tuple(object_ids):
            st.session_state[ids_key] = object_ids[: int(limit)]
            st.session_state[prefix + ":available"] = tuple(object_ids)
        previous = st.session_state.get(ids_key, object_ids[: int(limit)])
        st.session_state[ids_key] = [value for value in previous if value in set(object_ids)][
            : int(limit)
        ]
        mesh_ids = st.multiselect(
            "Meshes to display", object_ids, format_func=str, max_selections=int(limit), key=ids_key
        )
        st.caption(
            f"{len(mesh_ids)} meshes in this batch; {len(object_ids)} objects remain available for comparison."
        )
        faces = st.number_input(
            "Preview faces per mesh",
            1000,
            100000,
            12000,
            step=1000,
            key=prefix + ":faces",
            help="Display geometry is simplified if needed. A total limit of 200,000 preview faces also applies. Downloaded mesh files retain the source surface.",
        )
        signature = json.dumps([source_config, settings, mesh_ids, limit], sort_keys=True)
        needs_index = settings["source"] == "segmentation" and not settings.get("object_index")
        if check_pending and needs_index:
            st.caption(
                "The background check is building object bounds. Mesh rendering will be available when it finishes."
            )
        if st.button(
            "Render selected meshes",
            key=prefix + ":render",
            disabled=not mesh_ids or not comparable or (check_pending and needs_index),
        ):
            with st.spinner("Loading selected surfaces..."):
                st.session_state[prefix + ":batch"] = {
                    "signature": signature,
                    **load_mesh_batch(source_config, mesh_ids, settings, limit=int(limit)),
                }
        batch = st.session_state.get(prefix + ":batch", {})
        if batch.get("signature") != signature:
            return
        for failure in batch["failures"]:
            st.error(f"Object {failure['label_id']}: {failure['error']}")
        meshes = batch["meshes"]
        if not meshes:
            return
        if any(mesh["boundary_contact"] for mesh in meshes):
            st.warning(
                "Some surfaces touch the source region boundary and may be truncated. The archive records boundary contact for each object."
            )
        azimuth = st.slider("Mesh view azimuth", -180, 180, 35, key=prefix + ":azimuth")
        elevation = st.slider("Mesh view elevation", -90, 90, 25, key=prefix + ":elevation")
        budget = min(int(faces), 200000 // len(meshes))
        if batch.get("preview_budget") != budget:
            batch["previews"] = [preview_mesh(mesh, budget) for mesh in meshes]
            batch["preview_budget"] = budget
        previews = batch["previews"]
        st.plotly_chart(
            interactive_meshes(previews, azimuth=azimuth, elevation=elevation),
            use_container_width=True,
            key=prefix + ":chart",
            config={
                "toImageButtonOptions": {"format": "png", "filename": "mesh-comparison", "scale": 2}
            },
        )
        st.caption(
            "Drag to rotate and scroll to zoom each surface. All panels share a spatial scale and are centered for display. The camera button saves the current interactive view; SVG / PNG below use the selected view angles."
        )
        graphic_key = (signature, budget, azimuth, elevation)
        if st.button("Prepare mesh comparison SVG / PNG", key=prefix + ":graphic"):
            import matplotlib.pyplot as plt

            figure = mesh_figure(previews, azimuth=azimuth, elevation=elevation)
            try:
                with plt.rc_context({"svg.fonttype": "none"}):
                    st.session_state[prefix + ":graphic_result"] = {
                        "settings": graphic_key,
                        **{
                            extension: figure_bytes(figure, extension, dpi=300)
                            for extension in ("svg", "png")
                        },
                    }
            finally:
                plt.close(figure)
        graphic = st.session_state.get(prefix + ":graphic_result", {})
        if graphic.get("settings") == graphic_key:
            for extension in ("svg", "png"):
                st.download_button(
                    "Download mesh comparison " + extension.upper(),
                    graphic[extension],
                    f"mesh-comparison.{extension}",
                    key=prefix + ":download:" + extension,
                )
        format = st.selectbox("Mesh file format", ["ply", "obj", "glb"], key=prefix + ":format")
        st.caption(
            "The mesh archive contains this displayed batch in source coordinates, plus lossless NPZ arrays and a manifest with exact object IDs, units and source details."
        )
        if st.button("Prepare mesh files", key=prefix + ":archive"):
            st.session_state[prefix + ":archive_result"] = {
                "settings": (signature, format),
                "zip": mesh_archive(meshes, format=format),
            }
        archive = st.session_state.get(prefix + ":archive_result", {})
        if archive.get("settings") == (signature, format):
            st.download_button(
                "Download displayed meshes ZIP",
                archive["zip"],
                "selected-meshes.zip",
                key=prefix + ":zip",
            )


def _inspection_check_controls(embedding_ids, source_config, settings, root, prefix):
    from morphofeatures.analysis.inspection_validation import (
        inspection_check_key,
        start_inspection_check,
    )
    from morphofeatures.data.remote_n5 import is_remote_url

    options = source_config.get("inspection", {}).get("validation", {})
    enabled = st.checkbox(
        "Check object IDs in the background",
        value=bool(options.get("auto", True)),
        key=prefix + ":check_auto",
    )
    data = source_config.get("data", {})
    source = settings["source"]
    path = (
        settings.get("segmentation")
        if source == "segmentation"
        else (settings.get("mesh_table") or settings.get("mesh_directory"))
        if source == "files"
        else data.get("loss_masks")
    )
    ready = bool(path) and (is_remote_url(path) or Path(path).exists())
    if source == "prepared":
        ready = ready and bool(data.get("label_ids"))
    elif source == "segmentation" and path and Path(path).suffix.lower() != ".npy":
        ready = ready and bool(settings.get("segmentation_key"))
    task_key = prefix + ":check_task"
    task = st.session_state.get(task_key)
    if not enabled or not ready:
        if task and not task.future.done():
            task.cancel.set()
        return settings, False, settings.get("label_kind", "instances") == "instances"
    fields = (
        "source",
        "segmentation",
        "segmentation_key",
        "segmentation_axes",
        "segmentation_channel",
        "label_kind",
        "object_index",
        "index_ids",
        "id_mapping",
        "mesh_directory",
        "mesh_table",
        "remote_options",
        "remote_provenance",
        "index_reference",
        "spacing_zyx",
        "origin_zyx",
    )
    check_settings = {key: settings[key] for key in fields if key in settings}
    check_settings["data_version"] = str(options.get("data_version", "1"))
    if options.get("block_shape"):
        check_settings["check_block_shape"] = options["block_shape"]
    check_config = {
        "data": {
            key: data[key]
            for key in ("label_ids", "label_ids_key", "loss_masks", "loss_masks_key")
            if key in data
        }
    }
    signature = inspection_check_key(embedding_ids, check_config, check_settings)
    refresh = st.button(
        "Recheck object IDs",
        key=prefix + ":recheck",
        help="Recheck local volumes or the loaded remote object table. For updated remote files, change Cache version and reconnect first.",
    )
    if task is None or task.key != signature or refresh:
        if task and not task.future.done() and task.key != signature:
            task.cancel.set()
        task = start_inspection_check(
            embedding_ids, check_config, check_settings, root, force=refresh
        )
        st.session_state[task_key] = task
        st.session_state.pop(prefix + ":check_applied", None)
    _inspection_check_status(task, prefix)
    if not task.future.done():
        return settings, True, True
    report = task.future.result()
    effective = dict(settings)
    if (
        report.get("object_index")
        and not settings.get("object_index")
        and report["state"] in {"match", "subset", "mismatch"}
    ):
        effective.update(object_index=report["object_index"], index_ids="segmentation")
    return effective, False, report["state"] not in {"not_comparable", "error"}


@st.fragment(run_every="2s")
def _inspection_check_status(task, prefix):
    if not task.future.done():
        status = task.snapshot()
        total = max(1, status.get("total_blocks", 1))
        st.progress(
            min(1.0, status.get("blocks", 0) / total),
            text="Checking object IDs: " + status["phase"],
        )
        if st.button("Cancel object-ID check", key=prefix + ":cancel_check"):
            task.cancel.set()
        return
    if st.session_state.get(prefix + ":check_applied") != id(task):
        st.session_state[prefix + ":check_applied"] = id(task)
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if getattr(get_script_run_ctx(), "fragment_ids_this_run", None):
            st.rerun(scope="app")
    report = task.future.result()
    if report["state"] == "cancelled":
        st.info("Object-ID check cancelled. No partial scan is treated as a match.")
        return
    if report["state"] == "error":
        st.warning("Object-ID check could not finish: " + report["error"])
        return
    if report["state"] == "not_comparable":
        st.warning(report["interpretation"])
    else:
        catalog = report.get("validation_basis") == "object_table"
        summary = (
            f"Object-ID check: {report['embedded_objects']:,} embedded objects; "
            f"{report['source_objects']:,} {'catalog' if catalog else 'source'} objects; {report['matched_objects']:,} embedded IDs {'listed in the table' if catalog else 'found'}."
        )
        if report["state"] == "mismatch":
            st.warning(
                summary
                + f" {len(report['missing_embedding_ids']):,} embedded IDs are missing. Verify the dataset/version or supply an explicit ID mapping before trusting the mesh correspondence."
            )
        elif report["state"] == "subset":
            st.info(
                summary
                + f" {len(report['extra_source_ids']):,} source IDs are outside the embedded subset."
            )
        else:
            st.success(summary)
        st.caption(
            report["interpretation"]
            if catalog
            else "This compares actual IDs, not just counts. Matching IDs do not by themselves verify spatial alignment or biological identity."
        )
        if report.get("missing_embedding_ids"):
            st.caption(
                "Missing ID examples: " + ", ".join(map(str, report["missing_embedding_ids"][:20]))
            )
    st.download_button(
        "Download object-ID check",
        json.dumps(report, indent=2),
        "inspection-id-check.json",
        key=prefix + ":check_download",
    )
    if report.get("object_index"):
        st.caption("Object bounds cached for mesh rendering: " + report["object_index"])


def _volume_export_start(config, metadata, name, path, source_config):
    from morphofeatures.workflow_ui import navigate, use_draft
    from morphofeatures.workspace_state import new_draft

    directory = path.parent
    if "representations" in metadata:
        directory /= metadata["representations"][name]["directory"]
    label_file = directory / "object_labels.tsv"
    if not label_file.is_file():
        return
    with st.expander("Map labels back to the original segmentation"):
        st.caption(
            "Create a reviewed background export job for known types, predicted types and embedding clusters. Choose HDF5, Zarr v2 or Zarr v3 on the next page. A category/color lookup accompanies the volumes."
        )
        if st.button("Create volume export workflow"):
            stage = {"action": "export_labels", "labels": str(label_file), "output_format": "h5"}
            columns = pd.read_csv(label_file, sep="\t", nrows=0).columns
            stage["layers"] = [
                name for name in ("known_label", "predicted_label", "cluster") if name in columns
            ]
            prep = source_config.get("data", {}).get("preprocessing")
            if prep and Path(prep).is_file():
                settings = json.loads(Path(prep).read_text())["settings"]
                stage.update(
                    {
                        key: settings[key]
                        for key in (
                            "segmentation",
                            "segmentation_key",
                            "segmentation_axes",
                            "segmentation_channel",
                            "spacing_zyx",
                            "origin_zyx",
                            "unit",
                        )
                        if key in settings
                    }
                )
            draft = new_draft(
                "export_labels",
                document={
                    "stages": [stage],
                    "slurm": {"cpus": 4, "gpus": 0, "memory": "8G", "time": "01:00:00"},
                },
            )
            use_draft(config, draft, "Analyze")
            navigate("Workflow")
