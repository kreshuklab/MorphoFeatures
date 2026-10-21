"""Thin Streamlit views over shared configuration, job, and artifact APIs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from morphofeatures.config import repository_root
from morphofeatures.configuration_editor import load_document, parse_document, remember
from morphofeatures.registry import JobRegistry
from morphofeatures.ui_settings import setting_help


def retained_input(label, default, key, *, kind="text_input", **kwargs):
    durable = "value:" + key
    widget = "widget:" + key
    if widget not in st.session_state:
        st.session_state[widget] = deepcopy(st.session_state.get(durable, default))

    def changed():
        remember(st.session_state, durable, st.session_state[widget])

    return getattr(st, kind)(label, key=widget, on_change=changed, **kwargs)


def set_document(namespace, values):
    remember(st.session_state, "document:" + namespace, values)
    st.session_state["revision:" + namespace] = st.session_state.get("revision:" + namespace, 0) + 1


def edit_document(namespace, default):
    """All leaves are editable; the authoritative document is never a widget key."""
    document_key = "document:" + namespace
    if document_key not in st.session_state:
        set_document(namespace, default)
    revision = st.session_state["revision:" + namespace]
    errors = []

    def leaf(path, value):
        key = f"field:{namespace}:{revision}:" + ".".join(path)
        label = ".".join(path)
        if isinstance(value, bool):
            kind, options, initial = "checkbox", {}, value
        elif isinstance(value, (int, float)):
            kind, options, initial = "number_input", {}, value
            if isinstance(value, float):
                options["format"] = "%.8f"
        elif isinstance(value, str):
            kind, options, initial = "text_input", {}, value
        else:
            kind, options = (
                "text_input",
                {"help": "YAML value: use [a, b, c] for lists and null for unset values"},
            )
            initial = yaml.safe_dump(value, default_flow_style=True).replace("\n...\n", "").strip()
        if key not in st.session_state:
            st.session_state[key] = initial

        def changed():
            updated = deepcopy(st.session_state[document_key])
            current = updated
            for part in path[:-1]:
                current = current[part]
            replacement = st.session_state[key]
            if not isinstance(value, (str, bool, int, float)):
                try:
                    replacement = yaml.safe_load(replacement)
                except yaml.YAMLError:
                    return
            current[path[-1]] = replacement
            remember(st.session_state, document_key, updated)

        getattr(st, kind)(label, key=key, on_change=changed, help=setting_help(path), **options)
        if not isinstance(value, (str, bool, int, float)):
            try:
                yaml.safe_load(st.session_state[key])
            except yaml.YAMLError as error:
                errors.append(f"{label}: {error}")

    def walk(values, path):
        for key, value in values.items():
            if isinstance(value, dict) and value:
                walk(value, path + [str(key)])
            else:
                leaf(path + [str(key)], value)

    document = st.session_state[document_key]
    for section, value in document.items():
        if isinstance(value, dict):
            with st.expander(section, expanded=section in {"data", "training"}):
                walk(value, [section])
        else:
            leaf([section], value)
    with st.expander("Full configuration YAML / add fields"):
        edited = st.text_area(
            "Configuration YAML",
            yaml.safe_dump(st.session_state[document_key], sort_keys=False),
            height=280,
            key=f"yaml:{namespace}:{revision}",
        )
        if st.button("Apply YAML", key="apply:" + namespace):
            try:
                set_document(namespace, parse_document(edited))
                st.rerun()
            except Exception as error:
                st.error(str(error))
    for error in errors:
        st.error(error)
    return deepcopy(st.session_state[document_key]), bool(errors)


def extraction_defaults():
    return {
        "model": "mae",
        "checkpoint": "",
        "cache": str(repository_root() / "outputs" / "embedding_cache"),
        "config": load_document(repository_root() / "configs/smoke.yaml"),
        "model_repository": None,
        "variant": "dinov2_vits14",
        "views": {
            "axes": [0, 1, 2],
            "fractions": [0.25, 0.5, 0.75],
            "normalization": "foreground_percentile",
            "percentiles": [1.0, 99.0],
            "mask": True,
            "size": 224,
            "feature": "cls",
            "aggregation": "mean",
            "batch_size": 16,
        },
        "data_version": "1",
    }


def result_browser(config):
    from morphofeatures.job_status import reconcile_job_states
    from morphofeatures.run_deletion import DELETING
    from morphofeatures.workflow_outputs import ordered_artifacts, show_result_outputs

    registry = JobRegistry.under_output_root(config.paths.output_root)
    reconcile_job_states(registry)
    records = [r for r in registry.list(limit=500)
               if r.application_state not in DELETING]
    owners = {p: record for record in records for p in record.artifacts}
    options = ["Enter path"] + list(
        ordered_artifacts(
            p
            for record in records
            for p in record.artifacts
            if Path(p).suffix in {".json", ".npz", ".npy", ".tsv", ".csv"}
        )
    )
    if st.session_state.get("results:registered") not in options:
        st.session_state["results:registered"] = "Enter path"
    chosen = st.selectbox(
        "Registered results (local and Slurm)", options, key="results:registered",
        format_func=lambda value: value if value == "Enter path" else
        owners[value].display_name + " · " + value,
    )
    manual = retained_input(
        "Saved analysis.json, comparison.json, joint_analysis.json, or embedding file",
        "",
        "results:path",
        help="Open the top-level experiment.json for a complete DINO fine/coarse run. Also accepts distillation.json, transfer.json, bilateral.json, context.json, MoBIE reports and saved embedding files. Opening a report does not rerun the job.",
    )
    st.caption("For a DINO fine/coarse run, open experiment.json to see all representations, scores and vector downloads together.")
    path = Path(chosen if chosen != "Enter path" else manual).expanduser()
    if not str(manual).strip() and chosen == "Enter path":
        return
    path = (repository_root() / path).resolve()
    try:
        from morphofeatures.analysis.variants import REPORT_SCHEMAS
        from morphofeatures.analysis_variants_ui import result_controls, result_name_controls
        from morphofeatures.followup_ui import controls, download_file
        from morphofeatures.workspace_navigation import selector_key

        metadata = json.loads(path.read_text()) if path.suffix == ".json" else {}
        section = st.radio(
            "Result view", ["Explore results", "Follow-up analysis", "Files & provenance"],
            horizontal=True, key=selector_key("result-view:" + str(path)),
        )
        if metadata.get("schema") in REPORT_SCHEMAS:
            path, metadata = result_controls(path, metadata, exploration=section == "Follow-up analysis")
        elif metadata:
            result_name_controls(path, metadata)
        if section == "Follow-up analysis":
            controls(config, metadata, path)
            return
        if section == "Files & provenance":
            from morphofeatures.provenance_ui import provenance_view

            show_result_outputs(metadata or {"schema": "morphofeatures.embedding.v1"}, path)
            provenance_view(key="results:provenance:" + str(path), artifact=path)
            with st.expander("All saved settings and metrics"):
                st.json(metadata)
            download_file(path.parent / "export.zip", "Download complete results archive")
            return
        schema = metadata.get("schema")
        if schema == "morphofeatures.distillation.v1":
            from morphofeatures.distillation_ui import results

            results(config, metadata, path)
        elif schema == "morphofeatures.bilateral_export.v1":
            from morphofeatures.spatial_probe_ui import bilateral_export_results

            bilateral_export_results(config, metadata, path)
        elif schema in {"morphofeatures.bilateral.v1", "morphofeatures.context.v1"}:
            from morphofeatures.spatial_probe_ui import probe_results

            probe_results(config, metadata, path)
        elif schema in {"morphofeatures.dino_experiment.v1", "morphofeatures.transfer.v1"}:
            from morphofeatures.dino_experiment_ui import experiment_results

            experiment_results(config, metadata, path)
        elif schema == "morphofeatures.joint_analysis.v1":
            from morphofeatures.analysis_ui import joint_projection_results

            joint_projection_results(config, metadata, path)
        elif schema == "morphofeatures.mobie_export.v1":
            from morphofeatures.analysis_ui import mobie_export_results

            mobie_export_results(metadata, path)
        elif schema in {"morphofeatures.analysis.v1", "morphofeatures.comparison.v1"}:
            from morphofeatures.result_panels import analysis_panels

            analysis_panels(config, metadata, path)
        elif path.suffix == ".json":
            st.json(metadata)
        else:
            from morphofeatures.analysis_ui import inspect_results

            download_file(path, "Download embedding vectors")
            inspect_results(config, {path.stem: path}, {}, pd.DataFrame({"label_id": []}), path)
    except Exception as error:
        st.error(str(error))


def mesh_page(config):
    st.title("Mesh surfaces")
    source = retained_input("Mesh path (PLY, OBJ, GLB, STL)", "", "mesh:source")
    object_id = retained_input(
        "Object ID if absent from mesh (-1 = unknown)", -1, "mesh:id", kind="number_input"
    )
    if st.button("Load mesh"):
        try:
            from morphofeatures.mesh import load_surface

            st.session_state["mesh:surface"] = load_surface(Path(source), object_id=int(object_id))
            st.session_state["mesh:source_surface"] = st.session_state["mesh:surface"]
            st.session_state["mesh:provenance"] = {"source": str(Path(source).resolve())}
        except Exception as error:
            st.error(str(error))
    values, invalid = edit_document(
        "mesh-sampling",
        {
            "raw": "",
            "raw_key": "exported_data",
            "raw_axes": "zyxc",
            "raw_channel": 0,
            "spacing_zyx": [1.0, 1.0, 1.0],
            "origin_zyx": [0.0, 0.0, 0.0],
            "unit": "voxel",
            "interpolation": "linear",
            "mesh_to_world_xyz": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "block_shape": [64, 64, 64],
        },
    )
    st.caption(
        "Vertices use XYZ. The affine maps them to the same units as raw spacing/origin (ZYX). Sampling is nearest-voxel or trilinear at each vertex; outside-volume values are NaN."
    )
    surface = st.session_state.get("mesh:surface")
    if surface is None:
        return
    if st.button("Sample raw intensity on surface", disabled=invalid):
        try:
            from morphofeatures.mesh import sample_surface

            st.session_state["mesh:surface"] = sample_surface(
                st.session_state["mesh:source_surface"], values
            )
            st.session_state["mesh:provenance"]["sampling"] = values
            surface = st.session_state["mesh:surface"]
            st.success(
                "Surface sampled from the loaded mesh; displayed and exported vertices use world coordinates."
            )
        except Exception as error:
            st.error(str(error))
    try:
        import plotly.graph_objects as go

        figure = go.Figure(
            go.Mesh3d(
                x=surface.vertices[:, 0],
                y=surface.vertices[:, 1],
                z=surface.vertices[:, 2],
                i=surface.faces[:, 0],
                j=surface.faces[:, 1],
                k=surface.faces[:, 2],
                intensity=surface.intensity,
                customdata=surface.object_ids.astype(str),
                colorscale="Viridis",
                hovertemplate="object=%{customdata}<br>x=%{x}<br>y=%{y}<br>z=%{z}<extra></extra>",
            )
        )
        figure.update_layout(scene_aspectmode="data", height=600)
        st.plotly_chart(figure, use_container_width=True)
    except ImportError:
        st.error("Interactive mesh display requires plotly; install morphofeatures[workspace]")
    output = retained_input(
        "Mesh export path (.ply, .obj, .glb)",
        str(config.paths.output_root / "mesh.ply"),
        "mesh:output",
    )
    if st.button("Export mesh, intensities, and object IDs"):
        try:
            from morphofeatures.mesh import export_surface

            path = export_surface(surface, Path(output), st.session_state.get("mesh:provenance"))
            st.success(f"Saved {path} and its vertex table / metadata companions")
        except Exception as error:
            st.error(str(error))
