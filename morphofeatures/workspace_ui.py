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
    records = JobRegistry.under_output_root(config.paths.output_root).list(limit=500)
    options = ["Enter path"] + list(
        dict.fromkeys(
            p
            for record in records
            for p in record.artifacts
            if Path(p).suffix in {".json", ".npz", ".npy", ".tsv", ".csv"}
        )
    )
    if st.session_state.get("results:registered") not in options:
        st.session_state["results:registered"] = "Enter path"
    chosen = st.selectbox("Registered results (local and Slurm)", options, key="results:registered")
    manual = retained_input(
        "Saved analysis.json, comparison.json, or embedding file", "", "results:path"
    )
    path = Path(chosen if chosen != "Enter path" else manual).expanduser()
    if not str(manual).strip() and chosen == "Enter path":
        return
    path = (repository_root() / path).resolve()
    try:
        if path.suffix == ".json":
            metadata = json.loads(path.read_text())
            if metadata.get("schema") not in {
                "morphofeatures.analysis.v1",
                "morphofeatures.comparison.v1",
            }:
                st.json(metadata)
                return
            st.info(metadata["interpretation"])
            frame = pd.read_csv(path.parent / "coordinates.tsv", sep="\t")
            from morphofeatures.analysis_ui import projection_results

            projection_results(frame, metadata, path)
            if "representations" in metadata:
                scores = [
                    dict(representation=name, **row)
                    for name, value in metadata["representations"].items()
                    for row in value.get("evaluation", [])
                ]
                if scores:
                    st.dataframe(pd.DataFrame(scores))
                embeddings = {
                    name: path.parent / value["directory"] / "embeddings.npz"
                    for name, value in metadata["representations"].items()
                }
            else:
                embeddings = {"Embedding": path.parent / metadata["embedding"]}
            if metadata.get("classification"):
                evaluation = metadata["classification"]
                st.subheader("Classification check")
                st.caption(
                    f"{evaluation['evaluated_objects']} labeled objects evaluated; "
                    f"{len(evaluation['excluded_object_ids'])} objects excluded for missing labels. "
                    "Predictions are held out; scaling is fitted within training folds."
                )
                st.dataframe(pd.DataFrame(evaluation["fold_metrics"]))
                with st.expander("Per-class results and confusion matrices"):
                    for name, details in evaluation["classifiers"].items():
                        st.write(
                            "Linear classifier"
                            if name == "linear"
                            else "MLP classifier"
                            if name == "mlp"
                            else "Nearest-neighbor classifier"
                        )
                        st.caption(
                            "Confusion matrix: rows are true classes; columns are predicted classes."
                        )
                        st.dataframe(
                            pd.DataFrame(
                                details["confusion_matrix"],
                                index=details["classes"],
                                columns=details["classes"],
                            )
                        )
                        st.dataframe(
                            pd.DataFrame(
                                {
                                    key: value
                                    for key, value in details["per_class"].items()
                                    if isinstance(value, dict)
                                }
                            ).T
                        )
            st.download_button(
                "Export coordinates and clusters",
                frame.to_csv(sep="\t", index=False),
                "coordinates.tsv",
            )
            with st.expander("Provenance, settings, costs, exclusions, and metrics"):
                st.json(metadata)
            if (path.parent / "export.zip").exists():
                with (path.parent / "export.zip").open("rb") as stream:
                    st.download_button(
                        "Download report, embeddings, figures, and splits",
                        stream,
                        "morphofeatures-results.zip",
                    )
        else:
            embeddings = {path.stem: path}
        from morphofeatures.analysis_ui import inspect_results

        inspect_results(
            config,
            embeddings,
            metadata if path.suffix == ".json" else {},
            frame if path.suffix == ".json" else pd.DataFrame({"label_id": []}),
            path,
        )
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
