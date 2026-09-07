"""Thin Streamlit views over shared configuration, job, and artifact APIs."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from morphofeatures.artifacts import tail_text, write_json_atomic
from morphofeatures.config import repository_root
from morphofeatures.configuration_editor import load_document, parse_document, remember
from morphofeatures.registry import JobRegistry
from morphofeatures.workspace_jobs import submit_job


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

        getattr(st, kind)(label, key=key, on_change=changed, **options)
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


def launch_controls(document, config, namespace, *, invalid=False):
    root = retained_input("Output root", str(config.paths.output_root), namespace + ":root")
    run_id = retained_input(
        "Experiment / run ID (change for each variant)", namespace + "-001", namespace + ":run"
    )
    execution = retained_input(
        "Execution",
        "dry-run",
        namespace + ":execution",
        kind="selectbox",
        options=["dry-run", "local", "slurm"],
    )
    dependency = retained_input("SLURM afterok job ID (optional)", "", namespace + ":dependency")
    with st.expander("Resolved submission"):
        st.caption(
            "Output root and run ID place each stage under experiments/<run ID>/workspace. Run-specific checkpoint, log, and metadata paths replace the base configuration's output paths; both submitted settings and resolved configuration are saved."
        )
        st.code(yaml.safe_dump(document, sort_keys=False), language="yaml")
    if st.button(
        "Save dry run" if execution == "dry-run" else "Submit experiment",
        key="submit:" + namespace,
        disabled=invalid,
    ):
        try:
            record = submit_job(
                document,
                output_root=Path(root),
                run_id=run_id,
                execution=execution,
                base=repository_root(),
                dependency=dependency or None,
            )
            st.session_state["last_job:" + namespace] = record.working_directory
            write_json_atomic(Path(root) / ".morphofeatures" / (namespace + "-last.json"), document)
            st.success(f"{record.application_state}: {record.working_directory}")
        except Exception as error:
            st.error(str(error))
    folder = st.session_state.get("last_job:" + namespace)
    if folder:
        show_job(Path(folder), namespace)


def show_job(folder, namespace):
    st.caption(str(folder))
    st.button("Refresh progress and logs", key="refresh:" + namespace)
    status = folder / "status.json"
    if status.exists():
        state = json.loads(status.read_text())
        st.write(state["state"])
        if state.get("error"):
            st.error(state["error"])
        st.dataframe(pd.DataFrame(state.get("stages", [])))
    with st.expander("Logs"):
        st.code(tail_text(folder / "stdout.log") + "\n" + tail_text(folder / "stderr.log"))
        st.code(tail_text(folder / "metrics.jsonl"))
    if (folder / "job.slurm").exists():
        with st.expander("SLURM script"):
            st.code((folder / "job.slurm").read_text(), language="bash")


def training_page(config):
    st.title("Configure and train")
    st.caption(
        "Settings persist across pages. Every submission saves an independent configuration and runs outside the UI process."
    )
    source = retained_input("Base training config", "configs/smoke.yaml", "training:source")
    profile = retained_input(
        "Configuration profile (blank uses active profile)", "", "training:profile"
    )
    recent = (
        Path(st.session_state.get("value:training:root", config.paths.output_root))
        / ".morphofeatures"
        / "training-last.json"
    )
    left, right = st.columns(2)
    if left.button("Load base configuration"):
        try:
            set_document(
                "training", load_document(repository_root() / source, profile=profile or None)
            )
        except Exception as error:
            st.error(str(error))
    if right.button("Restore most recently submitted settings", disabled=not recent.exists()):
        saved = json.loads(recent.read_text())
        set_document("training", {**saved["stages"][0]["config"], "slurm": saved.get("slurm", {})})
    st.caption(
        "Loading resolves the active profile before editing. Restoring settings does not resume training; enable checkpoint resume explicitly below."
    )
    default = load_document(repository_root() / "configs/smoke.yaml")
    default.setdefault("data", {}).update({"crops": None, "label_ids": None, "loss_masks": None})
    default["training"].setdefault("weight_decay", 0.01)
    default["slurm"] = {
        "partition": "compute",
        "cpus": 4,
        "gpus": 0,
        "memory": "8G",
        "time": "01:00:00",
        "python_executable": sys.executable,
    }
    values, invalid = edit_document("training", default)
    st.download_button(
        "Export configuration",
        yaml.safe_dump(values, sort_keys=False),
        "training.yaml",
        "application/yaml",
    )
    resume = retained_input(
        "Resume training from training.resume_from", False, "training:resume", kind="checkbox"
    )
    if not resume:
        values.setdefault("training", {})["resume_from"] = None
    elif not values.get("training", {}).get("resume_from"):
        st.error("Set training.resume_from to an existing checkpoint in the configuration")
        invalid = True
    document = {"stages": [{"action": "train", "config": values}], "slurm": values.get("slurm", {})}
    launch_controls(document, config, "training", invalid=invalid)


def pipeline_page(config):
    st.title("Scientific pipeline")
    st.caption(
        "Ordered stages share outputs through from_preprocessing, from_training, and from_extraction. Stage failures stop the pipeline."
    )
    source = retained_input(
        "Pipeline YAML", "configs/workspace_pipeline.example.yaml", "pipeline:source"
    )
    if st.button("Load pipeline"):
        try:
            from morphofeatures.workspace_jobs import resolve_job

            path = (repository_root() / source).resolve()
            set_document("pipeline", resolve_job(parse_document(path.read_text()), path.parent))
        except Exception as error:
            st.error(str(error))
    values, invalid = edit_document(
        "pipeline",
        {
            "stages": [{"action": "train", "config": "configs/smoke.yaml"}],
            "slurm": {"partition": "compute", "gpus": 0},
        },
    )
    st.download_button("Export pipeline", yaml.safe_dump(values, sort_keys=False), "pipeline.yaml")
    launch_controls(values, config, "pipeline", invalid=invalid)


def workspace_jobs(config):
    records = [
        r
        for r in JobRegistry.under_output_root(config.paths.output_root).list(limit=500)
        if r.workflow == "workspace_run"
    ]
    if not records:
        return
    selected = st.selectbox(
        "Workspace pipeline details",
        records,
        format_func=lambda r: f"{r.run_id}: {r.application_state}",
    )
    show_job(Path(selected.working_directory), "registry")


def representations_page(config):
    st.title("Embeddings and comparison")
    extract_tab, analysis_tab, comparison_tab, results_tab = st.tabs(
        ("Extract", "Analyze", "Compare", "Reopen results")
    )
    with extract_tab:
        source = retained_input(
            "Model and target data config", "configs/smoke.yaml", "extract:source"
        )
        if st.button("Load model/data configuration"):
            try:
                current = st.session_state.get("document:extract", extraction_defaults())
                current["config"] = load_document(repository_root() / source)
                set_document("extract", current)
            except Exception as error:
                st.error(str(error))
        st.caption(
            "Edit config.data to choose the target objects. MAE architecture must match the checkpoint. DINO uses the same crop or grouped N5 object source."
        )
        stage, invalid = edit_document("extract", extraction_defaults())
        resources, resource_invalid = edit_document(
            "extract-resources",
            {
                "slurm": {
                    "partition": "compute",
                    "gpus": 0,
                    "cpus": 4,
                    "memory": "8G",
                    "time": "01:00:00",
                    "python_executable": sys.executable,
                }
            },
        )
        launch_controls(
            {"stages": [{**stage, "action": "extract"}], **resources},
            config,
            "extract",
            invalid=invalid or resource_invalid,
        )
    with analysis_tab:
        defaults = {
            "embedding": "",
            "normalization": "standardize",
            "umap": True,
            "neighbors": 15,
            "min_dist": 0.1,
            "umap_epochs": None,
            "clusters": 8,
            "cluster_method": "kmeans",
            "seed": 42,
        }
        stage, invalid = edit_document("analysis", defaults)
        launch_controls(
            {"stages": [{**stage, "action": "analyze"}], "slurm": resources["slurm"]},
            config,
            "analysis",
            invalid=invalid,
        )
    with comparison_tab:
        defaults = {
            "embeddings": {"MorphoFeatures": "", "DINOv2": "", "DINOv3": ""},
            "annotations": None,
            "label_column": "label",
            "group_column": None,
            "folds": 5,
            "knn_k": 5,
            "linear_c": 1.0,
            "evaluation_pca": None,
            "normalization": "standardize",
            "umap": True,
            "neighbors": 15,
            "min_dist": 0.1,
            "clusters": 8,
            "seed": 42,
        }
        st.caption(
            "Only IDs present in every representation are compared. Set group_column to keep related objects from the same specimen/acquisition together. Unlabeled results remain exploratory."
        )
        stage, invalid = edit_document("comparison", defaults)
        stage["embeddings"] = {k: v for k, v in stage["embeddings"].items() if v}
        launch_controls(
            {"stages": [{**stage, "action": "compare"}], "slurm": resources["slurm"]},
            config,
            "comparison",
            invalid=invalid,
        )
    with results_tab:
        result_browser(config)


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
    options = ["Enter path"] + list(dict.fromkeys(
        p for record in records for p in record.artifacts
        if Path(p).suffix in {".json", ".npz", ".npy", ".tsv", ".csv"}
    ))
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
            method = st.selectbox("Projection", ["pca"] + (["umap"] if "umap_1" in frame else []))
            if "representation" not in frame:
                frame["representation"] = "Embedding"
            names = frame.representation.unique().tolist()
            # Shared point selection is keyed by ID across all representation panels.
            charts = []
            for index, name in enumerate(names):
                subset = frame[frame.representation == name].copy()
                subset["label_id"] = subset.label_id.astype(str)
                chart = {
                    "title": name,
                    "data": {"values": subset.to_dict(orient="records")},
                    "mark": {"type": "point", "filled": True},
                    "width": 320,
                    "height": 360,
                    "encoding": {
                        "x": {"field": method + "_1", "type": "quantitative"},
                        "y": {"field": method + "_2", "type": "quantitative"},
                        "color": {"field": "cluster", "type": "nominal"},
                        "opacity": {
                            "condition": {"param": "object_selection", "value": 1},
                            "value": 0.12,
                        },
                        "tooltip": [{"field": "label_id"}, {"field": "cluster"}],
                    },
                }
                if index == 0:
                    chart["params"] = [
                        {
                            "name": "object_selection",
                            "select": {"type": "point", "fields": ["label_id"]},
                        }
                    ]
                charts.append(chart)
            st.caption(
                "Click a point in the first panel to highlight that object across panels; Shift-click selects multiple objects."
            )
            st.vega_lite_chart(
                {"$schema": "https://vega.github.io/schema/vega-lite/v5.json", "hconcat": charts}
            )
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
        from morphofeatures.data.io import load_embeddings
        from morphofeatures.representation_analysis import nearest_neighbors

        reference = load_embeddings(next(iter(embeddings.values())))
        selected_embedding = st.selectbox("Embedding to use in a new workflow", list(embeddings))
        if st.button("Analyze these embeddings in a new workflow"):
            from morphofeatures.workflow_ui import continue_from_artifact

            continue_from_artifact(config, embeddings[selected_embedding])
        neighbor_normalization = (
            metadata.get("settings", {}).get("normalization", "standardize")
            if path.suffix == ".json"
            else "standardize"
        )
        st.caption(
            f"Neighbor distances use Euclidean distance after {neighbor_normalization} preprocessing fitted to the displayed objects (exploratory)."
        )
        selected = st.selectbox(
            "Object for nearest-neighbor inspection", reference.label_ids.tolist(), format_func=str
        )
        columns = st.columns(len(embeddings))
        neighbor_tables = {}
        for column, (name, embedding) in zip(columns, embeddings.items()):
            with column:
                st.write(name)
                neighbor_tables[name] = nearest_neighbors(
                    embedding, selected, normalization=neighbor_normalization
                )
                st.dataframe(neighbor_tables[name])
        preview = retained_input(
            "Optional preprocessing config for object previews", "", "results:preview"
        )
        if preview:
            import numpy as np

            data = load_document(Path(preview))["data"]
            ids = np.load(data["label_ids"], mmap_mode="r")
            crops = np.load(data["crops"], mmap_mode="r")
            for column, (name, neighbors) in zip(columns, neighbor_tables.items()):
                with column:
                    for object_id in [selected] + neighbors.label_id.head(3).tolist():
                        row = np.flatnonzero(ids == object_id)
                        if len(row):
                            crop = np.asarray(crops[row[0]]).squeeze()
                            images = [
                                crop.take(crop.shape[axis] // 2, axis=axis) for axis in range(3)
                            ]
                            st.caption(f"{name}: object {object_id}")
                            st.image(
                                [((im - im.min()) / max(float(np.ptp(im)), 1e-8)) for im in images],
                                caption=["Axis 0", "Axis 1", "Axis 2"],
                            )
    except Exception as error:
        st.error(str(error))


def preprocessing_page(config):
    st.title("Prepare instance crops")
    st.caption(
        "Supply aligned raw and instance volumes. Zero segmentation is background. Spacing/origin describe the shared source grid; use unit=voxel when physical spacing is unknown."
    )
    values, invalid = edit_document(
        "preprocessing",
        {
            "raw": "",
            "raw_key": "exported_data",
            "raw_axes": "zyxc",
            "raw_channel": 0,
            "segmentation": "",
            "segmentation_key": "exported_data",
            "segmentation_axes": "zyxc",
            "segmentation_channel": 0,
            "spacing_zyx": [1.0, 1.0, 1.0],
            "origin_zyx": [0.0, 0.0, 0.0],
            "unit": "voxel",
            "roi": [[0, 0, 0], [64, 64, 64]],
            "block_shape": [32, 32, 32],
            "crop_shape": [32, 32, 32],
            "object_ids": [],
            "max_objects": 8,
            "min_voxels": 10,
            "center": "bbox",
            "oversized": "skip",
            "boundary": "pad",
            "roi_boundary": "skip",
            "background": 0.0,
            "normalization": "dtype",
        },
    )
    st.caption(
        "ROI bounds are half-open ZYX voxel indices. Objects touching the ROI boundary are skipped by default because their extent may be incomplete. oversized=clip and roi_boundary=allow retain explicitly marked partial objects."
    )
    resources, resource_invalid = edit_document(
        "preprocessing-resources",
        {
            "slurm": {
                "partition": "compute",
                "gpus": 0,
                "cpus": 4,
                "memory": "8G",
                "time": "01:00:00",
                "python_executable": sys.executable,
            }
        },
    )
    stages = [{**values, "action": "preprocess"}]
    if retained_input(
        "Train MAE after preprocessing", False, "preprocessing:train", kind="checkbox"
    ):
        train, training_invalid = edit_document(
            "preprocessing-training", load_document(repository_root() / "configs/smoke.yaml")
        )
        train.setdefault("mae", {})["input_shape"] = values["crop_shape"]
        stages.append({"action": "train", "from_preprocessing": True, "config": train})
        invalid |= training_invalid
        if retained_input(
            "Extract embeddings and analyze after training",
            False,
            "preprocessing:analyze",
            kind="checkbox",
        ):
            stages.extend(
                [
                    {"action": "extract", "model": "mae", "from_training": True},
                    {"action": "analyze", "from_extraction": True, "umap": False, "clusters": 2},
                ]
            )
    document = {"stages": stages, **resources}
    st.download_button(
        "Export preprocessing pipeline",
        yaml.safe_dump(document, sort_keys=False),
        "preprocess.yaml",
    )
    launch_controls(document, config, "preprocessing", invalid=invalid or resource_invalid)


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
