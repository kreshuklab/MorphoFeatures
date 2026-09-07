"""Streamlit workspace for the MorphoFeatures scientific pipeline."""

from __future__ import annotations

import importlib.util
import inspect
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

from morphofeatures.analysis.classification import (
    evaluate_embedding_classifier,
)
from morphofeatures.analysis.context import agglomerate_features, aggregate_neighbors
from morphofeatures.analysis.projection import cluster_embeddings, compute_umap
from morphofeatures.analysis.validation import validate_bundled_artifacts, validate_mobie_tables
from morphofeatures.artifacts import inspect_checkpoint, inspect_embedding, tail_text
from morphofeatures.config import load_config, repository_root
from morphofeatures.data.io import export_embeddings, load_embeddings, merge_embeddings
from morphofeatures.data.synthetic import save_synthetic_dataset
from morphofeatures.experiments import cancel_job, plan_job, refresh_jobs, submit_plan
from morphofeatures.mae_sweep import (
    load_prepared_sweep,
    plan_soft_grid,
    prepare_soft_grid,
    submit_soft_grid,
    summarize_soft_grid,
)
from morphofeatures.metrics import metric_series, read_metric_events
from morphofeatures.registry import DuplicateSubmissionError, JobRegistry
from morphofeatures.slurm import (
    ClusterProfile,
    DryRunScheduler,
    SlurmScheduler,
    load_cluster_profiles,
)
from morphofeatures.workflows import WORKFLOWS, WorkflowRequest, format_command

ROOT = repository_root()
DOCS = {
    "Dataset workspace workflows": ROOT / "docs" / "workspace_workflows.md",
    "Getting started": ROOT / "README.md",
    "Installation": ROOT / "docs" / "installation.md",
    "Data preparation": ROOT / "docs" / "data_preparation.md",
    "Legacy reproduction": ROOT / "docs" / "legacy_reproduction.md",
    "Training embeddings": ROOT / "docs" / "training_new_embeddings.md",
    "Modern MAE": ROOT / "docs" / "modern_mae_workflow.md",
    "Notebooks": ROOT / "docs" / "notebooks.md",
    "SLURM workspace": ROOT / "docs" / "slurm_workflow.md",
    "Analysis and MoBIE": ROOT / "docs" / "analysis_and_mobie.md",
    "Troubleshooting": ROOT / "docs" / "troubleshooting.md",
    "Reproducibility report": ROOT / "docs" / "reproducibility_report.md",
}
PAGES = (
    "Overview",
    "Analyze",
    "Build features",
    "Train and encode",
    "Experiments",
    "Scientific pipeline",
    "Embeddings and comparison",
    "Preprocessing",
    "Meshes",
    "Data and config",
    "Documentation",
)
CACHE_DATA = st.cache_data if hasattr(st, "cache_data") else st.experimental_memo
DATAFRAME_SUPPORTS_CONTAINER_WIDTH = (
    "use_container_width" in inspect.signature(st.dataframe).parameters
)


st.set_page_config(
    page_title="MorphoFeatures",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --mf-ink: #1d252b;
        --mf-muted: #617078;
        --mf-line: #d7dde0;
        --mf-teal: #0f766e;
        --mf-amber: #a55f0a;
        --mf-red: #b42318;
        --mf-surface: #11544d;
        --mf-bright-ink: #beebe6;
    }
    .stApp { color: var(--mf-bright-ink); }
    [data-testid="stSidebar"] { border-right: 1px solid var(--mf-line); }
    [data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
    .block-container { padding-top: 1.8rem; padding-bottom: 3rem; }
    h1, h2, h3 { letter-spacing: 0; color: var(--mf-bright-ink); }
    h1 { font-size: 1.75rem; margin-bottom: 0.2rem; }
    h2 { font-size: 1.25rem; margin-top: 0.5rem; }
    h3 { font-size: 1rem; }
    [data-testid="stMetric"] {
        border-top: 3px solid var(--mf-teal);
        border-radius: 4px;
        padding: 0.8rem 0.9rem;
        background: var(--mf-surface);
    }
    [data-testid="stMetricLabel"] { color: var(--mf-muted); }
    [data-testid="stForm"] { border-radius: 6px; border-color: var(--mf-line); }
    .mf-kicker { color: var(--mf-teal); font-weight: 700; font-size: 0.78rem; }
    .mf-note {
        border-left: 3px solid var(--mf-amber);
        background: #fff8eb;
        padding: 0.65rem 0.8rem;
        color: #5c411b;
        margin: 0.5rem 0 1rem;
    }
    .mf-ok { color: var(--mf-teal); font-weight: 700; }
    .mf-bad { color: var(--mf-red); font-weight: 700; }
    div.stButton > button { border-radius: 4px; }
    div[data-baseweb="select"] > div { border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _show_dataframe(data, *, height: Optional[int] = None):
    """Render a wide table across both legacy and current Streamlit releases."""
    options = {}
    if DATAFRAME_SUPPORTS_CONTAINER_WIDTH:
        options["use_container_width"] = True
    else:
        options["width"] = 1200
    if height is not None:
        options["height"] = height
    return st.dataframe(data, **options)


@CACHE_DATA(show_spinner=False)
def _artifact_frame(analysis_data: str) -> pd.DataFrame:
    results = validate_bundled_artifacts(Path(analysis_data))
    return pd.DataFrame(
        [
            {
                "artifact": name,
                "observed": " x ".join(str(value) for value in result.observed_shape),
                "expected": " x ".join(str(value) for value in result.expected_shape),
                "label_id": "valid" if result.label_ids_integral else "invalid",
                "status": "ready" if result.passed else "failed",
            }
            for name, result in results.items()
        ]
    )


@CACHE_DATA(show_spinner=False)
def _mobie_frame(mobie_data: str) -> pd.DataFrame:
    results = validate_mobie_tables(Path(mobie_data))
    return pd.DataFrame(
        {"table": list(results), "label_id": ["valid" if value else "missing" for value in results.values()]}
    )


def _capabilities() -> pd.DataFrame:
    modules = {
        "Published analysis": "sklearn",
        "UMAP": "umap",
        "Leiden": "leidenalg",
        "Torch training": "torch",
        "Mesh processing": "trimesh",
        "N5 / Zarr": "zarr",
        "WandB": "wandb",
    }
    capabilities = list(modules)
    statuses = [
        "available" if importlib.util.find_spec(module) is not None else "not installed"
        for module in modules.values()
    ]
    for command in ("sbatch", "squeue", "sacct", "scancel"):
        capabilities.append("SLURM {}".format(command))
        statuses.append("available" if shutil.which(command) else "not installed")
    return pd.DataFrame({"capability": capabilities, "status": statuses})


def _header(title: str, context: str) -> None:
    st.markdown('<div class="mf-kicker">MORPHOFEATURES WORKSPACE</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(context)


def _overview(config) -> None:
    _header("Pipeline overview", "Published artifacts, runtime capabilities, and workflow entry points.")
    artifacts = _artifact_frame(str(config.paths.analysis_data))
    mobie = _mobie_frame(str(config.paths.mobie_data))
    passed = int((artifacts["status"] == "ready").sum())
    columns = st.columns(4)
    columns[0].metric("Published arrays", "{}/{}".format(passed, len(artifacts)))
    columns[1].metric("MoBIE tables", "{}/{}".format((mobie["label_id"] == "valid").sum(), len(mobie)))
    columns[2].metric("MorphoFeatures", "480 dimensions")
    columns[3].metric("Context reference", "200 dimensions")

    left, right = st.columns((1.55, 1))
    with left:
        st.subheader("Artifact health")
        _show_dataframe(artifacts)
        st.subheader("Workflow")
        _show_dataframe(
            pd.DataFrame(
                [
                    ("1", "Validate inputs", "Data and config"),
                    ("2", "Train or load embeddings", "Train and encode"),
                    ("3", "Combine six feature groups", "Build features"),
                    ("4", "Classify, project, or cluster", "Analyze"),
                    ("5", "Export label-first tables", "Build features"),
                ],
                columns=("stage", "operation", "workspace"),
            )
        )
    with right:
        st.subheader("Runtime")
        _show_dataframe(_capabilities())
        st.subheader("Active paths")
        st.code(
            "analysis_data: {}\nmobie_data: {}\noutput_root: {}".format(
                _display_path(config.paths.analysis_data),
                _display_path(config.paths.mobie_data),
                _display_path(config.paths.output_root),
            ),
            language="yaml",
        )


def _classification(config) -> None:
    default_embedding = Path(
        st.session_state.get(
            "analysis_embedding_path",
            config.paths.analysis_data / "morphofeatures_all_cells.npy",
        )
    )
    default_labels = config.paths.analysis_data / "class_labels.tsv"
    with st.form("classification"):
        st.subheader("Cell-class prediction")
        path_column, label_column = st.columns(2)
        embedding_value = path_column.text_input("Embedding", _display_path(default_embedding))
        labels_value = label_column.text_input("Class labels", _display_path(default_labels))
        model_column, minimum_column = st.columns(2)
        classifier_model = model_column.selectbox(
            "Shallow model", ("logistic", "mlp"), help="Linear probe or one-hidden-layer MLP."
        )
        minimum_class_count = minimum_column.number_input(
            "Minimum matched cells per type", min_value=2, value=2
        )
        fold_column, seed_column, iteration_column = st.columns(3)
        folds = fold_column.number_input("Cross-validation folds", min_value=2, max_value=10, value=5)
        seed = seed_column.number_input("Seed", min_value=0, value=int(config.seed))
        max_iter = iteration_column.number_input("Maximum iterations", min_value=100, value=2000, step=100)
        submitted = st.form_submit_button("Run classification")
    if submitted:
        try:
            with st.spinner("Fitting an ID-aligned shallow classifier..."):
                result = evaluate_embedding_classifier(
                    _resolve(embedding_value),
                    _resolve(labels_value),
                    output_dir=config.paths.output_root / "analysis" / "shallow_classifier",
                    model=classifier_model,
                    folds=int(folds),
                    seed=int(seed),
                    max_iter=int(max_iter),
                    minimum_class_count=int(minimum_class_count),
                )
            st.session_state["classification_result"] = result
        except Exception as error:
            st.error(str(error))
    result = st.session_state.get("classification_result")
    if result is not None:
        metrics = st.columns(3)
        metrics[0].metric("Mean accuracy", "{:.3f}".format(result.mean_accuracy))
        metrics[1].metric("Standard deviation", "{:.3f}".format(result.std_accuracy))
        metrics[2].metric("Evaluated folds", str(len(result.scores)))
        score_frame = pd.DataFrame(
            {"fold": np.arange(1, len(result.scores) + 1), "accuracy": result.scores}
        ).set_index("fold")
        st.bar_chart(score_frame)
        confusion = pd.DataFrame(
            result.confusion,
            index=result.class_names,
            columns=result.class_names,
        )
        _show_dataframe(confusion)
        _show_dataframe(
            pd.DataFrame(
                {"cell_type": result.class_names, "recall": result.per_class_recall}
            )
        )
        st.caption(
            "Features are standardized inside each fold and labels are joined by label_id. "
            "Predictive association is not evidence of mechanism or cross-animal generalization."
        )


def _projection(config) -> None:
    default_embedding = Path(
        st.session_state.get(
            "analysis_embedding_path",
            config.paths.analysis_data / "morphofeatures_all_cells.npy",
        )
    )
    with st.form("projection"):
        st.subheader("Projection and clustering")
        input_column, output_column = st.columns(2)
        embedding_value = input_column.text_input(
            "Projection embedding", _display_path(default_embedding)
        )
        output_value = output_column.text_input(
            "Projection output", _display_path(config.paths.output_root / "projection.tsv")
        )
        annotation_value = st.text_input(
            "Biological metadata (optional, joined by label_id)",
            _display_path(config.paths.analysis_data / "class_labels.tsv"),
        )
        subset_column, method_column, cluster_column = st.columns(3)
        subset = subset_column.number_input("Subset cells (0 = all)", min_value=0, value=256, step=64)
        method = method_column.selectbox("Cluster method", ("kmeans", "leiden"))
        clusters = cluster_column.number_input("K-means clusters", min_value=2, value=8)
        neighbor_column, epoch_column, seed_column = st.columns(3)
        neighbors = neighbor_column.number_input("UMAP neighbors", min_value=2, value=15)
        epochs = epoch_column.number_input("UMAP epochs", min_value=10, value=50, step=10)
        seed = seed_column.number_input("Projection seed", min_value=0, value=int(config.seed))
        submitted = st.form_submit_button("Run projection")
    if submitted:
        try:
            with st.spinner("Computing standardized UMAP and clusters..."):
                table = load_embeddings(_resolve(embedding_value))
                rng = np.random.default_rng(int(seed))
                count = int(subset)
                if count and count < len(table.label_ids):
                    selected = np.sort(rng.choice(len(table.label_ids), size=count, replace=False))
                    ids, features = table.label_ids[selected], table.features[selected]
                else:
                    ids, features = table.label_ids, table.features
                features = StandardScaler().fit_transform(features)
                projection = compute_umap(
                    features,
                    n_neighbors=int(neighbors),
                    seed=int(seed),
                    n_epochs=int(epochs),
                )
                labels = cluster_embeddings(
                    features,
                    method=method,
                    n_neighbors=int(neighbors),
                    n_clusters=int(clusters),
                    seed=int(seed),
                )
            result_frame = pd.DataFrame(
                {"label_id": ids, "cluster": labels, "umap_1": projection[:, 0], "umap_2": projection[:, 1]}
            )
            if annotation_value.strip():
                annotation_path = _resolve(annotation_value)
                separator = "\t" if annotation_path.suffix.lower() == ".tsv" else ","
                annotations = pd.read_csv(annotation_path, sep=separator)
                if "label_id" not in annotations or annotations["label_id"].duplicated().any():
                    raise ValueError("Biological metadata requires unique label_id values")
                result_frame = result_frame.merge(
                    annotations, on="label_id", how="left", validate="one_to_one"
                )
            destination = export_embeddings(
                _resolve(output_value),
                ids,
                np.column_stack((labels, projection)),
                ("cluster", "umap_1", "umap_2"),
            )
            st.session_state["projection_result"] = result_frame
            st.success("Saved {}".format(_display_path(destination)))
        except Exception as error:
            st.error(str(error))
    result = st.session_state.get("projection_result")
    if result is not None:
        color_options = ["cluster"]
        color_options.extend(
            column for column in result.columns if column not in {"label_id", "cluster", "umap_1", "umap_2"}
        )
        color_field = st.selectbox("Color points by", color_options)
        tooltips = ["label_id", "cluster", "umap_1", "umap_2"]
        if color_field not in tooltips:
            tooltips.append(color_field)
        st.vega_lite_chart(
            result,
            {
                "mark": {"type": "point", "filled": True, "size": 45, "opacity": 0.75},
                "encoding": {
                    "x": {"field": "umap_1", "type": "quantitative"},
                    "y": {"field": "umap_2", "type": "quantitative"},
                    "color": {"field": color_field, "type": "nominal"},
                    "tooltip": tooltips,
                },
                "height": 500,
            },
            use_container_width=True,
        )
        _show_dataframe(result.head(100))


def _analyze(config) -> None:
    _header("Analyze embeddings", "Deterministic evaluation and exploratory structure.")
    classification_tab, projection_tab = st.tabs(("Classification", "Projection and clustering"))
    with classification_tab:
        _classification(config)
    with projection_tab:
        _projection(config)


def _combine_features(config) -> None:
    defaults = [
        config.paths.mobie_data / "features_shape_cell.tsv",
        config.paths.mobie_data / "features_shape_nucl.tsv",
        config.paths.mobie_data / "features_coarse_ultr_cell.tsv",
        config.paths.mobie_data / "features_coarse_ultr_nucl.tsv",
        config.paths.mobie_data / "features_fine_ultr_cell.tsv",
        config.paths.mobie_data / "features_fine_ultr_nucl.tsv",
    ]
    with st.form("combine"):
        st.subheader("Combine feature groups")
        values = []
        for index in range(0, len(defaults), 2):
            left, right = st.columns(2)
            values.append(left.text_input("Group {}".format(index + 1), _display_path(defaults[index])))
            values.append(right.text_input("Group {}".format(index + 2), _display_path(defaults[index + 1])))
        output_value = st.text_input(
            "Combined output", _display_path(config.paths.output_root / "morphofeatures.npy")
        )
        standardize = st.checkbox("Standardize before export", value=False)
        submitted = st.form_submit_button("Combine groups")
    if submitted:
        try:
            with st.spinner("Aligning label IDs and concatenating groups..."):
                table = merge_embeddings([_resolve(value) for value in values], standardize=standardize)
                destination = export_embeddings(_resolve(output_value), table.label_ids, table.features)
            st.success(
                "Saved {} cells x {} features to {}".format(
                    len(table.label_ids), table.features.shape[1], _display_path(destination)
                )
            )
        except Exception as error:
            st.error(str(error))


def _context_features(config) -> None:
    import pickle

    with st.form("context"):
        st.subheader("Aggregate neighbor context")
        input_column, neighbor_column = st.columns(2)
        embedding_value = input_column.text_input(
            "Context embedding",
            _display_path(config.paths.analysis_data / "morphofeatures_all_cells.npy"),
        )
        neighbor_value = neighbor_column.text_input(
            "Neighbor mapping", _display_path(config.paths.analysis_data / "bilateral_neighbors.pkl")
        )
        output_column, feature_column, reducer_column = st.columns(3)
        output_value = output_column.text_input(
            "Context output", _display_path(config.paths.output_root / "context.npy")
        )
        features = feature_column.number_input("Output features", min_value=1, value=200)
        reducer = reducer_column.selectbox("Neighbor reducer", ("mean", "max"))
        include_self = st.checkbox("Include source cell", value=True)
        submitted = st.form_submit_button("Build context features")
    if submitted:
        try:
            with st.spinner("Aggregating neighbors and clustering feature dimensions..."):
                table = load_embeddings(_resolve(embedding_value))
                with _resolve(neighbor_value).open("rb") as stream:
                    neighbors = pickle.load(stream)
                context = aggregate_neighbors(
                    table, neighbors, include_self=include_self, reducer=reducer
                )
                context = agglomerate_features(context, n_features=int(features))
                destination = export_embeddings(
                    _resolve(output_value), context.label_ids, context.features
                )
            st.success(
                "Saved {} cells x {} features to {}".format(
                    len(context.label_ids), context.features.shape[1], _display_path(destination)
                )
            )
        except Exception as error:
            st.error(str(error))


def _build_features(config) -> None:
    _header("Build feature sets", "Aligned six-group concatenation and neighbor aggregation.")
    combine_tab, context_tab = st.tabs(("MorphoFeatures", "MorphoContextFeatures"))
    with combine_tab:
        _combine_features(config)
    with context_tab:
        _context_features(config)


def _profiles(config) -> Dict[str, ClusterProfile]:
    configured = (config.raw or {}).get("slurm", {}).get(
        "profiles", "configs/slurm_profiles.example.yaml"
    )
    profile_value = os.environ.get("MORPHOFEATURES_SLURM_PROFILES", configured)
    profile_path = _resolve(profile_value)
    try:
        return load_cluster_profiles(profile_path)
    except Exception:
        return {"dry-run": ClusterProfile("dry-run", partition="compute")}


def _train_encode(config) -> None:
    from morphofeatures.workspace_ui import training_page

    training_tab, legacy_tab = st.tabs(("Persistent training configuration", "Legacy workflows and sweeps"))
    with training_tab:
        training_page(config)
    with legacy_tab:
        _legacy_train_encode(config)


def _legacy_train_encode(config) -> None:
    _header(
        "Train and encode",
        "Validate → snapshot → preview → explicitly save a dry run or submit to SLURM.",
    )
    definitions = tuple(WORKFLOWS.values())
    label_to_key = {definition.label: definition.key for definition in definitions}
    defaults = {
        "shape_train": "configs/shape_example.yaml",
        "shape_encode": "configs/shape_inference_example.yaml",
        "texture_train": "experiments/coarse_cell",
        "texture_encode": "experiments/coarse_cell",
        "mae_train": "configs/smoke.yaml",
        "mae_encode": "configs/smoke.yaml",
    }
    workflow_label = st.selectbox("Workflow", tuple(label_to_key), help="Scientific stage executed through the package CLI.")
    workflow = label_to_key[workflow_label]
    definition = WORKFLOWS[workflow]
    essential_left, essential_right = st.columns((2, 1))
    configuration_value = essential_left.text_input(
        "Configuration or texture experiment",
        defaults[workflow],
        key="workflow_config_{}".format(workflow),
        help="YAML for shape/MAE; a directory of the existing texture YAML files for texture workflows.",
    )
    run_id = essential_right.text_input(
        "Experiment / run ID",
        "mae-smoke" if workflow.startswith("mae_") else workflow.replace("_", "-"),
        key="workflow_run_{}".format(workflow),
        help="Stable registry identifier. Change it intentionally when resubmitting a modified experiment.",
    )
    input_value = st.text_input(
        "Input override (optional)",
        "",
        key="workflow_input_{}".format(workflow),
        help="Shape: manifest/root; texture: data root; MAE: crop .npy. The snapshot updates only the matching workflow-specific field.",
    )
    device_column, checkpoint_column, output_column = st.columns(3)
    device = device_column.selectbox(
        "Device", ("auto", "cpu", "cuda"), key="workflow_device_{}".format(workflow)
    )
    checkpoint_defaults = {
        "shape_encode": _display_path(config.paths.output_root / "shape_cell" / "checkpoints" / "best.pt"),
        "texture_encode": "experiments/coarse_cell/checkpoints/best.pt",
        "mae_train": "",
        "mae_encode": _display_path(config.paths.output_root / "mae" / "checkpoint.pt"),
    }
    checkpoint_default = checkpoint_defaults.get(workflow, "")
    checkpoint_value = checkpoint_column.text_input(
        "Checkpoint" + (" (input)" if definition.stage == "encoding" else " (output, if applicable)"),
        checkpoint_default,
        key="workflow_checkpoint_{}".format(workflow),
        help="Encoding reads this path. MAE training writes it; shape/texture training use their experiment checkpoint directories.",
    )
    output_value = output_column.text_input(
        "Embedding output",
        "",
        key="workflow_output_{}".format(workflow),
    )
    resume = False
    patches = False
    aggregate = False
    if workflow == "texture_train":
        resume = st.checkbox("Resume from last checkpoint", value=False)
    elif workflow == "texture_encode":
        patch_left, patch_right = st.columns(2)
        patches = patch_left.checkbox("Export fine-texture patches", value=False)
        aggregate = patch_right.checkbox("Aggregate patches per cell", value=patches, disabled=not patches)

    st.subheader("Cluster resources")
    profiles = _profiles(config)
    profile_name = st.selectbox(
        "Cluster profile",
        tuple(profiles),
        help="Profiles contain trusted module/environment setup. Free-form shell commands are not accepted.",
    )
    base_profile = profiles[profile_name]
    partition_column, time_column, memory_column = st.columns(3)
    partition = partition_column.text_input("Partition", base_profile.partition)
    walltime = time_column.text_input("Time", base_profile.time, help="HH:MM:SS or D-HH:MM:SS")
    memory = memory_column.text_input("Memory", base_profile.memory, help="For example 16G")
    cpu_column, gpu_column, account_column, qos_column = st.columns(4)
    cpus = cpu_column.number_input("CPUs", min_value=1, value=int(base_profile.cpus))
    gpus = gpu_column.number_input("GPUs", min_value=0, value=int(base_profile.gpus))
    account = account_column.text_input("Account (optional)", base_profile.account or "")
    qos = qos_column.text_input("QoS (optional)", base_profile.qos or "")
    with st.expander("Advanced scheduler and runtime settings"):
        topology_left, topology_middle, topology_right = st.columns(3)
        nodes = topology_left.number_input(
            "Nodes", min_value=1, value=int(base_profile.nodes)
        )
        ntasks = topology_middle.number_input(
            "Total tasks", min_value=1, value=int(base_profile.ntasks)
        )
        ntasks_per_node = topology_right.number_input(
            "Tasks per node", min_value=1, value=int(base_profile.ntasks_per_node)
        )
        gpu_directive = st.selectbox(
            "GPU request syntax",
            ("gres", "gpus"),
            index=("gres", "gpus").index(base_profile.gpu_directive),
            help="Use gres for clusters expecting --gres=gpu:N; use gpus for --gpus=N.",
        )
        mail_types = st.multiselect(
            "Mail notifications",
            (
                "BEGIN",
                "END",
                "FAIL",
                "REQUEUE",
                "TIME_LIMIT",
                "TIME_LIMIT_50",
                "TIME_LIMIT_80",
                "TIME_LIMIT_90",
                "INVALID_DEPEND",
                "STAGE_OUT",
                "ARRAY_TASKS",
                "ALL",
                "NONE",
            ),
            default=list(base_profile.mail_types),
            help="Notifications are emitted only when a mail address is also configured.",
        )
        mail_user = st.text_input("Mail address (optional)", base_profile.mail_user or "")
        python_executable = st.text_input(
            "Python executable (optional)",
            base_profile.python_executable or "",
            help=(
                "For example /path/to/env/bin/python. The launched module and subcommand "
                "remain fixed."
            ),
        )
        cpu_thread_env = st.checkbox(
            "Bind OMP/MKL threads to CPUs per task", value=base_profile.cpu_thread_env
        )
        local_runtime_directories = st.multiselect(
            "Run-local cache directories",
            ("wandb", "matplotlib"),
            default=list(base_profile.local_runtime_directories),
            help="Keeps optional W&B and Matplotlib files beneath the run directory.",
        )
        log_job_context = st.checkbox(
            "Print job ID, host, working directory, and command",
            value=base_profile.log_job_context,
        )

    registry = JobRegistry.under_output_root(config.paths.output_root)
    parent_candidates = [record for record in registry.list(limit=200) if record.slurm_job_id]
    parent_labels = ["No dependency"] + [
        "{} · {} · {} ({})".format(record.run_id, record.workflow, record.slurm_job_id, record.application_state)
        for record in parent_candidates
    ]
    parent_selection = st.selectbox(
        "Run after successful job (optional)",
        parent_labels,
        help="Adds an afterok dependency; the child starts only after the selected SLURM job succeeds.",
    )
    parent = parent_candidates[parent_labels.index(parent_selection) - 1] if parent_selection != "No dependency" else None
    request = WorkflowRequest(
        workflow,
        _resolve(configuration_value),
        input_path=_resolve(input_value) if input_value.strip() else None,
        device=device,
        checkpoint=_resolve(checkpoint_value) if checkpoint_value.strip() else None,
        output=_resolve(output_value) if output_value.strip() else None,
        resume=resume,
        save_patches=patches,
        aggregate_patches=aggregate,
    )
    plan = None
    profile = None
    try:
        profile = replace(
            base_profile,
            partition=partition,
            account=account or None,
            qos=qos or None,
            time=walltime,
            memory=memory,
            cpus=int(cpus),
            gpus=int(gpus),
            nodes=int(nodes),
            ntasks=int(ntasks),
            ntasks_per_node=int(ntasks_per_node),
            gpu_directive=gpu_directive,
            mail_types=tuple(mail_types),
            mail_user=mail_user or None,
            python_executable=python_executable or None,
            cpu_thread_env=cpu_thread_env,
            local_runtime_directories=tuple(local_runtime_directories),
            log_job_context=log_job_context,
        )
        plan = plan_job(
            request,
            profile,
            run_id=run_id,
            output_root=config.paths.output_root,
            protected_output_roots=(config.paths.analysis_data, config.paths.mobie_data),
            dependency_job_id=parent.slurm_job_id if parent else None,
            parent_job_id=parent.id if parent else None,
        )
        st.markdown('<span class="mf-ok">Configuration and paths validated</span>', unsafe_allow_html=True)
        for warning in plan.warnings:
            st.warning(warning)
    except Exception as error:
        st.error(str(error))

    preview_command, preview_script = st.tabs(("Exact command", "Generated SLURM script"))
    with preview_command:
        if plan:
            st.code(format_command(plan.command), language="bash")
            st.caption("Immutable snapshot: {}".format(_display_path(plan.snapshot_path)))
    with preview_script:
        if plan:
            st.code(plan.script, language="bash")

    dry_column, submit_column, runtime_column = st.columns(3)
    if dry_column.button("Save dry-run preview", disabled=plan is None):
        try:
            record = submit_plan(plan, registry, DryRunScheduler(), repository=ROOT)
            if record.application_state == "failed":
                st.error(record.error_message or "Could not save the dry-run snapshot")
            else:
                st.success("Saved dry-run record {}. No scheduler command was called.".format(record.id[:12]))
        except DuplicateSubmissionError as error:
            st.warning(str(error))
        except Exception as error:
            st.error(str(error))
    if submit_column.button("Submit to SLURM", disabled=plan is None):
        if not SlurmScheduler.available():
            st.error("sbatch is unavailable on this host. Save a dry run or launch the app on a SLURM login node.")
        else:
            try:
                record = submit_plan(plan, registry, SlurmScheduler(), repository=ROOT)
                if record.application_state == "failed":
                    st.error(record.error_message or "Submission failed")
                else:
                    st.success("Submitted SLURM job {}.".format(record.slurm_job_id))
            except DuplicateSubmissionError as error:
                st.warning(str(error))
            except Exception as error:
                st.error(str(error))
    if runtime_column.button("Check optional runtime"):
        _show_dataframe(_capabilities())
    st.markdown(
        '<div class="mf-note">Previewing and editing widgets never submits a job. Only “Submit to SLURM” invokes sbatch; dry runs persist the same snapshot and script without executing them.</div>',
        unsafe_allow_html=True,
    )
    if workflow == "mae_train":
        st.subheader("Optional bounded MAE soft grid")
        st.caption(
            "A one-at-a-time grid is usually easier to interpret than a full Cartesian grid. "
            "Every variant gets an immutable config, registry record, logs, metrics, and checkpoint."
        )
        sweep_left, sweep_right = st.columns(2)
        sweep_spec_value = sweep_left.text_input(
            "Sweep YAML",
            "configs/mae_nucleus_soft_grid.example.yaml",
            help="Only the documented scientific parameter paths are accepted.",
        )
        manifest_value = sweep_right.text_input(
            "Existing sweep manifest (optional)",
            st.session_state.get("mae_sweep_manifest", ""),
            help="Use this to reopen a prepared sweep after a browser or app restart.",
        )
        if st.button("Prepare immutable sweep configurations"):
            try:
                prepared = prepare_soft_grid(
                    _resolve(configuration_value),
                    _resolve(sweep_spec_value),
                    output_root=config.paths.output_root,
                )
                st.session_state["mae_sweep_manifest"] = str(prepared.manifest_path)
                st.success(
                    "Prepared {} variants. No job was submitted.".format(len(prepared.variants))
                )
            except Exception as error:
                st.error(str(error))
        active_manifest = manifest_value.strip() or st.session_state.get(
            "mae_sweep_manifest", ""
        )
        sweep_plans = None
        if active_manifest:
            try:
                if profile is None:
                    raise ValueError("Fix the cluster-resource validation errors before planning a sweep")
                prepared = load_prepared_sweep(_resolve(active_manifest))
                sweep_plans = plan_soft_grid(
                    prepared,
                    profile,
                    output_root=config.paths.output_root,
                    protected_output_roots=(config.paths.analysis_data, config.paths.mobie_data),
                )
                st.write("{} planned training jobs".format(len(sweep_plans)))
                _show_dataframe(
                    pd.DataFrame(
                        [
                            {
                                "variant": variant.name,
                                "run_id": variant.run_id,
                                "overrides": str(dict(variant.overrides)),
                            }
                            for variant in prepared.variants
                        ]
                    )
                )
                with st.expander("Exact generated SLURM scripts"):
                    for sweep_plan in sweep_plans:
                        st.caption(sweep_plan.run_id)
                        st.code(sweep_plan.script, language="bash")
                summary = summarize_soft_grid(prepared)
                if bool(summary["epochs_completed"].sum()):
                    st.caption("Existing post-training comparison")
                    _show_dataframe(summary)
            except Exception as error:
                st.error(str(error))
                sweep_plans = None
        encode_sweep = st.checkbox(
            "Queue encoding after each successful training job",
            value=True,
            help="Each encoder receives an afterok dependency on its corresponding training job.",
        )
        sweep_dry, sweep_submit = st.columns(2)
        if sweep_dry.button("Save all sweep jobs as dry runs", disabled=sweep_plans is None):
            try:
                records = submit_soft_grid(
                    sweep_plans,
                    registry,
                    DryRunScheduler(),
                    repository=ROOT,
                    encode_after_training=False,
                )
                st.success("Saved {} dry-run records; no scheduler command was called.".format(len(records)))
            except DuplicateSubmissionError as error:
                st.warning(str(error))
            except Exception as error:
                st.error(str(error))
        if sweep_submit.button("Submit soft grid to SLURM", disabled=sweep_plans is None):
            if not SlurmScheduler.available():
                st.error("sbatch is unavailable on this host; use dry run or a SLURM login node.")
            else:
                try:
                    records = submit_soft_grid(
                        sweep_plans,
                        registry,
                        SlurmScheduler(),
                        repository=ROOT,
                        encode_after_training=encode_sweep,
                    )
                    st.success("Submitted {} registered training/encoding jobs.".format(len(records)))
                except DuplicateSubmissionError as error:
                    st.warning(str(error))
                except Exception as error:
                    st.error(str(error))


def _experiments(config) -> None:
    _header("Experiments", "Persistent jobs, scheduler state, bounded logs, metrics, and artifacts.")
    from morphofeatures.workspace_ui import workspace_jobs

    workspace_jobs(config)
    registry = JobRegistry.under_output_root(config.paths.output_root)
    filter_column, state_column, refresh_column = st.columns((2, 2, 1))
    run_filter = filter_column.text_input("Filter run ID", "")
    states = state_column.multiselect(
        "Application state",
        ("dry_run", "queued", "running", "completed", "failed", "cancelled", "unknown"),
    )
    if refresh_column.button("Refresh SLURM"):
        if SlurmScheduler.accounting_available():
            try:
                updated, warnings = refresh_jobs(registry, SlurmScheduler())
                st.success("Refreshed {} active job(s).".format(updated))
                for warning in warnings:
                    st.warning(warning)
            except Exception as error:
                st.error(str(error))
        else:
            st.warning("squeue/sacct are unavailable; persisted records and artifacts remain viewable.")
    records = registry.list(run_id=run_filter or None, states=states, limit=500)
    if not records:
        st.info("No matching jobs. Create a dry run or submission on the Train and encode page.")
        return
    _show_dataframe(
        pd.DataFrame(
            [
                {
                    "run": record.run_id,
                    "workflow": record.workflow,
                    "stage": record.stage,
                    "state": record.application_state,
                    "SLURM": record.slurm_job_id or "—",
                    "raw state": record.raw_scheduler_state or "—",
                    "created": record.created_at,
                    "updated/completed": record.completed_at or record.started_at or record.submitted_at or "—",
                    "dependency": record.dependency_job_id or "—",
                }
                for record in records
            ]
        ),
        height=260,
    )
    labels = ["{} · {} · {}".format(record.run_id, record.workflow, record.id[:10]) for record in records]
    selected = records[labels.index(st.selectbox("Inspect job", labels))]
    details_tab, metrics_tab, logs_tab, artifacts_tab = st.tabs(("Details", "Metrics", "Logs", "Artifacts"))
    with details_tab:
        st.code(
            "command: {}\nworkdir: {}\nconfig: {}\ncheckpoint: {}\nembedding: {}\nparent: {}\nexit: {}".format(
                format_command(selected.command),
                selected.working_directory,
                selected.config_snapshot,
                selected.checkpoint_path or "—",
                selected.embedding_path or "—",
                selected.parent_job_id or "—",
                selected.exit_status or "—",
            ),
            language="yaml",
        )
        if selected.error_message:
            st.warning(selected.error_message)
        if selected.application_state in {"queued", "running", "unknown"}:
            confirmed = st.checkbox("Confirm cancellation of this SLURM job", value=False)
            if st.button("Cancel selected job", disabled=not confirmed):
                try:
                    cancel_job(selected, registry, SlurmScheduler())
                    st.success("Cancellation requested and retained in the registry.")
                except Exception as error:
                    st.error(str(error))
    with metrics_tab:
        try:
            events = read_metric_events(Path(selected.metrics_path))
            if events:
                latest_event = events[-1]
                st.caption(
                    "Latest structured event: {} at {}".format(
                        latest_event.get("event", "unknown"),
                        latest_event.get("timestamp", "unknown time"),
                    )
                )
                progress_events = [
                    event for event in events if event.get("event") == "batch_progress"
                ]
                if progress_events:
                    progress = progress_events[-1]
                    fraction = float(progress.get("progress_fraction", 0.0))
                    st.progress(max(0, min(100, int(round(fraction * 100)))))
                    st.caption(
                        "Epoch {epoch} · {phase} batch {batch}/{total} · "
                        "running loss {loss:.6f} · ETA {eta:.1f} min".format(
                            epoch=progress.get("epoch", "?"),
                            phase=progress.get("phase", "?"),
                            batch=progress.get("batch", "?"),
                            total=progress.get("total_batches", "?"),
                            loss=float(progress.get("running_loss", float("nan"))),
                            eta=float(progress.get("eta_seconds", 0.0)) / 60.0,
                        )
                    )
                with st.expander("Structured event timeline (latest 100)"):
                    _show_dataframe(pd.DataFrame(events[-100:]))
            series = metric_series(events)
            if series:
                frame = pd.DataFrame(series)
                index = "epoch" if "epoch" in frame else "step"
                chart_columns = [
                    column
                    for column in (
                        "train_loss",
                        "validation_loss",
                        "train_visible_mean_baseline_loss",
                        "validation_visible_mean_baseline_loss",
                    )
                    if column in frame
                ]
                st.line_chart(frame.set_index(index)[chart_columns])
                if "validation_improvement_over_visible_mean" in frame:
                    latest = float(frame["validation_improvement_over_visible_mean"].iloc[-1])
                    st.caption(
                        f"Latest validation improvement over the visible-mean baseline: "
                        f"{latest:.1%}. Negative values mean the model is worse than the baseline."
                    )
                _show_dataframe(frame)
            else:
                st.info("No structured epoch metrics are available yet.")
        except Exception as error:
            st.error(str(error))
    with logs_tab:
        stdout_tab, stderr_tab = st.tabs(("stdout (tail)", "stderr (tail)"))
        with stdout_tab:
            st.code(tail_text(Path(selected.stdout_path)) or "Log does not exist yet.")
        with stderr_tab:
            st.code(tail_text(Path(selected.stderr_path)) or "Log does not exist yet.")
    with artifacts_tab:
        st.write("Explicit run artifacts (no recursive guessing):")
        _show_dataframe(pd.DataFrame({"path": selected.artifacts or (
            selected.config_snapshot,
            selected.metrics_path,
            selected.checkpoint_path,
            selected.embedding_path,
            selected.stdout_path,
            selected.stderr_path,
        )}))
        if selected.checkpoint_path and Path(selected.checkpoint_path).is_file():
            try:
                st.json(inspect_checkpoint(Path(selected.checkpoint_path)))
            except Exception as error:
                st.warning(str(error))
        if selected.embedding_path and Path(selected.embedding_path).is_file():
            try:
                st.json(inspect_embedding(Path(selected.embedding_path)).to_dict())
                embedding = load_embeddings(Path(selected.embedding_path), mmap=True)
                class_metadata = pd.read_csv(
                    config.paths.analysis_data / "class_labels.tsv", sep="\t"
                )
                annotated = set(class_metadata["label_id"].astype(int))
                present = set(embedding.label_ids.astype(int))
                st.caption(
                    "Curated metadata coverage: {}/{} label IDs (explicit ID intersection).".format(
                        len(annotated.intersection(present)), len(annotated)
                    )
                )
                if st.button("Use this embedding on Analyze page"):
                    st.session_state["analysis_embedding_path"] = selected.embedding_path
                    st.success("Analysis defaults now point to this label-first embedding.")
            except Exception as error:
                st.error(str(error))


def _data_config(config, config_path: Path) -> None:
    _header("Data and configuration", "Input contracts, fixtures, and active repository paths.")
    paths_tab, fixture_tab, inspect_tab = st.tabs(("Paths", "Synthetic fixture", "Inspect embedding"))
    with paths_tab:
        st.subheader("Active configuration")
        st.code(
            "config: {}\nrepo_root: {}\ndata_root: {}\nanalysis_data: {}\nmobie_data: {}\noutput_root: {}".format(
                _display_path(config_path),
                config.paths.repo_root,
                config.paths.data_root,
                config.paths.analysis_data,
                config.paths.mobie_data,
                config.paths.output_root,
            ),
            language="yaml",
        )
        _show_dataframe(_mobie_frame(str(config.paths.mobie_data)))
    with fixture_tab:
        with st.form("synthetic"):
            destination_value = st.text_input(
                "Fixture output", _display_path(config.paths.output_root / "synthetic")
            )
            seed = st.number_input("Fixture seed", min_value=0, value=int(config.seed))
            submitted = st.form_submit_button("Generate fixture")
        if submitted:
            try:
                destination = save_synthetic_dataset(_resolve(destination_value), seed=int(seed))
                st.success("Generated {}".format(_display_path(destination)))
                _show_dataframe(
                    pd.DataFrame(
                        [
                            {"file": path.name, "bytes": path.stat().st_size}
                            for path in sorted(destination.iterdir())
                            if path.is_file()
                        ]
                    )
                )
            except Exception as error:
                st.error(str(error))
    with inspect_tab:
        value = st.text_input(
            "Embedding path",
            _display_path(config.paths.analysis_data / "morphofeatures_all_cells.npy"),
        )
        if st.button("Inspect embedding"):
            try:
                table = load_embeddings(_resolve(value), mmap=True)
                metrics = st.columns(4)
                metrics[0].metric("Cells", str(len(table.label_ids)))
                metrics[1].metric("Features", str(table.features.shape[1]))
                metrics[2].metric("First label", str(table.label_ids.min()))
                metrics[3].metric("Last label", str(table.label_ids.max()))
                _show_dataframe(
                    pd.DataFrame(table.features[:10, : min(12, table.features.shape[1])])
                )
            except Exception as error:
                st.error(str(error))


def _documentation() -> None:
    _header("Documentation", "Repository guides rendered alongside the workflow.")
    selection = st.selectbox("Guide", tuple(DOCS))
    path = DOCS[selection]
    if path.exists():
        st.markdown(path.read_text(encoding="utf-8"))
    else:
        st.error("Missing documentation: {}".format(path))


def _load_active_config(value: str):
    path = _resolve(value)
    try:
        return load_config(path), path, None
    except Exception as error:
        return load_config(), ROOT / "configs" / "default.yaml", error


def main() -> None:
    with st.sidebar:
        st.title("MorphoFeatures")
        st.caption("Scientific pipeline workspace")
        page = st.radio("Workspace", PAGES)
        st.markdown("---")
        config_value = st.text_input("Pipeline config", "configs/default.yaml")
        config, config_path, config_error = _load_active_config(config_value)
        if config_error:
            st.error(str(config_error))
        else:
            st.markdown('<span class="mf-ok">Config ready</span>', unsafe_allow_html=True)
        st.caption("v0.2.0")

    if page == "Overview":
        _overview(config)
    elif page == "Analyze":
        _analyze(config)
    elif page == "Build features":
        _build_features(config)
    elif page == "Train and encode":
        _train_encode(config)
    elif page == "Experiments":
        _experiments(config)
    elif page == "Scientific pipeline":
        from morphofeatures.workspace_ui import pipeline_page

        pipeline_page(config)
    elif page == "Embeddings and comparison":
        from morphofeatures.workspace_ui import representations_page

        representations_page(config)
    elif page == "Preprocessing":
        from morphofeatures.workspace_ui import preprocessing_page

        preprocessing_page(config)
    elif page == "Meshes":
        from morphofeatures.workspace_ui import mesh_page

        mesh_page(config)
    elif page == "Data and config":
        _data_config(config, config_path)
    else:
        _documentation()


main()
