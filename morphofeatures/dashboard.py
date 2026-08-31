"""Streamlit workspace for the MorphoFeatures scientific pipeline."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from sklearn.preprocessing import StandardScaler

from morphofeatures.analysis.classification import (
    cross_validate_logistic,
    load_class_labels,
    select_labeled_embeddings,
)
from morphofeatures.analysis.context import aggregate_neighbors, agglomerate_features
from morphofeatures.analysis.projection import cluster_embeddings, compute_umap
from morphofeatures.analysis.validation import validate_bundled_artifacts, validate_mobie_tables
from morphofeatures.config import load_config, repository_root
from morphofeatures.data.io import export_embeddings, load_embeddings, merge_embeddings
from morphofeatures.data.synthetic import save_synthetic_dataset


ROOT = repository_root()
DOCS = {
    "Getting started": ROOT / "README.md",
    "Installation": ROOT / "docs" / "installation.md",
    "Data preparation": ROOT / "docs" / "data_preparation.md",
    "Legacy reproduction": ROOT / "docs" / "legacy_reproduction.md",
    "Training embeddings": ROOT / "docs" / "training_new_embeddings.md",
    "Modern MAE": ROOT / "docs" / "modern_mae_workflow.md",
    "Analysis and MoBIE": ROOT / "docs" / "analysis_and_mobie.md",
    "Troubleshooting": ROOT / "docs" / "troubleshooting.md",
    "Reproducibility report": ROOT / "docs" / "reproducibility_report.md",
}
PAGES = (
    "Overview",
    "Analyze",
    "Build features",
    "Train and encode",
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
        --mf-surface: #f5f7f7;
    }
    .stApp { color: var(--mf-ink); }
    [data-testid="stSidebar"] { border-right: 1px solid var(--mf-line); }
    [data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
    .block-container { padding-top: 1.8rem; padding-bottom: 3rem; }
    h1, h2, h3 { letter-spacing: 0; color: var(--mf-ink); }
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
    return pd.DataFrame(
        {
            "capability": list(modules),
            "status": [
                "available" if importlib.util.find_spec(module) is not None else "not installed"
                for module in modules.values()
            ],
        }
    )


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
    default_embedding = config.paths.analysis_data / "morphofeatures_all_cells.npy"
    default_labels = config.paths.analysis_data / "class_labels.tsv"
    with st.form("classification"):
        st.subheader("Cell-class prediction")
        path_column, label_column = st.columns(2)
        embedding_value = path_column.text_input("Embedding", _display_path(default_embedding))
        labels_value = label_column.text_input("Class labels", _display_path(default_labels))
        fold_column, seed_column, iteration_column = st.columns(3)
        folds = fold_column.number_input("Cross-validation folds", min_value=2, max_value=10, value=5)
        seed = seed_column.number_input("Seed", min_value=0, value=int(config.seed))
        max_iter = iteration_column.number_input("Maximum iterations", min_value=100, value=2000, step=100)
        submitted = st.form_submit_button("Run classification")
    if submitted:
        try:
            with st.spinner("Fitting stratified logistic regression..."):
                table = load_embeddings(_resolve(embedding_value))
                ids, labels, class_names = load_class_labels(_resolve(labels_value))
                features, labels = select_labeled_embeddings(table, ids, labels)
                features = StandardScaler().fit_transform(features)
                result = cross_validate_logistic(
                    features,
                    labels,
                    class_names,
                    folds=int(folds),
                    seed=int(seed),
                    max_iter=int(max_iter),
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


def _projection(config) -> None:
    default_embedding = config.paths.analysis_data / "morphofeatures_all_cells.npy"
    with st.form("projection"):
        st.subheader("Projection and clustering")
        input_column, output_column = st.columns(2)
        embedding_value = input_column.text_input(
            "Projection embedding", _display_path(default_embedding)
        )
        output_value = output_column.text_input(
            "Projection output", _display_path(config.paths.output_root / "projection.tsv")
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
                destination = export_embeddings(
                    _resolve(output_value),
                    ids,
                    np.column_stack((labels, projection)),
                    ("cluster", "umap_1", "umap_2"),
                )
            st.session_state["projection_result"] = pd.DataFrame(
                {"label_id": ids, "cluster": labels, "umap_1": projection[:, 0], "umap_2": projection[:, 1]}
            )
            st.success("Saved {}".format(_display_path(destination)))
        except Exception as error:
            st.error(str(error))
    result = st.session_state.get("projection_result")
    if result is not None:
        st.vega_lite_chart(
            result,
            {
                "mark": {"type": "point", "filled": True, "size": 45, "opacity": 0.75},
                "encoding": {
                    "x": {"field": "umap_1", "type": "quantitative"},
                    "y": {"field": "umap_2", "type": "quantitative"},
                    "color": {"field": "cluster", "type": "nominal"},
                    "tooltip": ["label_id", "cluster", "umap_1", "umap_2"],
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


def _training_command(workflow: str, config_path: str, device: str, checkpoint: str, output: str) -> str:
    if workflow == "Shape training":
        return "python -m morphofeatures shape-train --config {}".format(config_path)
    if workflow == "Shape encoding":
        return "python -m morphofeatures shape-encode --config {} --output {}".format(
            config_path, output
        )
    if workflow == "Texture training":
        return "python -m morphofeatures texture-train {} --device {}".format(config_path, device)
    if workflow == "Texture encoding":
        return "python -m morphofeatures texture-encode {} --device {}".format(config_path, device)
    if workflow == "MAE training":
        return "python -m morphofeatures mae-train --config {} --output {}".format(
            config_path, checkpoint
        )
    return (
        "python -m morphofeatures mae-encode --config {} --checkpoint {} --output {}".format(
            config_path, checkpoint, output
        )
    )


def _train_encode(config) -> None:
    _header("Train and encode", "Validated configuration and explicit reproducible commands.")
    defaults = {
        "Shape training": "configs/shape_example.yaml",
        "Shape encoding": "configs/shape_inference_example.yaml",
        "Texture training": "experiments/coarse_cell",
        "Texture encoding": "experiments/coarse_cell",
        "MAE training": "configs/smoke.yaml",
        "MAE encoding": "configs/smoke.yaml",
    }
    workflow = st.selectbox("Workflow", tuple(defaults))
    input_column, device_column = st.columns((2, 1))
    config_path = input_column.text_input("Configuration or experiment", defaults[workflow])
    device = device_column.selectbox("Device", ("auto", "cpu", "cuda"))
    checkpoint_column, output_column = st.columns(2)
    checkpoint = checkpoint_column.text_input("Checkpoint", "outputs/mae/checkpoint.pt")
    output = output_column.text_input("Embedding output", "outputs/embeddings.npy")

    st.subheader("Command")
    command = _training_command(workflow, config_path, device, checkpoint, output)
    st.code(command, language="bash")
    check_column, dependency_column = st.columns(2)
    if check_column.button("Validate configuration"):
        candidate = _resolve(config_path)
        if workflow.startswith("Texture"):
            required = (candidate / "train_config.yml", candidate / "data_config.yml")
            missing = [path for path in required if not path.exists()]
            if missing:
                st.error("Missing: {}".format(", ".join(_display_path(path) for path in missing)))
            else:
                st.success("Texture experiment configuration is present.")
        elif not candidate.exists():
            st.error("Configuration does not exist: {}".format(_display_path(candidate)))
        else:
            try:
                with candidate.open("r", encoding="utf-8") as stream:
                    parsed = yaml.safe_load(stream) or {}
                st.success("Valid YAML with sections: {}".format(", ".join(parsed) or "none"))
            except Exception as error:
                st.error(str(error))
    if dependency_column.button("Check runtime"):
        _show_dataframe(_capabilities())
    st.markdown(
        '<div class="mf-note">Run training commands in a terminal so logs, GPU allocation, and interruption remain visible and controllable.</div>',
        unsafe_allow_html=True,
    )


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
    elif page == "Data and config":
        _data_config(config, config_path)
    else:
        _documentation()


main()
