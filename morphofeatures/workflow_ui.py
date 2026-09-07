"""Guided workflow editing and a single review/submit surface."""

from __future__ import annotations

import json
import socket
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from morphofeatures.config import repository_root
from morphofeatures.configuration_editor import load_document, parse_document
from morphofeatures.registry import JobRegistry
from morphofeatures.slurm import SlurmScheduler, load_cluster_profiles
from morphofeatures.workflows import format_command
from morphofeatures.workspace_jobs import (
    plan_workspace_job,
    resolve_job,
    submission_fingerprint,
    submit_workspace_plan,
)
from morphofeatures.workspace_state import (
    STAGE_LABELS,
    STARTING_POINTS,
    analysis_defaults,
    change_summary,
    import_workflow,
    load_draft,
    new_draft,
    save_draft,
    validate_draft_inputs,
    workflow_document,
)
from morphofeatures.workspace_ui import retained_input

STEPS = ("Start", "Prepare data", "Train", "Analyze", "Review & run")


def rerun():
    (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()


def navigate(page):
    st.session_state["navigate_to"] = page
    rerun()


def draft_key(config):
    return "workflow:" + str(config.paths.output_root)


def active_draft(config):
    return st.session_state.get(draft_key(config))


def use_draft(config, draft, step="Start"):
    st.session_state[draft_key(config)] = deepcopy(draft)
    st.session_state["workflow_step"] = step
    st.session_state.pop("workspace_plan", None)
    st.session_state["editor_epoch"] = st.session_state.get("editor_epoch", 0) + 1


def touch(draft):
    draft["revision"] += 1


def _get(values, path, default=None):
    try:
        for part in path:
            values = values[part]
        return values
    except (KeyError, IndexError, TypeError):
        return default


def _put(values, path, value):
    for part in path[:-1]:
        if isinstance(values, dict):
            values = values.setdefault(part, {})
        else:
            values = values[part]
    values[path[-1]] = value


def field(draft, path, label, default=None, *, options=None, help=None, optional=False):
    """Widgets mirror the draft; their cleanup on navigation cannot delete settings."""
    missing = object()
    value = _get(draft, path, missing)
    if value is missing:
        value = deepcopy(default)
        _put(draft, path, value)
        touch(draft)
    key = f"guided:{draft['id']}:{st.session_state.get('editor_epoch', 0)}:" + ".".join(
        map(str, path)
    )
    optional = optional and (value is None or isinstance(value, str))
    serialized = not isinstance(value, (str, int, float, bool)) and not optional and options is None
    initial = (
        yaml.safe_dump(value, default_flow_style=True).replace("\n...\n", "").strip()
        if serialized
        else ("" if optional and value is None else value)
    )
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool) and options is None
    if key not in st.session_state and not numeric:
        st.session_state[key] = initial

    def changed():
        replacement = st.session_state[key]
        if serialized:
            try:
                replacement = yaml.safe_load(replacement)
            except yaml.YAMLError:
                st.session_state.setdefault("guided_errors", {})[key] = (
                    f"{label}: enter valid YAML."
                )
                touch(draft)
                return
        st.session_state.setdefault("guided_errors", {}).pop(key, None)
        _put(draft, path, None if optional and replacement == "" else replacement)
        touch(draft)

    kwargs = {"key": key, "on_change": changed, "help": help}
    if options is not None:
        choices = list(options)
        if value not in choices:
            choices.insert(0, value)
        st.selectbox(label, choices, **kwargs)
    elif isinstance(value, bool):
        st.checkbox(label, **kwargs)
    elif isinstance(value, (int, float)):
        st.number_input(
            label, value=value, format="%d" if isinstance(value, int) else "%.8f", **kwargs
        )
    else:
        st.text_input(label, **kwargs)
    error = st.session_state.get("guided_errors", {}).get(key)
    if error:
        st.error(error)
    return _get(draft, path)


def triple_field(draft, path, label, default):
    value = _get(draft, path, default)
    if not isinstance(value, list) or len(value) != 3:
        return field(draft, path, label, default, help="Three values in Z, Y, X order.")
    if _get(draft, path) is None:
        _put(draft, path, deepcopy(default))
    st.caption(label + " · Z, Y, X")
    for axis, column in enumerate(st.columns(3)):
        with column:
            field(draft, path + [axis], f"{label} {'ZYX'[axis]}", value[axis])


def advanced_fields(draft, path, excluded=()):
    values = _get(draft, path, {})
    if not isinstance(values, dict):
        field(draft, path, "Additional settings", values)
        return
    for key, value in list(values.items()):
        if key in excluded:
            continue
        if isinstance(value, dict) and value:
            with st.expander(key.replace("_", " ").capitalize()):
                advanced_fields(draft, path + [key])
        else:
            field(
                draft,
                path + [key],
                key.replace("_", " ").capitalize(),
                value,
                help="Configuration field: " + ".".join(map(str, path[1:] + [key])),
            )


def _replace_document(draft, document):
    draft["document"] = deepcopy(document)
    touch(draft)
    st.session_state["editor_epoch"] = st.session_state.get("editor_epoch", 0) + 1
    st.session_state.pop("workspace_plan", None)
    st.session_state["guided_errors"] = {}


def yaml_editor(draft):
    with st.expander("Advanced: full pipeline YAML"):
        st.caption(
            "All stages and additional settings are preserved. Relative paths use the repository directory."
        )
        if not st.checkbox("Edit pipeline YAML", key="yaml_enabled:" + draft["id"]):
            st.code(yaml.safe_dump(draft["document"], sort_keys=False), language="yaml")
            return
        key = "pipeline_buffer:" + draft["id"]
        revision_key = key + ":revision"
        if key not in st.session_state:
            st.session_state[key] = yaml.safe_dump(draft["document"], sort_keys=False)
            st.session_state[revision_key] = draft["revision"]

        def reload_yaml():
            st.session_state[key] = yaml.safe_dump(draft["document"], sort_keys=False)
            st.session_state[revision_key] = draft["revision"]

        st.button("Reload YAML from current form settings", on_click=reload_yaml)
        edited = st.text_area("Pipeline YAML", key=key, height=320)
        stale = st.session_state[revision_key] != draft["revision"]
        if stale:
            st.warning(
                "The form changed since this YAML was opened. Reload it before applying YAML."
            )
        if st.button("Apply YAML to draft", disabled=stale):
            try:
                document = resolve_job(parse_document(edited), repository_root(), validate=False)
                _replace_document(draft, document)
                st.session_state[revision_key] = draft["revision"]
                rerun()
            except Exception as error:
                st.error(str(error))


def _start(config, draft):
    st.subheader("What are you starting with?")
    st.caption(
        "Choose an entry point, or load an existing configuration. Continue between stages without losing edits."
    )
    start = st.selectbox("Starting point", list(STARTING_POINTS), format_func=STARTING_POINTS.get)
    st.write(
        {
            "raw": "Prepare crops → train a model → extract embeddings → analyze.",
            "crops": "Use your prepared data → train → extract embeddings → analyze.",
            "checkpoint": "Use a matching model and data configuration → extract → analyze.",
            "embeddings": "Analyze one saved embedding or compare several representations.",
            "demo": "Train on eight generated synthetic crops, then extract and analyze. A short CPU demonstration.",
        }[start]
    )
    if draft:
        st.caption(
            "Starting again creates a separate draft. Save the current draft above to reopen it later."
        )
    if st.button("Create workflow" if not draft else "Create a new workflow", type="primary"):
        use_draft(
            config,
            new_draft(start),
            "Prepare data"
            if start == "raw"
            else "Train"
            if start in {"crops", "demo"}
            else "Analyze",
        )
        rerun()
    st.subheader("Load settings from a file")
    source = retained_input(
        "Training configuration or pipeline YAML",
        "configs/workspace_pipeline.example.yaml",
        "guided:source",
    )
    profile = retained_input(
        "Training profile (optional)",
        "",
        "guided:profile",
        help="For grouped N5 training configs. Blank uses the file's active profile.",
    )
    st.caption(
        "Loading copies settings into a draft. It does not change the source file or start a job."
    )
    if st.button("Preview settings to load"):
        try:
            path = (repository_root() / Path(source).expanduser()).resolve()
            document = import_workflow(path, profile or None)
            resolved_profiles = [
                s.get("config", {}).get("resolved_profile")
                for s in document["stages"]
                if isinstance(s.get("config"), dict)
            ]
            active_profile = profile or next((p for p in resolved_profiles if p), "")
            st.session_state["pending_import"] = {
                "document": document,
                "source": str(path),
                "profile": active_profile,
            }
        except Exception as error:
            st.session_state.pop("pending_import", None)
            st.error(str(error))
    pending = st.session_state.get("pending_import")
    if pending:
        st.info(
            "Ready to load: "
            + pending["source"]
            + (" · Profile: " + pending["profile"] if pending["profile"] else "")
        )
        st.write(
            " → ".join(
                STAGE_LABELS.get(s.get("action"), str(s.get("action")))
                for s in pending["document"]["stages"]
            )
        )
        if draft:
            changes = change_summary(draft["document"], pending["document"])
            with st.expander(f"Settings that will be replaced ({len(changes)})"):
                st.dataframe(pd.DataFrame(changes))
        if st.button("Replace current draft settings" if draft else "Load settings into draft"):
            loaded = new_draft(
                "import",
                document=pending["document"],
                source=pending["source"]
                + (" · Profile: " + pending["profile"] if pending["profile"] else ""),
            )
            use_draft(
                config,
                loaded,
                "Train"
                if any(s["action"] == "train" for s in loaded["document"]["stages"])
                else "Analyze",
            )
            st.session_state.pop("pending_import", None)
            rerun()
    saved = sorted((Path(config.paths.output_root) / ".morphofeatures" / "drafts").glob("*.json"))
    if saved:
        st.subheader("Reopen a saved draft")
        names = {}
        for path in saved:
            try:
                value = json.loads(path.read_text())
                names[path.stem] = value.get("name", path.stem) + " · " + value.get("run_id", "")
            except (OSError, ValueError):
                continue
        if names:
            identifier = st.selectbox("Saved drafts", list(names), format_func=names.get)
            if st.button("Open saved draft"):
                try:
                    saved = load_draft(config.paths.output_root, identifier)
                    first = saved["document"]["stages"][0].get("action")
                    step = {"preprocess": "Prepare data", "train": "Train"}.get(first, "Analyze")
                    use_draft(config, saved, step)
                    rerun()
                except Exception as error:
                    st.error(str(error))


def _load_stage_config(draft, index):
    with st.expander("Load model / data settings"):
        source = retained_input(
            "Model / data YAML path", "configs/smoke.yaml", f"stage:{index}:source"
        )
        profile = retained_input("Training profile", "", f"stage:{index}:profile")
        st.caption("This replaces this stage's model/data settings. It does not submit a job.")
        if st.button("Load settings into this stage", key=f"load-stage:{index}"):
            try:
                path = (repository_root() / Path(source).expanduser()).resolve()
                config = load_document(path, profile or None)
                document = deepcopy(draft["document"])
                document["stages"][index]["config"] = config
                _replace_document(draft, document)
                resolved_profile = config.get("resolved_profile", profile)
                draft.setdefault("stage_sources", {})[str(index)] = str(path) + (
                    " · Profile: " + resolved_profile if resolved_profile else ""
                )
                rerun()
            except Exception as error:
                st.error(str(error))
    if str(index) in draft.get("stage_sources", {}):
        st.caption("Model / data settings loaded: " + draft["stage_sources"][str(index)])


def _data_fields(draft, index, *, training=False):
    stage = draft["document"]["stages"][index]
    if stage.get("from_training") and "config" not in stage:
        st.info("Input: model, checkpoint and target data from Train in this workflow.")
        return
    path = ["document", "stages", index, "config", "data"]
    if stage.get("from_preprocessing"):
        st.info("Input: crops, object IDs and masks from Prepare data in this workflow.")
        return
    data = _get(draft, path, {})
    if data.get("source") == "n5_masked_patches":
        st.info("Input contract: grouped N5 patches. Keep this model paired with grouped data.")
        field(draft, path + ["patches_container"], "N5 patch container", "")
        field(draft, path + ["positions_container"], "N5 positions container", "")
    elif draft.get("allow_synthetic") and not data.get("crops"):
        st.info(
            "Synthetic demonstration: eight generated crops. No biological dataset is selected."
        )
    else:
        st.caption(
            "Input contract: whole-object crops. Crops, object IDs and masks must use the same row order."
        )
        field(draft, path + ["crops"], "Prepared crops (.npy)", None, optional=True)
        field(draft, path + ["label_ids"], "Object IDs (.npy)", None, optional=True)
        field(draft, path + ["loss_masks"], "Object masks (.npy, optional)", None, optional=True)


def _prepare(draft, index):
    path = ["document", "stages", index]
    st.caption("Use aligned raw and instance volumes. Label zero is background.")
    for kind, label in (("raw", "Raw image"), ("segmentation", "Instance segmentation")):
        field(draft, path + [kind], label + " path", "")
        with st.expander(label + " dataset, axes and channel", expanded=True):
            field(
                draft,
                path + [kind + "_key"],
                label + " dataset key",
                "exported_data",
                optional=True,
            )
            field(draft, path + [kind + "_axes"], label + " axis order", "zyxc")
            field(draft, path + [kind + "_channel"], label + " channel", 0)
    roi = _get(draft, path + ["roi"])
    if isinstance(roi, list) and len(roi) == 2:
        triple_field(draft, path + ["roi", 0], "ROI start (inclusive)", [0, 0, 0])
        triple_field(draft, path + ["roi", 1], "ROI stop (exclusive)", [64, 64, 64])
    else:
        field(
            draft,
            path + ["roi"],
            "ROI bounds",
            None,
            help="null scans the full volume; use [[z0,y0,x0],[z1,y1,x1]] to bound it.",
        )
    triple_field(draft, path + ["crop_shape"], "Crop shape", [32, 32, 32])
    field(draft, path + ["max_objects"], "Maximum objects (0 means all)", 8)
    field(
        draft,
        path + ["unit"],
        "Coordinate unit",
        "voxel",
        options=["voxel", "nm", "um", "micrometer"],
    )
    triple_field(draft, path + ["spacing_zyx"], "Voxel spacing", [1.0, 1.0, 1.0])
    st.caption(
        "Produces crops.npy, label_ids.npy, masks.npy and a model-ready data configuration. Objects touching ROI boundaries are skipped unless explicitly allowed."
    )
    with st.expander("Advanced preparation settings"):
        advanced_fields(
            draft,
            path,
            {
                "action",
                "raw",
                "segmentation",
                "raw_key",
                "segmentation_key",
                "raw_axes",
                "segmentation_axes",
                "raw_channel",
                "segmentation_channel",
                "roi",
                "crop_shape",
                "max_objects",
                "unit",
                "spacing_zyx",
            },
        )


def _train(draft, index):
    _artifact_picker(draft, index, "data")
    _load_stage_config(draft, index)
    _data_fields(draft, index, training=True)
    path = ["document", "stages", index, "config"]
    st.caption(
        "MAE learns a representation of your objects. Extraction turns the trained model into one feature vector per object."
    )
    field(draft, path + ["training", "epochs"], "Training epochs", 100)
    field(draft, path + ["training", "batch_size"], "Batch size", 4)
    field(draft, path + ["training", "learning_rate"], "Learning rate", 0.001)
    field(draft, path + ["device"], "Compute device", "auto", options=["auto", "cpu", "cuda"])
    with st.expander("Advanced model, data and training settings"):
        _model_advanced(draft, index, {"training", "device", "slurm"})
        advanced_fields(
            draft, path + ["training"], {"epochs", "batch_size", "learning_rate", "resume_from"}
        )
        field(
            draft,
            path + ["training", "resume_from"],
            "Resume checkpoint (grouped N5 only)",
            None,
            optional=True,
            help="Loading settings does not resume a model. This path explicitly requests checkpoint resume.",
        )
    st.caption(
        "Produces a checkpoint and resolved model configuration. Linked extraction uses them automatically."
    )


def _model_advanced(draft, index, excluded=()):
    path = ["document", "stages", index, "config"]
    linked = draft["document"]["stages"][index].get("from_preprocessing")
    advanced_fields(draft, path, set(excluded) | {"data", "mae", "config_schema", "resolved_profile"})
    with st.expander("MAE architecture"):
        if linked:
            st.caption("Input shape follows the crop shape in Prepare data.")
        advanced_fields(draft, path + ["mae"], {"input_shape"} if linked else ())
    with st.expander("Additional data settings"):
        advanced_fields(
            draft,
            path + ["data"],
            {"crops", "label_ids", "loss_masks", "patches_container", "positions_container"},
        )


def _extract(draft, index):
    path = ["document", "stages", index]
    stage = draft["document"]["stages"][index]
    if not stage.get("from_training"):
        _artifact_picker(draft, index, "checkpoint")
        _load_stage_config(draft, index)
    _data_fields(draft, index)
    if stage.get("from_training"):
        st.caption("Model: MAE · checkpoint from the preceding training stage.")
    else:
        field(
            draft, path + ["model"], "Embedding model", "mae", options=["mae", "dinov2", "dinov3"]
        )
        field(draft, path + ["checkpoint"], "Model checkpoint", "")
    if stage.get("model", "mae") != "mae":
        from morphofeatures.dino import VARIANTS

        field(draft, path + ["model_repository"], "Official local model repository", "")
        field(
            draft,
            path + ["variant"],
            "Backbone variant",
            sorted(VARIANTS[stage["model"]])[0],
            options=sorted(VARIANTS[stage["model"]]),
        )
        if "views" not in stage:
            from morphofeatures.workspace_ui import extraction_defaults

            stage["views"] = extraction_defaults()["views"]
            touch(draft)
    with st.expander("Advanced extraction settings"):
        field(
            draft,
            path + ["cache"],
            "Embedding cache directory",
            str(repository_root() / "outputs/embedding_cache"),
        )
        field(
            draft,
            path + ["data_version"],
            "Data version",
            "1",
            help="Change after editing source volumes in place to invalidate cached embeddings.",
        )
        field(
            draft, path + ["sequential_ids"], "Use sequential IDs when object IDs are absent", False
        )
        advanced_fields(
            draft,
            path,
            {
                "action",
                "model",
                "checkpoint",
                "from_training",
                "from_preprocessing",
                "model_repository",
                "variant",
                "data_version",
                "sequential_ids",
                "config",
                "cache",
            },
        )
        if "config" in stage:
            _model_advanced(draft, index)
    st.caption(
        "Produces ID-preserving embeddings. Analyze uses this output automatically when linked."
    )


def _analyze(draft, index):
    path = ["document", "stages", index]
    stage = draft["document"]["stages"][index]
    if stage["action"] == "analyze" and not stage.get("from_extraction"):
        _artifact_picker(draft, index, "embedding")
    if stage["action"] == "compare":
        st.caption(
            "Representations are matched by object ID. Annotation groups keep related specimens in the same evaluation fold."
        )
        advanced_fields(draft, path + ["embeddings"])
        field(draft, path + ["annotations"], "Annotation table (optional)", None, optional=True)
        field(draft, path + ["label_column"], "Label column", "label")
        field(
            draft,
            path + ["group_column"],
            "Specimen / acquisition group column (optional)",
            None,
            optional=True,
        )
        st.caption("Additional named representations can be added in the full pipeline YAML below.")
    elif stage.get("from_extraction"):
        st.info("Input: embeddings from Extract embeddings in this workflow.")
    else:
        field(draft, path + ["embedding"], "Saved embedding file", "")
    field(draft, path + ["umap"], "Include UMAP projection", False)
    field(
        draft,
        path + ["cluster_method"],
        "Clustering method",
        "kmeans",
        options=["kmeans", "leiden"],
    )
    field(draft, path + ["clusters"], "Number of clusters", 2)
    field(
        draft,
        path + ["normalization"],
        "Feature normalization",
        "standardize",
        options=["standardize", "l2", "none"],
    )
    with st.expander("Advanced analysis and evaluation settings"):
        advanced_fields(
            draft,
            path,
            {
                "action",
                "embedding",
                "embeddings",
                "from_extraction",
                "annotations",
                "label_column",
                "group_column",
                "umap",
                "clusters",
                "cluster_method",
                "normalization",
                "neighbors",
                "min_dist",
                "seed",
                "folds",
                "knn_k",
            },
        )
        field(draft, path + ["neighbors"], "UMAP neighbors", 15)
        field(draft, path + ["min_dist"], "UMAP minimum distance", 0.1)
        field(draft, path + ["seed"], "Random seed", 42)
        if stage["action"] == "compare":
            field(draft, path + ["folds"], "Evaluation folds", 5)
            field(draft, path + ["knn_k"], "Evaluation neighbors (K)", 5)
    st.caption(
        "Produces a saved report, coordinates, clusters and exports. Results reopens these without repeating computation."
    )


def _artifact_picker(draft, index, kind):
    root = Path(st.session_state.get("workspace_root", repository_root() / "outputs"))
    registry = root / ".morphofeatures" / "registry.sqlite3"
    if not registry.exists():
        return
    options = {}
    for record in JobRegistry(registry).list(states=["completed"], limit=500):
        for item in (*record.artifacts, record.checkpoint_path, record.embedding_path):
            if not item:
                continue
            path = Path(item)
            if (
                (kind == "data" and path.name == "mae_config.yaml")
                or (kind == "checkpoint" and path.suffix in {".pt", ".pth"})
                or (kind == "embedding" and path.suffix in {".npz", ".npy", ".tsv", ".csv"})
            ):
                options[str(path)] = record.run_id + " · " + path.name
    if not options:
        return
    with st.expander("Choose " + kind + " from a completed run"):
        selected = st.selectbox(
            "Completed " + kind + " outputs",
            list(options),
            format_func=options.get,
            key=f"artifact:{index}:{kind}",
        )
        if st.button("Use selected " + kind, key=f"use-artifact:{index}:{kind}"):
            try:
                path = Path(selected)
                if not path.exists():
                    raise ValueError("This output no longer exists: " + str(path))
                document = deepcopy(draft["document"])
                stage = document["stages"][index]
                if kind == "data":
                    prepared = load_document(path)
                    stage.setdefault("config", {})["data"] = prepared["data"]
                    stage["config"].setdefault("mae", {})["input_shape"] = prepared["mae"][
                        "input_shape"
                    ]
                    stage.pop("from_preprocessing", None)
                elif kind == "checkpoint":
                    stage["checkpoint"] = str(path)
                    config = path.parent / "resolved_config.yaml"
                    if config.exists():
                        stage["config"] = load_document(config)
                    stage.pop("from_training", None)
                else:
                    stage["embedding"] = str(path)
                    stage.pop("from_extraction", None)
                _replace_document(draft, document)
                draft.setdefault("stage_sources", {})[str(index)] = options[selected]
                rerun()
            except Exception as error:
                st.error(str(error))


def _stage_options(draft, step):
    document = draft["document"]
    stages = document["stages"]
    actions = [s["action"] for s in stages]
    with st.expander("Change which stages run"):
        st.caption(
            "Stage changes can be undone in this draft. Stopping early also removes later dependent stages."
        )

        def change(stages):
            draft.setdefault("stage_history", []).append(deepcopy(document))
            _replace_document(draft, {**document, "stages": stages})

        labels = [
            f"{i + 1}. {STAGE_LABELS.get(s['action'], s['action'])}" for i, s in enumerate(stages)
        ]
        stop = st.selectbox(
            "Stop this workflow after",
            list(range(len(stages))),
            format_func=lambda i: labels[i],
            index=len(stages) - 1,
        )
        if st.button("Keep stages through this point", disabled=stop == len(stages) - 1):
            change(stages[: stop + 1])
            rerun()
        if draft.get("stage_history") and st.button("Undo last stage change"):
            _replace_document(draft, draft["stage_history"].pop())
            rerun()
        if (
            "train" in actions
            and "extract" not in actions
            and st.button("Add embedding extraction after training")
        ):
            change(stages + [{"action": "extract", "model": "mae", "from_training": True}])
            rerun()
        if (
            "extract" in actions
            and not any(a in actions for a in ("analyze", "compare"))
            and st.button("Add analysis after extraction")
        ):
            change(stages + [analysis_defaults()])
            rerun()
        if len(stages) == 1 and actions == ["analyze"] and not stages[0].get("from_extraction"):
            if st.button("Compare multiple saved representations"):
                stage = {
                    **stages[0],
                    "action": "compare",
                    "embeddings": {
                        "Representation A": stages[0].get("embedding", ""),
                        "Representation B": "",
                    },
                }
                stage.pop("embedding", None)
                change([stage])
                rerun()
        if actions.count("train") == 1 and actions.count("extract") == 1:
            if st.button("Use an existing model instead of training"):
                model = deepcopy(next(s["config"] for s in stages if s["action"] == "train"))
                updated = deepcopy([s for s in stages if s["action"] != "train"])
                for stage in updated:
                    if stage["action"] == "extract":
                        stage.pop("from_training", None)
                        stage.update(config=model, checkpoint="", model="mae")
                        if "preprocess" in actions:
                            stage["from_preprocessing"] = True
                change(updated)
                rerun()


def _review(config, draft):
    st.subheader("Review & run")
    st.caption(
        "Review validates settings and generates the submission. It does not reserve a run ID or start work."
    )
    field(draft, ["run_id"], "Run ID", "run-001")
    st.caption(
        "Destination: "
        + str(Path(config.paths.output_root) / "experiments" / draft["run_id"] / "workspace")
    )
    field(draft, ["execution"], "Run on", "local", options=["local", "slurm"])
    path = ["document", "slurm"]
    if draft["execution"] == "slurm":
        with st.expander("Load a cluster resource preset"):
            source = retained_input(
                "Cluster presets YAML",
                "configs/slurm_profiles.example.yaml",
                "guided:cluster_profiles",
            )
            st.caption(
                "Example presets are placeholders. Loading a preset replaces resources for this workflow."
            )
            try:
                profiles = load_cluster_profiles(repository_root() / Path(source).expanduser())
                selected = st.selectbox("Cluster resource preset", list(profiles))
                if st.button("Use cluster resource preset"):
                    resources = asdict(profiles[selected])
                    resources.pop("name", None)
                    # Convert immutable profile tuples to portable YAML lists.
                    resources = json.loads(json.dumps(resources))
                    _replace_document(draft, {**draft["document"], "slurm": resources})
                    rerun()
            except Exception as error:
                st.error(str(error))
    field(draft, path + ["cpus"], "CPU threads / CPUs per task", 4)
    if draft["execution"] == "slurm":
        st.info("One Slurm job. Stages run in order and share the same resource allocation.")
        for key, label, default in (
            ("partition", "Partition", "compute"),
            ("account", "Account", None),
            ("gpus", "GPUs", 0),
            ("memory", "Memory", "8G"),
            ("time", "Time limit", "01:00:00"),
        ):
            field(draft, path + [key], label, default, optional=key == "account")
    else:
        st.info(
            f"A detached process runs on the app host, {socket.gethostname()}. CPU threads are applied; Slurm reservations and environment setup apply only on Slurm."
        )
    with st.expander("Advanced execution settings"):
        field(
            draft,
            path + ["python_executable"],
            "Worker Python interpreter (optional)",
            None,
            optional=True,
        )
        if draft["execution"] == "slurm":
            field(draft, ["dependency"], "Start after successful Slurm job ID (optional)", "")
            field(draft, path + ["qos"], "Quality of service (optional)", None, optional=True)
            field(
                draft,
                path + ["gpu_directive"],
                "GPU request syntax",
                "gpus",
                options=["gpus", "gres"],
            )
            field(
                draft,
                path + ["setup"],
                "Module / environment setup",
                [],
                help="Structured argument lists, for example [[module, load, CUDA/12.8]].",
            )
        advanced_fields(
            draft,
            path,
            {
                "cpus",
                "partition",
                "account",
                "gpus",
                "memory",
                "time",
                "python_executable",
                "qos",
                "gpu_directive",
                "setup",
            },
        )
    field(
        draft,
        ["allow_synthetic"],
        "Allow the synthetic demonstration when no real data is configured",
        False,
    )
    document = workflow_document(draft)
    kwargs = dict(
        output_root=config.paths.output_root,
        run_id=draft["run_id"],
        execution=draft["execution"],
        dependency=draft["dependency"] or None if draft["execution"] == "slurm" else None,
    )
    fingerprint = submission_fingerprint(document, **kwargs)
    form_errors = [
        value
        for key, value in st.session_state.get("guided_errors", {}).items()
        if draft["id"] in key
    ]
    buffer = st.session_state.get("pipeline_buffer:" + draft["id"])
    if (
        st.session_state.get("yaml_enabled:" + draft["id"])
        and buffer is not None
        and buffer != yaml.safe_dump(draft["document"], sort_keys=False)
    ):
        form_errors.append(
            "The full pipeline YAML differs from the form. Apply it or reload it from current settings before review."
        )
    for error in form_errors:
        st.error(error)
    if st.button("Review run", disabled=bool(form_errors), type="primary"):
        try:
            validate_draft_inputs(document, allow_synthetic=draft["allow_synthetic"])
            plan = plan_workspace_job(document, **kwargs)
            st.session_state["workspace_plan"] = (draft["id"], draft["revision"], plan)
        except Exception as error:
            st.session_state.pop("workspace_plan", None)
            st.error(str(error))
    saved = st.session_state.get("workspace_plan")
    if not saved or saved[0] != draft["id"]:
        return
    _, revision, plan = saved
    if revision != draft["revision"] or fingerprint != plan.fingerprint or form_errors:
        st.warning(
            "Settings changed after review. Review the run again to update its configuration and script."
        )
        return
    st.success("Review prepared. Inspect the stages, destination and execution details below.")
    rows = []
    for index, stage in enumerate(document["stages"]):
        links = [
            label
            for key, label in (
                ("from_preprocessing", "Prepare data"),
                ("from_training", "Train"),
                ("from_extraction", "Extract embeddings"),
            )
            if stage.get(key)
        ]
        rows.append(
            {
                "Order": index + 1,
                "Stage": STAGE_LABELS.get(stage["action"], stage["action"]),
                "Input": "From " + ", ".join(links) if links else "Configured files / data",
                "Stage directory": str(
                    Path(plan.record.working_directory) / f"{index:02d}-{stage['action']}"
                ),
            }
        )
    st.dataframe(pd.DataFrame(rows))
    st.caption(
        "Training output paths are scoped to this run. Extraction cache filenames are determined from the checkpoint and data fingerprints during execution."
    )
    changes = change_summary(draft["source_document"], document)
    with st.expander(f"Changes from loaded settings ({len(changes)})"):
        st.dataframe(pd.DataFrame(changes)) if changes else st.write(
            "No scientific settings changed."
        )
    with st.expander("Exact worker configuration and command"):
        st.code(format_command(plan.record.command), language="bash")
        st.code(plan.resolved_yaml, language="yaml")
    with st.expander("Generated Slurm script"):
        st.code(plan.script, language="bash")
    available = draft["execution"] != "slurm" or SlurmScheduler.available()
    if not available:
        st.warning(
            "sbatch is unavailable on the app host. You can save a dry-run bundle or run locally."
        )
    elif draft["execution"] == "slurm":
        st.caption(
            "Input checks use the app host. The selected interpreter, repository and data paths must also be accessible on compute nodes."
        )
    primary, secondary = st.columns(2)
    launch = primary.button(
        "Submit to Slurm" if draft["execution"] == "slurm" else "Run locally",
        disabled=not available,
        type="primary",
    )
    dry_run = secondary.button("Save dry-run bundle")
    st.caption(
        "Saving a dry-run bundle reserves this run ID and saves the configuration and script. Use a new run ID for a later submission."
    )
    if launch or dry_run:
        try:
            record = submit_workspace_plan(plan, expected_fingerprint=fingerprint, dry_run=dry_run)
            st.session_state["selected_run_id"] = record.id
            st.session_state["run_notice"] = (
                f"Saved dry-run bundle: {record.run_id}."
                if dry_run
                else f"Submitted Slurm job {record.slurm_job_id}."
                if record.slurm_job_id
                else f"Started local run {record.run_id}."
            )
            st.session_state.pop("workspace_plan", None)
            navigate("Runs")
        except Exception as error:
            st.session_state.pop("workspace_plan", None)
            st.error(str(error))


def workflow_page(config):
    st.title("Workflow")
    draft = active_draft(config)
    if not draft:
        _start(config, None)
        return
    field(draft, ["name"], "Workflow name", "My workflow")
    saved = draft["saved_revision"] == draft["revision"]
    st.caption(f"Loaded: {draft['source']} · {'Draft saved' if saved else 'Unsaved changes'}")
    left, middle, right = st.columns(3)
    if left.button("Save draft"):
        try:
            st.session_state[draft_key(config)] = save_draft(config.paths.output_root, draft)
            rerun()
        except Exception as error:
            st.error(str(error))
    middle.download_button(
        "Download pipeline YAML",
        yaml.safe_dump(workflow_document(draft), sort_keys=False),
        "pipeline.yaml",
        "application/yaml",
    )
    if right.button("Start / load another workflow"):
        st.session_state["workflow_step"] = "Start"
        st.session_state["step_widget"] = "Start"
        rerun()
    current = st.session_state.get("workflow_step", "Start")
    if st.session_state.get("step_widget") != current:
        st.session_state["step_widget"] = current

    def step_changed():
        st.session_state["workflow_step"] = st.session_state["step_widget"]

    step = st.radio(
        "Workflow steps", STEPS, horizontal=True, key="step_widget", on_change=step_changed
    )
    actions = [s.get("action") for s in draft["document"]["stages"]]
    st.caption(
        "Will run: " + " → ".join(STAGE_LABELS.get(action, str(action)) for action in actions)
    )
    if step == "Start":
        _start(config, draft)
        return
    if step == "Review & run":
        _review(config, draft)
    else:
        group = {
            "Prepare data": {"preprocess"},
            "Train": {"train"},
            "Analyze": {"extract", "analyze", "compare"},
        }[step]
        indices = [i for i, s in enumerate(draft["document"]["stages"]) if s.get("action") in group]
        if not indices:
            st.info(
                f"{step} is not included. This workflow starts from existing inputs or stops before this stage."
            )
        for index in indices:
            action = draft["document"]["stages"][index]["action"]
            st.subheader(STAGE_LABELS[action])
            {
                "preprocess": _prepare,
                "train": _train,
                "extract": _extract,
                "analyze": _analyze,
                "compare": _analyze,
            }[action](draft, index)
        _stage_options(draft, step)
    yaml_editor(draft)
    previous, following = st.columns(2)
    position = STEPS.index(step)
    if position > 0 and previous.button("Back to " + STEPS[position - 1]):
        st.session_state["workflow_step"] = STEPS[position - 1]
        rerun()
    if position < len(STEPS) - 1 and following.button("Continue to " + STEPS[position + 1]):
        st.session_state["workflow_step"] = STEPS[position + 1]
        rerun()


def continue_from_artifact(config, path, kind=None):
    path = Path(path).expanduser().resolve()
    if kind == "checkpoint" or path.suffix in {".pt", ".pth"}:
        draft = new_draft("checkpoint", source="Checkpoint: " + str(path))
        stage = draft["document"]["stages"][0]
        stage["checkpoint"] = str(path)
        source = path.parent / "resolved_config.yaml"
        if source.is_file():
            stage["config"] = load_document(source)
        step = "Analyze"
    elif path.name == "mae_config.yaml":
        draft = new_draft("crops", source="Prepared data: " + str(path))
        data = load_document(path)
        model = draft["document"]["stages"][0]["config"]
        model["data"] = data["data"]
        model["mae"]["input_shape"] = data["mae"]["input_shape"]
        step = "Train"
    else:
        draft = new_draft("embeddings", source="Embedding: " + str(path))
        draft["document"]["stages"][0]["embedding"] = str(path)
        step = "Analyze"
    draft["source_document"] = deepcopy(draft["document"])
    use_draft(config, draft, step)
    navigate("Workflow")


def run_actions(config, record):
    st.caption("Use this run's settings or completed outputs to continue the workflow.")
    if record.workflow == "workspace_run" and st.button("Use settings for a new run"):
        try:
            path = Path(record.working_directory) / "submitted_settings.yaml"
            draft = new_draft(
                "import", document=import_workflow(path), source="Run: " + record.run_id
            )
            use_draft(config, draft, "Review & run")
            navigate("Workflow")
        except Exception as error:
            st.error(str(error))
    if record.application_state != "completed":
        return
    artifacts = list(
        dict.fromkeys(
            p for p in (*record.artifacts, record.checkpoint_path, record.embedding_path) if p
        )
    )
    if artifacts:
        path = Path(st.selectbox("Continue from an output", artifacts))
        if (
            path.suffix in {".pt", ".pth", ".npz", ".npy", ".tsv", ".csv"}
            or path.name == "mae_config.yaml"
        ):
            if st.button("Use this output in a new workflow"):
                try:
                    if not path.exists():
                        raise ValueError(f"Output no longer exists: {path}")
                    continue_from_artifact(config, path)
                except Exception as error:
                    st.error(str(error))
        if path.suffix in {".json", ".npz", ".npy", ".tsv", ".csv"} and st.button(
            "Open this output in Results"
        ):
            st.session_state["value:results:path"] = str(path)
            st.session_state["widget:results:path"] = str(path)
            st.session_state["results:registered"] = "Enter path"
            navigate("Results")
