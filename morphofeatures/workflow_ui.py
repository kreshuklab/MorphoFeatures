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
from morphofeatures.configuration_editor import (
    load_document,
    parse_document,
    training_form_defaults,
)
from morphofeatures.registry import JobRegistry
from morphofeatures.slurm import SlurmScheduler, load_cluster_profiles
from morphofeatures.ui_settings import setting_help
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


def field(
    draft, path, label, default=None, *, options=None, help=None, optional=False, multiple=False
):
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

    if help is None:
        help = setting_help(path)
        if path[-1] == "normalization" and len(path) > 2 and path[1] == "stages":
            action = draft["document"]["stages"][path[2]].get("action")
            if action == "preprocess":
                help = "Crop intensity scaling: dtype maps the integer data range to [0,1] (float input must already be in [0,1]); percentile maps the object's 1st–99th intensity percentiles to [0,1]; none preserves intensities."
    kwargs = {"key": key, "on_change": changed, "help": help}
    if multiple:
        choices = list(dict.fromkeys([*(options or []), *(value or [])]))
        st.multiselect(label, choices, **kwargs)
    elif options is not None:
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
            "preprocess": "Prepare object crops, IDs and masks only. You can train or extract DINO features from these saved outputs later.",
            "crops": "Use your prepared data → train → extract embeddings → analyze.",
            "dino": "Use pretrained DINOv2 or DINOv3 → extract object features → visualize and optionally evaluate known classes. No MAE training is needed.",
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
            if start in {"raw", "preprocess"}
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
    stage = draft["document"]["stages"][index]
    dino = stage.get("action") == "extract" and stage.get("model") in {"dinov2", "dinov3"}
    with st.expander("Load model / data settings", expanded=dino):
        if dino:
            st.info(
                "Load mae_config.yaml from Prepare data to fill in crop paths and HDF5/N5 "
                "dataset keys. Already using grouped N5 patches for MAE? Load that same "
                "MAE YAML here to supply the patch container, positions/ID index and sampling. "
                "Your selected DINO weights, backbone and view settings are kept."
            )
        source = retained_input(
            "Model / data YAML path",
            "" if dino else "configs/smoke.yaml",
            f"stage:{draft['id']}:{index}:source",
            help="Load the generated mae_config.yaml for prepared crops. For the grouped Platynereis N5 inputs, use configs/sites/mae_platynereis_nuclei_embl.yaml. This field expects a YAML file."
            if dino
            else "YAML file containing this stage's model and input-data settings.",
        )
        profile = retained_input(
            "Data profile (optional)" if dino else "Training profile",
            "",
            f"stage:{draft['id']}:{index}:profile",
            help="Blank uses the file's active profile. Profiles also control which object IDs and how many patches per object are selected."
            if dino
            else "Blank uses the file's active training profile.",
        )
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
    dino = stage.get("action") == "extract" and stage.get("model") in {"dinov2", "dinov3"}
    if data.get("source") == "n5_masked_patches":
        st.info(
            "Input: existing grouped N5 patches. DINO encodes views of the stored patches and "
            "combines them into one feature vector per parent object ID from the positions index."
            if dino
            else "Input contract: grouped N5 patches. Keep this model paired with grouped data."
        )
        field(draft, path + ["patches_container"], "N5 patch container", "")
        field(draft, path + ["positions_container"], "N5 positions container", "")
        if dino:
            st.caption(
                f"Dataset keys: patches={data.get('patches_key', 'patches')}, "
                f"positions={data.get('positions_key', 'positions')}, IDs={data.get('ids_key', 'ids')}. "
                f"Sampling uses {data.get('group_size', 200)} patches per object and "
                f"split={stage.get('config', {}).get('inference', {}).get('split', 'all')}."
            )
    elif draft.get("allow_synthetic") and not data.get("crops"):
        st.info(
            "Synthetic demonstration: eight generated crops. No biological dataset is selected."
        )
    else:
        st.caption(
            "Input contract: whole-object crops. Crops, object IDs and masks must use the same row order."
        )
        if dino:
            st.caption(
                "Input: one crop per object, stored as NumPy, HDF5 or N5. For grouped N5 patches, load their MAE data YAML above."
            )
        for name, label, key_label in (
            ("crops", "Prepared crops (.npy / .h5 / .n5)", "Crop dataset key"),
            ("label_ids", "Object IDs (.npy / .h5 / .n5)", "Object ID dataset key"),
            ("loss_masks", "Object masks (optional)", "Mask dataset key"),
        ):
            field(draft, path + [name], label, None, optional=True)
            value = data.get(name)
            if isinstance(value, str) and Path(value).suffix.lower() in {
                ".h5",
                ".hdf5",
                ".hdf",
                ".n5",
            }:
                field(draft, path + [name + "_key"], key_label, None, optional=True)
        if (
            dino
            and data.get("crops")
            and Path(data["crops"]).suffix.lower() == ".n5"
            and not data.get("crops_key")
        ):
            st.warning(
                "N5 crops need a dataset key: load the mae_config.yaml produced by Prepare data. "
                "For grouped N5 patches, load their MAE YAML instead; the positions/ID index "
                "is needed alongside the patches. No additional preprocessing is required."
            )


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
    metadata = _inspect_preparation(draft, index)
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
    if metadata:
        _preparation_preview(draft, index, metadata)
    triple_field(draft, path + ["crop_shape"], "Crop shape", [32, 32, 32])
    field(draft, path + ["output_format"], "Output format", "npy", options=["npy", "h5", "n5"])
    output_format = _get(draft, path + ["output_format"], "npy")
    st.caption(
        "Saves crops.npy, masks.npy and label_ids.npy."
        if output_format == "npy"
        else f"Saves crops.{output_format} with compressed crops, masks and label_ids datasets."
    )
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
        "Every format includes mae_config.yaml with the paths and dataset keys needed by MAE, DINO and object previews. Objects touching ROI boundaries are skipped unless explicitly allowed."
    )
    with st.expander("Advanced preparation settings"):
        field(draft, path + ["object_ids"], "Specific object IDs (optional)", [])
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
                "output_format",
                "max_objects",
                "unit",
                "spacing_zyx",
                "object_ids",
            },
        )
    if len(draft["document"]["stages"]) > 1:
        if st.button(
            "Just Run Preprocessing",
            key=f"prepare-only:{index}",
            help="Keep this preparation stage and go to review. Later stages are removed from the draft; Undo last stage change restores them.",
        ):
            draft.setdefault("stage_history", []).append(deepcopy(draft["document"]))
            _replace_document(
                draft,
                {**draft["document"], "stages": [deepcopy(draft["document"]["stages"][index])]},
            )
            st.session_state["workflow_step"] = "Review & run"
            rerun()


def _inspect_preparation(draft, index):
    from morphofeatures.data.volumes import inspect_volume

    stage = draft["document"]["stages"][index]
    keys = (
        "raw",
        "segmentation",
        "raw_key",
        "segmentation_key",
        "raw_axes",
        "segmentation_axes",
        "raw_channel",
        "segmentation_channel",
    )
    signature = tuple(stage.get(key) for key in keys)
    key = f"volume-metadata:{draft['id']}:{index}"
    st.markdown("**Inspect data and choose a region**")
    st.caption(
        "Read dimensions first, then choose an ROI. Inspection reads metadata; the image preview reads one bounded slice."
    )
    if st.button("Inspect data dimensions", key=key + ":load"):
        metadata = {}
        for kind in ("raw", "segmentation"):
            try:
                if not stage.get(kind):
                    raise ValueError("Select a volume path first.")
                metadata[kind] = inspect_volume(
                    repository_root() / Path(stage[kind]).expanduser(),
                    stage.get(kind + "_key"),
                    stage.get(kind + "_axes", "zyx"),
                    stage.get(kind + "_channel", 0),
                )
            except Exception as error:
                metadata[kind] = {"error": str(error)}
        st.session_state[key] = (signature, metadata)
    stored = st.session_state.get(key)
    if not stored or stored[0] != signature:
        return None
    metadata = stored[1]
    for kind, value in metadata.items():
        if "error" in value:
            st.error(kind.capitalize() + ": " + value["error"])
    rows = [
        {
            "Volume": kind.capitalize(),
            "Stored dimensions": str(value["stored_shape"]),
            "Axis order": value["axes"],
            "Spatial Z, Y, X": str(value["spatial_shape"]),
            "Data type": value["dtype"],
            "Channel": value["channel"],
        }
        for kind, value in metadata.items()
        if "error" not in value
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    if any("error" in value for value in metadata.values()):
        return None
    if metadata["raw"]["spatial_shape"] != metadata["segmentation"]["spatial_shape"]:
        st.error(
            "The spatial dimensions differ. Select aligned volumes at the same resolution before preprocessing."
        )
        return None
    if st.button("Use full volume ROI", key=key + ":full"):
        document = deepcopy(draft["document"])
        document["stages"][index]["roi"] = [[0, 0, 0], list(metadata["raw"]["spatial_shape"])]
        _replace_document(draft, document)
        rerun()
    return metadata


def _preparation_preview(draft, index, metadata):
    import numpy as np

    from morphofeatures.data.volumes import preview_volume_pair

    stage = draft["document"]["stages"][index]
    key = f"volume-preview:{draft['id']}:{index}"
    if not st.checkbox("Show ROI image preview", value=True, key=key):
        return
    try:
        shape = metadata["raw"]["spatial_shape"]
        bounds = np.asarray(stage.get("roi") or [[0, 0, 0], list(shape)])
        if (
            bounds.shape != (2, 3)
            or not np.isfinite(bounds).all()
            or not np.equal(bounds, np.floor(bounds)).all()
        ):
            raise ValueError("Enter integer ROI start and stop coordinates.")
        start, stop = bounds.astype(int)
        if np.any(start < 0) or np.any(stop > shape) or np.any(stop <= start):
            raise ValueError(f"ROI bounds must be nonempty and inside {shape} (Z, Y, X).")
        st.caption("Selected ROI dimensions (Z, Y, X): " + str(tuple((stop - start).tolist())))
        axis = st.selectbox(
            "Preview slice axis",
            [0, 1, 2],
            format_func=lambda value: "ZYX"[value],
            key=key + ":axis",
        )
        lower, upper = int(start[axis]), int(stop[axis]) - 1
        plane = (
            st.slider(
                "Preview slice index",
                lower,
                upper,
                (lower + upper) // 2,
                key=f"{key}:slice:{axis}:{lower}:{upper}",
            )
            if lower < upper
            else lower
        )
        settings = deepcopy(stage)
        for kind in ("raw", "segmentation"):
            settings[kind] = str(repository_root() / Path(stage[kind]).expanduser())
        preview = preview_volume_pair(settings, axis=axis, index=plane)
        intensity, labels = preview["raw"], preview["segmentation"]
        if not np.isfinite(intensity).all():
            raise ValueError("The preview contains nonfinite raw intensities.")
        low, high = np.percentile(intensity, [1, 99])
        gray = np.clip((intensity.astype(float) - low) / max(float(high - low), 1e-8), 0, 1)
        ids, inverse = np.unique(labels, return_inverse=True)
        # Color indices are separate from scientific IDs, including large int64 IDs.
        import matplotlib.pyplot as plt

        colors = plt.get_cmap("tab20")(np.arange(len(ids)) % 20)[:, :3]
        colors[ids == 0] = 0
        colored = colors[inverse].reshape((*labels.shape, 3))
        overlay = np.repeat(gray[..., None], 3, axis=2)
        foreground = labels != 0
        overlay[foreground] = 0.55 * overlay[foreground] + 0.45 * colored[foreground]
        for column, image, caption in zip(
            st.columns(3),
            (gray, colored, overlay),
            ("Raw intensity", "Instance labels", "Alignment overlay"),
        ):
            with column:
                st.image(image, caption=caption, use_container_width=True, clamp=True)
        st.caption(
            f"Slice {'ZYX'[axis]}={plane}; displayed voxel window {preview['start']} → {preview['stop']}. "
            f"{int(np.count_nonzero(ids))} nonzero IDs intersect this slice; this is not a full object count."
        )
        if preview["cropped"]:
            st.info(
                "This ROI is larger than the preview limit. The image shows a central window of at most 512 × 512 pixels; preprocessing still uses the full selected ROI."
            )
    except Exception as error:
        st.error("Data preview: " + str(error))


def _train(draft, index):
    stage = draft["document"]["stages"][index]
    complete = training_form_defaults(stage.get("config", {}))
    if complete != stage.get("config"):
        stage["config"] = complete
        touch(draft)
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
    grouped = complete.get("data", {}).get("source") == "n5_masked_patches"
    _model_advanced(draft, index, {"training", "device", "slurm"})
    with st.expander("Optimization and data loading"):
        if grouped:
            field(
                draft,
                path + ["training", "scheduler"],
                "Learning-rate schedule",
                "constant",
                options=["constant", "cosine", "step"],
            )
        advanced_fields(
            draft,
            path + ["training"],
            {
                "epochs",
                "batch_size",
                "learning_rate",
                "resume_from",
                "early_stopping",
                "validation_fraction",
            }
            | ({"scheduler"} if grouped else set()),
        )
    with st.expander("Validation, early stopping and resume"):
        if grouped:
            advanced_fields(draft, path + ["training", "early_stopping"])
            field(
                draft,
                path + ["training", "resume_from"],
                "Resume checkpoint (grouped N5 only)",
                None,
                optional=True,
            )
        else:
            field(draft, path + ["training", "validation_fraction"], "Validation fraction", 0.25)
            st.caption(
                "Crop MAE trains for the requested epochs. Checkpoint resume and early stopping are available in the grouped N5 trainer."
            )
    st.caption(
        "Produces a checkpoint and resolved model configuration. Linked extraction uses them automatically."
    )


def _model_advanced(draft, index, excluded=()):
    path = ["document", "stages", index, "config"]
    linked = draft["document"]["stages"][index].get("from_preprocessing")
    with st.expander("Additional configuration, paths and reproducibility"):
        advanced_fields(
            draft,
            path,
            set(excluded) | {"data", "mae", "config_schema", "resolved_profile", "profiles"},
        )
        if _get(draft, path + ["profiles"]):
            st.caption(
                "Alternative profiles remain in the exported YAML. The resolved active settings above are the ones used for this run."
            )
    mae_path = path + ["mae"]
    mae = _get(draft, mae_path, {})
    grouped = _get(draft, path + ["data", "source"]) == "n5_masked_patches"
    with st.expander("MAE encoder and input geometry"):
        if linked:
            st.caption("Input shape follows the crop shape in Prepare data.")
        if mae.get("architecture_version"):
            st.caption("Checkpoint contract: " + mae["architecture_version"])
        if grouped:
            field(
                draft,
                mae_path + ["patch_encoder"],
                "Patch encoder",
                "linear",
                options=["linear", "resnet3d"],
            )
        decoder = {
            "decoder_dim",
            "decoder_depth",
            "decoder_heads",
            "reconstruction_shape",
            "norm_pix_loss",
        }
        advanced_fields(
            draft,
            mae_path,
            decoder
            | {"architecture_version"}
            | ({"patch_encoder"} if grouped else set())
            | ({"input_shape"} if linked else set()),
        )
    with st.expander("MAE decoder and reconstruction"):
        st.caption(
            "Grouped MAE reconstructs hidden stored patches using a transformer decoder."
            if grouped
            else "Crop MAE uses a fixed two-layer MLP decoder. Its hidden width is configurable; it has no decoder attention heads."
        )
        for key in (
            "decoder_dim",
            "decoder_depth",
            "decoder_heads",
            "reconstruction_shape",
            "norm_pix_loss",
        ):
            if key in mae:
                field(draft, mae_path + [key], key.replace("_", " ").capitalize(), mae[key])
    with st.expander("Additional data settings"):
        advanced_fields(
            draft,
            path + ["data"],
            {
                "crops",
                "label_ids",
                "loss_masks",
                "crops_key",
                "label_ids_key",
                "loss_masks_key",
                "patches_container",
                "positions_container",
            },
        )


def _extract(draft, index):
    path = ["document", "stages", index]
    stage = draft["document"]["stages"][index]
    if stage.get("from_training"):
        st.caption("Model: MAE · checkpoint from the preceding training stage.")
    else:
        field(
            draft, path + ["model"], "Embedding model", "mae", options=["mae", "dinov2", "dinov3"]
        )
    dino = stage.get("model", "mae") != "mae"
    if not stage.get("from_training"):
        _artifact_picker(draft, index, "data" if dino else "checkpoint")
        _load_stage_config(draft, index)
    _data_fields(draft, index)
    if not stage.get("from_training"):
        field(
            draft,
            path + ["checkpoint"],
            "Pretrained DINO weights" if dino else "Model checkpoint",
            "",
            help="Local backbone state dictionary matching the selected DINO family and variant. The workflow loads these frozen weights; it does not train DINO."
            if dino
            else None,
        )
    if dino:
        from morphofeatures.dino import VARIANTS

        st.info(
            "DINO features: convert each intensity crop into 2D views, encode with a frozen pretrained backbone, and combine the view vectors into one embedding per object. Supply intensity crops and aligned masks, rather than instance-label values as intensities."
        )
        field(draft, path + ["model_repository"], "Official local model repository", "")
        default_variant = "dinov2_vits14" if stage["model"] == "dinov2" else "dinov3_vits16"
        if stage.get("variant") in set.union(*VARIANTS.values()) - VARIANTS[stage["model"]]:
            stage["variant"] = default_variant
            _replace_document(draft, draft["document"])
            rerun()
        field(
            draft,
            path + ["variant"],
            "Backbone variant",
            default_variant,
            options=sorted(VARIANTS[stage["model"]]),
        )
        if "views" not in stage:
            from morphofeatures.workspace_ui import extraction_defaults

            stage["views"] = extraction_defaults()["views"]
            touch(draft)
        field(
            draft,
            path + ["config", "device"],
            "DINO compute device",
            "auto",
            options=["auto", "cpu", "cuda"],
        )
        with st.expander("DINO views and feature aggregation", expanded=True):
            views = path + ["views"]
            field(
                draft,
                views + ["normalization"],
                "DINO intensity normalization",
                "foreground_percentile",
                options=["foreground_percentile", "unit", "dtype"],
            )
            field(
                draft,
                views + ["feature"],
                "Feature vector per view",
                "cls",
                options=["cls", "patch_mean"],
            )
            field(
                draft,
                views + ["aggregation"],
                "Combine views per object",
                "mean",
                options=["mean", "max"],
            )
            field(
                draft,
                views + ["resize"],
                "Resize slices",
                "stretch",
                options=["letterbox", "stretch"],
            )
            advanced_fields(draft, views, {"normalization", "feature", "aggregation", "resize"})
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
                "views",
            },
        )
        if "config" in stage:
            if dino:
                advanced_fields(
                    draft,
                    path + ["config"],
                    {"data", "mae", "training", "slurm", "device", "profiles"},
                )
                with st.expander("Additional DINO data settings"):
                    advanced_fields(
                        draft,
                        path + ["config", "data"],
                        {
                            "crops",
                            "label_ids",
                            "loss_masks",
                            "crops_key",
                            "label_ids_key",
                            "loss_masks_key",
                            "patches_container",
                            "positions_container",
                        },
                    )
            else:
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
        field(draft, path + ["label_column"], "Label column", "auto")
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
    field(draft, path + ["clusters"], "Number of clusters", 8)
    field(
        draft,
        path + ["normalization"],
        "Feature normalization",
        "standardize",
        options=["standardize", "l2", "none"],
    )
    with st.expander(
        "Classification check against known labels", expanded=bool(stage.get("annotations"))
    ):
        st.caption(
            "Join known cell types by label_id for side-by-side cluster/class plots. Enable classification to evaluate the same logistic or MLP probes as Tools, plus optional K-nearest neighbors. Scaling and optional PCA are fitted within each training fold."
        )
        if stage["action"] != "compare":
            field(draft, path + ["annotations"], "Annotation table (optional)", None, optional=True)
            field(draft, path + ["label_column"], "Label column", "auto")
            field(
                draft,
                path + ["group_column"],
                "Specimen / acquisition group column (optional)",
                None,
                optional=True,
            )
        field(draft, path + ["folds"], "Evaluation folds", 5)
        field(draft, path + ["classify"], "Evaluate classifiers", True)
        field(
            draft,
            path + ["classifier_models"],
            "Classifiers",
            ["logistic", "knn"],
            options=["logistic", "mlp", "knn"],
            multiple=True,
        )
        field(draft, path + ["minimum_class_count"], "Minimum matched cells per type", 2)
        field(
            draft,
            path + ["fold_policy"],
            "Insufficient examples per fold",
            "reduce",
            options=["reduce", "strict"],
        )
        field(
            draft,
            path + ["class_weight"],
            "Logistic class weights",
            "balanced",
            options=["balanced", None],
        )
        field(draft, path + ["hidden_dimensions"], "MLP hidden layer dimensions", [64])
        field(draft, path + ["predict_unlabeled"], "Predict cell types for unlabeled objects", True)
        models = stage.get("classifier_models") or ["logistic"]
        if stage.get("prediction_model") not in models:
            stage["prediction_model"] = models[0]
        field(
            draft,
            path + ["prediction_model"],
            "Classifier for label plots and volume export",
            models[0],
            options=models,
        )
        field(draft, path + ["unlabeled_opacity"], "Unlabeled point opacity", 0.15)
        field(
            draft,
            path + ["input_config"],
            "Input data YAML for object inspection (optional)",
            None,
            optional=True,
        )
        field(draft, path + ["knn_k"], "Evaluation neighbors (K)", 5)
        field(draft, path + ["linear_c"], "Linear classifier inverse regularization", 1.0)
        field(draft, path + ["max_iter"], "Classifier maximum iterations", 2000)
        field(draft, path + ["evaluation_pca"], "Evaluation PCA dimensions (null disables)", None)
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
                "linear_c",
                "max_iter",
                "evaluation_pca",
                "classify",
                "classifier_models",
                "minimum_class_count",
                "fold_policy",
                "class_weight",
                "hidden_dimensions",
                "predict_unlabeled",
                "prediction_model",
                "unlabeled_opacity",
                "input_config",
                "subset",
                "umap_epochs",
                "resolution",
            },
        )
        field(draft, path + ["neighbors"], "UMAP neighbors", 15)
        field(draft, path + ["min_dist"], "UMAP minimum distance", 0.0)
        field(draft, path + ["umap_epochs"], "UMAP epochs (null uses automatic)", 50)
        field(draft, path + ["resolution"], "Leiden resolution", 0.004)
        field(draft, path + ["subset"], "Projection subset (0 means all)", 0)
        field(draft, path + ["seed"], "Random seed", 42)
    st.caption(
        "Produces a saved report, coordinates, clusters and exports. Results reopens these without repeating computation."
    )


def _export_labels(draft, index):
    path = ["document", "stages", index]
    st.caption(
        "Write categorical label volumes on the original segmentation grid. Zero means background or unassigned; lookup tables preserve class colors and original object IDs. This runs as a reviewed local/Slurm job."
    )
    for key, label, default in (
        ("labels", "Object label table", ""),
        ("segmentation", "Original instance segmentation", ""),
        ("segmentation_key", "Segmentation dataset key", "exported_data"),
        ("segmentation_axes", "Segmentation axis order", "zyx"),
        ("segmentation_channel", "Segmentation channel", 0),
    ):
        field(draft, path + [key], label, default)
    field(
        draft,
        path + ["layers"],
        "Label volumes to write",
        ["known_label", "predicted_label", "cluster"],
        options=["known_label", "predicted_label", "cluster"],
        multiple=True,
    )
    field(
        draft,
        path + ["output_format"],
        "Volume output format",
        "h5",
        options=["h5", "zarr2", "zarr3"],
        help="HDF5 file or a Zarr v2/v3 directory containing categorical label volumes and lookup metadata. V3 needs zarr-python 3 or a recent z5py backend.",
    )
    field(
        draft,
        path + ["include_rgb"],
        "Also write RGB color volumes",
        False,
        help="Adds uint8 Z/Y/X/RGB datasets using the exact plot colors. Uses extra disk space; categorical label volumes remain available for label-aware viewers.",
    )
    triple_field(draft, path + ["spacing_zyx"], "Voxel spacing", [1.0, 1.0, 1.0])
    triple_field(draft, path + ["origin_zyx"], "Voxel origin", [0.0, 0.0, 0.0])
    field(draft, path + ["unit"], "Coordinate unit", "voxel", options=["voxel", "nm", "um"])
    with st.expander("ID mapping and bounded I/O"):
        field(
            draft,
            path + ["id_mapping"],
            "Embedding-to-segmentation ID mapping (optional)",
            None,
            optional=True,
        )
        field(
            draft, path + ["mapping_column"], "Segmentation ID column in mapping", "segmentation_id"
        )
        triple_field(draft, path + ["block_shape"], "Export block shape", [64, 64, 64])


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
        if actions == ["preprocess"]:
            if st.button("Add MAE training after preprocessing"):
                from morphofeatures.workspace_state import training_defaults

                change(
                    stages
                    + [
                        {
                            "action": "train",
                            "config": training_defaults(),
                            "from_preprocessing": True,
                        },
                        {"action": "extract", "model": "mae", "from_training": True},
                        analysis_defaults(),
                    ]
                )
                rerun()
            if st.button("Add DINO features after preprocessing"):
                dino = new_draft("dino")["document"]["stages"]
                dino[0]["from_preprocessing"] = True
                change(stages + dino)
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


def _execution_fields(draft):
    path = ["document", "slurm"]
    if draft["execution"] != "slurm":
        field(draft, path + ["cpus"], "CPU threads / CPUs per task", 4)
        st.info(
            f"A detached process runs on the app host, {socket.gethostname()}. CPU threads are applied; Slurm reservations and environment setup apply only on Slurm."
        )
        with st.expander("Local Python environment"):
            field(
                draft,
                path + ["python_executable"],
                "Worker Python interpreter (optional)",
                None,
                optional=True,
            )
        return

    st.subheader("Cluster resources")
    st.caption("One Slurm job runs the stages in order. They share one resource allocation.")
    with st.expander("Load a cluster resource preset"):
        source = retained_input(
            "Cluster presets YAML", "configs/slurm_profiles.example.yaml", "guided:cluster_profiles"
        )
        st.caption(
            "Loading a preset copies its resources and environment into this draft. Example presets need your cluster's partition and account."
        )
        try:
            profiles = load_cluster_profiles(repository_root() / Path(source).expanduser())
            selected = st.selectbox("Cluster resource preset", list(profiles))
            if st.button("Use cluster resource preset"):
                resources = asdict(profiles[selected])
                resources.pop("name", None)
                _replace_document(
                    draft, {**draft["document"], "slurm": json.loads(json.dumps(resources))}
                )
                rerun()
        except Exception as error:
            st.error(str(error))
    for columns in (
        (
            ("partition", "Partition", "compute"),
            ("time", "Time limit", "01:00:00"),
            ("memory", "Memory", "8G"),
        ),
        (
            ("cpus", "CPUs per task", 4),
            ("gpus", "GPUs", 0),
            ("account", "Account (optional)", None),
            ("qos", "QoS (optional)", None),
        ),
    ):
        for column, (key, label, default) in zip(st.columns(len(columns)), columns):
            with column:
                field(draft, path + [key], label, default, optional=key in {"account", "qos"})
    with st.expander("Mail notifications", expanded=True):
        # The scheduler also accepts comma-separated values from imported YAML.
        settings = draft["document"].setdefault("slurm", {})
        if isinstance(settings.get("mail_types"), str):
            settings["mail_types"] = [
                v.strip().upper() for v in settings["mail_types"].split(",") if v.strip()
            ]
            touch(draft)
        left, right = st.columns(2)
        with left:
            field(
                draft,
                path + ["mail_types"],
                "Notify me when",
                [],
                multiple=True,
                options=[
                    "END",
                    "FAIL",
                    "BEGIN",
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
                ],
            )
        with right:
            field(draft, path + ["mail_user"], "Mail address (optional)", None, optional=True)
    with st.expander("Advanced scheduler and runtime settings"):
        for column, (key, label) in zip(
            st.columns(3),
            (("nodes", "Nodes"), ("ntasks", "Total tasks"), ("ntasks_per_node", "Tasks per node")),
        ):
            with column:
                field(draft, path + [key], label, 1)
        field(
            draft, path + ["gpu_directive"], "GPU request syntax", "gpus", options=["gres", "gpus"]
        )
        field(
            draft,
            ["dependency"],
            "Start after successful Slurm job ID (optional)",
            "",
            help="Add an afterok dependency: this workflow starts only when the specified Slurm job succeeds.",
        )
        field(
            draft,
            path + ["python_executable"],
            "Worker Python interpreter (optional)",
            None,
            optional=True,
        )
        field(draft, path + ["setup"], "Module / environment setup", [])
        field(draft, path + ["cpu_thread_env"], "Bind OMP/MKL threads to CPUs per task", False)
        field(
            draft,
            path + ["local_runtime_directories"],
            "Run-local cache directories",
            [],
            options=["wandb", "matplotlib"],
            multiple=True,
        )
        field(
            draft,
            path + ["log_job_context"],
            "Print job ID, host, working directory, and command",
            False,
        )
        advanced_fields(
            draft,
            path,
            {
                "partition",
                "time",
                "memory",
                "cpus",
                "gpus",
                "account",
                "qos",
                "nodes",
                "ntasks",
                "ntasks_per_node",
                "gpu_directive",
                "mail_types",
                "mail_user",
                "python_executable",
                "setup",
                "cpu_thread_env",
                "local_runtime_directories",
                "log_job_context",
            },
        )


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
    _execution_fields(draft)
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
    actions = [s.get("action") for s in draft["document"]["stages"]]
    steps = tuple(
        step
        for step in STEPS
        if step in {"Start", "Review & run"}
        or (step == "Prepare data" and "preprocess" in actions)
        or (step == "Train" and "train" in actions)
        or (
            step == "Analyze"
            and any(a in actions for a in ("extract", "analyze", "compare", "export_labels"))
        )
    )
    current = st.session_state.get("workflow_step", "Start")
    if current not in steps:
        current = st.session_state["workflow_step"] = steps[1]
    if st.session_state.get("step_widget") != current:
        st.session_state["step_widget"] = current

    def step_changed():
        st.session_state["workflow_step"] = st.session_state["step_widget"]

    step = st.radio(
        "Workflow steps", steps, horizontal=True, key="step_widget", on_change=step_changed
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
            "Analyze": {"extract", "analyze", "compare", "export_labels"},
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
                "export_labels": _export_labels,
            }[action](draft, index)
        _stage_options(draft, step)
    yaml_editor(draft)
    previous, following = st.columns(2)
    position = steps.index(step)
    if position > 0 and previous.button("Back to " + steps[position - 1]):
        st.session_state["workflow_step"] = steps[position - 1]
        rerun()
    if position < len(steps) - 1 and following.button("Continue to " + steps[position + 1]):
        st.session_state["workflow_step"] = steps[position + 1]
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
        draft = new_draft(
            "dino" if kind == "dino" else "crops", source="Prepared data: " + str(path)
        )
        data = load_document(path)
        model = draft["document"]["stages"][0]["config"]
        model["data"] = data["data"]
        if kind != "dino":
            model["mae"]["input_shape"] = data["mae"]["input_shape"]
        step = "Analyze" if kind == "dino" else "Train"
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
        if path.name == "mae_config.yaml" and st.button(
            "Extract DINO features from these prepared objects"
        ):
            try:
                continue_from_artifact(config, path, kind="dino")
            except Exception as error:
                st.error(str(error))
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
