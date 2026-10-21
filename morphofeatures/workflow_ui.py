"""Guided workflow editing and a single review/submit surface."""

from __future__ import annotations

import json
import socket
from contextlib import contextmanager
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
from morphofeatures.data.remote_n5 import is_remote_url
from morphofeatures.registry import JobRegistry
from morphofeatures.slurm import (
    SlurmScheduler,
    cluster_profiles_from_mapping,
    load_cluster_profiles,
)
from morphofeatures.ui_settings import (
    WORKFLOW_REQUIREMENTS,
    setting_help,
    workflow_requirements,
)
from morphofeatures.workflow_validation import (
    WorkflowIssue,
    check_workflow,
    execution_issue,
    stage_issue,
    stage_page,
)
from morphofeatures.workflows import format_command
from morphofeatures.workspace_jobs import (
    embedding_names,
    plan_workspace_job,
    resolve_job,
    submission_fingerprint,
    submit_workspace_plan,
)
from morphofeatures.workspace_state import (
    STAGE_LABELS,
    analysis_defaults,
    change_summary,
    import_workflow,
    load_draft,
    new_draft,
    save_draft,
    stages_after_preparation,
    workflow_document,
)
from morphofeatures.workspace_ui import retained_input

STEPS = ("Start", "Prepare data", "Train", "Extract", "Build features", "Learn", "Analyze", "Export", "Review & run")


def rerun():
    (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()


def navigate(page):
    st.session_state["navigate_to"] = page
    rerun()


def draft_key(config):
    return "workflow:" + str(config.paths.output_root)


def active_draft(config):
    draft = st.session_state.get(draft_key(config))
    if draft:
        st.session_state.setdefault("drafts:" + str(config.paths.output_root), {})[draft["id"]] = draft
        st.session_state["workflow_active_id"] = draft["id"]
    return draft


def use_draft(config, draft, step="Start", *, resume=False):
    from morphofeatures.workflow_catalog import entry_page, retain_draft

    # Older callers used Analyze for every post-training operation.
    if step == "Analyze" and not resume:
        step = entry_page(draft["document"]["stages"])
    previous = st.session_state.get(draft_key(config))
    if previous:
        previous["editor_epoch"] = st.session_state.get("editor_epoch", 0)
    retain_draft(st.session_state, config.paths.output_root, draft, step)
    st.session_state["editor_epoch"] = draft.get("editor_epoch", 0)


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


def _local_stage_error(draft, index, error, area):
    issue = stage_issue(error, index, draft["document"]["stages"][index]).to_dict()
    issue["message"] = area + ": " + issue["message"]
    _show_issues(draft, [issue], context="local", navigate_to_fields=False)


def _check_key(draft):
    return "workflow-checks:" + draft["id"]


def _clear_checks(draft):
    st.session_state.pop(_check_key(draft), None)


def _field_path(path):
    return tuple(path[1:] if path and path[0] == "document" else path)


def _field_locations(draft):
    return st.session_state.setdefault("workflow-fields:" + draft["id"], {})


def _stored_issues(draft):
    return st.session_state.get(_check_key(draft), [])


def _field_matches(path, target):
    path, target = tuple(path), tuple(target)
    return path == target or (
        path[: len(target)] == target and all(isinstance(v, int) for v in path[len(target) :])
    )


def _issue_location(draft, issue):
    parts = [issue["page"]]
    if issue.get("stage_index") is not None:
        index = issue["stage_index"]
        stages = draft["document"]["stages"]
        name = _stage_title(stages, index) if index < len(stages) else issue["stage_name"]
        parts.append(f"Stage {index + 1}: {name}")
    registry = _field_locations(draft)
    labels, sections = [], []
    for path in issue.get("fields", []):
        path = tuple(path)
        location = registry.get(path)
        if location:
            if location["section"] and location["section"] not in sections:
                sections.append(location["section"])
            labels.append(location["label"])
        else:
            tail = path[2:] if path[:1] == ("stages",) else path
            tail = tuple(p for p in tail if p != "config")
            labels.append(" / ".join(str(p).replace("_", " ").capitalize() for p in tail))
    parts.extend(sections)
    parts.extend(dict.fromkeys(label for label in labels if label))
    return " → ".join(parts)


def _show_issues(draft, issues, *, context, navigate_to_fields=True):
    for index, issue in enumerate(issues):
        message = _issue_location(draft, issue) + ": " + issue["message"]
        getattr(st, issue.get("severity", "error"))(message)
        stage_key = "workflow-stage:" + draft["id"] + ":" + issue["page"]
        if navigate_to_fields and (issue["page"] != st.session_state.get("workflow_step") or
                                  issue.get("stage_index") is not None and st.session_state.get(stage_key) != issue["stage_index"]):
            if st.button("Go to " + issue["page"], key=f"{context}:{draft['id']}:{index}"):
                st.session_state["workflow_step"] = issue["page"]
                if issue.get("stage_index") is not None:
                    st.session_state[stage_key] = issue["stage_index"]
                rerun()


@contextmanager
def validation_expander(draft, label, *, expanded=False, **kwargs):
    # Remember actual section labels, rather than maintaining a second form layout.
    previous = st.session_state.get("workflow-section", "")
    section = " → ".join(p for p in (previous, label) if p)
    if draft:
        locations = _field_locations(draft)
        expanded = expanded or any(
            locations.get(tuple(path), {}).get("section", "").startswith(section)
            for issue in _stored_issues(draft)
            for path in issue.get("fields", [])
        )
    with st.expander(label, expanded=expanded, **kwargs):
        st.session_state["workflow-section"] = section
        try:
            yield
        finally:
            st.session_state["workflow-section"] = previous


def _form_issues(draft):
    issues = [v for k, v in st.session_state.get("guided_errors", {}).items() if draft["id"] in k]
    issues = [
        v if isinstance(v, dict) else WorkflowIssue(str(v), "Review & run").to_dict()
        for v in issues
    ]
    editor = draft.get("yaml_editor", {})
    buffer = st.session_state.get("pipeline_buffer:" + draft["id"], editor.get("buffer"))
    if (
        st.session_state.get("yaml_enabled:" + draft["id"], editor.get("enabled", False))
        and buffer is not None
        and buffer != yaml.safe_dump(draft["document"], sort_keys=False)
    ):
        issues.append(
            WorkflowIssue(
                "Advanced: full pipeline YAML → Pipeline YAML: apply your YAML edits or reload it from the form before checking or continuing.",
                st.session_state.get("workflow_step", "Review & run"),
            ).to_dict()
        )
    return issues


def _check_page(draft, step):
    document = workflow_document(draft)
    indices = {i for i, s in enumerate(document["stages"]) if stage_page(s.get("action")) == step}
    issues = [v for v in _form_issues(draft) if v["page"] == step]
    if step == "Prepare data":
        from morphofeatures.dino_experiments import resolve_paths, validate_experiment
        from morphofeatures.workflow_validation import stage_issue

        for i, stage in enumerate(document["stages"]):
            if stage.get("action") == "dino_experiment":
                try:
                    validate_experiment(resolve_paths(stage, repository_root()), sources_only=True)
                except Exception as error:
                    issue = stage_issue(error, i, stage).to_dict()
                    issue["page"] = step
                    issues.append(issue)
    if not issues:
        try:
            issues = [
                i.to_dict()
                for i in check_workflow(
                    document,
                    repository_root(),
                    indices=indices,
                    allow_synthetic=draft["allow_synthetic"],
                )
            ]
        except Exception as error:
            issues = [WorkflowIssue(str(error), step).to_dict()]
    retained = [i for i in _stored_issues(draft) if i["page"] != step]
    st.session_state[_check_key(draft)] = retained + issues
    st.session_state["page-check:" + draft["id"]] = (step, draft["revision"], not issues)
    return not any(i["severity"] == "error" for i in issues)


def field(
    draft,
    path,
    label,
    default=None,
    *,
    options=None,
    help=None,
    optional=False,
    multiple=False,
    format_func=None,
):
    """Widgets mirror the draft; their cleanup on navigation cannot delete settings."""
    location_path = _field_path(path)
    _field_locations(draft)[location_path] = {
        "label": label,
        "section": st.session_state.get("workflow-section", ""),
    }
    missing = object()
    value = _get(draft, path, missing)
    if value is missing:
        value = deepcopy(default)
        _put(draft, path, value)
        touch(draft)
    key = f"guided:{draft['id']}:{st.session_state.get('editor_epoch', 0)}:" + ".".join(
        map(str, path)
    )
    if format_func is not None:
        key += ":choices"
    optional = optional and (value is None or isinstance(value, str))
    serialized = not isinstance(value, (str, int, float, bool)) and not optional and options is None
    initial = (
        yaml.safe_dump(value, default_flow_style=True).replace("\n...\n", "").strip()
        if serialized
        else ("" if optional and value is None else value)
    )
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool) and options is None
    if key not in st.session_state and not numeric:
        # Invalid serialized edits are not in the draft yet. Keep their text
        # when Streamlit removes widgets on navigation, so users can correct it.
        invalid = st.session_state.get("guided_errors", {}).get(key, {})
        st.session_state[key] = (
            invalid.get("input_value", initial) if isinstance(invalid, dict) else initial
        )

    def changed():
        replacement = st.session_state[key]
        if serialized:
            try:
                replacement = yaml.safe_load(replacement)
            except yaml.YAMLError:
                if location_path[:1] == ("stages",):
                    i = location_path[1]
                    stage = draft["document"]["stages"][i]
                    issue = stage_issue(ValueError("Enter valid YAML for " + label + "."), i, stage)
                    issue = WorkflowIssue(
                        issue.message, issue.page, i, issue.stage_name, (location_path,)
                    )
                else:
                    issue = WorkflowIssue(
                        "Enter valid YAML for " + label + ".",
                        "Review & run",
                        fields=(location_path,),
                    )
                st.session_state.setdefault("guided_errors", {})[key] = {
                    **issue.to_dict(),
                    "input_value": replacement,
                }
                _clear_checks(draft)
                touch(draft)
                return
        st.session_state.setdefault("guided_errors", {}).pop(key, None)
        _put(draft, path, None if optional and replacement == "" else replacement)
        _clear_checks(draft)
        touch(draft)

    if help is None:
        action = None
        if location_path[:1] == ("stages",) and len(location_path) > 1:
            action = draft["document"]["stages"][location_path[1]].get("action")
        help = setting_help(path, action=action)
        if path[-1] == "normalization" and action == "preprocess":
            help = "Crop intensity scaling: dtype maps the integer data range to [0,1] (float input must already be in [0,1]); percentile maps the object's 1st–99th intensity percentiles to [0,1]; none preserves intensities."
    kwargs = {"key": key, "on_change": changed, "help": help}
    if multiple:
        choices = list(dict.fromkeys([*(options or []), *(value or [])]))
        st.multiselect(
            label, choices, **({"format_func": format_func} if format_func else {}), **kwargs
        )
    elif options is not None:
        choices = list(options)
        if value not in choices:
            choices.insert(0, value)
        st.selectbox(
            label, choices, **({"format_func": format_func} if format_func else {}), **kwargs
        )
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
        _show_issues(draft, [error], context=key, navigate_to_fields=False)
    for issue in _stored_issues(draft):
        if any(_field_matches(location_path, target) for target in issue.get("fields", [])):
            getattr(st, issue.get("severity", "error"))(label + ": " + issue["message"])
    return _get(draft, path)


def triple_field(draft, path, label, default, *, help_text=None):
    value = _get(draft, path, default)
    if not isinstance(value, list) or len(value) != 3:
        return field(
            draft, path, label, default, help=help_text or "Three values in Z, Y, X order."
        )
    if _get(draft, path) is None:
        _put(draft, path, deepcopy(default))
    st.caption(label + " · Z, Y, X")
    for axis, column in enumerate(st.columns(3)):
        with column:
            field(draft, path + [axis], f"{label} {'ZYX'[axis]}", value[axis], help=help_text)


def advanced_fields(draft, path, excluded=()):
    values = _get(draft, path, {})
    if not isinstance(values, dict):
        field(draft, path, "Additional settings", values)
        return
    for key, value in list(values.items()):
        if key in excluded:
            continue
        if isinstance(value, dict) and value:
            with validation_expander(draft, key.replace("_", " ").capitalize()):
                advanced_fields(draft, path + [key])
        else:
            field(
                draft,
                path + [key],
                key.replace("_", " ").capitalize(),
                value,
            )


def _replace_document(draft, document):
    _clear_checks(draft)
    _field_locations(draft).clear()
    draft["document"] = deepcopy(document)
    touch(draft)
    st.session_state["editor_epoch"] = st.session_state.get("editor_epoch", 0) + 1
    st.session_state.pop("workspace_plan", None)
    st.session_state["guided_errors"] = {k: v for k, v in st.session_state.get("guided_errors", {}).items() if draft["id"] not in k}


def yaml_editor(draft):
    editor = draft.setdefault("yaml_editor", {})
    enabled_key = "yaml_enabled:" + draft["id"]
    st.session_state.setdefault(enabled_key, editor.get("enabled", False))

    def remember_enabled():
        editor["enabled"] = st.session_state[enabled_key]

    with validation_expander(draft, "Advanced: full pipeline YAML"):
        st.caption(
            "All stages and additional settings are preserved. Relative paths use the repository directory."
        )
        if not st.checkbox("Edit pipeline YAML", key=enabled_key, on_change=remember_enabled):
            st.code(yaml.safe_dump(draft["document"], sort_keys=False), language="yaml")
            return
        key = "pipeline_buffer:" + draft["id"]
        revision_key = key + ":revision"
        if key not in st.session_state:
            st.session_state[key] = editor.get("buffer", yaml.safe_dump(draft["document"], sort_keys=False))
            st.session_state[revision_key] = editor.get("revision", draft["revision"])

        def remember_buffer():
            editor.update(buffer=st.session_state[key], revision=st.session_state[revision_key])

        def reload_yaml():
            st.session_state[key] = yaml.safe_dump(draft["document"], sort_keys=False)
            st.session_state[revision_key] = draft["revision"]
            remember_buffer()

        st.button("Reload YAML from current form settings", on_click=reload_yaml)
        edited = st.text_area("Pipeline YAML", key=key, height=320, on_change=remember_buffer)
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
                remember_buffer()
                rerun()
            except Exception as error:
                st.error(str(error))


def _preparation_checklist(items):
    if items:
        st.markdown(
            "**What you should have for this workflow**\n\n"
            + "\n".join("- " + item for item in items)
        )


def _start(config, draft):
    from morphofeatures.workflow_catalog import entry_page
    from morphofeatures.workflow_selection_ui import selection

    start, proposed = selection(config)
    if start is None:
        return
    description = {
            "published": "Explore bundled original features, preserve their published UMAP/clusters, and optionally evaluate labels or bilateral partners.",
            "bilateral": "Compare MAE, DINO and published features on the same cell IDs using candidate-partner retrieval ranks. No training is required.",
            "context": "Combine each object's own embedding with its physical neighbors' mean → analyze or compare the resulting context vectors.",
            "raw": "Prepare crops → train a model → extract embeddings → analyze.",
            "platybrowser": "Select published raw and segmentation sources → stream object crops → train MAE or embed with MAE/DINO → analyze and compare.",
            "ome_zarr": "Connect raw and/or segmentation OME-Zarr layers → select resolutions and preview. With both layers ready, prepare an ROI and run MAE/DINO and analysis.",
            "preprocess": "Prepare object crops, IDs and masks only. You can train or extract DINO features from these saved outputs later.",
            "crops": "Use your prepared data → train → extract embeddings → analyze.",
            "dino": "Use pretrained DINOv2 or DINOv3 → extract object features → visualize and optionally evaluate known classes. No MAE training is needed.",
            "dino_experiment": "Select physically registered raw/instance volumes → compare fine, coarse and combined DINO vectors → joint maps and optional label transfer.",
            "fusion": "Align saved fine/coarse vectors by object ID → concatenate with explicit weights → project and inspect the combined representation.",
            "distill": "Reuse saved vectors → reserve validation objects → compare original, PCA and supervised adapter embeddings, with optional bilateral retrieval.",
            "transfer": "Train or load a labeled reference classifier → predict independent datasets → compare uncorrected and CORAL-adapted results.",
            "checkpoint": "Use a matching model and data configuration → extract → analyze.",
            "embeddings": "Analyze one saved embedding or compare several representations.",
            "joint_embeddings": "Pool independent datasets encoded by the same checkpoint → fit one PCA/UMAP and clustering → compare dataset and cell-type distributions. Accepts saved embeddings or new extractions.",
            "demo": "Train on eight generated synthetic crops, then extract and analyze. A short CPU demonstration.",
        }[start]
    if proposed:
        st.caption("The plan below starts with your available inputs and includes the operations you selected.")
    else:
        st.write(description)
    _preparation_checklist(workflow_requirements(proposed["document"]["stages"]) if proposed else WORKFLOW_REQUIREMENTS[start])
    from morphofeatures.workflow_outputs import show_expected_outputs

    if proposed:
        st.markdown("**Proposed plan:** " + " → ".join(STAGE_LABELS.get(s["action"], s["action"]) for s in proposed["document"]["stages"]))
        show_expected_outputs(proposed["document"]["stages"])
    else:
        show_expected_outputs(start=start)
    if draft:
        st.caption(
            "Your current draft stays available in this session. Save drafts to retain them after restarting the app."
        )
    if st.button("Create workflow" if not draft else "Create a new workflow", type="primary"):
        created = proposed or new_draft(start, model_defaults=st.session_state.get("workspace_dino_defaults"))
        use_draft(config, created, entry_page(created["document"]["stages"]))
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
            with validation_expander(draft, f"Settings that will be replaced ({len(changes)})"):
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
                "Prepare data"
                if any(s["action"] in {"preprocess", "dino_experiment"} for s in loaded["document"]["stages"])
                else "Train"
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
                    step = entry_page(saved["document"]["stages"])
                    use_draft(config, saved, step)
                    rerun()
                except Exception as error:
                    st.error(str(error))


def _load_stage_config(draft, index):
    stage = draft["document"]["stages"][index]
    dino = stage.get("action") == "extract" and stage.get("model") in {"dinov2", "dinov3"}
    with validation_expander(draft, "Load model / data settings", expanded=dino):
        if dino:
            st.info(
                "Load mae_config.yaml (single crops) or data_config.yaml (distributed patches) from Prepare data to fill in paths and HDF5/N5 "
                "dataset keys. Already using grouped N5 patches for MAE? Load that same "
                "MAE YAML here to supply the patch container, positions/ID index and sampling. "
                "Your DINO weights and backbone are kept. Distributed patches set stored views, unit normalization and letterbox resizing to match their saved data."
            )
        source = retained_input(
            "Model / data YAML path",
            "" if dino else "configs/smoke.yaml",
            f"stage:{draft['id']}:{index}:source",
            help="Load the generated mae_config.yaml for single crops or data_config.yaml for distributed patches. For the grouped Platynereis N5 inputs, use configs/sites/mae_platynereis_nuclei_embl.yaml. This field expects a YAML file."
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
                if dino and config.get("data", {}).get("source") == "grouped_crops":
                    document["stages"][index].setdefault("views", {}).update(
                        mode="stored", normalization="unit", resize="letterbox"
                    )
                _replace_document(draft, document)
                resolved_profile = config.get("resolved_profile", profile)
                draft.setdefault("stage_sources", {})[str(index)] = str(path) + (
                    " · Profile: " + resolved_profile if resolved_profile else ""
                )
                rerun()
            except Exception as error:
                _local_stage_error(
                    draft,
                    index,
                    error,
                    "Load model / data settings → Model / data YAML path and profile",
                )
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
    if data.get("source") == "grouped_crops":
        st.info(
            "Input: saved distributed fine patches. Each object has several masked 3D patches, pooled into one DINO vector. Load data_config.yaml to keep patch IDs, offsets and physical centers together."
        )
        st.caption(
            "Per-object contrast was fixed during preparation; use unit intensity normalization. Single-crop MAE training cannot consume this layout."
        )
        st.json({k: v for k, v in data.items() if k not in {"volume"}}, expanded=False)
    elif data.get("source") == "n5_masked_patches":
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
                f"{stage_page(stage['action'])} → Stage {index + 1}: {_stage_title(draft['document']['stages'], index)} → Crop dataset key: "
                "N5 crops need a dataset key: load the mae_config.yaml produced by Prepare data. "
                "For grouped N5 patches, load their MAE YAML instead; the positions/ID index "
                "is needed alongside the patches. No additional preprocessing is required."
            )


def _platybrowser_preparation_source(draft, index):
    from morphofeatures.analysis.platybrowser import (
        DEFAULTS,
        list_platybrowser_sources,
        resolve_platybrowser_preprocessing,
    )

    path = ["document", "stages", index, "platybrowser"]
    stage = draft["document"]["stages"][index]
    key = f"platy-workflow:{draft['id']}:{index}"
    st.caption(
        "Connect reads published metadata and object tables. Crops stream when the job runs; the optional preview reads a bounded slice."
    )
    field(
        draft,
        path + ["project_url"],
        "PlatyBrowser project URL",
        DEFAULTS["project_url"],
        help="GitHub repository containing the published MoBIE project metadata.",
    )
    field(
        draft,
        path + ["dataset"],
        "Published dataset version",
        "1.0.1",
        help="Dataset folder inside the project's data directory, for example 1.0.1.",
    )
    field(
        draft,
        path + ["revision"],
        "Metadata revision",
        DEFAULTS["revision"],
        help="Commit, tag or branch. Connecting pins this to a commit for the saved workflow.",
    )
    with validation_expander(draft, "Streaming cache and request limits"):
        for name, label, default, help_text in (
            (
                "cache_directory",
                "Shared download cache directory",
                str(repository_root() / "outputs/remote_cache/platybrowser"),
                "Use a directory accessible to the UI and Slurm workers. Prepared crops are saved separately in the run directory.",
            ),
            (
                "cache_size_mb",
                "Download cache size (MiB)",
                1024,
                "Persistent image chunks are reused and evicted as the cache fills. This does not limit the size of the prepared crop dataset.",
            ),
            (
                "timeout_seconds",
                "Request timeout (seconds)",
                20,
                "Maximum wait for each remote HTTP request.",
            ),
            (
                "cache_version",
                "Cache version",
                "1",
                "Change to refresh hosted image data if it has changed in place.",
            ),
        ):
            field(draft, path + [name], label, default, help=help_text)
    options = stage["platybrowser"]

    def catalog_signature():
        return json.dumps(
            {
                k: options.get(k)
                for k in ("project_url", "dataset", "revision", "cache_directory", "cache_version")
            },
            sort_keys=True,
        )

    if st.button("Load available PlatyBrowser datasets", key=key + ":catalog"):
        try:
            catalog = list_platybrowser_sources(options)
            options["revision"] = catalog["options"]["revision"]
            st.session_state[key + ":catalog-data"] = (catalog_signature(), catalog)
            _replace_document(draft, draft["document"])
            rerun()
        except (ValueError, OSError, KeyError) as error:
            _local_stage_error(
                draft, index, error, "Published source selection → Connect Workflow to PlatyBrowser"
            )
    stored = st.session_state.get(key + ":catalog-data")
    catalog = stored[1] if stored and stored[0] == catalog_signature() else {}
    field(
        draft,
        path + ["raw_source"],
        "Published raw image source",
        "raw",
        options=catalog.get("image"),
        help="Select an intensity image from the catalog, or enter its source name (for example raw).",
    )
    field(
        draft,
        path + ["segmentation_source"],
        "Published instance segmentation source",
        "nuclei",
        options=catalog.get("segmentation"),
        help="Select nuclei, cells or another instance source with a published object-bounds table. Tissue-region segmentations produce one crop per region, not per cell.",
    )
    for kind in ("raw", "segmentation"):
        field(
            draft,
            path + [kind + "_level"],
            kind.capitalize() + " resolution level",
            DEFAULTS[kind + "_level"],
            help="Larger numbers are coarser. Distributed patches can use different raw/segmentation resolutions in the same physical frame; choose fine enough levels to preserve texture and mask boundaries. Single crops require matching grids (raw 3 / nuclei 0 in dataset 1.0.1).",
        )
    if stage.get("layout") != "distributed_patches":
        field(
            draft,
            ["document", "stages", index, "read_budget_side"],
            "Maximum streamed object side (voxels)",
            256,
            help="Per-object read limit, 8–512 voxels along each axis. Objects exceeding this limit are reported as failed. Choose coarser aligned levels for large cells.",
        )
    signature = json.dumps(options, sort_keys=True)
    if st.button("Connect Workflow to PlatyBrowser", key=key + ":connect", type="primary"):
        try:
            with st.spinner("Resolving published sources and checking alignment…"):
                if stage.get("layout") == "distributed_patches":
                    from morphofeatures.data.dino_views import resolve_source

                    resolved = resolve_source(stage)
                else:
                    resolved = resolve_platybrowser_preprocessing(stage)
            document = deepcopy(draft["document"])
            document["stages"][index] = resolved
            st.session_state[key + ":connected"] = json.dumps(
                resolved["platybrowser"], sort_keys=True
            )
            _replace_document(draft, document)
            rerun()
        except (ValueError, OSError, KeyError) as error:
            _local_stage_error(
                draft, index, error, "Published source selection → Connect Workflow to PlatyBrowser"
            )
    ready = st.session_state.get(key + ":connected") == signature
    if ready:
        st.success(
            "Connected: published object bounds will locate crops without a whole-volume scan. Distributed patches use physical coordinates; single crops require aligned grids."
        )
        st.caption(
            f"Raw: {stage['raw']} / {stage['raw_key']} · Segmentation: {stage['segmentation']} / {stage['segmentation_key']}"
        )
    else:
        st.info(
            "Connect to check this source pair before previewing. Review also validates and pins the selected sources."
        )
    st.caption(
        "Prepared crops and embeddings use stored segmentation IDs (nucleus IDs for nuclei). Cell-ID annotations from grouped patch datasets need the published cell-to-nucleus mapping first."
    )
    return ready


def _ome_zarr_preparation_source(draft, index):
    from morphofeatures.data.ome_zarr import discover_ome_zarr, resolve_ome_zarr_preprocessing

    stage = draft["document"]["stages"][index]
    path = ["document", "stages", index]
    key = f"ome-workflow:{draft['id']}:{index}"
    st.caption(
        "Paste an OME-Zarr image-group URL for either layer. Metadata lists the resolutions; previews stream one bounded slice. Add both aligned layers when ready to prepare objects."
    )
    with validation_expander(draft, "OME-Zarr cache and request limits"):
        for name, label, default, help_text in (
            (
                "cache_directory",
                "OME-Zarr download cache directory",
                str(repository_root() / "outputs/remote_cache/ome_zarr"),
                "Use a directory accessible to the UI and Slurm worker. Old chunks are evicted when the cache fills.",
            ),
            (
                "cache_size_mb",
                "OME-Zarr cache size (MiB)",
                1024,
                "Limits persistent downloads; prepared crops are stored separately.",
            ),
            (
                "timeout_seconds",
                "OME-Zarr request timeout (seconds)",
                20,
                "Maximum wait for each metadata or chunk request.",
            ),
            (
                "cache_version",
                "OME-Zarr cache version",
                "1",
                "Change after the hosted image or unfinished segmentation is updated to fetch fresh data.",
            ),
        ):
            field(draft, path + ["remote_options", name], label, default, help=help_text)
    options = stage["remote_options"]
    options["format"] = "ome_zarr"
    for kind in ("raw", "segmentation"):
        source_path = path + ["ome_zarr", kind]
        with validation_expander(draft, kind.capitalize() + " OME-Zarr source", expanded=True):
            field(
                draft,
                source_path + ["url"],
                kind.capitalize() + " OME-Zarr URL",
                None,
                optional=True,
                help="Paste the image-group/layer link containing multiscales metadata, rather than an individual chunk. Leave blank to preview only the other layer.",
            )
            field(
                draft,
                source_path + ["image_index"],
                kind.capitalize() + " image index",
                0,
                help="Zero selects the first multiscale image in this group. Most image-layer URLs contain only one.",
            )
            source = stage["ome_zarr"][kind]
            signature = json.dumps(
                [source.get("url"), source.get("image_index"), options], sort_keys=True
            )
            if st.button(
                "Find " + kind + " resolutions",
                key=key + ":" + kind + ":discover",
                disabled=not source.get("url"),
            ):
                try:
                    metadata = discover_ome_zarr(
                        source["url"], options, image_index=source.get("image_index", 0)
                    )
                    st.session_state[key + ":" + kind + ":catalog"] = (signature, metadata)
                except (ValueError, OSError, KeyError) as error:
                    _local_stage_error(
                        draft,
                        index,
                        error,
                        "OME-Zarr source selection → Connect / Find resolutions",
                    )
            stored = st.session_state.get(key + ":" + kind + ":catalog")
            metadata = stored[1] if stored and stored[0] == signature else None
            if metadata:
                labels = {
                    v[
                        "level"
                    ]: f"{v['level']}: {v['key']} · {v['spacing_zyx']} {metadata['unit']} Z,Y,X"
                    for v in metadata["levels"]
                }
                field(
                    draft,
                    source_path + ["level"],
                    kind.capitalize() + " OME-Zarr resolution",
                    0,
                    options=list(labels),
                    format_func=labels.get,
                    help="Choose a listed resolution. The label shows the stored array path and physical voxel spacing.",
                )
                st.caption(
                    f"Stored axes: {metadata['axes']}; {metadata['image_count']} multiscale image(s). Display and ROI coordinates use Z,Y,X."
                )
            else:
                field(
                    draft,
                    source_path + ["level"],
                    kind.capitalize() + " OME-Zarr resolution",
                    0,
                    help="Zero-based index in the metadata's resolution list, not a magnification factor. Use Find resolutions to see the available levels.",
                )
            field(
                draft,
                source_path + ["channel"],
                kind.capitalize() + " OME-Zarr channel",
                0,
                help="Zero-based channel index, or 0 for images without a channel axis.",
            )
            field(
                draft,
                source_path + ["timepoint"],
                kind.capitalize() + " OME-Zarr timepoint",
                0,
                help="Zero-based time index, or 0 for images without a time axis.",
            )
    signature = json.dumps([stage["ome_zarr"], options], sort_keys=True)
    if st.button("Connect OME-Zarr sources", key=key + ":connect", type="primary"):
        try:
            with st.spinner("Reading selected array metadata…"):
                if stage.get("layout") == "distributed_patches" and all(
                    stage.get("ome_zarr", {}).get(kind, {}).get("url")
                    for kind in ("raw", "segmentation")
                ):
                    from morphofeatures.data.dino_views import resolve_source

                    resolved = resolve_source(stage)
                else:
                    resolved = resolve_ome_zarr_preprocessing(stage, require_pair=False)
            st.session_state[key + ":connected"] = json.dumps(
                [resolved["ome_zarr"], resolved.get("remote_options", {})], sort_keys=True
            )
            document = deepcopy(draft["document"])
            document["stages"][index] = resolved
            _replace_document(draft, document)
            rerun()
        except (ValueError, OSError, KeyError) as error:
            _local_stage_error(
                draft, index, error, "OME-Zarr source selection → Connect / Find resolutions"
            )
    ready = st.session_state.get(key + ":connected") == signature
    if ready:
        st.success(
            "Connected. Inspect dimensions and preview either layer below; no image chunks have been read by connection."
        )
    else:
        st.info(
            "Connect the selected resolutions to enable previews. Changing a URL, level, channel or timepoint invalidates the previous preview."
        )
    with validation_expander(draft, "Remote object preparation limits"):
        if stage.get("layout") != "distributed_patches":
            field(
                draft,
                path + ["read_budget_side"],
                "OME-Zarr maximum crop side (voxels)",
                256,
                help="Maximum crop side for streamed preparation; each read also has a chunk and byte limit.",
            )
        field(
            draft,
            path + ["max_scan_voxels"],
            "Maximum segmentation ROI voxels",
            128**3,
            help="Generic sources lack PlatyBrowser's object catalog. Preparation scans only an explicit ROI, capped here (at most 256³ voxels). Object count limits extraction after this scan.",
        )
        st.caption(
            "Previews work with either layer alone. Preparation requires both sources and an explicit segmentation ROI. Single crops need matching grids; distributed patches resample using physical metadata."
        )
    return ready


def _after_preparation_controls(draft, index):
    distributed = draft["document"]["stages"][index].get("layout") == "distributed_patches"
    with validation_expander(
        draft,
        "Choose training and embedding stages",
        expanded=len(draft["document"]["stages"]) == 1,
    ):
        st.caption(
            "Prepare crops once and reuse them for each model. Applying these choices replaces the later stages; Undo last stage change restores them."
        )
        key = f"after-prepare:{draft['id']}:{index}"
        mae = st.selectbox(
            "MAE workflow",
            ["none"] if distributed else ["none", "train", "checkpoint"],
            format_func={
                "none": "No MAE",
                "train": "Train a new MAE and embed",
                "checkpoint": "Embed with an existing MAE",
            }.get,
            key=key + ":mae",
        )
        if distributed:
            st.caption(
                "Distributed patch outputs currently support DINO. For crop MAE training, choose Single object crop above. Existing grouped N5 MAE workflows keep their original layout."
            )
        dino = st.checkbox(
            "Include DINO embeddings",
            key=key + ":dino",
            help="Use frozen DINOv2 or DINOv3 weights. Configure the variant, local weights and backbone checkout under Analyze → Extract embeddings.",
        )
        compare = st.checkbox(
            "Compare MAE and DINO representations",
            value=True,
            disabled=mae == "none" or not dino,
            key=key + ":compare",
        )
        if mae == "checkpoint":
            st.info(
                "Use a checkpoint trained on whole-object crops and load its matching architecture configuration. Grouped N5 patch checkpoints require their original input layout."
            )
        if st.button("Apply training and embedding choices", key=key + ":apply"):
            draft.setdefault("stage_history", []).append(deepcopy(draft["document"]))
            preparation = draft["document"]["stages"][index]
            later = stages_after_preparation(preparation, mae=mae, dino=dino, compare=compare, model_defaults=st.session_state.get("workspace_dino_defaults"))
            _replace_document(
                draft, {**draft["document"], "stages": draft["document"]["stages"][:index] + later}
            )
            rerun()


def _distributed_patch_controls(draft, index):
    from morphofeatures.data.distributed_patches import PATCH_DEFAULTS

    path = ["document", "stages", index]
    native = _get(draft, path + ["patches", "sampling"]) == "native_voxels"
    st.markdown("**Distributed fine patches**")
    controls = (
        (
            "tile_size_um",
            "Fine patch width (µm)",
            "Physical side length of each cubic patch. 1 µm is an experimental starting point, not a universal optimum. Use the same width across datasets for a matched field of view; 32³ saved voxels at 1 µm means 31.25 nm per voxel.",
        ),
        (
            "min_foreground_fraction",
            "Minimum object content per patch (0–1)",
            "Fraction of the entire 3D patch occupied by this object's segmentation. 0.25 means at least a quarter, 0.5 half, and 1.0 fully inside. Higher cutoffs favor interior texture and can exclude small objects; this is not a cutoff for each 2D DINO slice.",
        ),
        (
            "tile_anchor",
            "Patch grid placement",
            "centroid centers one tile on the mean foreground position. bbox_center centers one on the bounding-box midpoint. bbox_min starts the grid at the lowest bounding-box corner. All extend a non-overlapping lattice through the object; the cap selects spatially separated eligible tiles.",
        ),
        (
            "max_tiles",
            "Maximum fine patches per object",
            "Upper limit on saved patches for each object, not the GPU batch size. Selection starts near the chosen center, then repeatedly chooses the most distant remaining tile. Small objects may have fewer eligible patches. 64 is the default; use 200 to compare a larger sampling budget.",
        ),
        (
            "exhaustive",
            "Save every eligible fine patch",
            "Ignore the patch-count cap and keep all tiles passing the foreground cutoff. This can substantially increase storage and DINO inference time. Read, object-mask and candidate-grid safety limits still apply.",
        ),
    )
    for name, label, help_text in controls:
        if native and name == "tile_size_um":
            continue
        field(
            draft,
            path + ["patches", name],
            label,
            PATCH_DEFAULTS[name],
            help=help_text,
            **(
                {
                    "options": ["centroid", "bbox_center", "bbox_min"],
                    "format_func": {
                        "centroid": "Foreground centroid",
                        "bbox_center": "Bounding-box center",
                        "bbox_min": "Bounding-box minimum",
                    }.get,
                }
                if name == "tile_anchor"
                else {}
            ),
        )
    with validation_expander(draft, "Fine patch contrast and safety limits"):
        field(
            draft,
            path + ["patches", "percentiles"],
            "Object contrast percentiles",
            [1.0, 99.0],
            help="Estimate these percentiles once from up to 65,536 reproducibly sampled foreground intensities per object. Every patch of that object uses the same scale; pixels outside its mask are zero. DINO must use unit normalization for these already scaled patches.",
        )
        field(
            draft,
            path + ["patches", "max_candidate_tiles"],
            "Maximum candidate tiles per object",
            250000,
            help="Safety limit on the complete lattice before the foreground cutoff. If exceeded, the object is skipped with a reason. Increase physical patch width to reduce the lattice; this is separate from the saved-patch cap.",
        )
        field(
            draft,
            path + ["max_object_voxels"],
            "Maximum object bounding-box voxels",
            256**3,
            help="Limit on the full segmentation box plus a one-voxel halo, including background. Larger objects are reported and skipped, never clipped. Raising this needs additional RAM for labels, masks and coordinates; it does not increase the SLURM memory request. Up to one billion voxels may be configured.",
        )
    stage = draft["document"]["stages"][index]
    width = stage.get("patches", {}).get("tile_size_um", 1.0)
    if native:
        spacing = stage.get("raw_spacing_zyx") or stage.get("spacing_zyx", [1, 1, 1])
        factor = {"nm": 0.001, "um": 1, "micrometer": 1}.get(stage.get("unit"))
        if factor and isinstance(spacing, list) and len(spacing) == 3:
            extent = [s * n * factor for s, n in zip(spacing, stage.get("crop_shape", [32] * 3))]
            st.caption(
                "Native patch field of view Z/Y/X (µm): " + " / ".join(f"{v:.4g}" for v in extent)
            )
        st.info(
            "Native mode cuts raw voxels without spatial interpolation or smoothing. "
            "Contrast normalization and foreground masking still apply. If the segmentation grid differs, "
            "its labels are sampled onto the raw grid with nearest neighbors. "
            "DINO later resizes 2D slices to its input canvas; the saved 3D patches remain native."
        )
    if not native and isinstance(width, (int, float)):
        unit_factor = {"nm": 0.001, "um": 1, "micrometer": 1}.get(stage.get("unit"))
        for kind in ("raw", "segmentation"):
            spacing = stage.get(kind + "_spacing_zyx") or stage.get("spacing_zyx")
            if (
                unit_factor
                and isinstance(spacing, list)
                and len(spacing) == 3
                and all(isinstance(v, (int, float)) and v > 0 for v in spacing)
            ):
                native = [width / (v * unit_factor) for v in spacing]
                st.caption(
                    f"Approximate native {kind} voxels per patch, Z/Y/X: "
                    + " / ".join(f"{v:.1f}" for v in native)
                    + ". Source resolution limits the available detail."
                )
                if kind == "segmentation" and min(native) < 4:
                    st.warning(
                        "This patch spans fewer than four segmentation voxels along at least one axis. Masks may look blocky and foreground estimates may be coarse; preview them or select a finer segmentation level."
                    )
    st.caption(
        "Default DINO handoff uses one center slice per axis: up to 3 × the saved patch count in images per object. Other axes or slice fractions can be chosen under Analyze. Empty masked slices are omitted."
    )
    _distributed_preview_controls(draft, index)


def _distributed_preview_controls(draft, index):
    from morphofeatures.data.patch_preview import start_patch_preview

    path = ["document", "stages", index]
    field(
        draft,
        path + ["object_index"],
        "Existing object bounds table (optional)",
        None,
        help="Reuse objects.tsv from preparation, physical DINO extraction or a previous preview of this exact segmentation grid. Bounds let the preview jump directly to one object. Without a table, Preview builds and caches one in the background; this may take time for a large volume.",
    )
    field(
        draft,
        path + ["data_version"],
        "Source data version",
        "1",
        help="Change this value after replacing segmentation chunks in place to force a new preview lookup. File metadata is checked, but individual volume chunks are not fully hashed. Existing external bounds must also be rebuilt if objects changed.",
    )
    prefix = f"distributed-preview:{draft['id']}:{index}"
    requested = st.text_input(
        "Preview object ID (blank chooses an eligible object)",
        key=prefix + ":object",
        help="Enter an exact label ID to inspect that object only; this does not change the workflow's object selection. An ID alone does not locate its voxels: existing bounds are reused, or the segmentation ROI is scanned once in bounded chunks. No complete raw volume or DINO model is loaded.",
    ).strip()
    stage = draft["document"]["stages"][index]
    signature = json.dumps([stage, requested], sort_keys=True)
    state = st.session_state.get(prefix + ":task")
    if state and state[0] != signature:
        state[1].cancel.set()
        st.session_state.pop(prefix + ":task", None)
        state = None
    st.caption(
        "Preview runs in the background: locate objects once, then read one mask and up to six raw patches. "
        "Large local ROIs are allowed; initial lookup time depends on segmentation size and storage speed. "
        "Remote scans still obey the configured ROI scan budget. Reusing bounds avoids that scan."
    )
    if st.button(
        "Preview distributed fine patches",
        key=prefix,
        help="Build/reuse an object lookup, then show one eligible object and at most six fine patches. Progress and cancellation are available while the UI stays responsive. No training, embedding or SLURM job is started.",
    ):
        try:
            object_id = int(requested) if requested else None
            if object_id is not None and not 1 <= object_id <= 2**63 - 1:
                raise ValueError("Enter a positive integer object ID within int64 range")
            if state:
                state[1].cancel.set()
            from morphofeatures.dino_experiments import resolve_paths

            root = Path(st.session_state.get("workspace_root", repository_root() / "outputs"))
            task = start_patch_preview(
                resolve_paths(stage, repository_root()),
                root / "preview_indices",
                object_id=object_id,
            )
            state = (signature, task)
            st.session_state[prefix + ":task"] = state
        except (ValueError, OSError) as error:
            st.error(f"Prepare data → Distributed fine patches → Preview object ID: {error}")
    if not state:
        return
    task = state[1]
    if not task.future.done():
        _distributed_preview_progress(task, prefix)
        return
    result = task.future.result()
    if result["state"] == "cancelled":
        st.info("Preview cancelled. An incomplete scan is never reused as a complete object index.")
    elif result["state"] == "error":
        from morphofeatures.workflow_validation import FieldError

        _local_stage_error(
            draft,
            index,
            FieldError(result["error"], *result["fields"]),
            "Distributed fine patches → Preview",
        )
    else:
        _show_distributed_preview(result["images"], result["summary"])
        bounds = result["summary"].get("object_index")
        if bounds:
            st.caption(f"Reusable object bounds: {bounds}")
            if st.button("Use preview bounds for preparation", key=prefix + ":use-index"):
                _put(draft, path + ["object_index"], bounds)
                touch(draft)
                st.session_state["editor_epoch"] = st.session_state.get("editor_epoch", 0) + 1
                rerun()


@st.fragment(run_every="2s")
def _distributed_preview_progress(task, prefix):
    if task.future.done():
        rerun()
    status = task.snapshot()
    st.progress(
        min(1.0, status.get("blocks", 0) / max(1, status.get("total_blocks", 1))),
        text="Object preview: " + status["phase"],
    )
    if st.button("Cancel patch preview", key=prefix + ":cancel"):
        task.cancel.set()
        st.info("Cancellation requested; the current bounded read/computation will finish first.")


def _show_distributed_preview(previews, summary):
    import matplotlib.pyplot as plt

    st.caption(
        f"Object {summary['label_id']}: {summary['eligible']} eligible, {summary['selected']} selected; showing {len(previews)}. Images are actual saved-grid center slices with foreground contours, displayed at their physical aspect ratio."
    )
    if summary.get("coverage_scope") == "partial_object":
        st.warning(
            "This object touches the volume/ROI boundary; only its available portion is shown."
        )
    if summary.get("overview"):
        st.caption(
            "Whole-object context: display-sized center sections, with the selected label outlined. These overview images do not change saved patch resolution."
        )
        fig, axes = plt.subplots(1, 3, figsize=(8, 2.8), squeeze=False)
        for ax, (raw, mask, info) in zip(axes[0], summary["overview"]):
            ax.imshow(raw, cmap="gray", vmin=0, vmax=1)
            if mask.any() and not mask.all():
                ax.contour(mask.astype(float), levels=[0.5], colors=["#e6ab02"], linewidths=0.7)
            ax.set_title(
                f"{['XY', 'XZ', 'YZ'][info['axis']]} · {raw.shape[0] * info['pixel_size_um']:.3g} µm canvas",
                fontsize=9,
            )
            ax.set_axis_off()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    fig, axes = plt.subplots(3, len(previews), figsize=(2.4 * len(previews), 6.8), squeeze=False)
    spacing = summary["saved_voxel_size_um"]
    for col, (raw, mask, _center) in enumerate(previews):
        for axis in range(3):
            image = raw.take(raw.shape[axis] // 2, axis=axis)
            foreground = mask.take(mask.shape[axis] // 2, axis=axis)
            dims = [d for d in range(3) if d != axis]
            ax = axes[axis, col]
            # Equal physical distances have equal lengths even with anisotropic native voxels.
            ax.imshow(
                image, cmap="gray", vmin=0, vmax=1, aspect=spacing[dims[0]] / spacing[dims[1]]
            )
            if foreground.any() and not foreground.all():
                ax.contour(
                    foreground.astype(float), levels=[0.5], colors=["#e6ab02"], linewidths=0.7
                )
            ax.set_title(f"Patch {col + 1} · {['XY', 'XZ', 'YZ'][axis]}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "patch": i + 1,
                    "center_zyx_um": list(center),
                    "sampled_foreground_fraction": float(mask.mean()),
                }
                for i, (_, mask, center) in enumerate(previews)
            ]
        ),
        hide_index=True,
    )


def _prepare(draft, index):
    path = ["document", "stages", index]
    field(
        draft,
        path + ["layout"],
        "Object preparation layout",
        "single_crop",
        options=["single_crop", "distributed_patches"],
        format_func={
            "single_crop": "Single object crop",
            "distributed_patches": "Distributed fine patches (DINO)",
        }.get,
        help="single_crop saves one fixed voxel crop per object for MAE or DINO. distributed_patches saves several small, physically sized patches spread through each object for DINO; it retains local texture at the cost of losing the arrangement between patches when vectors are averaged.",
    )
    distributed = _get(draft, path + ["layout"]) == "distributed_patches"
    if distributed:
        st.markdown("**What you should have for distributed fine patches**")
        st.markdown(
            "- A raw image and an integer instance segmentation (one nonzero ID per cell or nucleus).\n- Measured voxel sizes and origins in nm or µm; both volumes must describe the same physical coordinate frame.\n- A region containing complete objects, and either a physical patch width or native voxel counts.\n- Writable storage; existing object bounds are optional and speed up previews.\n- For embeddings: local DINO weights and the matching code checkout. No MAE checkpoint is needed."
        )
    field(
        draft,
        path + ["source"],
        "Volume data location",
        "local",
        options=["local", "platybrowser", "ome_zarr"],
        help="Use local files, published PlatyBrowser sources, or an HTTP OME-Zarr image group.",
    )
    source = _get(draft, path + ["source"])
    remote = source in {"platybrowser", "ome_zarr"}
    ready = (
        _ome_zarr_preparation_source(draft, index)
        if source == "ome_zarr"
        else _platybrowser_preparation_source(draft, index)
        if remote
        else True
    )
    if not remote:
        st.caption(
            "Use raw and instance volumes in the same physical frame. Label zero is background."
            if distributed
            else "Use aligned raw and instance volumes. Label zero is background."
        )
        for kind, label in (("raw", "Raw image"), ("segmentation", "Instance segmentation")):
            field(draft, path + [kind], label + " path", "")
            with validation_expander(draft, label + " dataset, axes and channel", expanded=True):
                field(
                    draft,
                    path + [kind + "_key"],
                    label + " dataset key",
                    "exported_data",
                    optional=True,
                )
                field(draft, path + [kind + "_axes"], label + " axis order", "zyxc")
                field(draft, path + [kind + "_channel"], label + " channel", 0)
                if "t" in _get(draft, path + [kind + "_axes"], ""):
                    field(
                        draft,
                        path + [kind + "_timepoint"],
                        label + " timepoint",
                        0,
                        help="Zero-based index along the stored time axis.",
                    )
    metadata = _inspect_preparation(draft, index) if ready else None
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
            help="null previews the full extent through a bounded slice. OME-Zarr preparation requires explicit [[z0,y0,x0],[z1,y1,x1]] bounds; PlatyBrowser uses published objects and local data can be scanned in full.",
        )
    if metadata:
        _preparation_preview(draft, index, metadata)
    if distributed:
        field(
            draft,
            path + ["patches", "sampling"],
            "Fine patch sampling",
            "physical_resampled",
            options=["physical_resampled", "native_voxels"],
            format_func={
                "physical_resampled": "Fixed physical width — resample to saved grid",
                "native_voxels": "Native raw voxels — do not resample",
            }.get,
            help="Physical mode fits the requested µm box onto the saved grid with interpolation and downsampling filters. Native mode directly crops the chosen number of raw voxels and ignores physical patch width. With anisotropic voxels, equal voxel counts span different physical lengths. Measured spacing is still needed to align the segmentation and record physical locations.",
        )
    native = distributed and _get(draft, path + ["patches", "sampling"]) == "native_voxels"
    triple_field(
        draft,
        path + ["crop_shape"],
        "Native fine patch shape (Z, Y, X voxels)"
        if native
        else "Saved fine patch shape (Z, Y, X voxels)"
        if distributed
        else "Crop shape",
        [32, 32, 32],
        help_text=(
            "Number of original raw voxels along this axis (1–128). No spatial interpolation occurs during preparation. Physical field of view is this count multiplied by the raw voxel spacing."
            if native
            else "Number of saved voxels along this axis (up to 128). Physical mode requires equal sides and resamples the chosen µm width onto this grid."
            if distributed
            else None
        ),
    )
    if distributed:
        st.caption(
            "Choose 1–128 raw voxels per axis; unequal sides are allowed. The grid snaps to source voxel boundaries. Physical size follows the measured raw spacing, and may differ between datasets."
            if native
            else "Use equal sides, for example 32,32,32 (maximum 128³). This is the saved sampling grid; the physical width below determines which tissue fits in it. Upsampling cannot recover detail missing from the raw image."
        )
    field(draft, path + ["output_format"], "Output format", "npy", options=["npy", "h5", "n5"])
    output_format = _get(draft, path + ["output_format"], "npy")
    st.caption(
        "Includes masked crops, per-patch IDs, unique object IDs, grouping offsets and physical centers, plus data_config.yaml for reuse with DINO."
        if distributed
        else (
            "Saves crops.npy, masks.npy and label_ids.npy."
            if output_format == "npy"
            else f"Saves crops.{output_format} with compressed crops, masks and label_ids datasets."
        )
    )
    field(draft, path + ["max_objects"], "Maximum objects (0 means all)", 8)
    if remote:
        stage = draft["document"]["stages"][index]
        if ready and stage.get("remote_provenance") and not distributed:
            st.caption(
                f"Resolved voxel spacing Z,Y,X: {stage['spacing_zyx']} {stage['unit']}; origin {stage['origin_zyx']}. Change resolution levels and reconnect to change the grid."
            )
    else:
        field(
            draft,
            path + ["unit"],
            "Coordinate unit",
            "voxel",
            options=["voxel", "nm", "um", "micrometer"],
        )
        triple_field(draft, path + ["spacing_zyx"], "Voxel spacing", [1.0, 1.0, 1.0])
        if distributed:
            with validation_expander(draft, "Different raw and segmentation grids (optional)"):
                st.caption(
                    "Leave these absent to use the shared spacing/origin. Set them when resolutions differ; units follow Coordinate unit above. Origins describe the center of voxel zero."
                )
                for kind in ("raw", "segmentation"):
                    field(
                        draft,
                        path + [kind + "_spacing_zyx"],
                        kind.capitalize() + " spacing override",
                        None,
                        help="Three measured Z,Y,X voxel sizes in the chosen coordinate unit; omit to use Voxel spacing.",
                    )
                    field(
                        draft,
                        path + [kind + "_origin_zyx"],
                        kind.capitalize() + " origin override",
                        None,
                        help="Three Z,Y,X coordinates of voxel zero in the chosen unit; omit to use the shared origin.",
                    )
    if distributed:
        _distributed_patch_controls(draft, index)
    st.caption(
        "Objects touching volume/ROI boundaries are skipped; coverage.tsv explains exclusions. DINO averages valid 2D slice features across each object's saved patches. This does not preserve the arrangement between patches."
        if distributed
        else "Every format includes mae_config.yaml with the paths and dataset keys needed by MAE, DINO and object previews. Objects touching ROI boundaries are skipped unless explicitly allowed."
    )
    from morphofeatures.object_selection_ui import object_selection_fields

    object_selection_fields(draft, path, physical=True, limit_path=path + ["max_objects"])
    with validation_expander(draft, "Advanced preparation settings"):
        field(draft, path + ["object_ids"], "Specific object IDs (optional)", [])
        advanced_fields(
            draft,
            path,
            {
                "source",
                "layout",
                "object_selection",
                "patches",
                "max_object_voxels",
                "ome_zarr",
                "max_scan_voxels",
                "raw_timepoint",
                "segmentation_timepoint",
                "raw_unit",
                "segmentation_unit",
                "platybrowser",
                "remote_options",
                "remote_provenance",
                "object_index",
                "data_version",
                "index_reference",
                "inspection_config",
                "read_budget_side",
                "raw_spacing_zyx",
                "raw_origin_zyx",
                "segmentation_spacing_zyx",
                "segmentation_origin_zyx",
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
            }
            | (
                {
                    "center",
                    "oversized",
                    "boundary",
                    "roi_boundary",
                    "normalization",
                    "background",
                    "block_shape",
                }
                if distributed
                else set()
            )
            | (
                {"origin_zyx"} | ({"block_shape"} if source == "platybrowser" else set())
                if remote
                else set()
            ),
        )
    _after_preparation_controls(draft, index)
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
            if _check_page(draft, "Prepare data"):
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
        "raw_timepoint",
        "segmentation_timepoint",
    )
    signature = json.dumps(
        {key: stage.get(key) for key in (*keys, "remote_options")}, sort_keys=True
    )
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
                    continue
                metadata[kind] = inspect_volume(
                    stage[kind]
                    if is_remote_url(stage[kind])
                    else repository_root() / Path(stage[kind]).expanduser(),
                    stage.get(kind + "_key"),
                    stage.get(kind + "_axes", "zyx"),
                    stage.get(kind + "_channel", 0),
                    remote_options=stage.get("remote_options"),
                    timepoint=stage.get(kind + "_timepoint", 0),
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
            _local_stage_error(
                draft,
                index,
                ValueError(value["error"]),
                "Inspect data dimensions → " + kind.capitalize() + " volume / dataset key",
            )
    rows = [
        {
            "Volume": kind.capitalize(),
            "Stored dimensions": str(value["stored_shape"]),
            "Axis order": value["axes"],
            "Spatial Z, Y, X": str(value["spatial_shape"]),
            "Data type": value["dtype"],
            "Channel": value["channel"],
            "Timepoint": value.get("timepoint", 0),
            "Voxel spacing Z,Y,X": str(
                stage.get(kind + "_spacing_zyx", stage.get("spacing_zyx", "not specified"))
            ),
            "Coordinate unit": stage.get(kind + "_unit", stage.get("unit", "not specified")),
        }
        for kind, value in metadata.items()
        if "error" not in value
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    metadata = {kind: value for kind, value in metadata.items() if "error" not in value}
    if not metadata:
        st.info("Enter at least one volume source to inspect.")
        return None
    if not _preview_aligned(stage, metadata) and len(metadata) == 2:
        st.warning(
            "The grids differ. Preview each layer separately. Distributed patches can resample physically aligned grids with correct spacings/origins; single-crop preparation and this overlay require matching grids."
        )
    reference = metadata.get("segmentation", next(iter(metadata.values())))
    if st.button("Use full volume ROI", key=key + ":full"):
        document = deepcopy(draft["document"])
        document["stages"][index]["roi"] = [[0, 0, 0], list(reference["spatial_shape"])]
        _replace_document(draft, document)
        rerun()
    if st.button("Use a central 64-voxel ROI", key=key + ":central"):
        shape = reference["spatial_shape"]
        start = [max(0, (s - 64) // 2) for s in shape]
        stop = [min(s, a + 64) for s, a in zip(shape, start)]
        document = deepcopy(draft["document"])
        document["stages"][index]["roi"] = [start, stop]
        _replace_document(draft, document)
        rerun()
    return metadata


def _preview_aligned(stage, metadata):
    import numpy as np

    if (
        len(metadata) != 2
        or metadata["raw"]["spatial_shape"] != metadata["segmentation"]["spatial_shape"]
    ):
        return False
    if stage.get("raw_unit") != stage.get("segmentation_unit"):
        return False
    return all(
        np.allclose(
            stage.get("raw_" + key) or stage.get(key, default),
            stage.get("segmentation_" + key) or stage.get(key, default),
        )
        for key, default in (("spacing_zyx", [1, 1, 1]), ("origin_zyx", [0, 0, 0]))
    )


def _preparation_preview(draft, index, metadata):
    import numpy as np

    from morphofeatures.data.volumes import preview_volume_pair, preview_volume_region

    stage = draft["document"]["stages"][index]
    key = f"volume-preview:{draft['id']}:{index}"
    if not st.checkbox(
        "Show ROI image preview",
        value=stage.get("source") not in {"platybrowser", "ome_zarr"},
        key=key,
    ):
        return
    try:
        choices = (["overlay"] if _preview_aligned(stage, metadata) else []) + list(metadata)
        mode = st.selectbox(
            "Preview layer",
            choices,
            format_func={
                "overlay": "Raw + segmentation overlay",
                "raw": "Raw intensity",
                "segmentation": "Instance segmentation",
            }.get,
            key=key + ":layer",
        )
        shape = metadata["raw" if mode == "overlay" else mode]["spatial_shape"]
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
            if stage.get(kind) and not is_remote_url(stage[kind]):
                settings[kind] = str(repository_root() / Path(stage[kind]).expanduser())
        preview = (
            preview_volume_pair(settings, axis=axis, index=plane)
            if mode == "overlay"
            else preview_volume_region(settings, mode, axis=axis, index=plane)
        )
        images, captions = [], []
        if "raw" in preview:
            intensity = preview["raw"]
            if not np.isfinite(intensity).all():
                raise ValueError("The preview contains nonfinite raw intensities.")
            low, high = np.percentile(intensity, [1, 99])
            gray = np.clip((intensity.astype(float) - low) / max(float(high - low), 1e-8), 0, 1)
            images.append(gray)
            captions.append("Raw intensity · 1st–99th percentile display contrast")
        if "segmentation" in preview:
            import matplotlib.pyplot as plt

            labels = preview["segmentation"]
            ids, inverse = np.unique(labels, return_inverse=True)
            colors = plt.get_cmap("tab20")(np.arange(len(ids)) % 20)[:, :3]
            colors[ids == 0] = 0
            colored = colors[inverse].reshape((*labels.shape, 3))
            images.append(colored)
            captions.append("Instance labels")
            if mode == "overlay":
                overlay = np.repeat(gray[..., None], 3, axis=2)
                foreground = labels != 0
                overlay[foreground] = 0.55 * overlay[foreground] + 0.45 * colored[foreground]
                images.append(overlay)
                captions.append("Alignment overlay")
            st.caption(
                f"{int(np.count_nonzero(ids))} nonzero IDs intersect this slice; this is not a full object count."
            )
        for column, pixels, caption in zip(st.columns(len(images)), images, captions):
            with column:
                st.image(pixels, caption=caption, use_container_width=True, clamp=True)
        st.caption(
            f"Slice {'ZYX'[axis]}={plane}; displayed voxel window {preview['start']} → {preview['stop']}."
        )
        if preview["cropped"]:
            st.info(
                "This ROI is larger than the preview limit. The image shows a central window of at most 512 × 512 pixels; preprocessing still uses the full selected ROI."
            )
    except Exception as error:
        _local_stage_error(draft, index, error, "Inspect data and choose a region → Data preview")


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
    with validation_expander(draft, "Optimization and data loading"):
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
    with validation_expander(draft, "Validation, early stopping and resume"):
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
    with validation_expander(draft, "Additional configuration, paths and reproducibility"):
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
    with validation_expander(draft, "MAE encoder and input geometry"):
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
    with validation_expander(draft, "MAE decoder and reconstruction"):
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
    with validation_expander(draft, "Additional data settings"):
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
    if not stage.get("from_training") and not dino:
        field(
            draft,
            path + ["checkpoint"],
            "Model checkpoint",
            "",
        )
    if dino:
        from morphofeatures.dino_defaults import backbone_fields

        st.info(
            "DINO features: convert each intensity crop into 2D views, encode with a frozen pretrained backbone, and combine the view vectors into one embedding per object. Supply intensity crops and aligned masks, rather than instance-label values as intensities."
        )
        backbone_fields(draft, path, show_family=False, checkpoint_label="Pretrained DINO weights", repository_label="Official local model repository")
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
        with validation_expander(draft, "DINO views and feature aggregation", expanded=True):
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
                options=["cls", "patch_mean", "foreground_mean"],
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
        from morphofeatures.object_selection_ui import object_selection_fields

        object_selection_fields(draft, path, physical=stage.get("views", {}).get("mode", "stored") != "stored")
        if stage.get("views", {}).get("mode", "stored") != "stored":
            from morphofeatures.dino_runtime_ui import runtime_fields

            runtime_fields(draft, path)
    else:
        from morphofeatures.data.object_selection import selection_active

        if selection_active(stage.get("object_selection")):
            from morphofeatures.object_selection_ui import object_selection_fields

            st.warning("This extraction still has a DINO target selection. Switch back to DINO, or choose your MAE cohort on Prepare data and reset this extraction to All objects.")
            object_selection_fields(draft, path)
    with validation_expander(draft, "Advanced extraction settings"):
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
                "runtime",
                "object_selection",
            },
        )
        if "config" in stage:
            if dino:
                advanced_fields(
                    draft,
                    path + ["config"],
                    {"data", "mae", "training", "slurm", "device", "profiles"},
                )
                with validation_expander(draft, "Additional DINO data settings"):
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


def _joint_analyze(draft, index):
    path = ["document", "stages", index]
    stage = draft["document"]["stages"][index]
    st.caption(
        "Each dataset keeps its own object IDs. PCA, UMAP and clustering are fitted once to pooled "
        "objects, with a dataset-colored panel. Use identical weights, feature settings and comparable "
        "crop preparation across datasets. This is an exploratory projection; it does not train a classifier."
    )
    earlier = [name for s in draft["document"]["stages"][:index] for name in embedding_names(s)]
    for i, dataset in enumerate(stage.get("datasets", [])):
        dp = path + ["datasets", i]
        with validation_expander(
            draft, f"Dataset {i + 1} · {dataset.get('name', '')}", expanded=True
        ):
            field(draft, dp + ["name"], "Dataset name", f"Dataset {i + 1}")
            if dataset.get("from_embedding"):
                field(
                    draft,
                    dp + ["from_embedding"],
                    "Extraction stage for this dataset",
                    earlier[0] if earlier else "",
                    options=earlier or [dataset["from_embedding"]],
                )
                if st.button("Use saved embeddings instead", key=f"joint-file:{index}:{i}"):
                    document = deepcopy(draft["document"])
                    row = document["stages"][index]["datasets"][i]
                    row.pop("from_embedding", None)
                    row["embedding"] = ""
                    _replace_document(draft, document)
                    rerun()
            else:
                field(
                    draft,
                    dp + ["embedding"],
                    "Saved embedding file",
                    "",
                    help="An NPZ, NPY, TSV or CSV of features with object IDs. Keep the extraction metadata beside the file so the model can be verified.",
                )
                if earlier and st.button(
                    "Use an earlier extraction", key=f"joint-link:{index}:{i}"
                ):
                    document = deepcopy(draft["document"])
                    row = document["stages"][index]["datasets"][i]
                    row.pop("embedding", None)
                    row["from_embedding"] = earlier[0]
                    _replace_document(draft, document)
                    rerun()
                if st.button("Add extraction for this dataset", key=f"joint-extract:{index}:{i}"):
                    from morphofeatures.workspace_state import add_dataset_extraction

                    _replace_document(draft, add_dataset_extraction(draft["document"], index, i))
                    rerun()
                st.caption(
                    "Configure the first extraction's MAE/DINO model and checkpoint before adding others. Later extractions copy those settings; select each dataset's own prepared inputs in its extraction stage."
                )
            field(
                draft,
                dp + ["annotations"],
                "This dataset's annotation table (optional)",
                None,
                optional=True,
                help="Labels are joined by this dataset's original label_id, so identical IDs in other datasets remain separate.",
            )
            field(draft, dp + ["label_column"], "Cell-type label column", "auto")
            field(
                draft,
                dp + ["input_config"],
                "This dataset's inspection YAML (optional)",
                None,
                optional=True,
                help="Used only for this dataset's image/mesh inspection and MoBIE export. Extraction metadata supplies it automatically when available.",
            )
            if len(stage["datasets"]) > 2 and st.button(
                "Remove dataset from projection", key=f"joint-remove:{index}:{i}"
            ):
                document = deepcopy(draft["document"])
                document["stages"][index]["datasets"].pop(i)
                _replace_document(draft, document)
                rerun()
    if st.button("Add another dataset", key=f"joint-add:{index}"):
        document = deepcopy(draft["document"])
        document["stages"][index]["datasets"].append({"name": "", "embedding": ""})
        _replace_document(draft, document)
        rerun()
    field(
        draft,
        path + ["assume_same_model"],
        "I confirm identical models/settings when metadata is missing",
        False,
        help="Only needed for externally generated embeddings without checkpoint provenance. Recorded as unverified in the report. Known checkpoint or feature-setting mismatches always block the run.",
    )
    field(
        draft,
        path + ["max_objects_per_dataset"],
        "Maximum projected objects per dataset (0 = all)",
        0,
        help="A reproducible random cap per dataset can reduce memory use and dominance by larger datasets. Unsampled objects retain their IDs and labels but have no shared coordinates or clusters.",
    )
    field(draft, path + ["umap"], "Include UMAP projection", False)
    umap_pca_fields(draft, path)
    field(
        draft,
        path + ["normalization"],
        "Feature normalization",
        "standardize",
        options=["standardize", "l2", "none"],
    )
    field(
        draft,
        path + ["cluster_method"],
        "Clustering method",
        "kmeans",
        options=["kmeans", "leiden"],
    )
    field(draft, path + ["clusters"], "Number of shared clusters", 8)
    with validation_expander(draft, "Shared projection settings"):
        field(draft, path + ["neighbors"], "UMAP / Leiden neighbors", 15)
        field(draft, path + ["min_dist"], "UMAP minimum distance", 0.0)
        field(draft, path + ["umap_epochs"], "UMAP epochs (null uses automatic)", 50)
        field(draft, path + ["resolution"], "Leiden resolution", 0.004)
        field(draft, path + ["seed"], "Random seed", 42)
        field(draft, path + ["unlabeled_opacity"], "Unlabeled point opacity", 0.15)


def umap_pca_fields(draft, path, *, include_variants=True):
    """Independent PCA inputs, optionally followed by named analysis combinations."""
    from morphofeatures.analysis_variants_ui import variant_fields

    cluster_path = path + ["cluster_pca_components"]
    if _get(draft, cluster_path) is None:
        _put(draft, cluster_path, 0)
        touch(draft)
    field(draft, cluster_path, "PCA components before clustering (0 = off)", 0)
    if _get(draft, path + ["umap"], True):
        count_path = path + ["umap_pca_components"]
        if _get(draft, count_path) is None:
            _put(draft, count_path, 0)
            touch(draft)
        field(draft, count_path, "PCA components before UMAP (0 = off)", 0)
    if include_variants:
        with validation_expander(draft, "Additional analysis combinations"):
            variant_fields(draft, path)


def projection_settings_fields(
    draft,
    path,
    *,
    umap_label="Generate UMAP",
    clusters_label="Number of clusters",
    seed_label="Random seed",
    include_variants=True,
):
    """Expose the main transfer/experiment map settings before optional variants."""
    from morphofeatures.analysis_variants_ui import variant_fields

    with validation_expander(draft, "Clustering, PCA and UMAP", expanded=True):
        st.caption(
            "Configure the main maps here; additional combinations below are optional. "
            "Clustering groups objects by their embedding features or selected PCA components. "
            "UMAP minimum distance controls how tightly points appear on the map. "
            "Classifier inputs have separate controls above or below this section."
        )
        method = field(
            draft,
            path + ["cluster_method"],
            "Clustering method",
            "kmeans",
            options=["kmeans", "leiden"],
        )
        if method == "kmeans":
            field(draft, path + ["clusters"], clusters_label, 8)
        elif method == "leiden":
            field(draft, path + ["resolution"], "Leiden resolution", 0.004)
        umap = field(draft, path + ["umap"], umap_label, True)
        umap_pca_fields(draft, path, include_variants=False)
        if umap or method == "leiden":
            field(draft, path + ["neighbors"], "UMAP / Leiden neighbors", 15)
        if umap:
            field(draft, path + ["min_dist"], "UMAP minimum distance", 0.0)
            field(draft, path + ["umap_epochs"], "UMAP epochs (null uses automatic)", 50)
        field(draft, path + ["seed"], seed_label, 42)
        field(draft, path + ["unlabeled_opacity"], "Unlabeled point opacity", 0.15)
    if include_variants:
        with validation_expander(draft, "Additional analysis combinations"):
            variant_fields(draft, path)


def classifier_input_fields(draft, path):
    from morphofeatures.analysis.classification import classifier_space

    settings = _get(draft, path, {})
    space = field(
        draft,
        path + ["classifier_space"],
        "Classification input",
        classifier_space(settings),
        options=["embeddings", "pca", "umap"],
        format_func={
            "embeddings": "Full embedding vectors",
            "pca": "PCA components",
            "umap": "UMAP coordinates",
        }.get,
    )
    if space != "embeddings":
        field(
            draft,
            path + ["classifier_dimensions"],
            "Classifier dimensions",
            settings.get("evaluation_pca") or 2,
        )
    field(
        draft,
        path + ["classifier_standardize"],
        "Standardize selected classifier dimensions",
        False,
    )
    if space == "umap":
        field(draft, path + ["classifier_umap_neighbors"], "Classifier UMAP neighbors", 15)
        field(draft, path + ["classifier_umap_min_dist"], "Classifier UMAP minimum distance", 0.1)
        field(draft, path + ["classifier_umap_epochs"], "Classifier UMAP epochs", 200)
    st.caption(
        "The classifier's projection and scaling are fitted inside each training fold. These coordinates are separate from the exploratory plot fitted to all objects. Transferred objects use the fitted reference transformation."
    )


def _analyze(draft, index):
    from morphofeatures.analysis.classification import CLASSIFIER_NAMES

    path = ["document", "stages", index]
    stage = draft["document"]["stages"][index]
    if stage.get("published_reference"):
        from morphofeatures.spatial_probe_ui import published_fields

        published_fields(draft, index)
    if stage["action"] == "analyze" and not (
        stage.get("from_extraction") or stage.get("from_embedding")
    ):
        _artifact_picker(draft, index, "embedding")
    if stage["action"] == "compare":
        st.caption(
            "Representations are matched by object ID. Annotation groups keep related specimens in the same evaluation fold."
        )
        if stage.get("from_embeddings"):
            st.info(
                "Input: named embeddings from this workflow: " + ", ".join(stage["from_embeddings"])
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
    elif stage.get("from_embedding"):
        st.info("Input: named embedding output " + stage["from_embedding"])
    elif stage.get("from_extraction"):
        st.info("Input: embeddings from Extract embeddings in this workflow.")
    else:
        field(draft, path + ["embedding"], "Saved embedding file", "")
    with validation_expander(draft, "Bilateral-partner retrieval (optional)"):
        from morphofeatures.spatial_probe_ui import bilateral_fields

        bilateral_fields(draft, path + ["bilateral"], stage.setdefault("bilateral", {}), inline=True)
    field(draft, path + ["umap"], "Include UMAP projection", False)
    umap_pca_fields(draft, path)
    field(
        draft,
        path + ["cluster_method"],
        "Clustering method",
        "kmeans",
        options=["kmeans", "leiden"],
    )
    field(draft, path + ["clusters"], "Number of clusters", 8)
    if stage.get("analysis_normalizations") is None:
        field(
            draft, path + ["normalization"], "Feature normalization", "standardize",
            options=["standardize", "l2", "none"],
        )
    from morphofeatures.analysis_variants_ui import scaling_fields

    scaling_fields(draft, path)
    with validation_expander(
        draft, "Classification check against known labels", expanded=bool(stage.get("annotations"))
    ):
        st.caption(
            "Join known cell types by label_id for side-by-side cluster/class plots. Evaluate linear logistic regression, MLP or K-nearest neighbors on full features or training-fitted PCA/UMAP dimensions. Select only logistic to run a linear classifier alone."
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
            format_func=lambda value: CLASSIFIER_NAMES.get(value, value),
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
        classifier_input_fields(draft, path)
    with validation_expander(draft, "Advanced analysis and evaluation settings"):
        advanced_fields(
            draft,
            path,
            {
                "action",
                "embedding",
                "embeddings",
                "bilateral",
                "published_reference",
                "use_published_layout",
                "from_extraction",
                "annotations",
                "label_column",
                "group_column",
                "umap",
                "umap_pca_components",
                "cluster_pca_components",
                "analysis_variants",
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
                "classifier_space",
                "classifier_dimensions",
                "classifier_standardize",
                "classifier_umap_neighbors",
                "classifier_umap_min_dist",
                "classifier_umap_epochs",
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


def _export_mobie(draft, index):
    from morphofeatures.analysis.mobie_export import resolve_mobie_export

    path = ["document", "stages", index]
    stage = draft["document"]["stages"][index]
    st.caption(
        "Create a MoBIE project linking the original objects to saved clusters, class labels "
        "and UMAP/PCA coordinates. Export runs in a reviewed local or Slurm job. "
        "See Docs → Analysis and MoBIE for preparation, sharing and Fiji instructions."
    )
    if stage.get("from_bilateral"):
        st.info("Uses the earlier bilateral result; queries and partners are cell IDs.")
    elif stage.get("from_analysis"):
        st.info("Uses the last Analyze or Compare result produced earlier in this workflow.")
    elif stage.get("action") != "export_bilateral":
        field(draft, path + ["analysis"], "Saved analysis.json or comparison.json", "")
    field(
        draft,
        path + ["input_config"],
        "Original data / inspection YAML (optional)",
        None,
        optional=True,
        help="Recover spatial sources from an inspection or training config. Prepared crop stacks alone cannot place objects back in the animal.",
    )
    field(
        draft,
        path + ["mode"],
        "Images in the MoBIE project",
        "portable",
        options=["portable", "platybrowser"],
        format_func=lambda v: {
            "portable": "Copy a region — portable project",
            "platybrowser": "Stream published PlatyBrowser images",
        }.get(v, v),
        help="Portable copies the selected region at the selected source resolution. PlatyBrowser copies metadata and tables; Fiji streams its existing image pyramid.",
    )
    field(
        draft,
        path + ["dataset_name"],
        "Dataset name",
        "morphofeatures",
        help="A short folder name using letters, numbers, underscores or hyphens.",
    )
    field(
        draft,
        path + ["description"],
        "Biological sample and analysis description",
        "MorphoFeatures analysis",
    )
    if stage.get("mode") == "platybrowser":
        from morphofeatures.analysis.platybrowser import DEFAULTS

        for key in ("project_url", "revision", "dataset", "raw_source", "segmentation_source"):
            field(
                draft,
                path + ["platybrowser", key],
                "PlatyBrowser " + key.replace("_", " "),
                DEFAULTS[key],
            )
        field(
            draft,
            path + ["platybrowser", "object_ids"],
            "IDs used by the embeddings",
            "segmentation",
            options=["segmentation", "cell"],
            help="New PlatyBrowser preprocessing uses segmentation IDs. Older cell-indexed patches need the published cell-to-nucleus mapping when displaying nuclei.",
        )
        field(
            draft,
            path + ["platybrowser", "cache_directory"],
            "Published metadata cache directory",
            str(repository_root() / "outputs/remote_cache/platybrowser"),
            help="A writable directory shared with the export worker. Only metadata and object tables are needed for this export.",
        )
    else:
        if stage.get("roi") is None:
            stage["roi"] = [[0, 0, 0], [64, 64, 64]]
            touch(draft)
        st.caption(
            "Enter ROI start (inclusive) and stop (exclusive) in the selected volume's Z, Y, X voxels. No resampling is performed."
        )
        for kind in ("raw", "segmentation"):
            field(
                draft,
                path + [kind],
                f"{kind.capitalize()} container or URL" + (" (optional)" if kind == "raw" else ""),
                "",
                help="Original spatial volume, not the N × Z × Y × X prepared crop stack. Segmentation must contain a distinct positive integer ID for each cell/nucleus; zero is background.",
            )
            field(draft, path + [kind + "_key"], f"{kind.capitalize()} array key", "0")
        triple_field(draft, path + ["roi", 0], "ROI start", [0, 0, 0])
        triple_field(draft, path + ["roi", 1], "ROI stop", [64, 64, 64])
        triple_field(draft, path + ["spacing_zyx"], "Voxel spacing", [1.0, 1.0, 1.0])
        triple_field(draft, path + ["origin_zyx"], "Original grid origin", [0.0, 0.0, 0.0])
        field(draft, path + ["unit"], "Coordinate unit", "voxel", options=["voxel", "nm", "um"])
        field(
            draft,
            path + ["max_voxels"],
            "Maximum voxels to copy",
            256**3,
            help="Safeguard checked before reading pixels. Raw float32 plus segmentation int64 need about 12 bytes per voxel before compression, plus the optional ZIP copy.",
        )
        with validation_expander(draft, "Axis order, channels, timepoints and remote reading"):
            for kind in ("raw", "segmentation"):
                field(
                    draft, path + [kind + "_axes"], kind.capitalize() + " stored axis order", "zyx"
                )
                field(
                    draft,
                    path + [kind + "_channel"],
                    kind.capitalize() + " channel (zero based)",
                    0,
                )
                field(
                    draft,
                    path + [kind + "_timepoint"],
                    kind.capitalize() + " timepoint (zero based)",
                    0,
                )
            field(
                draft,
                path + ["remote_options"],
                "Remote reader options",
                {},
                help="Inherited from Prepare data. For generic HTTP OME-Zarr set {format: ome_zarr}; cache_directory, cache_size_mb and timeout_seconds are optional.",
            )
            triple_field(draft, path + ["block_shape"], "Copy block size", [32, 32, 32])
    field(
        draft,
        path + ["id_mapping"],
        "Embedding-to-segmentation ID table (optional)",
        None,
        optional=True,
        help="TSV/CSV with label_id and segmentation_id (or nucleus_id). Positive mappings must be one-to-one. Leave empty when embeddings already use segmentation IDs.",
    )
    field(
        draft,
        path + ["require_all_ids"],
        "Stop export if any embedding objects are missing",
        False,
        help="Enable for whole-sample exports. For an ROI, unmatched objects are listed in excluded-ID tables. Zero matched objects always stops the export.",
    )
    with validation_expander(draft, "Evidence files, display and download"):
        field(
            draft,
            path + ["include_artifacts"],
            "Include saved embeddings, figures and analysis reports",
            True,
        )
        field(
            draft,
            path + ["include_feature_table"],
            "Also export every embedding dimension as a MoBIE table",
            False,
            help="Adds a wide features table per representation. NPZ embeddings are already included with evidence; wide tables can slow the viewer.",
        )
        field(
            draft,
            path + ["attachments"],
            "Additional files to include",
            [],
            help="YAML list of files, for example [outputs/run/checkpoint.pt, outputs/run/metrics.jsonl]. Checkpoints are copied only when explicitly listed.",
        )
        field(draft, path + ["max_artifact_gb"], "Evidence file size limit (GiB)", 4.0)
        field(
            draft,
            path + ["contrast_limits"],
            "Raw display contrast [low, high]",
            [0.0, 255.0],
            help="Display only; exported raw intensities are not normalized. For floating-point raw images choose limits based on the data preview.",
        )
        field(
            draft,
            path + ["make_zip"],
            "Create a ZIP for transfer",
            True,
            help="Uses extra disk space. Large projects can be copied as folders; the browser offers downloads only up to 200 MiB.",
        )
    if st.button("Check export settings and recover source metadata", key=f"mobie-check:{index}"):
        try:
            if stage.get("input_config"):
                from morphofeatures.analysis.mobie_export import export_source_defaults

                stage.update(export_source_defaults(load_document(stage["input_config"])))
                # A deliberate recovery refreshes source widgets as well as the draft.
                st.session_state["editor_epoch"] = st.session_state.get("editor_epoch", 0) + 1
            resolved = resolve_mobie_export(stage, require_analysis=not (stage.get("from_analysis") or stage.get("from_bilateral")))
            stage.update(resolved)
            touch(draft)
            st.success(
                "Settings checked. Review source fields and ROI before submitting. Object coverage is checked by the background export job."
            )
        except Exception as error:
            _local_stage_error(
                draft,
                index,
                error,
                "Original data / inspection YAML → Check export settings and recover source metadata",
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
    with validation_expander(draft, "ID mapping and bounded I/O"):
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
    from morphofeatures.artifact_availability import selected_artifact
    from morphofeatures.result_catalog import run_artifacts

    for record in JobRegistry(registry).list(limit=500):
        for item in run_artifacts(record):
            if not item:
                continue
            path = Path(item)
            status = selected_artifact(path)
            if not status.ready:
                continue
            if (
                (kind == "data" and path.name in {"mae_config.yaml", "data_config.yaml"})
                or (kind == "checkpoint" and path.suffix in {".pt", ".pth"})
                or (kind == "embedding" and status.kind == "vectors")
            ):
                options[str(path)] = record.display_name + " · " + path.name
    if not options:
        return
    with validation_expander(draft, "Choose " + kind + " from a completed run"):
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
                    if prepared["data"].get("source") == "grouped_crops":
                        if stage.get("action") != "extract" or stage.get("model") not in {
                            "dinov2",
                            "dinov3",
                        }:
                            raise ValueError(
                                "These distributed patches feed DINO. Choose a DINO extraction stage, or select single-object crops for crop MAE training."
                            )
                        stage.setdefault("views", {}).update(
                            mode="stored", normalization="unit", resize="letterbox"
                        )
                    stage.setdefault("config", {})["data"] = prepared["data"]
                    if prepared.get("mae"):
                        stage["config"].setdefault("mae", {})["input_shape"] = prepared["mae"][
                            "input_shape"
                        ]
                    if prepared.get("inspection"):
                        stage["config"]["inspection"] = prepared["inspection"]
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
                _local_stage_error(draft, index, error, "Reuse saved inputs → Selected artifact")


def _stage_title(stages, index):
    stage = stages[index]
    title = STAGE_LABELS.get(stage["action"], stage["action"])
    name = stage.get("name")
    if stage.get("from_extraction"):
        name = next(
            (
                s.get("name", s.get("model"))
                for s in reversed(stages[:index])
                if s["action"] == "extract"
            ),
            None,
        )
    if stage.get("from_embeddings"):
        name = " + ".join(stage["from_embeddings"])
    return title + (" · " + str(name) if name else "")


def _stage_options(draft, step):
    document = draft["document"]
    stages = document["stages"]
    actions = [s["action"] for s in stages]
    with validation_expander(draft, "Change which stages run"):
        st.caption(
            "Stage changes can be undone in this draft. Stopping early also removes later dependent stages."
        )

        def change(stages):
            draft.setdefault("stage_history", []).append(deepcopy(document))
            _replace_document(draft, {**document, "stages": stages})

        if step in {"Extract", "Build features", "Learn", "Analyze"}:
            if st.button("Add embedding distillation (adapter / PCA)"):
                from morphofeatures.analysis.distillation import defaults

                names = [name for stage in stages for name in embedding_names(stage)]
                addition = defaults()
                addition["name"] = f"Distilled {len(stages) + 1}"
                if names:
                    addition["from_embedding"] = names[-1]
                change(stages + [addition])
                rerun()
            if "bilateral" in actions and st.button("Add bilateral 3D export"):
                change(stages + [{"action": "export_bilateral", "from_bilateral": True, "make_mobie": True, "mode": "platybrowser", "platybrowser": {"segmentation_source": "cells", "object_ids": "segmentation"}}])
                rerun()
            if st.button("Add a bilateral-partner probe"):
                from morphofeatures.analysis.bilateral import defaults

                names = [name for s in stages for name in (embedding_names(s)[1:] if s["action"] == "distill" else embedding_names(s))]
                change(stages + [{"action": "bilateral", **defaults(), "from_embeddings": names, "embeddings": {}}])
                rerun()
            if st.button("Add spatial context features"):
                change(stages + [{"action": "context", "name": "Spatial context", "embedding": "", "graph_mode": "adjacency_table", "adjacency_table": ""}])
                rerun()
            if st.button("Add a feature-combination stage"):
                change(
                    stages
                    + [
                        {
                            "action": "fuse",
                            "name": "Combined",
                            "embeddings": {},
                            "weights": [0.5, 0.5],
                        }
                    ]
                )
                rerun()
            if st.button("Add a label-transfer stage"):
                change(stages + [{"action": "transfer", "reference": {}, "targets": []}])
                rerun()

        labels = [f"{i + 1}. {_stage_title(stages, i)}" for i in range(len(stages))]
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
        if any(a in actions for a in ("analyze", "compare")) and "export_mobie" not in actions:
            if st.button("Add MoBIE export after analysis"):
                from morphofeatures.analysis.mobie_export import SOURCE_FIELDS

                prepared = next((s for s in stages if s["action"] == "preprocess"), {})
                change(
                    stages
                    + [
                        {
                            **{k: v for k, v in prepared.items() if k in SOURCE_FIELDS},
                            "action": "export_mobie",
                            "from_analysis": True,
                            "mode": "platybrowser"
                            if prepared.get("source") == "platybrowser"
                            else "portable",
                        }
                    ]
                )
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
                dino = new_draft(
                    "dino", model_defaults=st.session_state.get("workspace_dino_defaults")
                )["document"]["stages"]
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
            and not any(a in actions for a in ("analyze", "compare", "joint_analyze"))
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
                stage.pop("published_reference", None)
                stage.pop("use_published_layout", None)
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
        with validation_expander(draft, "Local Python environment"):
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
    with validation_expander(draft, "Load a cluster resource preset"):
        workspace_presets = st.session_state.get("workspace_resource_presets")
        preset_source = "file"
        if workspace_presets:
            preset_source = retained_input(
                "Preset source", "workspace", "guided:preset_source", kind="selectbox",
                options=["workspace", "file"],
                format_func={"workspace": "Workspace settings", "file": "Another YAML file"}.get,
                help="Applied workspace presets are available here. You can also load resources from a separate preset YAML on this machine.",
            )
        if preset_source == "file":
            source = retained_input(
                "Cluster presets YAML", st.session_state.get("workspace_cluster_profiles", "configs/slurm_profiles.example.yaml"), "guided:cluster_profiles"
            )
        st.caption(
            "Loading a preset copies its resources and environment into this draft. Example presets need your cluster's partition and account."
        )
        try:
            profiles = (
                cluster_profiles_from_mapping({"profiles": workspace_presets})
                if preset_source == "workspace"
                else load_cluster_profiles(repository_root() / Path(source).expanduser())
            )
            selected = st.selectbox("Cluster resource preset", list(profiles))
            if st.button("Use cluster resource preset"):
                resources = asdict(profiles[selected])
                resources.pop("name", None)
                _replace_document(
                    draft, {**draft["document"], "slurm": json.loads(json.dumps(resources))}
                )
                rerun()
        except Exception as error:
            st.error(
                "Review & run → Load a cluster resource preset → "
                + ("Workspace settings: " if preset_source == "workspace" else "Cluster presets YAML: ")
                + str(error)
            )
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
    with validation_expander(draft, "Mail notifications", expanded=True):
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
    with validation_expander(draft, "Advanced scheduler and runtime settings"):
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
    from morphofeatures.dino_runtime_ui import memory_review

    memory_review(document, draft["execution"])
    for stage in document["stages"]:
        if stage.get("analysis_normalizations"):
            from morphofeatures.analysis.projection import NORMALIZATION_NAMES

            st.caption("Full analysis scaling comparisons for " + stage.get("name", stage["action"]) + ": " + "; ".join(NORMALIZATION_NAMES.get(v, str(v)) for v in stage["analysis_normalizations"]) + ". Classifiers are evaluated again with the same held-out folds; extraction is reused.")
        if stage.get("analysis_variants"):
            st.caption(
                "Additional analysis combinations for " + stage.get("name", stage["action"])
                + ". Each reuses the main analysis's plotted cohort and saved embeddings."
            )
            st.dataframe(pd.DataFrame(stage["analysis_variants"]))
        if stage.get("action") == "transfer":
            from morphofeatures.provenance_ui import transfer_input_check

            transfer_input_check(stage)
        transfer = stage if stage.get("action") == "transfer" else stage.get("transfer", {})
        if stage.get("action") == "transfer" or transfer.get("enabled"):
            from morphofeatures.analysis.transfer import CONDITION_NAMES

            targets = (
                transfer.get("targets", [])
                if stage.get("action") == "transfer"
                else [
                    d for d in stage.get("datasets", []) if d["name"] != transfer.get("reference")
                ]
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Target": target.get("name", "Target"),
                            "Transfer condition": CONDITION_NAMES.get(condition, condition),
                            "Uses unlabeled target features": condition
                            in {"coral", "coral_source_to_target"}
                            or (
                                condition == "deep_coral"
                                and transfer.get("adapter", {}).get("coral_weight", 1.0) != 0
                            ),
                            "Neural training epochs per fit": transfer.get("adapter", {}).get(
                                "epochs", 100
                            )
                            if condition in {"adapter", "deep_coral"}
                            else "—",
                        }
                        for target in targets
                        for condition in transfer.get("conditions", ["uncorrected", "coral"])
                    ]
                )
            )
            st.caption(
                "Transfer conditions share source folds and object cohorts. New classifiers/adapters are fitted once per reference validation fold and once on the complete labeled reference. Fine/coarse experiments repeat this comparison for each representation, checkpoint and physical scale. No target labels guide fitting or stopping."
            )
        if stage.get("action") == "dino_experiment":
            from morphofeatures.dino_experiments import FUSION_METHODS, fusion_recipes

            try:
                recipes = fusion_recipes(stage)
                st.dataframe(pd.DataFrame([
                    {"Combined output": r["name"], "Feature blocks": ", ".join(r["blocks"]),
                     "Fusion method": FUSION_METHODS[r["combination"]],
                     "Weights": "Ignored (raw)" if r["combination"] == "raw" else str(r["weights"])}
                    for r in recipes
                ]), hide_index=True)
                st.caption(f"{4 + len(recipes)} generated representations per dataset/backbone/scale, before any saved baselines. Each selected analysis scaling adds a separately fitted map and enabled classification evaluation.")
            except ValueError as error:
                st.warning(str(error))
            count = int(stage.get("max_objects", 100))
            views = stage.get("views", {})
            axes = len(views.get("axes", [0, 1, 2]))
            bound = axes * (
                int(views.get("max_tiles", 64)) * len(views.get("fine_fractions", [0.5]))
                + len(views.get("coarse_fractions", [0.1, 0.3, 0.5, 0.7, 0.9]))
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Dataset": d["name"],
                            "Object cap": count or "All selected",
                            "Backbone": b.get("variant"),
                            "Fine width (µm)": scale,
                            "Minimum nucleus / 3D tile": views.get("min_foreground_fraction", 0.01),
                            "Minimum nucleus / 2D image": views.get(
                                "min_slice_foreground_fraction", 0.0
                            ),
                            "Fine grid anchor": views.get("tile_anchor", "bbox_min"),
                            "View bound per object": "Exhaustive"
                            if views.get("exhaustive")
                            else bound,
                            "Outputs": "Fine CLS / foreground; coarse CLS / foreground; combined",
                            "Fine/coarse weights": str(stage.get("fusion_weights", [0.5, 0.5])),
                        }
                        for d in stage["datasets"]
                        for b in stage["backbones"]
                        for scale in stage.get("tile_sizes_um", [1.0])
                    ]
                )
            )
            st.caption(
                "View counts are upper bounds before mask filtering. Objects are selected once per dataset. Metadata checks do not read or encode full objects."
            )
    from morphofeatures.provenance_ui import provenance_view

    provenance_view(
        key="review:provenance:" + draft["id"], document=document, base=repository_root()
    )
    kwargs = dict(
        output_root=config.paths.output_root,
        run_id=draft["run_id"],
        execution=draft["execution"],
        dependency=draft["dependency"] or None if draft["execution"] == "slurm" else None,
    )
    fingerprint = submission_fingerprint(document, **kwargs)
    form_errors = _form_issues(draft)
    _show_issues(draft, form_errors, context="review-form")
    if st.button("Review run", disabled=bool(form_errors), type="primary"):
        collected = []
        issues = []
        try:
            plan = plan_workspace_job(
                document, **kwargs, issues=collected, allow_synthetic=draft["allow_synthetic"]
            )
            st.session_state["workspace_plan"] = (draft["id"], draft["revision"], plan)
        except Exception as error:
            issues = [v.to_dict() for v in collected] or [
                execution_issue(error, resources=document.get("slurm", {})).to_dict()
            ]
        st.session_state[_check_key(draft)] = issues
        if issues:
            st.session_state.pop("workspace_plan", None)
        rerun()
    _show_issues(draft, _stored_issues(draft), context="review-issues")
    saved = st.session_state.get("workspace_plan")
    if not saved or saved[0] != draft["id"]:
        return
    _, revision, plan = saved
    if revision != draft["revision"] or fingerprint != plan.fingerprint or form_errors:
        st.warning(
            "Review & run → Review run: Settings changed after review. Review again to update the configuration and script."
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
                ("from_analysis", "last Analyze / Compare result"),
            )
            if stage.get(key)
        ]
        rows.append(
            {
                "Order": index + 1,
                "Stage": _stage_title(document["stages"], index),
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
    with validation_expander(draft, f"Changes from loaded settings ({len(changes)})"):
        st.dataframe(pd.DataFrame(changes)) if changes else st.write(
            "No scientific settings changed."
        )
    with validation_expander(draft, "Exact worker configuration and command"):
        st.code(format_command(plan.record.command), language="bash")
        st.code(plan.resolved_yaml, language="yaml")
    with validation_expander(draft, "Generated Slurm script"):
        st.code(plan.script, language="bash")
    available = draft["execution"] != "slurm" or SlurmScheduler.available()
    if not available:
        st.warning(
            "Review & run → Run on: Slurm submission is unavailable because sbatch was not found on this app host. Choose local execution or save a dry-run bundle."
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
            issue = execution_issue(error, resources=document.get("slurm", {})).to_dict()
            st.session_state[_check_key(draft)] = [issue]
            _show_issues(draft, [issue], context="submission")


def workflow_page(config):
    from morphofeatures.dino_defaults import workspace_defaults

    st.session_state["workspace_dino_defaults"] = workspace_defaults(config)
    st.title("Workflow")
    draft = active_draft(config)
    bank = st.session_state.get("drafts:" + str(config.paths.output_root), {})
    if len(bank) > 1:
        with st.expander("Switch draft · edits are kept in this session"):
            chosen = st.selectbox("Session drafts", list(bank), format_func=lambda i: bank[i]["name"] + " · " + i[:8], key="draft-switch")
            if st.button("Resume selected draft"):
                use_draft(config, bank[chosen], bank[chosen].get("last_page", "Start"), resume=True)
                rerun()
    if not draft:
        _start(config, None)
        return
    initial_locations = len(_field_locations(draft))
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
    from morphofeatures.workflow_catalog import PAGE_ACTIONS, workflow_steps

    steps = workflow_steps(draft["document"]["stages"])
    current = st.session_state.get("workflow_step", "Start")
    if current not in steps:
        current = st.session_state["workflow_step"] = steps[1]
    if current != "Start":
        with st.expander("Workflow prerequisites and expected outputs"):
            _preparation_checklist(
                workflow_requirements(
                    draft["document"]["stages"], allow_synthetic=draft["allow_synthetic"]
                )
            )
            from morphofeatures.workflow_outputs import show_expected_outputs

            show_expected_outputs(draft["document"]["stages"])
    if st.session_state.get("step_widget") != current:
        st.session_state["step_widget"] = current

    def step_changed():
        st.session_state["workflow_step"] = st.session_state["step_widget"]

    step = st.radio(
        "Workflow steps", steps, horizontal=True, key="step_widget", on_change=step_changed
    )
    draft["last_page"] = step
    draft["editor_epoch"] = st.session_state.get("editor_epoch", 0)
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
        _show_issues(
            draft, [i for i in _stored_issues(draft) if i["page"] == step], context="page-issues"
        )
        st.caption(
            "Check this page or Continue validates this step. You can still switch pages directly to edit linked settings; final Review checks the complete workflow."
        )
        checked = st.session_state.get("page-check:" + draft["id"])
        if checked == (step, draft["revision"], True):
            st.success(
                "This page passed its checks. Final Review also checks cross-stage inputs and execution settings."
            )
        group = PAGE_ACTIONS[step]
        indices = [i for i, s in enumerate(draft["document"]["stages"]) if s.get("action") in group]
        if len(indices) > 1:
            key = "workflow-stage:" + draft["id"] + ":" + step
            if st.session_state.get(key) not in indices:
                st.session_state.pop(key, None)
            indices = [st.selectbox("Stage to configure", indices, key=key,
                                   format_func=lambda i: f"{i + 1}. " + _stage_title(draft["document"]["stages"], i))]
        if not indices:
            st.info(
                f"{step} is not included. This workflow starts from existing inputs or stops before this stage."
            )
        for index in indices:
            action = draft["document"]["stages"][index]["action"]
            st.subheader(f"Stage {index + 1} · " + _stage_title(draft["document"]["stages"], index))
            stage = draft["document"]["stages"][index]
            links = [label for flag, label in (("from_preprocessing", "prepared data"), ("from_training", "trained model"),
                      ("from_extraction", "extracted vectors"), ("from_embedding", "named vectors"), ("from_analysis", "analysis")) if stage.get(flag)]
            st.caption("Input: " + ("linked " + ", ".join(links) if links else "existing inputs configured below") + ". Only the stages shown in the plan will run.")
            if action == "analysis_variant":
                field(draft, ["document", "stages", index, "analysis"], "Saved parent analysis", "")
                vp = ["document", "stages", index, "variant"]
                field(draft, vp + ["name"], "Map / clustering combination name", "New map")
                field(draft, vp + ["normalization"], "Embedding normalization", "standardize", options=["standardize", "l2", "none"])
                projection_settings_fields(draft, vp, include_variants=False)
                st.caption("Exact plotted cohort and saved classifier scores are retained. Only the map and clusters are recomputed.")
                continue
            if action == "dino_experiment":
                from morphofeatures.dino_experiment_ui import experiment_form

                experiment_form(draft, index, page=step)
                continue
            if action == "distill":
                from morphofeatures.distillation_ui import stage_form

                stage_form(draft, index)
                continue
            if action in {"fuse", "transfer"}:
                from morphofeatures.dino_experiment_ui import artifact_stage_form

                artifact_stage_form(draft, index)
                continue
            if action in {"context", "bilateral"}:
                from morphofeatures.spatial_probe_ui import stage_form

                stage_form(draft, index)
                continue
            if action == "export_bilateral":
                from morphofeatures.spatial_probe_controls import export_controls

                export_controls(draft, index)
                continue
            {
                "preprocess": _prepare,
                "train": _train,
                "extract": _extract,
                "analyze": _analyze,
                "compare": _analyze,
                "joint_analyze": _joint_analyze,
                "export_labels": _export_labels,
                "export_mobie": _export_mobie,
            }[action](draft, index)
        _stage_options(draft, step)
        if st.button("Check this page", key="check-page:" + step):
            _check_page(draft, step)
            rerun()
    yaml_editor(draft)
    if len(_field_locations(draft)) != initial_locations and any(
        issue["page"] == step for issue in _stored_issues(draft)
    ):
        rerun()
    previous, following = st.columns(2)
    position = steps.index(step)
    if position > 0 and previous.button("Back to " + steps[position - 1]):
        st.session_state["workflow_step"] = steps[position - 1]
        rerun()
    if position < len(steps) - 1 and following.button("Continue to " + steps[position + 1]):
        if _check_page(draft, step):
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
    elif path.suffix.lower() in {".yaml", ".yml"}:
        data = load_document(path)
        if not isinstance(data.get("data"), dict):
            raise ValueError("Choose a prepared-data configuration with a data section.")
        grouped = data.get("data", {}).get("source") == "grouped_crops"
        if grouped:
            kind = "dino"
        from morphofeatures.dino_defaults import workspace_defaults

        draft = new_draft(
            "dino" if kind == "dino" else "crops", source="Prepared data: " + str(path), model_defaults=workspace_defaults(config)
        )
        model = draft["document"]["stages"][0]["config"]
        model["data"] = data["data"]
        if data.get("inspection"):
            model["inspection"] = data["inspection"]
        if grouped:
            draft["document"]["stages"][0].setdefault("views", {}).update(
                mode="stored", normalization="unit", resize="letterbox", fractions=[0.5]
            )
        if kind != "dino" and data.get("mae"):
            model["mae"].update(data["mae"])
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
    from morphofeatures.artifact_availability import probe, selected_artifact
    from morphofeatures.result_catalog import run_artifacts, run_results

    declared = run_artifacts(record)
    artifacts = [p for p in declared if probe(p).state not in {"missing", "invalid", "unreadable"}]
    results, unavailable = run_results(record)
    if unavailable:
        with st.expander("Unavailable saved outputs"):
            st.dataframe([{"Path": item.path, "Status": item.state, "Reason": item.reason} for item in unavailable], hide_index=True)
            st.caption("Open moved results at their new location in Results. Other completed stages can still be reused.")

    def open_result(path):
        st.session_state["value:results:path"] = str(path)
        st.session_state["widget:results:path"] = str(path)
        st.session_state["results:registered"] = "Enter path"
        st.session_state["results:source"] = "Open a file"
        navigate("Results")

    # A completed report must be reachable independently of the advanced artifact
    # selector, whose first entry historically defaulted to job.yaml or a log.
    reports = [Path(row["path"]) for row in results if row["kind"] == "report" or
               (row["kind"] in {"vectors", "coordinates"} and selected_artifact(row["path"]).ready)]
    if reports:
        st.markdown("**Available results**")
        names = {Path(row["path"]): row["name"] for row in results}
        path = reports[0] if len(reports) == 1 else st.selectbox(
            "Completed result", reports, key="run-result-choice:" + record.id,
            format_func=lambda p: names[p] + " · " + p.parent.name,
        )
        title = names[path]
        if st.button(
            "Open results · " + title,
            key="run-result:" + record.id + ":" + str(path),
            disabled=not path.is_file(),
            help=f"Open {path}. Uses saved vectors, plots and scores; no extraction is repeated.",
        ):
            open_result(path)
        if not path.is_file():
            st.warning(f"Saved report is unavailable at {path}. Restore the file or open its new location in Results.")
    if artifacts:
        with st.expander("Other saved outputs and workflow inputs"):
            path = Path(st.selectbox("Continue from an output", artifacts))
            artifact_status = selected_artifact(path)
            if artifact_status.kind == "prepared" and artifact_status.ready and st.button(
                "Extract DINO features from these prepared objects"
            ):
                try:
                    continue_from_artifact(config, path, kind="dino")
                except Exception as error:
                    st.error(str(error))
            if artifact_status.ready and artifact_status.kind in {"prepared", "checkpoint", "vectors"}:
                if st.button("Use this output in a new workflow"):
                    try:
                        if not path.exists():
                            raise ValueError(f"Output no longer exists: {path}")
                        continue_from_artifact(config, path)
                    except Exception as error:
                        st.error(str(error))
            if path not in reports and probe(path).kind in {"report", "coordinates", "vectors"} and selected_artifact(path).ready and st.button(
                "Open this output in Results"
            ):
                open_result(path)
