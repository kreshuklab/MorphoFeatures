"""Editable, shareable workspace defaults, separate from scientific job settings."""

import os
import tempfile
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import yaml

from morphofeatures.config import config_from_mapping, repository_root
from morphofeatures.slurm import (
    cluster_profile_source,
    cluster_profile_values,
    cluster_profiles_from_mapping,
)

PATH_LABELS = {
    "repo_root": (
        "Repository directory",
        "MorphoFeatures checkout. Relative paths in this document are resolved from this directory.",
    ),
    "data_root": (
        "Data directory",
        "Shared data location used by tools that read workspace defaults.",
    ),
    "analysis_data": (
        "Analysis tables and embeddings directory",
        "Folder containing bundled or shared analysis tables, labels and embeddings.",
    ),
    "mobie_data": (
        "MoBIE tables directory",
        "Folder containing the MoBIE tables used by data inspection tools.",
    ),
    "output_root": (
        "Workspace output directory",
        "Drafts, experiment outputs and the run registry live here. Switching directories opens that workspace; existing files are not moved.",
    ),
}
DATA_LABELS = {
    "raw_volume": "Default raw volume",
    "cell_segmentation": "Default cell segmentation",
    "nucleus_segmentation": "Default nucleus segmentation",
    "cell_to_nucleus": "Default cell-to-nucleus mapping",
    "cell_table": "Default cell table",
    "nucleus_table": "Default nucleus table",
}


def settings_document(config):
    document = deepcopy(config.raw or {})
    document["paths"] = {
        **document.get("paths", {}),
        **{k: str(v) for k, v in asdict(config.paths).items()},
    }
    document.update(seed=config.seed, device=config.device)
    return document


def validate_settings(document, base):
    """Return effective paths, preserving additional deployment-specific fields."""
    if not isinstance(document, dict) or "stages" in document:
        raise ValueError(
            "Load workspace defaults here. A pipeline with stages belongs in Workflow → Start."
        )
    if not isinstance(document.get("paths", {}), dict):
        raise ValueError("Paths must be a YAML mapping of directory names to paths.")
    for key, value in document.get("paths", {}).items():
        if key in PATH_LABELS and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{PATH_LABELS[key][0]} must be a nonempty path.")
    seed = document.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Default random seed must be a whole number of at least 0.")
    if document.get("device", "auto") not in {"auto", "cpu", "cuda"}:
        raise ValueError("Default compute device must be auto, cpu or cuda.")
    for section in ("data", "slurm"):
        if not isinstance(document.get(section, {}), dict):
            raise ValueError(f"{section} must be a YAML mapping.")
    config = config_from_mapping(document, base=base, apply_environment=False)
    if config.paths.output_root.is_file():
        raise ValueError("Workspace output directory points to a file; choose a directory.")
    result = settings_document(config)
    if "dino" in result:
        from morphofeatures.dino_defaults import normalize_defaults

        result["dino"] = normalize_defaults(result["dino"], config.paths.repo_root)
    slurm = result.get("slurm", {})
    if cluster_profile_source(slurm) == "workspace":
        presets = slurm.get("resource_presets")
        try:
            parsed = cluster_profiles_from_mapping({"profiles": presets})
        except ValueError as error:
            raise ValueError(f"Cluster resource presets → {error}") from error
        slurm["resource_presets"] = {
            name: {**(presets[name] or {}), **cluster_profile_values(profile)}
            for name, profile in parsed.items()
        }
    # Absolute paths keep exports independent of where the recipient saves the YAML.
    for section, names in (("data", DATA_LABELS), ("slurm", ("profiles",))):
        for key in names:
            value = result.get(section, {}).get(key)
            if value:
                if not isinstance(value, str):
                    raise ValueError(f"{section}.{key} must be a path or null.")
                if "://" not in value:
                    path = Path(value).expanduser()
                    result[section][key] = str((config.paths.repo_root / path).resolve())
    return result


def save_settings(path, document, *, replace_existing=False):
    """Save a validated export; replacing an existing file is atomic and explicit."""
    path = Path(path).expanduser().resolve()
    document = validate_settings(document, path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(document, sort_keys=False)
    if not replace_existing:
        with path.open("x") as stream:
            stream.write(content)
        return path
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def settings_page(config, config_path):
    import streamlit as st

    from morphofeatures.cluster_presets_ui import cluster_presets_editor
    from morphofeatures.workspace_ui import retained_input

    st.title("Workspace settings")
    st.caption(
        "Edit shared locations and tool defaults, then apply them or share a workspace YAML. Existing workflow drafts keep their explicit scientific settings."
    )
    notice = st.session_state.pop("settings:notice", None)
    if notice:
        st.success(notice)
    key = "workspace:settings:document"
    if key not in st.session_state:
        st.session_state[key] = settings_document(config)
    with st.expander("Load shared workspace settings"):
        source = st.text_input(
            "Workspace defaults YAML", str(config_path), key="settings:import_path"
        )
        uploaded = st.file_uploader("Upload workspace defaults YAML", type=["yaml", "yml"])
        if st.button("Load workspace YAML"):
            try:
                path = (repository_root() / Path(source).expanduser()).resolve()
                content = (
                    uploaded.getvalue().decode("utf-8")
                    if uploaded is not None
                    else path.read_text()
                )
                document = validate_settings(
                    yaml.safe_load(content),
                    repository_root() if uploaded is not None else path.parent,
                )
                st.session_state[key] = document
                st.session_state["settings:epoch"] = st.session_state.get("settings:epoch", 0) + 1
                st.session_state["settings:notice"] = (
                    "Loaded into the editor. Apply workspace settings activates it; saving or downloading only exports it."
                )
                st.rerun()
            except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
                st.error(f"Load workspace YAML: {error}")

    document = deepcopy(st.session_state[key])
    prefix = "workspace-settings:" + str(st.session_state.get("settings:epoch", 0)) + ":"
    st.subheader("Workspace directories")
    for name, (label, help_text) in PATH_LABELS.items():
        document.setdefault("paths", {})[name] = retained_input(
            label, document["paths"][name], prefix + name, help=help_text
        )
    st.subheader("Defaults for interactive tools")
    document["seed"] = retained_input(
        "Default random seed",
        document.get("seed", 42),
        prefix + "seed",
        kind="number_input",
        min_value=0,
        step=1,
        help="Initial seed for interactive classification, projection and synthetic fixtures. Workflow stages retain their own seeds.",
    )
    document["device"] = retained_input(
        "Default compute device",
        document.get("device", "auto"),
        prefix + "device",
        kind="selectbox",
        options=["auto", "cpu", "cuda"],
        help="Shared device preference. Explicit model and workflow device settings take precedence.",
    )
    from morphofeatures.dino_defaults import defaults_editor

    defaults_editor(document, prefix)
    preset_errors = cluster_presets_editor(document, prefix)
    with st.expander("Default dataset locations"):
        data = document.setdefault("data", {})
        for name, label in DATA_LABELS.items():
            data[name] = (
                retained_input(
                    label,
                    data.get(name) or "",
                    prefix + name,
                    help="Optional shared source location. Blank leaves this unset. Select data for each scientific workflow inside Workflow.",
                )
                or None
            )
        data["coordinate_order"] = retained_input(
            "Default coordinate order",
            data.get("coordinate_order", "zyx"),
            prefix + "coordinate_order",
            help="Order used by shared dataset metadata, normally zyx. Each source must still declare its actual axes.",
        )
        data["resolution_unit"] = retained_input(
            "Default spatial unit",
            data.get("resolution_unit", "micrometer"),
            prefix + "resolution_unit",
            help="Unit recorded for the shared dataset metadata. This does not resample images or override source-specific physical metadata.",
        )
    st.session_state[key] = document
    if preset_errors:
        st.info(
            "Correct the cluster preset errors above before applying or exporting these workspace settings."
        )
        return
    try:
        effective = validate_settings(document, repository_root())
    except (ValueError, TypeError, OSError) as error:
        st.error(f"Workspace settings: {error}")
        return
    if st.button("Apply workspace settings", type="primary"):
        changed_workspace = effective["paths"]["output_root"] != str(config.paths.output_root)
        st.session_state["workspace_config_override"] = effective
        st.session_state["workspace_root"] = effective["paths"]["output_root"]
        st.session_state.pop("workspace_plan", None)
        for name in (
            "value:guided:cluster_profiles",
            "widget:guided:cluster_profiles",
            "value:guided:preset_source",
            "widget:guided:preset_source",
        ):
            st.session_state.pop(name, None)
        if changed_workspace:
            for name in (
                "navigation:current",
                "navigation:history",
                "runs:selected",
                "selected_run_id",
                "results:registered",
                "value:results:path",
                "widget:results:path",
            ):
                st.session_state.pop(name, None)
        st.session_state["settings:notice"] = (
            "Workspace settings applied. Save or download the YAML to reuse them after restarting the app."
        )
        st.rerun()
    st.caption("Active workspace: " + str(config.paths.output_root))
    content = yaml.safe_dump(effective, sort_keys=False)
    st.download_button(
        "Download workspace YAML", content, "workspace_settings.yaml", "application/yaml"
    )
    with st.expander("Save YAML on this machine / inspect export"):
        target = retained_input(
            "Save workspace YAML to",
            str(config.paths.output_root / ".morphofeatures/workspace_settings.yaml"),
            "settings:save_path",
            help="File on the app host. Save creates a reusable YAML; it does not submit a job or activate the settings.",
        )
        path = (repository_root() / Path(target).expanduser()).resolve()
        overwrite = (
            st.checkbox("Replace existing settings file", value=False) if path.exists() else False
        )
        if st.button("Save workspace YAML"):
            try:
                saved = save_settings(path, effective, replace_existing=overwrite)
                st.success("Saved workspace settings: " + str(saved))
            except (OSError, ValueError) as error:
                st.error(f"Save workspace YAML: {error}")
        st.code(content, language="yaml")
    st.caption(
        "Exports include resolved paths and any extra fields from the loaded defaults. Other users can load this YAML, adjust paths for their machine, then Apply workspace settings. Model checkpoints, images and credentials are not bundled."
    )
