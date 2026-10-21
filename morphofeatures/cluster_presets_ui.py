"""Workspace-owned SLURM presets, edited without submitting jobs or rewriting imports."""

from copy import deepcopy
from pathlib import Path

import streamlit as st
import yaml

from morphofeatures.config import repository_root
from morphofeatures.slurm import (
    ClusterProfile,
    cluster_profile_source,
    cluster_profile_values,
    cluster_profiles_from_mapping,
)
from morphofeatures.ui_settings import HELP
from morphofeatures.workspace_ui import retained_input


def import_presets(content):
    """Validate uploads before replacing any editor state; retain extra fields."""
    raw = yaml.safe_load(content)
    profiles = cluster_profiles_from_mapping(raw)
    values = raw.get("profiles", raw)
    return {
        name: {**(values[name] or {}), **cluster_profile_values(profile)}
        for name, profile in profiles.items()
    }


def cluster_presets_editor(document, prefix):
    """Mutate only the pending workspace document; Apply activates its presets."""
    slurm = document.setdefault("slurm", {})
    editor_key = prefix + f"presets:{st.session_state.get('cluster-presets:epoch', 0)}:"
    st.subheader("Cluster resource presets")
    source_labels = {
        "workspace": "Edit presets in this workspace",
        "file": "Use a presets YAML file",
    }
    slurm["preset_source"] = retained_input(
        "Cluster preset source",
        cluster_profile_source(slurm),
        editor_key + "source",
        kind="radio",
        options=list(source_labels),
        format_func=source_labels.get,
        help="Workspace presets are edited below and included in your workspace YAML. File mode loads presets from the selected path when you review a job.",
    )
    slurm["profiles"] = retained_input(
        "Cluster resource presets YAML",
        slurm.get("profiles") or "configs/slurm_profiles.example.yaml",
        prefix + "profiles",
        help="Existing preset YAML on the app host. In file mode this is the active source. Import presets into editor copies its contents into this workspace without modifying the original file. An uploaded YAML takes precedence for import.",
    )

    def replace_presets(values, selected=None):
        slurm["resource_presets"] = values
        slurm["preset_source"] = "workspace"
        st.session_state["workspace:settings:document"] = deepcopy(document)
        # Fresh widget keys avoid stale form values after imports and renames.
        st.session_state["cluster-presets:epoch"] = (
            st.session_state.get("cluster-presets:epoch", 0) + 1
        )
        if selected:
            st.session_state["cluster-presets:selected"] = selected
        st.rerun()

    with st.expander("Import existing cluster presets"):
        uploaded = st.file_uploader("Upload cluster resource presets YAML", type=["yaml", "yml"])
        st.caption(
            "Import replaces the presets in this editor. Apply workspace settings activates them; no job is submitted and the source file is kept."
        )
        if st.button("Import presets into editor"):
            try:
                base = Path(
                    document.get("paths", {}).get("repo_root", repository_root())
                ).expanduser()
                content = (
                    uploaded.getvalue().decode("utf-8")
                    if uploaded is not None
                    else (base / Path(slurm["profiles"]).expanduser()).read_text()
                )
                values = import_presets(content)
                replace_presets(values, next(iter(values)))
            except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
                st.error(f"Cluster resource presets → Import YAML: {error}")
    if slurm["preset_source"] == "file":
        st.caption(
            "Use Edit presets in this workspace to create presets here, or import your existing YAML. The file path remains available for future imports."
        )
        return False

    if not slurm.get("resource_presets"):
        slurm["resource_presets"] = {
            "my-cluster": cluster_profile_values(ClusterProfile("my-cluster", "compute"))
        }
    presets = slurm["resource_presets"]
    names = list(presets)
    if st.session_state.get("cluster-presets:selected") not in names:
        st.session_state["cluster-presets:selected"] = names[0]
    selected = retained_input(
        "Preset to edit",
        st.session_state["cluster-presets:selected"],
        editor_key + "selected",
        kind="selectbox",
        options=names,
    )
    st.session_state["cluster-presets:selected"] = selected
    values = presets[selected]
    st.caption(
        "Enter your cluster's partition/account and resource limits. These are reusable templates; each job receives a copy only when you choose Use cluster resource preset in Review & run."
    )
    key = editor_key + selected + ":"

    def control(name, label, default, *, kind="text_input", **kwargs):
        values[name] = retained_input(
            label,
            values.get(name, default),
            key + name,
            kind=kind,
            help=HELP.get(name),
            **kwargs,
        )

    for row in (
        (
            ("partition", "Partition", "compute"),
            ("time", "Time limit", "01:00:00"),
            ("memory", "Memory", "8G"),
        ),
        (("account", "Account (optional)", ""), ("qos", "QoS (optional)", "")),
    ):
        for column, (name, label, default) in zip(st.columns(len(row)), row):
            with column:
                if values.get(name) is None:
                    values[name] = default
                control(name, label, default)
                if name in {"account", "qos"}:
                    values[name] = values[name] or None
    for column, name, label, default, minimum in zip(
        st.columns(2), ("cpus", "gpus"), ("CPUs per task", "GPUs"), (4, 0), (1, 0)
    ):
        with column:
            control(name, label, default, kind="number_input", min_value=minimum, step=1)
    with st.expander("Preset mail notifications", expanded=True):
        mail = values.get("mail_types", [])
        if isinstance(mail, str):
            values["mail_types"] = [v.strip().upper() for v in mail.split(",") if v.strip()]
        control(
            "mail_types",
            "Notify me when",
            [],
            kind="multiselect",
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
        values["mail_user"] = values.get("mail_user") or ""
        control("mail_user", "Mail address (optional)", "")
        values["mail_user"] = values["mail_user"] or None
    with st.expander("Preset environment and advanced resources"):
        for column, name, label in zip(
            st.columns(3),
            ("nodes", "ntasks", "ntasks_per_node"),
            ("Nodes", "Total tasks", "Tasks per node"),
        ):
            with column:
                control(name, label, 1, kind="number_input", min_value=1, step=1)
        control(
            "gpu_directive",
            "GPU request syntax",
            "gpus",
            kind="selectbox",
            options=["gres", "gpus"],
        )
        values["python_executable"] = values.get("python_executable") or ""
        control("python_executable", "Worker Python interpreter (optional)", "")
        values["python_executable"] = values["python_executable"] or None
        setup = retained_input(
            "Module / environment setup",
            values["setup"]
            if isinstance(values.get("setup"), str)
            else yaml.safe_dump(values.get("setup", []), sort_keys=False),
            key + "setup",
            kind="text_area",
            help=HELP["setup"],
        )
        try:
            parsed = yaml.safe_load(setup)
            if not isinstance(parsed, list) or any(
                not isinstance(command, list) for command in parsed
            ):
                raise ValueError(
                    "Enter a list of argument lists, for example [[module, load, CUDA/12.8]]."
                )
            values["setup"] = parsed
        except (ValueError, yaml.YAMLError) as error:
            # Keep invalid text in the pending document, so switching presets cannot
            # silently allow Apply/Save to export older, apparently valid setup.
            values["setup"] = setup
            st.error(f"Preset {selected} → Module / environment setup: {error}")
        control("cpu_thread_env", "Bind OMP/MKL threads to CPUs per task", False, kind="checkbox")
        control(
            "local_runtime_directories",
            "Run-local cache directories",
            [],
            kind="multiselect",
            options=["wandb", "matplotlib"],
        )
        control(
            "log_job_context",
            "Print job ID, host, working directory, and command",
            False,
            kind="checkbox",
        )

    with st.expander("Add, copy or rename presets"):
        new_name = st.text_input(
            "New preset name",
            "new-preset",
            key=key + "new_name",
            help="A unique short name using letters, digits, dots, underscores or hyphens, for example gpu-long.",
        )
        buttons = st.columns(3)
        operations = {
            name: button.button(name)
            for button, name in zip(buttons, ("Add preset", "Duplicate preset", "Rename preset"))
        }
        operation = next((name for name, clicked in operations.items() if clicked), None)
        if operation:
            try:
                if new_name in presets:
                    raise ValueError("A preset with this name already exists.")
                cluster_profiles_from_mapping({"profiles": {new_name: {}}})
                replacement = deepcopy(presets)
                replacement[new_name] = (
                    cluster_profile_values(ClusterProfile(new_name, "compute"))
                    if operation == "Add preset"
                    else deepcopy(values)
                )
                if operation == "Rename preset":
                    replacement.pop(selected)
                replace_presets(replacement, new_name)
            except (ValueError, TypeError) as error:
                st.error(f"Cluster resource presets → New preset name: {error}")
        if st.button(
            "Remove selected preset",
            disabled=len(presets) < 2,
            help="Removes this preset from the editor. Existing jobs and the imported YAML are unchanged. Keep at least one preset.",
        ):
            replacement = deepcopy(presets)
            replacement.pop(selected)
            replace_presets(replacement, next(iter(replacement)))
    try:
        cluster_profiles_from_mapping({"profiles": presets})
    except (ValueError, TypeError) as error:
        st.error(f"Cluster resource presets → {error}")
        return True
    st.download_button(
        "Download cluster presets YAML",
        yaml.safe_dump({"profiles": presets}, sort_keys=False),
        "slurm_profiles.yaml",
        "application/yaml",
    )
    st.caption(
        "These presets are also included in Download / Save workspace YAML. Apply workspace settings makes them available to job forms in this session."
    )
    return False
