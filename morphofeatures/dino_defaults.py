"""Shareable local DINO locations, selected by family and exact architecture.

These are UI defaults, not worker-side overrides. Concrete paths are copied into
new workflow stages so saved jobs remain independent of later workspace changes.
No directory scanning, code loading or weight downloads happen here.
"""

from copy import deepcopy
from pathlib import Path

from morphofeatures.dino import VARIANTS

DEFAULT_VARIANTS = {"dinov2": "dinov2_vits14", "dinov3": "dinov3_vits16"}


def normalize_defaults(values, base):
    if not isinstance(values, dict):
        raise ValueError("DINO defaults must be a mapping with dinov2/dinov3 entries.")
    result = deepcopy(values)
    for family in DEFAULT_VARIANTS:
        item = result.setdefault(family, {})
        if not isinstance(item, dict):
            raise ValueError(f"DINO defaults → {family}: use a YAML mapping.")
        variant = item.setdefault("variant", DEFAULT_VARIANTS[family])
        if variant not in VARIANTS[family]:
            raise ValueError(f"DINO defaults → {family}: select a supported default backbone.")

        def path(value, directory, field, family=family):
            if value is None or (isinstance(value, str) and not value.strip()):
                return None
            if not isinstance(value, str) or "://" in value:
                raise ValueError(
                    f"DINO defaults → {family} → {field}: enter a local path or leave blank."
                )
            return str((Path(directory) / Path(value.strip()).expanduser()).resolve())

        for key in ("model_repository", "weights_directory"):
            item[key] = path(item.get(key), base, key)
        checkpoints = item.setdefault("checkpoints", {})
        if not isinstance(checkpoints, dict) or any(v not in VARIANTS[family] for v in checkpoints):
            raise ValueError(
                f"DINO defaults → {family}: checkpoints must map supported backbone names to weight files."
            )
        item["checkpoints"] = {
            name: path(value, item["weights_directory"] or base, name)
            for name, value in checkpoints.items()
        }
    return result


def workspace_defaults(config):
    return normalize_defaults((config.raw or {}).get("dino", {}), config.paths.repo_root)


def backbone_defaults(defaults, family="dinov2", variant=None):
    item = (defaults or {}).get(family, {})
    variant = variant or item.get("variant", DEFAULT_VARIANTS[family])
    return {
        "model": family,
        "variant": variant,
        "model_repository": item.get("model_repository") or "",
        "checkpoint": item.get("checkpoints", {}).get(variant) or "",
    }


def defaults_editor(document, prefix):
    import streamlit as st

    from morphofeatures.workspace_ui import retained_input

    with st.expander("DINO backbone repositories and weights", expanded=True):
        st.caption(
            "Set local locations once for new DINO workflows. Save/download the workspace YAML to share them. Existing drafts and per-run paths remain editable. These fields do not download or load a model."
        )
        defaults = document.setdefault("dino", {})
        for family, default in DEFAULT_VARIANTS.items():
            title = "DINOv2" if family == "dinov2" else "DINOv3"
            item = defaults.setdefault(family, {})
            key = prefix + family + ":"
            with st.expander(title + " defaults", expanded=family == "dinov2"):
                item["model_repository"] = (
                    retained_input(
                        title + " backbone repository",
                        item.get("model_repository") or "",
                        key + "repository",
                        help="Official local code checkout containing hubconf.py. Select the matching DINO family. The same path must be accessible on your local/SLURM worker.",
                    )
                    or None
                )
                item["weights_directory"] = (
                    retained_input(
                        title + " weights directory (optional)",
                        item.get("weights_directory") or "",
                        key + "directory",
                        help="Base folder for relative weight filenames below. Absolute filenames also work. Files are selected explicitly by architecture; this folder is not searched automatically.",
                    )
                    or None
                )
                variant = retained_input(
                    "Default " + title + " backbone",
                    item.get("variant", default),
                    key + "variant",
                    kind="selectbox",
                    options=sorted(VARIANTS[family]),
                    help="Architecture to start with for a new backbone of this family. Changing it shows that architecture's saved weight file; other entries remain saved.",
                )
                item["variant"] = variant
                checkpoints = item.setdefault("checkpoints", {})
                checkpoints[variant] = (
                    retained_input(
                        title + " weights for " + variant,
                        checkpoints.get(variant) or "",
                        key + variant,
                        help="Matching pretrained checkpoint file (.pth/.pt), either absolute or relative to the weights directory. Blank means choose a file in each workflow. Register and non-register architectures need matching weights.",
                    )
                    or None
                )
                st.caption(
                    "To save several checkpoints, select each architecture and enter its file, then return to the architecture you want as the default."
                )


def backbone_fields(
    draft,
    path,
    *,
    show_family=True,
    checkpoint_label="Local DINO weights",
    repository_label="Official local backbone checkout",
):
    """Keep selections synchronized without replacing imported or manual paths."""
    import streamlit as st

    from morphofeatures.workflow_ui import _get, _replace_document, field, rerun, touch

    item = _get(draft, path)
    defaults = st.session_state.get("workspace_dino_defaults", {})
    key = f"dino-choice:{draft['id']}:{st.session_state.get('editor_epoch', 0)}:{path}"
    previous = st.session_state.get(key)
    if show_family:
        field(draft, path + ["model"], "DINO family", "dinov2", options=list(DEFAULT_VARIANTS))
    family = item["model"]
    if previous and previous[0] != family:
        item.update(backbone_defaults(defaults, family))
        _replace_document(draft, draft["document"])
        rerun()
    preset = backbone_defaults(defaults, family, item.get("variant"))
    field(
        draft,
        path + ["variant"],
        "Backbone variant",
        preset["variant"],
        options=sorted(VARIANTS[family]),
    )
    if previous and previous[1] != item["variant"]:
        # A checkpoint for the previous architecture must not follow this choice.
        item["checkpoint"] = backbone_defaults(defaults, family, item["variant"])["checkpoint"]
        _replace_document(draft, draft["document"])
        rerun()
    if previous is None:
        for name in ("model_repository", "checkpoint"):
            if not item.get(name) and preset[name]:
                item[name] = preset[name]
                touch(draft)
    st.session_state[key] = (family, item["variant"])
    field(
        draft,
        path + ["model_repository"],
        repository_label,
        "",
        help="Official local DINO checkout containing hubconf.py. Filled from Workspace settings when available; you can override it for this workflow.",
    )
    field(
        draft,
        path + ["checkpoint"],
        checkpoint_label,
        "",
        help="Pretrained weights matching this exact family and architecture. Workspace settings can remember one file per architecture. Selecting another architecture selects its saved file or leaves this blank.",
    )
    if st.button(
        "Use saved DINO locations for this backbone",
        key=key + ":apply",
        help="Replace this backbone's repository and weights with the current Workspace settings defaults. The selected family and architecture stay the same.",
    ):
        preset = backbone_defaults(defaults, family, item["variant"])
        item.update({k: preset[k] for k in ("model_repository", "checkpoint")})
        _replace_document(draft, draft["document"])
        rerun()
