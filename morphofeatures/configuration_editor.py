"""Lossless configuration documents, independent of Streamlit widget lifetimes."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


def parse_document(text):
    value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return value


def absolute_paths(values, base):
    """Resolve known file fields before moving a document into a run directory."""
    from morphofeatures.data.remote_n5 import is_remote_url

    values = deepcopy(values)
    fields = {
        "data": (
            "crops",
            "label_ids",
            "loss_masks",
            "preprocessing",
            "patches_container",
            "positions_container",
            "qc_raw_container",
            "foreground_mask_container",
            "masks_container",
            "manifest",
            "root",
            "raw",
            "object_table",
            "id_mapping",
        ),
        "training": ("resume_from",),
        "model": ("checkpoint",),
        "annotations": ("cell_table", "cell_to_nucleus", "symmetric_cells"),
    }
    for section, keys in fields.items():
        for key in keys:
            value = values.get(section, {}).get(key)
            if isinstance(value, str) and value and not is_remote_url(value):
                import os

                path = Path(os.path.expandvars(value)).expanduser()
                values[section][key] = str((base / path).resolve())
    mesh = values.get("inspection", {}).get("mesh", {})
    for key in ("segmentation", "object_index", "id_mapping", "mesh_directory", "mesh_table"):
        if isinstance(mesh.get(key), str) and mesh[key] and not is_remote_url(mesh[key]):
            import os

            mesh[key] = str((base / Path(os.path.expandvars(mesh[key])).expanduser()).resolve())
    for section in (
        values.get("inspection", {}).get("platybrowser", {}),
        mesh.get("remote_options", {}),
        values.get("data", {}).get("remote_options", {}),
    ):
        if section.get("cache_directory"):
            section["cache_directory"] = str(
                (base / Path(section["cache_directory"]).expanduser()).resolve()
            )
    paths = values.get("paths", {})
    repo = (base / Path(paths.get("repo_root", str(base)))).resolve()
    for key, value in list(paths.items()):
        if isinstance(value, str) and value:
            anchor = (
                repo if key in {"data_root", "output_root", "analysis_data", "mobie_data"} else base
            )
            paths[key] = str((anchor / Path(value).expanduser()).resolve())
    return values


def load_document(path, profile=None):
    path = Path(path).expanduser().resolve()
    values = parse_document(path.read_text())
    if (
        values.get("data", {}).get("source") == "n5_masked_patches"
        and values.get("schema") != "morphofeatures.inspection.v1"
    ):
        from morphofeatures.real_mae import resolve_real_mae_config

        resolved = dict(resolve_real_mae_config(path, profile=profile, require_data=False).values)
        # Keep alternative profiles for export; the resolved schema prevents them
        # from silently overriding edits when this document is submitted.
        if "profiles" in values:
            resolved["profiles"] = values["profiles"]
        values = resolved
    return absolute_paths(values, path.parent)


def remember(state, key, value):
    """Store document data under a non-widget key (widgets are removed on navigation)."""
    state[key] = deepcopy(value)


def training_form_defaults(values):
    """Expose omitted, supported options using the trainers' effective defaults.

    Crop MAE has an MLP decoder; attention heads and decoder depth belong only
    to grouped MAE. Imported values and extension fields always take precedence.
    """
    values = deepcopy(values)
    grouped = values.get("data", {}).get("source") == "n5_masked_patches"
    mae = values.setdefault("mae", {})
    defaults = {
        "input_channels": 1,
        "input_shape": [32, 32, 32] if grouped else [16, 16, 16],
        "embedding_dim": 80 if grouped else 128,
        "encoder_depth": 4 if grouped else 2,
        "encoder_heads": 8 if grouped else 4,
        "mask_ratio": 0.75,
    }
    for key, value in defaults.items():
        mae.setdefault(key, value)
    mae.setdefault("decoder_dim", mae["embedding_dim"])
    if grouped:
        defaults = {
            "reconstruction_shape": [8, 8, 8],
            "decoder_depth": 4,
            "decoder_heads": mae["encoder_heads"],
            "norm_pix_loss": True,
            "patch_encoder": "linear",
            "resnet_channels": [16, 32, 64],
            "resnet_blocks": [1, 1, 1],
        }
    else:
        defaults = {"patch_size": [4, 4, 4]}
    for key, value in defaults.items():
        mae.setdefault(key, value)
    training = values.setdefault("training", {})
    defaults = {
        "epochs": 2,
        "batch_size": 4 if grouped else 2,
        "learning_rate": 0.001,
        "weight_decay": 0.05 if grouped else 0.01,
    }
    if grouped:
        defaults.update(
            workers=0,
            pin_memory=False,
            scheduler="constant",
            warmup_epochs=0,
            min_learning_rate=1e-6,
            scheduler_step_size=25,
            scheduler_gamma=0.5,
            mixed_precision=False,
            checkpoint_interval=10,
            progress_interval_batches=25,
            resume_from=None,
        )
        early = training.setdefault("early_stopping", {})
        if isinstance(early, bool):
            early = training["early_stopping"] = {"enabled": early}
        if isinstance(early, dict):
            for key, value in dict(
                enabled=False, patience=10, min_delta=0.0, restore_best=True
            ).items():
                early.setdefault(key, value)
    else:
        defaults["validation_fraction"] = 0.25
    for key, value in defaults.items():
        training.setdefault(key, value)
    inference = values.setdefault("inference", {})
    inference.setdefault("batch_size", training["batch_size"] if grouped else 4)
    if grouped:
        inference.setdefault("workers", training["workers"])
        inference.setdefault("split", "all")
    values.setdefault("seed", 42)
    return values


def validate_training(values, base, *, require_data=True):
    """Validate settings before allocating an experiment or launching a process."""
    import math
    import tempfile

    values = absolute_paths(values, Path(base))
    if "mae" not in values:
        raise ValueError(
            "This training stage requires an MAE configuration with a mae section; use the legacy workflows for shape/texture models"
        )
    training = values.get("training", {})
    mae = values.get("mae", {})
    for key in ("epochs", "batch_size"):
        value = training.get(key, 2)
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    for key, default in (("learning_rate", 0.001), ("weight_decay", 0.01)):
        value = float(training.get(key, default))
        if not math.isfinite(value) or value < 0 or (key == "learning_rate" and value == 0):
            raise ValueError(
                f"training.{key} must be finite and {'positive' if key == 'learning_rate' else 'non-negative'}"
            )
    if values.get("device", "auto") not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if not 0 < float(mae.get("mask_ratio", 0.75)) < 1:
        raise ValueError("mae.mask_ratio must be between zero and one")
    dimensions = [("embedding_dim", "encoder_heads", 4)]
    if values.get("data", {}).get("source") == "n5_masked_patches":
        dimensions.append(("decoder_dim", "decoder_heads", mae.get("encoder_heads", 8)))
    elif mae.get("decoder_dim") is not None and int(mae["decoder_dim"]) < 1:
        raise ValueError("mae.decoder_dim must be positive")
    for dimension, heads, default in dimensions:
        if dimension in mae:
            if (
                int(mae.get(heads, default)) < 1
                or int(mae[dimension]) < 1
                or int(mae[dimension]) % int(mae.get(heads, default))
            ):
                raise ValueError(f"mae.{dimension} must be positive and divisible by mae.{heads}")
    if values.get("data", {}).get("source") == "n5_masked_patches":
        from morphofeatures.real_mae import resolve_real_mae_config

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "config.yaml"
            source.write_text(yaml.safe_dump(values))
            resolved = dict(resolve_real_mae_config(source, require_data=require_data).values)
        if "profiles" in values:
            resolved["profiles"] = values["profiles"]
        resume = resolved.get("training", {}).get("resume_from")
        if require_data and resume and not Path(resume).is_file():
            raise ValueError(f"training.resume_from checkpoint does not exist: {resume}")
        return resolved
    if training.get("resume_from"):
        raise ValueError(
            "Checkpoint resume is supported by grouped N5 MAE training; crop MAE does not support resume"
        )
    shape = mae.get("input_shape", [16, 16, 16])
    patch = mae.get("patch_size", [4, 4, 4])
    if (
        len(shape) != 3
        or len(patch) != 3
        or any(int(s) < 1 or int(p) < 1 or int(s) % int(p) for s, p in zip(shape, patch))
    ):
        raise ValueError(
            "mae.input_shape requires three positive dimensions divisible by mae.patch_size"
        )
    if not 0 <= float(training.get("validation_fraction", 0.25)) < 1:
        raise ValueError("training.validation_fraction must be in [0, 1)")
    for key in ("crops", "label_ids", "loss_masks"):
        value = values.get("data", {}).get(key)
        if require_data and isinstance(value, str) and not Path(value).exists():
            raise ValueError(f"data.{key} file does not exist: {value}")
    if values.get("data", {}).get("crops") and require_data:
        from morphofeatures.data.crop_storage import open_crop_array
        from morphofeatures.mae3d import _load_label_ids, _load_loss_masks

        with open_crop_array(values["data"]) as crops:
            if crops.ndim not in (4, 5) or not len(crops):
                raise ValueError("data.crops must contain nonempty (N,Z,Y,X) or (N,C,Z,Y,X) crops")
            if tuple(crops.shape[-3:]) != tuple(mae.get("input_shape", (16, 16, 16))):
                raise ValueError("mae.input_shape must match data.crops spatial shape")
            _load_label_ids(values, len(crops))
            _load_loss_masks(values, len(crops), crops.shape[-3:])
    return values
