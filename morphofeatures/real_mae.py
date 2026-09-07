"""Configured real-data MAE workflow for grouped, masked N5 patches.

This is intentionally an adapter around :mod:`morphofeatures.mae3d`.  The
scientific model remains in that module; this one resolves configuration,
builds leakage-safe lazy datasets, manages resume/checkpoint metadata, and
aggregates patch encodings to one row per biological ``label_id``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from morphofeatures.artifacts import write_json_atomic
from morphofeatures.data.io import export_embeddings
from morphofeatures.data.n5 import (
    N5GroupedPatchDataset,
    N5MaskedPatchDataset,
    PatchIndex,
    deterministic_label_split,
    discover_n5_metadata,
    load_patch_index,
    preprocess_masked_patch,
    select_patch_indices,
)
from morphofeatures.mae_contract import GROUPED_PATCH_MAE_ARCHITECTURE_VERSION
from morphofeatures.metrics import MetricWriter, configured_metrics_path
from morphofeatures.training_runtime import resolve_device, save_checkpoint

REAL_N5_SOURCE = "n5_masked_patches"


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = {key: value for key, value in base.items()}
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expand_strings(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_strings(item) for key, item in value.items()}
    return value


def _path(value: Any, base: Path, name: str) -> Path:
    if value in {None, ""}:
        raise ValueError(f"Configuration requires {name}")
    text = str(value)
    if "${" in text:
        raise ValueError(f"Environment variable in {name} is not set: {text}")
    candidate = Path(text)
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def _triple(value: Sequence[Any], name: str, numeric=float):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three z, y, x values")
    result = tuple(numeric(item) for item in value)
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain positive values")
    return result


@dataclass(frozen=True)
class ResolvedRealMAEConfig:
    """One validated configuration shared by YAML, notebook, CLI, and SLURM."""

    values: Mapping[str, Any]
    source_path: Path
    profile: str

    @property
    def data(self) -> Mapping[str, Any]:
        return self.values["data"]

    @property
    def run_dir(self) -> Path:
        return Path(self.values["paths"]["run_dir"])

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.values["paths"]["checkpoint"])

    @property
    def metrics_path(self) -> Path:
        return Path(self.values["paths"]["metrics"])

    @property
    def embedding_path(self) -> Path:
        return Path(self.values["paths"]["embedding"])

    def save(self, path: Path | None = None) -> Path:
        destination = Path(path or self.values["paths"]["resolved_config"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = yaml.safe_dump(dict(self.values), sort_keys=False)
        if destination.exists():
            if destination.read_text(encoding="utf-8") == encoded:
                return destination
            digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
            destination = destination.with_name(f"{destination.stem}-{digest}{destination.suffix}")
            if destination.exists():
                if destination.read_text(encoding="utf-8") != encoded:
                    raise ValueError(f"Configuration snapshot hash collision: {destination}")
                return destination
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(str(temporary), str(destination))
        return destination


@dataclass(frozen=True)
class PreparedRealMAEData:
    config: ResolvedRealMAEConfig
    index: PatchIndex
    label_splits: Mapping[str, np.ndarray]
    patch_indices: Mapping[str, np.ndarray]
    patch_metadata: Mapping[str, Any]
    positions_metadata: Mapping[str, Any]

    def dataset(self, split: str, mode: str = "train", augment: bool | None = None):
        if split not in self.patch_indices:
            raise ValueError(f"Unknown split: {split}")
        data = self.config.data
        labels = (
            np.sort(np.concatenate(list(self.label_splits.values())))
            if split == "all"
            else self.label_splits[split]
        )
        return N5GroupedPatchDataset(
            Path(data["patches_container"]),
            str(data["patches_key"]),
            self.index,
            labels,
            group_size=int(data["group_size"]),
            position_stride_zyx=data["position_stride_zyx"],
            normalization=str(data["normalization"]),
            mask_mode=str(data["mask_mode"]),
            min_foreground_fraction=float(data["min_foreground_fraction"]),
            mode=mode,
        )


def resolve_real_mae_config(
    path: Path,
    *,
    profile: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    require_data: bool = True,
) -> ResolvedRealMAEConfig:
    """Resolve a YAML profile and explicit notebook overrides into one snapshot."""

    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError("Real MAE configuration root must be a YAML mapping")
    already_resolved = raw.get("config_schema") in {
        "morphofeatures.real_mae.v1",
        "morphofeatures.real_mae.v2",
    }
    profiles = raw.pop("profiles", {})
    selected_profile = str(
        profile or raw.pop("active_profile", None) or raw.get("resolved_profile", "quick")
    )
    if selected_profile and not already_resolved:
        if selected_profile not in profiles:
            raise ValueError(
                f"Unknown execution profile {selected_profile!r}; available: {sorted(profiles)}"
            )
        raw = _deep_merge(raw, profiles[selected_profile])
    if overrides:
        raw = _deep_merge(raw, overrides)
    raw = _expand_strings(raw)
    base = source.parent
    data = raw.setdefault("data", {})
    if data.get("source") != REAL_N5_SOURCE:
        raise ValueError(f"data.source must be {REAL_N5_SOURCE!r}")
    for key in ("patches_container", "positions_container"):
        data[key] = str(_path(data.get(key), base, f"data.{key}"))
    if data.get("qc_raw_container"):
        data["qc_raw_container"] = str(
            _path(data["qc_raw_container"], base, "data.qc_raw_container")
        )
        data.setdefault("qc_raw_key", "volumes/raw/s1")
        _triple(
            data.setdefault("position_radius_zyx", [4, 4, 4]),
            "data.position_radius_zyx",
            int,
        )
        _triple(
            data.setdefault("position_to_raw_scale_zyx", [4, 4, 4]),
            "data.position_to_raw_scale_zyx",
            int,
        )
    data.setdefault("patches_key", "patches")
    data.setdefault("positions_key", "positions")
    data.setdefault("ids_key", "ids")
    data.setdefault("axes", "zyx")
    data.setdefault("modality", "EM")
    data.setdefault("biological_unit", "nucleus")
    data.setdefault("mask_mode", "nonzero")
    data.setdefault("normalization", "dtype")
    data.setdefault("min_foreground_fraction", 0.01)
    data.setdefault("group_size", 200)
    data.setdefault("position_stride_zyx", [8, 8, 8])
    if data["axes"] != "zyx":
        raise ValueError("Maintained MAE arrays use explicit z, y, x axis order")
    resolution = _triple(data.get("resolution_zyx_um", ()), "data.resolution_zyx_um")
    position_resolution = _triple(
        data.get("position_resolution_zyx_um", ()), "data.position_resolution_zyx_um"
    )
    ratios = np.asarray(position_resolution) / np.asarray(resolution)
    if not np.allclose(ratios, np.rint(ratios)):
        raise ValueError("Position and patch resolutions must have integer scale factors")
    patch_shape = _triple(data.get("patch_shape_zyx", ()), "data.patch_shape_zyx", int)
    if data["modality"] != "EM":
        raise ValueError("This real-data workflow is configured and validated for EM patches")
    if data["biological_unit"] != "nucleus":
        raise ValueError("The audited patch store represents nucleus-derived samples")
    if data["mask_mode"] not in {"nonzero", "all"}:
        raise ValueError("data.mask_mode must be nonzero or all")
    if data["normalization"] not in {
        "dtype",
        "foreground_percentile",
        "foreground_zscore",
        "none",
    }:
        raise ValueError("Unsupported data.normalization")
    if not 0 <= float(data["min_foreground_fraction"]) <= 1:
        raise ValueError("data.min_foreground_fraction must be between zero and one")
    if int(data["group_size"]) < 2:
        raise ValueError("data.group_size must be at least two")
    _triple(data["position_stride_zyx"], "data.position_stride_zyx")

    split = raw.setdefault("split", {})
    split.setdefault("train_fraction", 0.8)
    split.setdefault("validation_fraction", 0.1)
    split.setdefault("test_fraction", 0.1)
    fractions = [
        float(split["train_fraction"]),
        float(split["validation_fraction"]),
        float(split["test_fraction"]),
    ]
    if any(item < 0 for item in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("Train, validation, and test fractions must sum to one")
    if split.get("max_labels") is not None and int(split["max_labels"]) < 3:
        raise ValueError("split.max_labels must be at least three or null")

    mae = raw.setdefault("mae", {})
    mae.setdefault("architecture_version", GROUPED_PATCH_MAE_ARCHITECTURE_VERSION)
    mae.setdefault("input_channels", 1)
    mae.setdefault("input_shape", list(patch_shape))
    mae.setdefault("reconstruction_shape", [8, 8, 8])
    mae.setdefault("embedding_dim", 80)
    mae.setdefault("encoder_depth", 4)
    mae.setdefault("encoder_heads", 8)
    mae.setdefault("decoder_dim", mae["embedding_dim"])
    mae.setdefault("decoder_depth", 4)
    mae.setdefault("decoder_heads", mae["encoder_heads"])
    mae.setdefault("norm_pix_loss", True)
    mae.setdefault("mask_ratio", 0.75)
    mae.setdefault("patch_encoder", "linear")
    mae.setdefault("resnet_channels", [16, 32, 64])
    mae.setdefault("resnet_blocks", [1, 1, 1])
    if str(mae["architecture_version"]) != GROUPED_PATCH_MAE_ARCHITECTURE_VERSION:
        raise ValueError(
            "Real N5 nucleus patches require mae.architecture_version "
            f"{GROUPED_PATCH_MAE_ARCHITECTURE_VERSION!r}. The earlier single-patch "
            "objective used the wrong biological sampling unit and must be retrained."
        )
    if tuple(_triple(mae["input_shape"], "mae.input_shape", int)) != patch_shape:
        raise ValueError("mae.input_shape must match data.patch_shape_zyx")
    reconstruction_shape = _triple(
        mae["reconstruction_shape"], "mae.reconstruction_shape", int
    )
    if any(output > source for source, output in zip(patch_shape, reconstruction_shape)):
        raise ValueError("mae.reconstruction_shape cannot exceed the stored patch shape")
    if int(mae["embedding_dim"]) <= 0 or int(mae["encoder_depth"]) <= 0:
        raise ValueError("MAE embedding dimension and depth must be positive")
    if int(mae["embedding_dim"]) % int(mae["encoder_heads"]):
        raise ValueError("mae.embedding_dim must be divisible by mae.encoder_heads")
    if int(mae["decoder_dim"]) % int(mae["decoder_heads"]):
        raise ValueError("mae.decoder_dim must be divisible by mae.decoder_heads")
    if not 0 < float(mae["mask_ratio"]) < 1:
        raise ValueError("mae.mask_ratio must be between zero and one")
    if mae["patch_encoder"] not in {"linear", "resnet3d"}:
        raise ValueError("mae.patch_encoder must be linear or resnet3d")
    for key in ("resnet_channels", "resnet_blocks"):
        if not isinstance(mae[key], (list, tuple)) or not mae[key]:
            raise ValueError(f"mae.{key} must be a non-empty sequence")
        if any(int(value) <= 0 for value in mae[key]):
            raise ValueError(f"mae.{key} must contain positive integers")
    if len(mae["resnet_channels"]) != len(mae["resnet_blocks"]):
        raise ValueError("mae.resnet_channels and mae.resnet_blocks must have equal length")

    training = raw.setdefault("training", {})
    training.setdefault("epochs", 2)
    training.setdefault("batch_size", 4)
    training.setdefault("workers", 0)
    training.setdefault("learning_rate", 1e-3)
    training.setdefault("weight_decay", 0.05)
    training.setdefault("scheduler", "constant")
    training.setdefault("warmup_epochs", 0)
    training.setdefault("min_learning_rate", 1e-6)
    training.setdefault("scheduler_step_size", 25)
    training.setdefault("scheduler_gamma", 0.5)
    training.setdefault("mixed_precision", False)
    training.setdefault("checkpoint_interval", 10)
    training.setdefault("progress_interval_batches", 25)
    training.setdefault("resume_from", None)
    for key in ("epochs", "batch_size"):
        if int(training[key]) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if int(training["workers"]) < 0:
        raise ValueError("training.workers must be non-negative")
    if int(training["checkpoint_interval"]) < 0:
        raise ValueError("training.checkpoint_interval must be non-negative")
    if int(training["progress_interval_batches"]) < 0:
        raise ValueError("training.progress_interval_batches must be non-negative")
    if float(training["learning_rate"]) <= 0 or float(training["weight_decay"]) < 0:
        raise ValueError("Learning rate must be positive and weight decay non-negative")
    if training["scheduler"] not in {"constant", "cosine", "step"}:
        raise ValueError("training.scheduler must be constant, cosine, or step")
    if not 0 <= int(training["warmup_epochs"]) < int(training["epochs"]):
        raise ValueError("training.warmup_epochs must be non-negative and less than epochs")
    if float(training["min_learning_rate"]) < 0:
        raise ValueError("training.min_learning_rate must be non-negative")
    if int(training["scheduler_step_size"]) <= 0:
        raise ValueError("training.scheduler_step_size must be positive")
    if not 0 < float(training["scheduler_gamma"]) <= 1:
        raise ValueError("training.scheduler_gamma must be in (0, 1]")
    early_stopping = training.setdefault("early_stopping", {})
    if isinstance(early_stopping, bool):
        early_stopping = {"enabled": early_stopping}
        training["early_stopping"] = early_stopping
    if not isinstance(early_stopping, dict):
        raise ValueError("training.early_stopping must be a mapping or boolean")
    early_stopping.setdefault("enabled", False)
    early_stopping.setdefault("patience", 10)
    early_stopping.setdefault("min_delta", 0.0)
    early_stopping.setdefault("restore_best", True)
    for key in ("enabled", "restore_best"):
        if not isinstance(early_stopping[key], bool):
            raise ValueError(f"training.early_stopping.{key} must be true or false")
    if int(early_stopping["patience"]) <= 0:
        raise ValueError("training.early_stopping.patience must be positive")
    if float(early_stopping["min_delta"]) < 0:
        raise ValueError("training.early_stopping.min_delta must be non-negative")
    if training.get("resume_from"):
        training["resume_from"] = str(_path(training["resume_from"], base, "training.resume_from"))

    inference = raw.setdefault("inference", {})
    inference.setdefault("batch_size", training["batch_size"])
    inference.setdefault("workers", training["workers"])
    inference.setdefault("split", "all")
    if inference["split"] not in {"train", "validation", "test", "all"}:
        raise ValueError("inference.split must be train, validation, test, or all")

    annotations = raw.setdefault("annotations", {})
    for key in ("cell_table", "cell_to_nucleus", "symmetric_cells"):
        if annotations.get(key):
            annotations[key] = str(_path(annotations[key], base, f"annotations.{key}"))

    paths = raw.setdefault("paths", {})
    run_dir = _path(paths.get("run_dir", "../outputs/real_mae"), base, "paths.run_dir")
    paths["run_dir"] = str(run_dir)
    defaults = {
        "resolved_config": run_dir / "resolved_config.yaml",
        "metrics": run_dir / "metrics.jsonl",
        "checkpoint": run_dir / "checkpoints" / "checkpoint-last.pt",
        "best_checkpoint": run_dir / "checkpoints" / "checkpoint-best.pt",
        "embedding": run_dir / "embeddings" / "nucleus_texture_mae.npy",
        "split_manifest": run_dir / "split_manifest.tsv",
        "run_metadata": run_dir / "run_metadata.json",
    }
    for key, default in defaults.items():
        paths[key] = str(
            _path(paths[key], base, f"paths.{key}") if paths.get(key) else default.resolve()
        )
    if early_stopping["enabled"] and paths["best_checkpoint"] == paths["checkpoint"]:
        raise ValueError("paths.best_checkpoint must differ from paths.checkpoint")
    raw.setdefault("seed", 42)
    raw.setdefault("device", "auto")
    augmentation = raw.setdefault("augmentation", {"random_flip": False, "random_rot90": False})
    if augmentation.get("random_flip") or augmentation.get("random_rot90"):
        raise ValueError(
            "Grouped real-data MAE augmentation is disabled until patch pixels and physical "
            "z,y,x coordinates can be transformed together"
        )
    raw.setdefault("runtime", {})["metrics_path"] = paths["metrics"]
    raw["resolved_profile"] = selected_profile
    raw["config_schema"] = "morphofeatures.real_mae.v2"

    if require_data:
        for key in ("patches_container", "positions_container"):
            if not Path(data[key]).is_dir():
                raise ValueError(f"Configured N5 container does not exist: {data[key]}")
        if data.get("qc_raw_container") and not Path(data["qc_raw_container"]).is_dir():
            raise ValueError(
                f"Configured optional raw N5 container does not exist: {data['qc_raw_container']}"
            )
        for value in annotations.values():
            if value and not Path(value).is_file():
                raise ValueError(f"Configured optional annotation table does not exist: {value}")
    return ResolvedRealMAEConfig(raw, source, selected_profile)


def prepare_real_mae_data(
    config: ResolvedRealMAEConfig,
    progress: Callable[..., None] | None = None,
) -> PreparedRealMAEData:
    """Validate metadata, IDs, coordinates, and leakage-safe patch selections."""

    def emit(event: str, **values: Any) -> None:
        if progress is not None:
            progress(event, **values)

    data = config.data
    stage_started = time.perf_counter()
    patch_metadata = discover_n5_metadata(
        Path(data["patches_container"]), {str(data["patches_key"]): "nzyx"}
    )
    positions_metadata = discover_n5_metadata(
        Path(data["positions_container"]),
        {str(data["positions_key"]): "nc", str(data["ids_key"]): "n"},
    )
    patch_rows = [item for item in patch_metadata if item.key == data["patches_key"]]
    position_rows = [item for item in positions_metadata if item.key == data["positions_key"]]
    id_rows = [item for item in positions_metadata if item.key == data["ids_key"]]
    if len(patch_rows) != 1 or len(position_rows) != 1 or len(id_rows) != 1:
        raise ValueError("Configured patch, position, or ID dataset key was not discovered")
    patch_row, position_row = patch_rows[0], position_rows[0]
    expected_shape = tuple(int(item) for item in data["patch_shape_zyx"])
    if patch_row.shape[1:] != expected_shape or patch_row.dtype != "uint8":
        raise ValueError(
            f"Expected uint8 patch shape (n, {expected_shape}), observed {patch_row.shape} "
            f"and {patch_row.dtype}"
        )
    if position_row.shape != (patch_row.shape[0], 4) or position_row.dtype != "int64":
        raise ValueError("positions must be int64 with one label_id,z,y,x row per patch")
    emit(
        "metadata_validated",
        stage="preprocessing",
        patch_count=int(patch_row.shape[0]),
        patch_shape_zyx=list(patch_row.shape[1:]),
        patch_dtype=patch_row.dtype,
        duration_seconds=time.perf_counter() - stage_started,
    )
    stage_started = time.perf_counter()
    emit(
        "index_loading_started",
        stage="preprocessing",
        patch_count=int(patch_row.shape[0]),
    )
    index = load_patch_index(
        Path(data["positions_container"]),
        str(data["positions_key"]),
        str(data["ids_key"]),
        expected_patch_count=patch_row.shape[0],
    )
    emit(
        "index_loaded",
        stage="preprocessing",
        patch_count=int(len(index.labels)),
        label_count=int(len(index.unique_label_ids)),
        duration_seconds=time.perf_counter() - stage_started,
    )
    stage_started = time.perf_counter()
    split = config.values["split"]
    labels = deterministic_label_split(
        index.unique_label_ids,
        (
            split["train_fraction"],
            split["validation_fraction"],
            split["test_fraction"],
        ),
        seed=int(config.values["seed"]),
        max_labels=split.get("max_labels"),
    )
    emit(
        "splits_created",
        stage="preprocessing",
        train_labels=int(len(labels["train"])),
        validation_labels=int(len(labels["validation"])),
        test_labels=int(len(labels["test"])),
        duration_seconds=time.perf_counter() - stage_started,
    )
    stage_started = time.perf_counter()
    patch_indices = {
        name: select_patch_indices(
            index,
            ids,
            patches_per_label=int(data["group_size"]),
            seed=int(config.values["seed"]) + offset,
        )
        for offset, (name, ids) in enumerate(labels.items())
    }
    all_labels = np.concatenate(list(labels.values()))
    patch_indices["all"] = select_patch_indices(
        index,
        np.sort(all_labels),
        patches_per_label=int(data["group_size"]),
        seed=int(config.values["seed"]) + 100,
    )
    emit(
        "patch_selection_completed",
        stage="preprocessing",
        train_patches=int(len(patch_indices["train"])),
        validation_patches=int(len(patch_indices["validation"])),
        test_patches=int(len(patch_indices["test"])),
        all_patches=int(len(patch_indices["all"])),
        duration_seconds=time.perf_counter() - stage_started,
    )
    return PreparedRealMAEData(
        config,
        index,
        labels,
        patch_indices,
        patch_row.to_dict(),
        position_row.to_dict(),
    )


def save_split_manifest(prepared: PreparedRealMAEData, path: Path | None = None) -> Path:
    import pandas as pd

    rows = []
    group_size = int(prepared.config.data["group_size"])
    for split, label_ids in prepared.label_splits.items():
        for label_id in label_ids:
            location = int(np.searchsorted(prepared.index.unique_label_ids, label_id))
            available = int(prepared.index.counts[location])
            rows.append(
                {
                    "label_id": int(label_id),
                    "split": split,
                    "available_patches": available,
                    "selected_training_patches": min(available, group_size),
                }
            )
    destination = Path(path or prepared.config.values["paths"]["split_manifest"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("label_id").to_csv(destination, sep="\t", index=False)
    return destination


def summarize_patch_qc(
    dataset: N5MaskedPatchDataset | N5GroupedPatchDataset, max_samples: int = 64
):
    """Read a bounded, evenly spaced subset and return one QC row per actual patch."""

    if len(dataset) == 0:
        return []
    selected = np.linspace(0, len(dataset) - 1, min(int(max_samples), len(dataset)), dtype=int)
    return [dataset.read_sample(int(index))[2].to_dict() for index in selected]


def read_patch_qc_views(
    prepared: PreparedRealMAEData,
    patch_indices: Sequence[int],
) -> Mapping[str, Any]:
    """Read bounded stored/model/raw views using the audited center transform.

    The unmasked raw source is optional and used only for quality control. Training
    always reads the immutable masked patch store. A nonzero masked voxel must equal
    its raw-source voxel exactly; mismatches are returned for explicit inspection.
    """

    try:
        import z5py
    except ImportError as error:  # pragma: no cover - optional legacy N5 reader
        raise RuntimeError("N5 quality-control views require z5py") from error

    data = prepared.config.data
    indices = np.asarray(patch_indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(prepared.index.labels)):
        raise ValueError("patch_indices must be valid rows in the patch index")
    patch_store = z5py.File(str(data["patches_container"]), "r")
    patches = patch_store[str(data["patches_key"])]
    raw_container = data.get("qc_raw_container")
    raw_store = z5py.File(str(raw_container), "r") if raw_container else None
    raw_dataset = raw_store[str(data["qc_raw_key"])] if raw_store is not None else None
    radius = np.asarray(data.get("position_radius_zyx", [4, 4, 4]), dtype=np.int64)
    scale = np.asarray(data.get("position_to_raw_scale_zyx", [4, 4, 4]), dtype=np.int64)

    raw_views = []
    masked_views = []
    model_inputs = []
    masks = []
    matches = []
    for patch_index in indices:
        masked = np.asarray(patches[int(patch_index)])
        model_input, loss_mask, _ = preprocess_masked_patch(
            masked,
            normalization=str(data["normalization"]),
            mask_mode=str(data["mask_mode"]),
        )
        if raw_dataset is not None:
            center = prepared.index.positions_zyx[int(patch_index)]
            bounds = tuple(
                slice(int((coordinate - rad) * factor), int((coordinate + rad) * factor))
                for coordinate, rad, factor in zip(center, radius, scale)
            )
            raw = np.asarray(raw_dataset[bounds])
            if raw.shape != masked.shape:
                raise ValueError(
                    f"Raw QC crop shape {raw.shape} does not match masked patch {masked.shape}"
                )
            nonzero = masked != 0
            matches.append(bool(np.array_equal(masked[nonzero], raw[nonzero])))
            raw_views.append(raw)
        masked_views.append(masked)
        model_inputs.append(model_input)
        masks.append(np.ones_like(model_input) if loss_mask is None else loss_mask)
    return {
        "patch_indices": indices,
        "label_ids": prepared.index.labels[indices].astype(np.int64, copy=False),
        "positions_zyx": prepared.index.positions_zyx[indices].astype(np.int64, copy=False),
        "raw": None if raw_dataset is None else np.stack(raw_views),
        "masked": np.stack(masked_views),
        "model_inputs": np.stack(model_inputs),
        "loss_masks": np.stack(masks),
        "masked_nonzero_matches_raw": None
        if raw_dataset is None
        else np.asarray(matches, dtype=bool),
    }


def _data_loader(dataset, config: Mapping[str, Any], *, shuffle: bool, seed: int):
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config.get("workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)),
        generator=generator,
        drop_last=False,
    )


def _batch_output(model, batch, device, mask_ratio, amp_enabled):
    import torch

    if not isinstance(batch, (tuple, list)) or len(batch) != 3:
        raise ValueError("Grouped real MAE batches must contain patches, positions, and valid mask")
    patches, positions, valid = [value.to(device, non_blocking=True) for value in batch]
    context = torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp_enabled,
    )
    with context:
        output = model(patches, positions, valid, mask_ratio=mask_ratio)
    return output


def _batch_loss(model, batch, device, mask_ratio, amp_enabled):
    """Backward-compatible scalar-loss helper used by older callers."""

    return _batch_output(model, batch, device, mask_ratio, amp_enabled).loss


def _write_batch_progress(
    writer: MetricWriter,
    *,
    phase: str,
    epoch: int,
    batch_number: int,
    total_batches: int,
    epoch_started: float,
    losses: Sequence[float],
    groups_completed: int,
    patches_completed: int,
    step: int,
    interval: int,
) -> None:
    """Write bounded live progress without changing the scientific loss series."""

    if interval <= 0 or not (
        batch_number == 1 or batch_number % interval == 0 or batch_number == total_batches
    ):
        return
    elapsed = max(time.perf_counter() - epoch_started, 1e-9)
    batches_per_second = batch_number / elapsed
    remaining = max(total_batches - batch_number, 0)
    writer.write(
        "batch_progress",
        stage="training",
        phase=phase,
        epoch=epoch,
        batch=batch_number,
        total_batches=total_batches,
        progress_fraction=batch_number / max(total_batches, 1),
        running_loss=float(np.mean(losses)),
        groups_completed=groups_completed,
        patches_completed=patches_completed,
        batches_per_second=batches_per_second,
        groups_per_second=groups_completed / elapsed,
        patches_per_second=patches_completed / elapsed,
        elapsed_seconds=elapsed,
        eta_seconds=remaining / max(batches_per_second, 1e-12),
        step=step,
    )


def build_learning_rate_scheduler(optimizer, training: Mapping[str, Any]):
    """Build the small, validated scheduler vocabulary used by real MAE runs."""

    import torch

    name = str(training.get("scheduler", "constant"))
    if name == "constant":
        return None
    epochs = int(training["epochs"])
    warmup_epochs = int(training.get("warmup_epochs", 0))
    if name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(training.get("scheduler_step_size", 25)),
            gamma=float(training.get("scheduler_gamma", 0.5)),
        )
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup_epochs),
            eta_min=float(training.get("min_learning_rate", 1e-6)),
        )
    else:  # configuration validation normally catches this first
        raise ValueError(f"Unsupported learning-rate scheduler: {name}")
    if not warmup_epochs:
        return scheduler
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / max(2, warmup_epochs + 1),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, scheduler],
        milestones=[warmup_epochs],
    )


def train_real_mae(
    config: ResolvedRealMAEConfig,
    *,
    checkpoint: Path | None = None,
) -> Path:
    """Train with validation, append-only metrics, portable resume, and explicit artifacts."""

    import torch

    from morphofeatures.mae3d import build_mae_model, load_mae_checkpoint

    config.run_dir.mkdir(parents=True, exist_ok=True)
    writer = MetricWriter(config.metrics_path)
    preprocessing_started = time.perf_counter()
    writer.write(
        "preprocessing_started",
        workflow="real_mae_train",
        stage="preprocessing",
        profile=config.profile,
    )
    try:
        prepared = prepare_real_mae_data(config, progress=writer.write)
        resolved_path = config.save()
        manifest_started = time.perf_counter()
        split_path = save_split_manifest(prepared)
        writer.write(
            "split_manifest_written",
            stage="preprocessing",
            path=str(split_path),
            rows=int(sum(len(values) for values in prepared.label_splits.values())),
            duration_seconds=time.perf_counter() - manifest_started,
        )
        writer.write(
            "preprocessing_completed",
            workflow="real_mae_train",
            stage="preprocessing",
            duration_seconds=time.perf_counter() - preprocessing_started,
        )
    except Exception as error:
        writer.write(
            "failed",
            workflow="real_mae_train",
            stage="preprocessing",
            error=str(error),
        )
        raise

    training = config.values["training"]
    try:
        train_loader = _data_loader(
            prepared.dataset("train", "train", augment=True),
            training,
            shuffle=True,
            seed=int(config.values["seed"]),
        )
        validation_loader = _data_loader(
            prepared.dataset("validation", "train", augment=False),
            training,
            shuffle=False,
            seed=int(config.values["seed"]),
        )
        writer.write(
            "dataloader_ready",
            stage="setup",
            train_batches=len(train_loader),
            validation_batches=len(validation_loader),
            batch_size=int(training["batch_size"]),
            workers=int(training["workers"]),
            pin_memory=bool(training.get("pin_memory", False)),
            group_size=int(config.data["group_size"]),
        )
        torch.manual_seed(int(config.values["seed"]))
        np.random.seed(int(config.values["seed"]))
        device = resolve_device(str(config.values["device"]))
        model = build_mae_model(dict(config.values)).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        scheduler = build_learning_rate_scheduler(optimizer, training)
        amp_enabled = bool(training.get("mixed_precision", False) and device.type == "cuda")
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except AttributeError:  # pragma: no cover - compatibility with older torch
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        writer.write(
            "model_ready",
            stage="setup",
            device=str(device),
            parameter_count=int(sum(parameter.numel() for parameter in model.parameters())),
            cuda_devices_visible=int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            cuda_devices_used=1 if device.type == "cuda" else 0,
            mixed_precision=amp_enabled,
        )
    except Exception as error:
        writer.write(
            "failed",
            workflow="real_mae_train",
            stage="setup",
            error=str(error),
        )
        raise

    start_epoch = 0
    resume_payload: Mapping[str, Any] = {}
    resume = training.get("resume_from")
    if resume:
        resume_payload = load_mae_checkpoint(
            Path(resume),
            model,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        start_epoch = int(resume_payload.get("epoch", 0))
        writer.write(
            "checkpoint_resumed",
            stage="setup",
            path=str(resume),
            start_epoch=start_epoch,
            step=int(resume_payload.get("step", 0)),
        )

    output = Path(checkpoint or config.checkpoint_path)
    early_stopping = training["early_stopping"]
    early_enabled = bool(early_stopping["enabled"])
    best_output = Path(config.values["paths"]["best_checkpoint"])
    if early_enabled and best_output.resolve() == output.resolve():
        raise ValueError("Early-stopping best and final checkpoint paths must differ")
    total_epochs = int(training["epochs"])
    step = int(resume_payload.get("step", start_epoch * len(train_loader)))
    resume_metrics = resume_payload.get("metrics", {})
    if not isinstance(resume_metrics, Mapping):
        resume_metrics = {}
    best_validation_loss = float(resume_metrics.get("best_validation_loss", np.inf))
    best_epoch = int(resume_metrics.get("best_epoch", 0))
    epochs_without_improvement = int(resume_metrics.get("epochs_without_improvement", 0))
    best_checkpoint_available: Path | None = best_output if best_output.is_file() else None
    previous_best = resume_metrics.get("best_checkpoint")
    if best_checkpoint_available is None and previous_best and Path(previous_best).is_file():
        best_checkpoint_available = Path(previous_best)
    progress_interval = int(training["progress_interval_batches"])
    writer.write(
        "started",
        workflow="real_mae_train",
        profile=config.profile,
        mae_architecture_version=GROUPED_PATCH_MAE_ARCHITECTURE_VERSION,
        patch_encoder=str(config.values["mae"]["patch_encoder"]),
        learning_rate_scheduler=str(training["scheduler"]),
        device=str(device),
        start_epoch=start_epoch,
        target_epochs=total_epochs,
        train_labels=len(prepared.label_splits["train"]),
        validation_labels=len(prepared.label_splits["validation"]),
        train_patches=len(prepared.patch_indices["train"]),
        progress_interval_batches=progress_interval,
        early_stopping_enabled=early_enabled,
        early_stopping_patience=(int(early_stopping["patience"]) if early_enabled else None),
        early_stopping_min_delta=(
            float(early_stopping["min_delta"]) if early_enabled else None
        ),
    )
    last_metrics: dict[str, Any] = {}
    current_epoch = start_epoch
    completed_epochs = start_epoch
    stopped_early = False
    try:
        for epoch in range(start_epoch, total_epochs):
            current_epoch = epoch
            train_loader.dataset.set_epoch(epoch)
            train_loader.generator.manual_seed(int(config.values["seed"]) + epoch)
            torch.manual_seed(int(config.values["seed"]) + epoch)
            model.train()
            train_losses = []
            train_baselines = []
            train_groups = 0
            train_patches = 0
            phase_started = time.perf_counter()
            for batch_index, batch in enumerate(train_loader, start=1):
                optimizer.zero_grad(set_to_none=True)
                batch_output = _batch_output(
                    model,
                    batch,
                    device,
                    float(config.values["mae"]["mask_ratio"]),
                    amp_enabled,
                )
                loss = batch_output.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(float(loss.detach().cpu()))
                train_baselines.append(
                    float(batch_output.visible_mean_baseline_loss.detach().cpu())
                )
                train_groups += int(batch[0].shape[0])
                train_patches += int(batch[2].sum().item())
                step += 1
                _write_batch_progress(
                    writer,
                    phase="train",
                    epoch=epoch + 1,
                    batch_number=batch_index,
                    total_batches=len(train_loader),
                    epoch_started=phase_started,
                    losses=train_losses,
                    groups_completed=train_groups,
                    patches_completed=train_patches,
                    step=step,
                    interval=progress_interval,
                )
            model.eval()
            torch.manual_seed(int(config.values["seed"]) + 10_000_000 + epoch)
            validation_losses = []
            validation_baselines = []
            validation_groups = 0
            validation_patches = 0
            phase_started = time.perf_counter()
            with torch.no_grad():
                for batch_index, batch in enumerate(validation_loader, start=1):
                    batch_output = _batch_output(
                        model,
                        batch,
                        device,
                        float(config.values["mae"]["mask_ratio"]),
                        amp_enabled,
                    )
                    validation_losses.append(float(batch_output.loss.detach().cpu()))
                    validation_baselines.append(
                        float(batch_output.visible_mean_baseline_loss.detach().cpu())
                    )
                    validation_groups += int(batch[0].shape[0])
                    validation_patches += int(batch[2].sum().item())
                    _write_batch_progress(
                        writer,
                        phase="validation",
                        epoch=epoch + 1,
                        batch_number=batch_index,
                        total_batches=len(validation_loader),
                        epoch_started=phase_started,
                        losses=validation_losses,
                        groups_completed=validation_groups,
                        patches_completed=validation_patches,
                        step=step,
                        interval=progress_interval,
                    )
            train_loss = float(np.mean(train_losses))
            validation_loss = float(np.mean(validation_losses))
            train_baseline = float(np.mean(train_baselines))
            validation_baseline = float(np.mean(validation_baselines))
            last_metrics = {
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_visible_mean_baseline_loss": train_baseline,
                "validation_visible_mean_baseline_loss": validation_baseline,
                "train_improvement_over_visible_mean": (
                    1.0 - train_loss / max(train_baseline, 1e-12)
                ),
                "validation_improvement_over_visible_mean": (
                    1.0 - validation_loss / max(validation_baseline, 1e-12)
                ),
            }
            improved = validation_loss < (
                best_validation_loss - float(early_stopping["min_delta"])
            )
            if early_enabled:
                if improved:
                    best_validation_loss = validation_loss
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                last_metrics.update(
                    {
                        "best_validation_loss": best_validation_loss,
                        "best_epoch": best_epoch,
                        "epochs_without_improvement": epochs_without_improvement,
                        "best_checkpoint": str(best_output),
                    }
                )
            writer.write(
                "epoch",
                epoch=epoch + 1,
                step=step,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                **last_metrics,
            )
            if scheduler is not None:
                scheduler.step()
            completed_epochs = epoch + 1
            if early_enabled and improved:
                save_checkpoint(
                    best_output,
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=completed_epochs,
                    step=step,
                    config=config.values,
                    metrics=last_metrics,
                )
                best_checkpoint_available = best_output
                writer.write(
                    "checkpoint",
                    kind="best",
                    path=str(best_output),
                    epoch=completed_epochs,
                    step=step,
                    validation_loss=validation_loss,
                )
            interval = int(training.get("checkpoint_interval", 0))
            if interval > 0 and completed_epochs % interval == 0 and completed_epochs < total_epochs:
                periodic = output.parent / f"checkpoint-{completed_epochs:04d}.pt"
                save_checkpoint(
                    periodic,
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=completed_epochs,
                    step=step,
                    config=config.values,
                    metrics=last_metrics,
                )
                writer.write(
                    "checkpoint",
                    kind="periodic",
                    path=str(periodic),
                    epoch=completed_epochs,
                    step=step,
                )
            if early_enabled and epochs_without_improvement >= int(
                early_stopping["patience"]
            ):
                stopped_early = True
                writer.write(
                    "early_stopping",
                    stage="training",
                    epoch=completed_epochs,
                    step=step,
                    patience=int(early_stopping["patience"]),
                    min_delta=float(early_stopping["min_delta"]),
                    best_epoch=best_epoch,
                    best_validation_loss=best_validation_loss,
                    epochs_without_improvement=epochs_without_improvement,
                )
                break
        final_checkpoint_epoch = completed_epochs
        final_checkpoint_step = step
        final_checkpoint_metrics = last_metrics
        if (
            stopped_early
            and bool(early_stopping["restore_best"])
            and best_checkpoint_available is not None
        ):
            restored = load_mae_checkpoint(
                best_checkpoint_available,
                model,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
            final_checkpoint_epoch = int(restored.get("epoch", best_epoch))
            final_checkpoint_step = int(restored.get("step", step))
            restored_metrics = restored.get("metrics", last_metrics)
            if isinstance(restored_metrics, Mapping):
                final_checkpoint_metrics = dict(restored_metrics)
            writer.write(
                "best_checkpoint_restored",
                stage="training",
                path=str(best_checkpoint_available),
                best_epoch=best_epoch,
                best_validation_loss=best_validation_loss,
            )
        save_checkpoint(
            output,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=final_checkpoint_epoch,
            step=final_checkpoint_step,
            config=config.values,
            metrics=final_checkpoint_metrics,
        )
        writer.write(
            "checkpoint",
            kind="final",
            path=str(output),
            epoch=final_checkpoint_epoch,
            step=final_checkpoint_step,
        )
        writer.write(
            "completed",
            workflow="real_mae_train",
            checkpoint=str(output),
            epochs_completed=completed_epochs,
            target_epochs=total_epochs,
            stopped_early=stopped_early,
            best_epoch=best_epoch if early_enabled else None,
            best_validation_loss=best_validation_loss if early_enabled else None,
        )
    except Exception as error:
        writer.write(
            "failed",
            workflow="real_mae_train",
            stage="training",
            error=str(error),
            epoch=current_epoch + 1,
            step=step,
        )
        raise
    metadata = {
        "run_id": config.run_dir.name,
        "workflow": "real_mae_nucleus_texture",
        "biological_unit": "cell-associated nucleus",
        "feature_scope": "nucleus-derived EM texture; not the six-group representation",
        "profile": config.profile,
        "mae_architecture_version": GROUPED_PATCH_MAE_ARCHITECTURE_VERSION,
        "patch_encoder": str(config.values["mae"]["patch_encoder"]),
        "source_config": str(config.source_path),
        "artifacts": {
            "resolved_config": str(resolved_path),
            "split_manifest": str(split_path),
            "metrics": str(config.metrics_path),
            "checkpoint": str(output),
            "best_checkpoint": (
                str(best_checkpoint_available) if best_checkpoint_available else None
            ),
        },
        "training": {
            "epochs_completed": completed_epochs,
            "target_epochs": total_epochs,
            "stopped_early": stopped_early,
            "best_epoch": best_epoch if early_enabled else None,
            "best_validation_loss": best_validation_loss if early_enabled else None,
        },
        "data": {
            "patches": prepared.patch_metadata,
            "positions": prepared.positions_metadata,
            "n_available_label_ids": int(len(prepared.index.unique_label_ids)),
            "n_selected_label_ids": int(
                sum(len(label_ids) for label_ids in prepared.label_splits.values())
            ),
            "split_label_counts": {
                name: int(len(label_ids)) for name, label_ids in prepared.label_splits.items()
            },
            "patch_selection": (
                "deterministic central spatial group per parent; whole stored patches are MAE tokens"
            ),
        },
    }
    write_json_atomic(Path(config.values["paths"]["run_metadata"]), metadata)
    return output


def encode_real_mae(
    config: ResolvedRealMAEConfig,
    checkpoint: Path,
    *,
    output: Path | None = None,
) -> Path:
    """Encode each spatial parent group and export one label-first row per ID."""

    import torch

    from morphofeatures.mae3d import build_mae_model, load_mae_checkpoint

    prepared = prepare_real_mae_data(config)
    inference = config.values["inference"]
    split = str(inference["split"])
    dataset = prepared.dataset(split, "encode", augment=False)
    loader = _data_loader(
        dataset,
        inference,
        shuffle=False,
        seed=int(config.values["seed"]),
    )
    device = resolve_device(str(config.values["device"]))
    model = build_mae_model(dict(config.values)).to(device)
    load_mae_checkpoint(Path(checkpoint), model, device=device)
    model.eval()
    ids = []
    feature_batches = []
    valid_patch_counts = []
    progress = MetricWriter(configured_metrics_path(config.values, config.metrics_path))
    completed_objects = 0
    with torch.no_grad():
        for label_ids, patches, positions, valid in loader:
            features = model.encode(
                patches.to(device, non_blocking=True),
                positions.to(device, non_blocking=True),
                valid.to(device, non_blocking=True),
            ).cpu().numpy()
            ids.append(label_ids.numpy())
            feature_batches.append(features)
            valid_patch_counts.append(valid.sum(dim=1).numpy())
            completed_objects += len(label_ids)
            progress.write("encoding_progress", processed=completed_objects, total=len(dataset))
    ids = np.concatenate(ids).astype(np.int64, copy=False)
    features = np.concatenate(feature_batches)
    valid_counts = np.concatenate(valid_patch_counts).astype(np.int64, copy=False)
    order = np.argsort(ids)
    ids, features, valid_counts = ids[order], features[order], valid_counts[order]
    expected = (
        np.sort(np.concatenate(list(prepared.label_splits.values())))
        if split == "all"
        else prepared.label_splits[split]
    )
    if not np.array_equal(ids, expected):
        raise ValueError("Aggregated embedding IDs do not exactly cover the configured label split")
    destination = Path(output or config.embedding_path)
    export_embeddings(destination, ids, features)
    metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
    write_json_atomic(
        metadata_path,
        {
            "artifact_type": "label_first_embedding",
            "scope": "nucleus-derived EM texture",
            "mae_architecture_version": GROUPED_PATCH_MAE_ARCHITECTURE_VERSION,
            "patch_encoder": str(config.values["mae"]["patch_encoder"]),
            "checkpoint": str(Path(checkpoint).resolve()),
            "rows": int(len(ids)),
            "features": int(features.shape[1]),
            "label_ids_unique": True,
            "finite": bool(np.all(np.isfinite(features))),
            "group_size": int(config.data["group_size"]),
            "valid_patches_per_label": {
                str(label_id): int(count) for label_id, count in zip(ids, valid_counts)
            },
        },
    )
    run_metadata_path = Path(config.values["paths"]["run_metadata"])
    if run_metadata_path.exists():
        metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        metadata.setdefault("artifacts", {}).update(
            {"embedding": str(destination), "embedding_metadata": str(metadata_path)}
        )
        write_json_atomic(run_metadata_path, metadata)
    return destination


def reconstruct_real_samples(
    config: ResolvedRealMAEConfig,
    checkpoint: Path,
    *,
    split: str = "test",
    count: int = 3,
) -> Mapping[str, np.ndarray]:
    """Return hidden whole-patch targets and reconstructions for notebook QC."""

    import torch

    from morphofeatures.mae3d import build_mae_model, load_mae_checkpoint

    prepared = prepare_real_mae_data(config)
    dataset = prepared.dataset(split, "inspect", augment=False)
    if not len(dataset):
        raise ValueError(f"No samples are configured for split {split!r}")
    selected = np.linspace(0, len(dataset) - 1, min(int(count), len(dataset)), dtype=int)
    ids, groups, positions, valid_masks = [], [], [], []
    for item in selected:
        sample = dataset[int(item)]
        ids.append(int(sample["label_id"]))
        groups.append(sample["patches"])
        positions.append(sample["positions_zyx"])
        valid_masks.append(sample["valid"])
    device = resolve_device(str(config.values["device"]))
    model = build_mae_model(dict(config.values)).to(device)
    load_mae_checkpoint(Path(checkpoint), model, device=device)
    model.eval()
    inputs = torch.from_numpy(np.stack(groups)).to(device)
    position_tensor = torch.from_numpy(np.stack(positions)).to(device)
    valid_tensor = torch.from_numpy(np.stack(valid_masks)).to(device)
    torch.manual_seed(int(config.values["seed"]))
    with torch.no_grad():
        prediction = model(
            inputs,
            position_tensor,
            valid_tensor,
            mask_ratio=float(config.values["mae"]["mask_ratio"]),
        )
    target = prediction.target
    composite = torch.where(
        prediction.mask[:, :, None, None, None, None],
        prediction.reconstruction,
        target,
    )
    hidden_indices = prediction.mask.to(torch.int64).argmax(dim=1)
    visible_indices = []
    for sample, hidden_index in enumerate(hidden_indices):
        candidates = torch.nonzero(
            ~prediction.mask[sample] & valid_tensor[sample], as_tuple=False
        ).flatten()
        distance = (
            position_tensor[sample, candidates] - position_tensor[sample, hidden_index]
        ).square().sum(dim=1)
        visible_indices.append(candidates[distance.argmin()])
    visible_indices = torch.stack(visible_indices)
    rows = torch.arange(len(hidden_indices), device=device)
    hidden_target = target[rows, hidden_indices]
    hidden_reconstruction = prediction.reconstruction[rows, hidden_indices]
    hidden_composite = composite[rows, hidden_indices]
    visible_context = target[rows, visible_indices]
    voxel_mask = torch.ones(
        (len(rows),) + tuple(model.reconstruction_shape), dtype=torch.bool, device=device
    )
    model_loss = np.asarray(float(prediction.loss.cpu()))
    baseline_loss = np.asarray(float(prediction.visible_mean_baseline_loss.cpu()))
    return {
        "label_ids": np.asarray(ids, dtype=np.int64),
        "inputs": hidden_target.cpu().numpy(),
        "foreground_masks": (hidden_target != 0).cpu().numpy(),
        "reconstructions": hidden_reconstruction.cpu().numpy(),
        "composites": hidden_composite.cpu().numpy(),
        "masked_voxels": voxel_mask.cpu().numpy(),
        "visible_context": visible_context.cpu().numpy(),
        "hidden_group_indices": hidden_indices.cpu().numpy(),
        "hidden_positions_zyx": position_tensor[rows, hidden_indices].cpu().numpy(),
        "hidden_patch_normalized_mse": model_loss,
        "visible_patch_mean_baseline_mse": baseline_loss,
        # Backward-compatible aliases for notebook code saved before v3.
        "masked_foreground_mse": model_loss,
        "visible_mean_baseline_mse": baseline_loss,
    }
