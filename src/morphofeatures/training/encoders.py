"""Training entry points for creating new embedding encoders.

This module is intentionally lightweight. It validates training inputs and
describes the artifacts produced by training without importing the heavy GPU
stack until the user actually launches training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from morphofeatures.config.loading import load_yaml_config

EncoderKind = Literal["shape", "texture"]


@dataclass(slots=True)
class EncoderTrainingRun:
    """Description of an encoder training run and its expected artifacts."""

    kind: EncoderKind
    run_dir: Path
    train_config: Path
    checkpoint_dir: Path
    data_config: Path | None = None
    embedding_command: str = ""

    def describe(self) -> str:
        """Return a human-readable run summary."""

        data_line = f"\nData config: {self.data_config}" if self.data_config else ""
        return (
            f"Encoder type: {self.kind}\n"
            f"Run directory: {self.run_dir}\n"
            f"Training config: {self.train_config}"
            f"{data_line}\n"
            f"Checkpoint directory: {self.checkpoint_dir}\n"
            f"Embedding command after training: {self.embedding_command}"
        )


def _require_keys(config: dict[str, Any], keys: set[str], context: str) -> None:
    """Raise a helpful error when a config section is missing keys."""

    missing = sorted(key for key in keys if key not in config)
    if missing:
        raise KeyError(f"{context} is missing required key(s): {', '.join(missing)}")


def _check_path(path: str | Path, strict: bool, context: str) -> Path:
    """Resolve a path and optionally require it to exist."""

    resolved = Path(path).expanduser()
    if strict and not resolved.exists():
        raise FileNotFoundError(f"{context} does not exist: {resolved}")
    return resolved


def validate_shape_training_config(config_path: str | Path, strict_paths: bool = False) -> EncoderTrainingRun:
    """Validate a shape encoder training config.

    Args:
        config_path: YAML config consumed by ``morphofeatures-shape train``.
        strict_paths: When true, referenced input array paths must exist.

    Returns:
        A training run description.
    """

    train_config = Path(config_path).expanduser()
    config = load_yaml_config(train_config)
    _require_keys(config, {"experiment_dir", "data", "model", "optimizer", "criterion", "training"}, "Shape config")

    data_config = config["data"]
    if "npz_path" in data_config:
        _check_path(data_config["npz_path"], strict_paths, "Shape npz_path")
    elif "points_path" in data_config:
        _check_path(data_config["points_path"], strict_paths, "Shape points_path")
        if data_config.get("features_path"):
            _check_path(data_config["features_path"], strict_paths, "Shape features_path")
        if data_config.get("ids_path"):
            _check_path(data_config["ids_path"], strict_paths, "Shape ids_path")
    else:
        raise KeyError("Shape config data section requires either 'npz_path' or 'points_path'.")

    run_dir = Path(config["experiment_dir"]).expanduser()
    checkpoint_dir = run_dir / "checkpoints"
    embedding_command = (
        "After training, create an inference config whose model.checkpoint points "
        f"to a file in {checkpoint_dir}, then run: "
        "morphofeatures-shape embed --config <inference_config.yaml> --save-to <embeddings.npy>"
    )
    return EncoderTrainingRun(
        kind="shape",
        run_dir=run_dir,
        train_config=train_config,
        checkpoint_dir=checkpoint_dir,
        embedding_command=embedding_command,
    )


def validate_texture_training_run(
    project_directory: str | Path,
    train_config: str | Path | None = None,
    data_config: str | Path | None = None,
    strict_paths: bool = False,
) -> EncoderTrainingRun:
    """Validate a texture encoder training run directory.

    Args:
        project_directory: Experiment directory where weights/logs are written.
        train_config: Optional training config path. Defaults to
            ``project_directory/train_config.yml``.
        data_config: Optional data config path. Defaults to
            ``project_directory/data_config.yml``.
        strict_paths: When true, config files and explicit data paths must exist.

    Returns:
        A training run description.
    """

    run_dir = Path(project_directory).expanduser()
    train_config_path = Path(train_config).expanduser() if train_config else run_dir / "train_config.yml"
    data_config_path = Path(data_config).expanduser() if data_config else run_dir / "data_config.yml"

    _check_path(train_config_path, True, "Texture train_config")
    _check_path(data_config_path, True, "Texture data_config")

    train = load_yaml_config(train_config_path)
    data = load_yaml_config(data_config_path)
    _require_keys(train, {"model_name", "model_kwargs", "loss", "training_optimizer_kwargs"}, "Texture train_config")
    if not data.get("contrastive", False) and not data.get("texture_contrastive", False):
        raise KeyError("Texture data_config must set either 'contrastive: true' or 'texture_contrastive: true'.")

    data_section = data.get("data", data.get("data_config", {}))
    explicit_path_keys = {
        "raw_data",
        "cell_segmentation_xml",
        "nucleus_segmentation_xml",
        "cell_to_nucleus_table",
        "cell_table",
        "nucleus_table",
    }
    path_section = data_section.get("paths", {})
    has_explicit_paths = explicit_path_keys.issubset(path_section) or explicit_path_keys.issubset(data_section)
    has_legacy_version = "version" in data_section
    if not has_explicit_paths and not has_legacy_version:
        raise KeyError(
            "Texture data_config must provide explicit volume/table paths or a legacy 'version' under data/data_config."
        )
    if strict_paths and has_explicit_paths:
        source = path_section if explicit_path_keys.issubset(path_section) else data_section
        for key in explicit_path_keys:
            _check_path(source[key], True, f"Texture {key}")

    checkpoint_dir = run_dir / "Weights"
    embedding_command = f"morphofeatures-texture predict {run_dir} --devices <gpu_ids>"
    return EncoderTrainingRun(
        kind="texture",
        run_dir=run_dir,
        train_config=train_config_path,
        data_config=data_config_path,
        checkpoint_dir=checkpoint_dir,
        embedding_command=embedding_command,
    )


def train_shape_encoder(config_path: str | Path, dry_run: bool = False, strict_paths: bool = False) -> EncoderTrainingRun:
    """Validate and optionally train a new shape encoder.

    Args:
        config_path: Shape training YAML config.
        dry_run: When true, only validate and return artifact locations.
        strict_paths: When true, referenced input arrays must exist.

    Returns:
        Training run description.
    """

    run = validate_shape_training_config(config_path, strict_paths=strict_paths)
    if dry_run:
        return run

    from morphofeatures.shape.trainer import ShapeTrainer

    ShapeTrainer(load_yaml_config(run.train_config)).run()
    return run


def train_texture_encoder(
    project_directory: str | Path,
    train_config: str | Path | None = None,
    data_config: str | Path | None = None,
    devices: str = "0",
    from_checkpoint: bool = False,
    dry_run: bool = False,
    strict_paths: bool = False,
) -> EncoderTrainingRun:
    """Validate and optionally train a new texture encoder.

    Args:
        project_directory: Experiment directory where ``Weights`` and ``Logs``
            are written.
        train_config: Optional training config path.
        data_config: Optional data config path.
        devices: CUDA device list passed to the texture trainer.
        from_checkpoint: Resume from ``Weights/checkpoint.pytorch``.
        dry_run: When true, only validate and return artifact locations.
        strict_paths: When true, explicit volume/table paths must exist.

    Returns:
        Training run description.
    """

    run = validate_texture_training_run(
        project_directory,
        train_config=train_config,
        data_config=data_config,
        strict_paths=strict_paths,
    )
    if dry_run:
        return run

    from morphofeatures.texture.trainer import train_texture_model

    train_texture_model(
        run.run_dir,
        run.train_config,
        run.data_config if run.data_config is not None else Path(run.run_dir) / "data_config.yml",
        devices=devices,
        from_checkpoint=from_checkpoint,
    )
    return run
