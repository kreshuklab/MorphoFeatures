"""Safe construction and validation of supported MorphoFeatures CLI workflows."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import shlex
from typing import Dict, Optional, Sequence, Tuple

import yaml


@dataclass(frozen=True)
class WorkflowDefinition:
    key: str
    label: str
    stage: str
    configuration_kind: str


WORKFLOWS: Dict[str, WorkflowDefinition] = {
    definition.key: definition
    for definition in (
        WorkflowDefinition("shape_train", "Shape training", "training", "yaml"),
        WorkflowDefinition("shape_encode", "Shape encoding", "encoding", "yaml"),
        WorkflowDefinition("texture_train", "Texture training", "training", "directory"),
        WorkflowDefinition("texture_encode", "Texture encoding", "encoding", "directory"),
        WorkflowDefinition("mae_train", "MAE training", "training", "yaml"),
        WorkflowDefinition("mae_encode", "MAE encoding", "encoding", "yaml"),
    )
}

_SUBCOMMANDS = {
    "workspace_run": "workspace-run",
    "shape_train": "shape-train",
    "shape_encode": "shape-encode",
    "texture_train": "texture-train",
    "texture_encode": "texture-encode",
    "mae_train": "mae-train",
    "mae_encode": "mae-encode",
}
_DEVICES = {"auto", "cpu", "cuda"}


@dataclass(frozen=True)
class WorkflowRequest:
    workflow: str
    configuration: Path
    input_path: Optional[Path] = None
    device: str = "auto"
    checkpoint: Optional[Path] = None
    output: Optional[Path] = None
    resume: bool = False
    save_patches: bool = False
    aggregate_patches: bool = False

    @property
    def definition(self) -> WorkflowDefinition:
        try:
            return WORKFLOWS[self.workflow]
        except KeyError as error:
            raise ValueError("Unsupported workflow: {}".format(self.workflow)) from error


@dataclass(frozen=True)
class ValidationReport:
    errors: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors


def _safe_text(value: object, name: str) -> Optional[str]:
    text = str(value)
    if not text or "\x00" in text or "\n" in text or "\r" in text:
        return "{} contains an empty value or a control character".format(name)
    return None


def _configured_path(value: object, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def validate_workflow_request(
    request: WorkflowRequest,
    *,
    protected_output_roots: Sequence[Path] = (),
    allow_missing_checkpoint: bool = False,
) -> ValidationReport:
    errors = []
    warnings = []
    try:
        definition = request.definition
    except ValueError as error:
        return ValidationReport((str(error),), ())
    if request.device not in _DEVICES:
        errors.append("Device must be one of: auto, cpu, cuda")
    optional_modules = ["torch"]
    if request.workflow.startswith("shape_"):
        optional_modules.append("torch_cluster")
    if request.workflow.startswith("texture_"):
        optional_modules.append("skimage")
        if importlib.util.find_spec("zarr") is None and importlib.util.find_spec("z5py") is None:
            warnings.append("Texture volume loading requires zarr or legacy z5py at runtime")
    missing_modules = [name for name in optional_modules if importlib.util.find_spec(name) is None]
    if missing_modules:
        warnings.append("Optional runtime modules are missing here: {}".format(", ".join(missing_modules)))
    for name, value in (
        ("configuration", request.configuration),
        ("input", request.input_path),
        ("checkpoint", request.checkpoint),
        ("output", request.output),
    ):
        if value is not None:
            problem = _safe_text(value, name)
            if problem:
                errors.append(problem)

    if request.input_path is not None:
        input_path = Path(request.input_path)
        if not input_path.exists():
            errors.append("Input override does not exist: {}".format(input_path))
        elif request.workflow.startswith("mae_") and input_path.suffix.lower() != ".npy":
            errors.append("MAE input override must be a NumPy crop array")
        elif request.workflow.startswith("texture_") and not input_path.is_dir():
            errors.append("Texture input override must be a data-root directory")
        elif request.workflow.startswith("shape_") and not (
            input_path.is_dir() or input_path.suffix.lower() in {".tsv", ".csv"}
        ):
            errors.append("Shape input override must be a manifest or point-cloud root directory")

    configuration = Path(request.configuration)
    if definition.configuration_kind == "yaml":
        if not configuration.is_file():
            errors.append("Configuration file does not exist: {}".format(configuration))
        elif configuration.suffix.lower() not in {".yaml", ".yml"}:
            errors.append("Configuration must be a YAML file: {}".format(configuration))
        else:
            try:
                with configuration.open("r", encoding="utf-8") as stream:
                    parsed = yaml.safe_load(stream) or {}
                if not isinstance(parsed, dict):
                    errors.append("Configuration root must be a YAML mapping")
                elif request.input_path is None and request.workflow.startswith("shape_"):
                    data = parsed.get("data", {})
                    configured_input = data.get("manifest") or data.get("root")
                    if not configured_input:
                        errors.append("Shape configuration requires data.manifest or data.root")
                    elif not _configured_path(configured_input, configuration.resolve().parent).exists():
                        errors.append("Configured shape input does not exist: {}".format(configured_input))
                elif request.input_path is None and request.workflow.startswith("mae_"):
                    data = parsed.get("data", {})
                    if data.get("source") == "n5_masked_patches":
                        try:
                            from morphofeatures.real_mae import resolve_real_mae_config

                            resolve_real_mae_config(
                                configuration,
                                profile=parsed.get("active_profile")
                                or parsed.get("resolved_profile"),
                            )
                        except (OSError, ValueError) as error:
                            errors.append("Invalid real-data MAE configuration: {}".format(error))
                    else:
                        for key in ("crops", "label_ids"):
                            if data.get(key) and isinstance(data[key], str):
                                candidate = _configured_path(data[key], configuration.resolve().parent)
                                if not candidate.is_file():
                                    errors.append(
                                        "Configured MAE {} does not exist: {}".format(key, candidate)
                                    )
            except (OSError, yaml.YAMLError) as error:
                errors.append("Invalid YAML configuration: {}".format(error))
    else:
        if not configuration.is_dir():
            errors.append("Texture experiment directory does not exist: {}".format(configuration))
        else:
            required = [configuration / "train_config.yml", configuration / "data_config.yml"]
            if request.workflow == "texture_encode":
                test_name = "test_config_patches.yml" if request.save_patches else "test_config.yml"
                required.append(configuration / test_name)
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                errors.append("Missing texture configuration: {}".format(", ".join(missing)))
            for path in required:
                if not path.is_file():
                    continue
                try:
                    with path.open("r", encoding="utf-8") as stream:
                        parsed = yaml.safe_load(stream) or {}
                    if not isinstance(parsed, dict):
                        errors.append("Texture config root must be a mapping: {}".format(path))
                    elif path.name != "train_config.yml":
                        data_config = parsed.get("data_config")
                        if not isinstance(data_config, dict):
                            errors.append("Texture data config requires a data_config mapping: {}".format(path))
                        elif request.input_path is None:
                            root = data_config.get("data_root") or os.environ.get(
                                "MORPHOFEATURES_DATA_ROOT"
                            )
                            if not root:
                                errors.append("Texture data config requires data_root or an input override")
                            elif not _configured_path(root, path.resolve().parent).exists():
                                errors.append("Configured texture data root does not exist: {}".format(root))
                except (OSError, yaml.YAMLError) as error:
                    errors.append("Invalid texture YAML {}: {}".format(path, error))

    if definition.stage == "encoding":
        if request.checkpoint is None:
            errors.append("Encoding requires a checkpoint path")
        if request.checkpoint is not None and not Path(request.checkpoint).is_file():
            message = "Checkpoint does not exist yet: {}".format(request.checkpoint)
            if allow_missing_checkpoint:
                warnings.append(message)
            else:
                errors.append(message)
        if request.output is None:
            errors.append("Encoding requires an output path")
    elif request.workflow == "texture_train":
        if request.resume and request.checkpoint is None:
            errors.append("Texture resume requires the previous last.pt checkpoint")
        if request.resume and request.checkpoint is not None and not Path(request.checkpoint).is_file():
            errors.append("Resume checkpoint does not exist: {}".format(request.checkpoint))
        if not request.resume and request.checkpoint is not None:
            errors.append("A texture training checkpoint is used only with resume enabled")
    elif request.workflow == "mae_train" and request.checkpoint is not None:
        checkpoint_output = Path(request.checkpoint).resolve()
        if checkpoint_output.suffix.lower() not in {".pt", ".pth"}:
            errors.append("MAE checkpoint output must end in .pt or .pth")
        if checkpoint_output.exists():
            errors.append("Checkpoint output already exists; choose a new path: {}".format(checkpoint_output))
        for root in protected_output_roots:
            try:
                checkpoint_output.relative_to(Path(root).resolve())
            except ValueError:
                continue
            errors.append("Checkpoint would overwrite a protected bundled-data directory")
            break

    if request.aggregate_patches and not request.save_patches:
        errors.append("Patch aggregation requires save_patches")
    if request.save_patches and request.workflow != "texture_encode":
        errors.append("Patch export is supported only for texture encoding")

    if request.output is not None:
        output = Path(request.output).resolve()
        if output.exists() and output.is_dir():
            errors.append("Output points to a directory rather than a file: {}".format(output))
        elif output.exists():
            errors.append("Output already exists; choose a new destination: {}".format(output))
        nearest_parent = output.parent
        while not nearest_parent.exists() and nearest_parent != nearest_parent.parent:
            nearest_parent = nearest_parent.parent
        if nearest_parent.exists() and not os.access(nearest_parent, os.W_OK):
            errors.append("Output parent is not writable: {}".format(nearest_parent))
        suffixes = {".npz"} if request.save_patches and not request.aggregate_patches else {
            ".npy",
            ".tsv",
            ".csv",
        }
        if output.suffix.lower() not in suffixes:
            errors.append("Output must end in {}".format(", ".join(sorted(suffixes))))
        for root in protected_output_roots:
            try:
                output.relative_to(Path(root).resolve())
            except ValueError:
                continue
            errors.append("Output would overwrite a protected bundled-data directory: {}".format(root))
            break
    return ValidationReport(tuple(errors), tuple(warnings))


def build_workflow_command(request: WorkflowRequest, configuration: Optional[Path] = None) -> list[str]:
    """Return an argument list for one of the six supported package workflows."""
    _ = request.definition
    config = str(Path(configuration or request.configuration))
    command = ["python", "-m", "morphofeatures", _SUBCOMMANDS[request.workflow]]
    if request.workflow == "shape_train":
        command.extend(("--config", config))
    elif request.workflow == "shape_encode":
        command.extend(("--config", config, "--output", str(request.output)))
        if request.checkpoint is not None:
            command.extend(("--checkpoint", str(request.checkpoint)))
    elif request.workflow == "texture_train":
        command.extend((config, "--device", request.device))
        if request.resume:
            command.append("--from-checkpoint")
            command.extend(("--checkpoint", str(request.checkpoint)))
    elif request.workflow == "texture_encode":
        command.extend((config, "--device", request.device))
        if request.checkpoint is not None:
            command.extend(("--checkpoint", str(request.checkpoint)))
        if request.output is not None:
            command.extend(("--output", str(request.output)))
        if request.save_patches:
            command.append("--patches")
        if request.aggregate_patches:
            command.append("--aggregate")
    elif request.workflow == "mae_train":
        command.extend(("--config", config))
        if request.checkpoint is not None:
            command.extend(("--output", str(request.checkpoint)))
    else:
        command.extend(
            (
                "--config",
                config,
                "--checkpoint",
                str(request.checkpoint),
                "--output",
                str(request.output),
            )
        )
    validate_command(command)
    return command


def validate_command(command: Sequence[str]) -> None:
    """Reject commands that were not constructed for the package CLI."""
    if len(command) < 4:
        raise ValueError("Command is incomplete")
    for index, argument in enumerate(command):
        problem = _safe_text(argument, "argument {}".format(index))
        if problem:
            raise ValueError(problem)
    executable = Path(command[0]).name
    if executable not in {"python", "python3"} and not executable.startswith("python3."):
        raise ValueError("Only a Python interpreter may launch workflows")
    if list(command[1:3]) != ["-m", "morphofeatures"]:
        raise ValueError("Only the morphofeatures module may be launched")
    if command[3] not in set(_SUBCOMMANDS.values()):
        raise ValueError("Unsupported MorphoFeatures subcommand: {}".format(command[3]))


def format_command(command: Sequence[str]) -> str:
    validate_command(command)
    return shlex.join(list(command))
