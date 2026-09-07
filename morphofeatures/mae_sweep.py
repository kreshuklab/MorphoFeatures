"""Bounded MAE ablations using the existing experiment and scheduler lifecycle."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from morphofeatures.artifacts import write_json_atomic
from morphofeatures.experiments import JobPlan, plan_job, submit_plan
from morphofeatures.metrics import read_metric_events
from morphofeatures.real_mae import resolve_real_mae_config
from morphofeatures.registry import JobRecord, JobRegistry
from morphofeatures.slurm import ClusterProfile, SchedulerBackend
from morphofeatures.workflows import WorkflowRequest

SUPPORTED_SWEEP_PARAMETERS = {
    "training.learning_rate",
    "training.scheduler",
    "training.weight_decay",
    "mae.patch_encoder",
    "mae.embedding_dim",
    "mae.reconstruction_shape",
    "mae.mask_ratio",
    "data.normalization",
}
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$")


@dataclass(frozen=True)
class SweepVariant:
    name: str
    run_id: str
    config_path: Path
    run_dir: Path
    overrides: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedSweep:
    name: str
    directory: Path
    manifest_path: Path
    variants: tuple[SweepVariant, ...]


def _nested_get(values: Mapping[str, Any], dotted: str) -> Any:
    current: Any = values
    for key in dotted.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _nested_override(dotted: str, value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current = result
    keys = dotted.split(".")
    for key in keys[:-1]:
        child: dict[str, Any] = {}
        current[key] = child
        current = child
    current[keys[-1]] = value
    return result


def _merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _slug(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        text = "x".join(str(item) for item in value)
    else:
        text = str(value)
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return text[:24] or "value"


def load_sweep_spec(path: Path) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        spec = yaml.safe_load(stream) or {}
    if "sweep" in spec:
        spec = spec["sweep"]
    if not isinstance(spec, Mapping):
        raise ValueError("Sweep YAML must contain a mapping")
    return spec


def expand_soft_grid(
    base_values: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    """Expand a bounded one-at-a-time or Cartesian ablation.

    ``one_at_a_time`` is the default: it includes the base configuration and
    changes one parameter at a time. This is intentionally less combinatorial
    than a conventional grid and makes the source of an observed effect easier
    to interpret.
    """

    parameters = spec.get("parameters", {})
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError("Sweep parameters must be a non-empty mapping")
    unsupported = sorted(set(parameters).difference(SUPPORTED_SWEEP_PARAMETERS))
    if unsupported:
        raise ValueError("Unsupported sweep parameters: {}".format(", ".join(unsupported)))
    choices: dict[str, list[Any]] = {}
    for key, values in parameters.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Sweep parameter {key} must be a non-empty YAML list")
        choices[str(key)] = values
    mode = str(spec.get("mode", "one_at_a_time"))
    include_base = bool(spec.get("include_base", True))
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    if include_base:
        candidates.append(("baseline", {}))
    if mode == "one_at_a_time":
        for key, values in choices.items():
            for value in values:
                if value == _nested_get(base_values, key):
                    continue
                candidates.append((f"{key.split('.')[-1]}-{_slug(value)}", {key: value}))
    elif mode == "cartesian":
        keys = tuple(choices)
        for combination in itertools.product(*(choices[key] for key in keys)):
            dotted = dict(zip(keys, combination))
            digest = hashlib.sha256(
                json.dumps(dotted, sort_keys=True).encode("utf-8")
            ).hexdigest()[:8]
            candidates.append(("grid-" + digest, dotted))
    else:
        raise ValueError("Sweep mode must be one_at_a_time or cartesian")
    max_runs = int(spec.get("max_runs", 16))
    if max_runs < 1 or max_runs > 64:
        raise ValueError("sweep.max_runs must be between 1 and 64")
    if len(candidates) > max_runs:
        raise ValueError(
            f"Sweep expands to {len(candidates)} runs, exceeding max_runs={max_runs}"
        )
    seen: set[str] = set()
    result = []
    for name, dotted in candidates:
        unique_name = name
        if unique_name in seen:
            digest = hashlib.sha256(
                json.dumps(dotted, sort_keys=True).encode("utf-8")
            ).hexdigest()[:6]
            unique_name = f"{name[:32]}-{digest}"
        seen.add(unique_name)
        nested: Mapping[str, Any] = {}
        for key, value in dotted.items():
            nested = _merge(nested, _nested_override(key, value))
        result.append((unique_name, nested))
    return tuple(result)


def prepare_soft_grid(
    base_config_path: Path,
    sweep_spec_path: Path,
    *,
    output_root: Path,
    profile: str | None = None,
    require_data: bool = True,
) -> PreparedSweep:
    """Validate and save immutable resolved configurations plus a manifest."""

    spec = load_sweep_spec(sweep_spec_path)
    name = str(spec.get("name", "mae-soft-grid"))
    if not _NAME.fullmatch(name):
        raise ValueError("Sweep name must contain 1-40 letters, numbers, dots, underscores, or dashes")
    base = resolve_real_mae_config(
        Path(base_config_path), profile=profile, require_data=require_data
    )
    expanded = expand_soft_grid(base.values, spec)
    root = Path(output_root).expanduser().resolve()
    directory = root / "sweeps" / name
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Sweep manifest already exists; choose a new name: {manifest_path}")
    config_directory = directory / "configs"
    config_directory.mkdir(parents=True, exist_ok=False)
    variants = []
    manifest_variants = []
    for variant_name, scientific_overrides in expanded:
        run_id = f"{name}-{variant_name}"[:80]
        run_dir = root / "experiments" / run_id / "mae_train"
        path_overrides = {
            "paths": {
                "run_dir": str(run_dir),
                "resolved_config": str(run_dir / "resolved_config.yaml"),
                "metrics": str(run_dir / "metrics.jsonl"),
                "checkpoint": str(run_dir / "checkpoints" / "checkpoint.pt"),
                "embedding": str(run_dir / "embeddings" / "nucleus_texture_mae.npy"),
                "split_manifest": str(run_dir / "split_manifest.tsv"),
                "run_metadata": str(run_dir / "run_metadata.json"),
            }
        }
        overrides = _merge(scientific_overrides, path_overrides)
        resolved = resolve_real_mae_config(
            Path(base_config_path),
            profile=profile,
            overrides=overrides,
            require_data=require_data,
        )
        config_path = config_directory / f"{variant_name}.yaml"
        resolved.save(config_path)
        variant = SweepVariant(
            variant_name,
            run_id,
            config_path,
            run_dir,
            scientific_overrides,
        )
        variants.append(variant)
        manifest_variants.append(
            {
                "name": variant.name,
                "run_id": variant.run_id,
                "config_path": str(variant.config_path),
                "run_dir": str(variant.run_dir),
                "overrides": variant.overrides,
                "metrics": str(run_dir / "metrics.jsonl"),
                "checkpoint": str(run_dir / "checkpoints" / "checkpoint.pt"),
                "embedding": str(run_dir / "embeddings" / "nucleus_texture_mae.npy"),
            }
        )
    write_json_atomic(
        manifest_path,
        {
            "schema": "morphofeatures.mae_soft_grid.v1",
            "name": name,
            "mode": spec.get("mode", "one_at_a_time"),
            "base_config": str(Path(base_config_path).expanduser().resolve()),
            "profile": base.profile,
            "sweep_spec": str(Path(sweep_spec_path).expanduser().resolve()),
            "variants": manifest_variants,
        },
    )
    return PreparedSweep(name, directory, manifest_path, tuple(variants))


def load_prepared_sweep(path: Path) -> PreparedSweep:
    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "morphofeatures.mae_soft_grid.v1":
        raise ValueError("Unsupported or missing MAE soft-grid manifest schema")
    variants = tuple(
        SweepVariant(
            str(item["name"]),
            str(item["run_id"]),
            Path(item["config_path"]),
            Path(item["run_dir"]),
            item.get("overrides", {}),
        )
        for item in payload["variants"]
    )
    return PreparedSweep(str(payload["name"]), manifest_path.parent, manifest_path, variants)


def plan_soft_grid(
    sweep: PreparedSweep,
    cluster_profile: ClusterProfile,
    *,
    output_root: Path,
    protected_output_roots: Iterable[Path] = (),
) -> tuple[JobPlan, ...]:
    return tuple(
        plan_job(
            WorkflowRequest("mae_train", variant.config_path),
            cluster_profile,
            run_id=variant.run_id,
            output_root=output_root,
            protected_output_roots=protected_output_roots,
        )
        for variant in sweep.variants
    )


def submit_soft_grid(
    plans: Sequence[JobPlan],
    registry: JobRegistry,
    scheduler: SchedulerBackend,
    *,
    repository: Path,
    encode_after_training: bool = False,
) -> tuple[JobRecord, ...]:
    """Submit explicitly prepared jobs and optional ``afterok`` encoders."""

    records: list[JobRecord] = []
    for plan in plans:
        training_record = submit_plan(plan, registry, scheduler, repository=repository)
        records.append(training_record)
        if not encode_after_training or not training_record.slurm_job_id:
            continue
        embedding_path = plan.working_directory / "embeddings" / "nucleus_texture_mae.npy"
        encoding = plan_job(
            WorkflowRequest(
                "mae_encode",
                plan.snapshot_path,
                checkpoint=plan.checkpoint_path,
                output=embedding_path,
            ),
            plan.profile,
            run_id=plan.run_id,
            output_root=plan.working_directory.parents[2],
            dependency_job_id=training_record.slurm_job_id,
            parent_job_id=training_record.id,
        )
        records.append(submit_plan(encoding, registry, scheduler, repository=repository))
    return tuple(records)


def summarize_soft_grid(
    sweep: PreparedSweep | Path,
    *,
    output: Path | None = None,
) -> pd.DataFrame:
    """Compare raw final/best losses without assuming that low loss means useful biology."""

    prepared = load_prepared_sweep(sweep) if isinstance(sweep, (str, Path)) else sweep
    rows = []
    for variant in prepared.variants:
        with variant.config_path.open("r", encoding="utf-8") as stream:
            effective_config = yaml.safe_load(stream) or {}
        metrics_path = variant.run_dir / "metrics.jsonl"
        events = read_metric_events(metrics_path)
        epochs = [event for event in events if event.get("event") == "epoch"]
        validation = [event for event in epochs if event.get("validation_loss") is not None]
        best = min(validation, key=lambda event: float(event["validation_loss"])) if validation else None
        last = epochs[-1] if epochs else None
        row: dict[str, Any] = {
            "variant": variant.name,
            "run_id": variant.run_id,
            "epochs_completed": len(epochs),
            "last_train_loss": None if last is None else last.get("train_loss"),
            "last_validation_loss": None if last is None else last.get("validation_loss"),
            "best_validation_loss": None if best is None else best.get("validation_loss"),
            "best_epoch": None if best is None else best.get("epoch"),
            "last_validation_improvement_over_visible_mean": None
            if last is None
            else last.get("validation_improvement_over_visible_mean"),
            "checkpoint_exists": (variant.run_dir / "checkpoints" / "checkpoint.pt").is_file(),
            "embedding_exists": (
                variant.run_dir / "embeddings" / "nucleus_texture_mae.npy"
            ).is_file(),
        }
        for dotted in sorted(SUPPORTED_SWEEP_PARAMETERS):
            value = _nested_get(effective_config, dotted)
            if value is not None:
                row[dotted] = json.dumps(value) if isinstance(value, (list, tuple)) else value
        rows.append(row)
    frame = pd.DataFrame(rows)
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, sep="\t" if destination.suffix == ".tsv" else ",", index=False)
    return frame


def plot_soft_grid_metrics(sweep: PreparedSweep | Path, output: Path):
    """Save comparable train/validation curves for every readable variant."""

    import matplotlib.pyplot as plt

    prepared = load_prepared_sweep(sweep) if isinstance(sweep, (str, Path)) else sweep
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    plotted = 0
    for variant in prepared.variants:
        events = [
            event
            for event in read_metric_events(variant.run_dir / "metrics.jsonl")
            if event.get("event") == "epoch"
        ]
        if not events:
            continue
        epochs = [event["epoch"] for event in events]
        axes[0].plot(epochs, [event["train_loss"] for event in events], label=variant.name)
        axes[1].plot(
            epochs,
            [event.get("validation_loss", np.nan) for event in events],
            label=variant.name,
        )
        plotted += 1
    if not plotted:
        plt.close(figure)
        raise ValueError("No completed epoch metrics were found in the sweep")
    for axis, title in zip(axes, ("training loss", "validation loss")):
        axis.set(xlabel="epoch", ylabel="normalized masked-patch MSE", title=title)
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize="small", bbox_to_anchor=(1.02, 1), loc="upper left")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    return figure


def plot_soft_grid_reconstructions(
    sweep: PreparedSweep | Path,
    output: Path,
    *,
    split: str = "test",
    count: int = 1,
    device: str = "cpu",
):
    """Save matched target/reconstruction/error slices for completed checkpoints."""

    import matplotlib.pyplot as plt

    from morphofeatures.real_mae import reconstruct_real_samples

    prepared = load_prepared_sweep(sweep) if isinstance(sweep, (str, Path)) else sweep
    completed = [
        variant
        for variant in prepared.variants
        if (variant.run_dir / "checkpoints" / "checkpoint.pt").is_file()
    ]
    if not completed:
        raise ValueError("No completed sweep checkpoints were found")
    figure, axes = plt.subplots(
        len(completed) * int(count), 3, figsize=(9, 3 * len(completed) * int(count)), squeeze=False
    )
    row = 0
    for variant in completed:
        config = resolve_real_mae_config(
            variant.config_path,
            profile=None,
            overrides={"device": device},
        )
        views = reconstruct_real_samples(
            config,
            variant.run_dir / "checkpoints" / "checkpoint.pt",
            split=split,
            count=count,
        )
        for sample in range(len(views["label_ids"])):
            target = views["inputs"][sample, 0]
            reconstruction = views["reconstructions"][sample, 0]
            middle = target.shape[0] // 2
            images = (target[middle], reconstruction[middle], np.abs(target - reconstruction)[middle])
            for column, (image, title) in enumerate(
                zip(images, ("target", "reconstruction", "absolute error"))
            ):
                axes[row, column].imshow(image, cmap="magma" if column == 2 else "gray")
                axes[row, column].set_title(f"{variant.name}: {title}")
                axes[row, column].axis("off")
            row += 1
    figure.tight_layout()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    return figure


def compare_soft_grid_classifiers(
    sweep: PreparedSweep | Path,
    labels_path: Path,
    *,
    output: Path | None = None,
    model: str = "logistic",
    folds: int = 5,
    seed: int = 42,
    minimum_class_count: int = 2,
) -> pd.DataFrame:
    """Run the same ID-aligned shallow probe for every available sweep embedding."""

    from morphofeatures.analysis.classification import evaluate_embedding_classifier

    prepared = load_prepared_sweep(sweep) if isinstance(sweep, (str, Path)) else sweep
    rows = []
    for variant in prepared.variants:
        embedding = variant.run_dir / "embeddings" / "nucleus_texture_mae.npy"
        if not embedding.is_file():
            continue
        result = evaluate_embedding_classifier(
            embedding,
            Path(labels_path),
            output_dir=variant.run_dir / "analysis" / f"classifier_{model}",
            model=model,
            folds=folds,
            seed=seed,
            minimum_class_count=minimum_class_count,
        )
        row = {
            "variant": variant.name,
            "run_id": variant.run_id,
            "model": model,
            "n_matched": int(len(result.labels)),
            "mean_accuracy": result.mean_accuracy,
            "std_accuracy": result.std_accuracy,
        }
        row.update(
            {
                f"recall_{name}": float(recall)
                for name, recall in zip(result.class_names, result.per_class_recall)
            }
        )
        rows.append(row)
    if not rows:
        raise ValueError("No encoded sweep embeddings are available for classification")
    frame = pd.DataFrame(rows)
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, sep="\t" if destination.suffix == ".tsv" else ",", index=False)
    return frame
