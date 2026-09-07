"""Persistent local/SLURM workers for shared scientific workflows."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from morphofeatures.artifacts import write_json_atomic
from morphofeatures.config import repository_root
from morphofeatures.configuration_editor import absolute_paths, load_document, validate_training
from morphofeatures.experiments import collect_provenance
from morphofeatures.metrics import METRICS_ENV, MetricWriter, utc_now
from morphofeatures.registry import JobRecord, JobRegistry
from morphofeatures.slurm import ClusterProfile, SlurmScheduler, render_slurm_script


def resolve_job(document, base, *, validate=True):
    document = deepcopy(document)
    stages = document.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("Job requires a nonempty stages list")
    available = set()
    named_embeddings = set()
    prepared = None
    for stage in stages:
        if not isinstance(stage, dict):
            raise ValueError("Each pipeline stage must be a YAML mapping")
        for link, required in (
            ("from_preprocessing", "preprocess"),
            ("from_training", "train"),
            ("from_extraction", "extract"),
        ):
            if validate and stage.get(link) and required not in available:
                raise ValueError(f"{link} requires an earlier {required} stage")
        if validate and stage.get("action") not in {
            "train",
            "extract",
            "analyze",
            "compare",
            "preprocess",
        }:
            raise ValueError(f"Unsupported stage action: {stage.get('action')}")
        if "config" in stage:
            stage["config"] = (
                load_document(Path(base) / stage["config"])
                if isinstance(stage["config"], str)
                else absolute_paths(stage["config"], Path(base))
            )
        for key in (
            "checkpoint",
            "embedding",
            "annotations",
            "raw",
            "segmentation",
            "crops",
            "label_ids",
            "masks",
            "model_repository",
            "cache",
        ):
            if stage.get(key):
                stage[key] = str((Path(base) / Path(stage[key]).expanduser()).resolve())
        if stage.get("embeddings"):
            stage["embeddings"] = {
                k: str((Path(base) / Path(v).expanduser()).resolve())
                for k, v in stage["embeddings"].items()
            }
        if not validate:
            continue
        if stage["action"] == "train":
            if "config" not in stage:
                raise ValueError("Training requires config")
            if stage.get("from_preprocessing"):
                if stage["config"].get("data", {}).get("source") == "n5_masked_patches":
                    raise ValueError(
                        "Prepared whole-object crops require a crop MAE configuration, not grouped N5"
                    )
                config = deepcopy(stage["config"])
                shape = prepared.get("crop_shape", [32, 32, 32])
                config.setdefault("mae", {}).setdefault("input_shape", shape)
                if list(config["mae"]["input_shape"]) != list(shape):
                    raise ValueError(
                        "Training input shape must match the preceding preparation crop shape"
                    )
                stage["config"] = validate_training(config, base, require_data=False)
            else:
                stage["config"] = validate_training(stage["config"], base)
        if stage["action"] == "preprocess":
            from morphofeatures.data.preprocessing import validate_preprocessing

            validate_preprocessing(stage)
            prepared = stage
        if stage["action"] == "extract":
            if stage.get("model", "mae") not in {"mae", "dinov2", "dinov3"}:
                raise ValueError("Extraction model must be mae, dinov2, or dinov3")
            if not stage.get("checkpoint") and not stage.get("from_training"):
                raise ValueError("Extraction requires checkpoint or from_training")
            if stage.get("checkpoint") and not Path(stage["checkpoint"]).is_file():
                raise ValueError(f"Checkpoint does not exist: {stage['checkpoint']}")
            if (
                not stage.get("config")
                and not stage.get("from_preprocessing")
                and not stage.get("from_training")
            ):
                raise ValueError("Extraction requires the model/data config or from_preprocessing")
            if stage.get("model", "mae") != "mae":
                from morphofeatures.dino import VARIANTS

                if stage.get("variant") not in VARIANTS[stage["model"]]:
                    raise ValueError(
                        f"Choose a supported {stage['model']} variant: {sorted(VARIANTS[stage['model']])}"
                    )
                if not (Path(stage.get("model_repository") or "") / "hubconf.py").is_file():
                    raise ValueError(
                        "DINO extraction requires model_repository pointing to an official local checkout"
                    )
        if stage["action"] == "analyze" and not stage.get("from_extraction"):
            if not stage.get("embedding") or not Path(stage["embedding"]).is_file():
                raise ValueError("Analysis requires an existing embedding file or from_extraction")
        if stage["action"] == "compare":
            references = stage.get("from_embeddings", [])
            if any(name not in named_embeddings for name in references):
                raise ValueError("from_embeddings must name earlier extraction stages")
            if len(set(stage.get("embeddings", {})) | set(references)) < 2:
                raise ValueError("Comparison requires at least two named embedding files")
            for name, path in stage.get("embeddings", {}).items():
                if not Path(path).is_file():
                    raise ValueError(f"Embedding for {name} does not exist: {path}")
        if stage.get("annotations") and not Path(stage["annotations"]).is_file():
            raise ValueError("Annotation table does not exist")
        available.add(stage["action"])
        if stage["action"] == "extract" and stage.get("name"):
            if stage["name"] in named_embeddings:
                raise ValueError("Extraction stage names must be unique")
            named_embeddings.add(stage["name"])
    return document


def scope_training_outputs(config, destination):
    config = deepcopy(config)
    paths = config.setdefault("paths", {})
    paths.update(
        {
            "run_dir": str(destination),
            "checkpoint": str(destination / "checkpoint.pt"),
            "best_checkpoint": str(destination / "checkpoint-best.pt"),
            "resolved_config": str(destination / "resolved_config.yaml"),
            "metrics": str(destination / "metrics.jsonl"),
            "split_manifest": str(destination / "split_manifest.tsv"),
            "run_metadata": str(destination / "run_metadata.json"),
            "embedding": str(destination / "embeddings.npz"),
        }
    )
    config.setdefault("runtime", {})["metrics_path"] = paths["metrics"]
    return config


@dataclass(frozen=True)
class WorkspacePlan:
    """A reviewed snapshot. Planning never allocates a run or opens the registry."""

    record: JobRecord
    submitted_yaml: str
    resolved_yaml: str
    script: str
    environment: tuple
    execution: str
    cpus: int
    fingerprint: str


def submission_fingerprint(document, *, output_root, run_id, execution, dependency=None):
    payload = {
        "document": document,
        "output_root": str(Path(output_root).expanduser().resolve()),
        "run_id": run_id,
        "execution": execution,
        "dependency": dependency or None,
    }
    return hashlib.sha256(yaml.safe_dump(payload, sort_keys=True).encode()).hexdigest()


def plan_workspace_job(
    document, *, output_root, run_id, execution="local", base=None, dependency=None
):
    """Validate and render the exact submission without persisting anything."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("Run ID requires 1–80 letters, digits, dots, underscores or dashes")
    if execution not in {"local", "slurm", "dry-run"}:
        raise ValueError("execution must be local, slurm, or dry-run")
    if dependency and execution != "slurm":
        raise ValueError("Job dependencies require SLURM; use stages for a local pipeline")
    if dependency and not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", dependency):
        raise ValueError("Dependency must be a Slurm job ID")
    fingerprint = submission_fingerprint(
        document, output_root=output_root, run_id=run_id, execution=execution, dependency=dependency
    )
    document = resolve_job(document, base or repository_root())
    profile = ClusterProfile.from_mapping("workspace", document.get("slurm", {}))
    output_root = Path(output_root).expanduser().resolve()
    folder = output_root / "experiments" / run_id / "workspace"
    source = folder / "job.yaml"
    interpreter = profile.python_executable or sys.executable
    command = (interpreter, "-m", "morphofeatures", "workspace-run", "--config", str(source))
    record = JobRecord(
        run_id=run_id,
        workflow="workspace_run",
        stage="pipeline",
        command=command,
        submission_key=hashlib.sha256(str(folder).encode()).hexdigest(),
        working_directory=str(folder),
        config_snapshot=str(source),
        stdout_path=str(folder / "stdout.log"),
        stderr_path=str(folder / "stderr.log"),
        metrics_path=str(folder / "metrics.jsonl"),
        dependency_job_id=dependency,
        application_state="dry_run" if execution == "dry-run" else "queued",
        provenance=collect_provenance(repository_root()),
    )
    if folder.exists():
        raise FileExistsError(f"Experiment already exists; choose a new run ID: {folder}")
    submitted = deepcopy(document)
    for index, stage in enumerate(document["stages"]):
        if stage["action"] == "train" and stage.get("config"):
            stage["config"] = scope_training_outputs(stage["config"], folder / f"{index:02d}-train")
    document["worker"] = {
        "registry": str(output_root / ".morphofeatures" / "registry.sqlite3"),
        "record_id": record.id,
        "directory": str(folder),
        "execution": execution,
    }
    environment = {
        METRICS_ENV: record.metrics_path,
        "PYTHONUNBUFFERED": "1",
        "MPLCONFIGDIR": str(folder / ".matplotlib"),
        "PYTHONPATH": str(repository_root()) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    script = render_slurm_script(
        command,
        profile,
        job_name="mf-workspace",
        working_directory=folder,
        stdout_path=Path(record.stdout_path),
        stderr_path=Path(record.stderr_path),
        environment=environment,
    )
    return WorkspacePlan(
        record=record,
        submitted_yaml=yaml.safe_dump(submitted, sort_keys=False),
        resolved_yaml=yaml.safe_dump(document, sort_keys=False),
        script=script,
        environment=tuple(environment.items()),
        execution=execution,
        cpus=profile.cpus,
        fingerprint=fingerprint,
    )


def submit_workspace_plan(plan, *, expected_fingerprint, dry_run=False):
    """Persist a reviewed plan and launch once; reject stale reviews and collisions."""
    if expected_fingerprint != plan.fingerprint:
        raise ValueError("Settings changed after review. Review the run again before submitting.")
    document = yaml.safe_load(plan.resolved_yaml)
    execution = "dry-run" if dry_run else plan.execution
    document["worker"]["execution"] = execution
    record = replace(
        plan.record, application_state="dry_run" if execution == "dry-run" else "queued"
    )
    folder = Path(record.working_directory)
    # Atomic reservation also protects against double clicks and concurrent sessions.
    folder.mkdir(parents=True, exist_ok=False)
    registry = JobRegistry(Path(document["worker"]["registry"]))
    (folder / "submitted_settings.yaml").write_text(plan.submitted_yaml)
    Path(record.config_snapshot).write_text(yaml.safe_dump(document, sort_keys=False))
    script_path = folder / "job.slurm"
    script_path.write_text(plan.script)
    registry.create(record)
    environment = dict(plan.environment)
    try:
        if execution == "local":
            environment.update(OMP_NUM_THREADS=str(plan.cpus), MKL_NUM_THREADS=str(plan.cpus))
            with (
                Path(record.stdout_path).open("ab") as stdout,
                Path(record.stderr_path).open("ab") as stderr,
            ):
                process = subprocess.Popen(
                    record.command,
                    cwd=folder,
                    env={**os.environ, **environment},
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
            write_json_atomic(folder / "process.json", {"pid": process.pid, "started": utc_now()})
        elif execution == "slurm":
            job_id = SlurmScheduler().submit(script_path, record.dependency_job_id)
            record = registry.update(record.id, slurm_job_id=job_id, submitted_at=utc_now())
    except Exception as error:
        registry.update(
            record.id, application_state="failed", error_message=str(error), completed_at=utc_now()
        )
        raise
    return record


def submit_job(document, *, output_root, run_id, execution="local", base=None, dependency=None):
    """CLI-compatible convenience wrapper around shared planning and submission."""
    plan = plan_workspace_job(
        document,
        output_root=output_root,
        run_id=run_id,
        execution=execution,
        base=base,
        dependency=dependency,
    )
    return submit_workspace_plan(plan, expected_fingerprint=plan.fingerprint)


def run_job(path):
    """Worker entry point used unchanged by local processes and SLURM scripts."""
    path = Path(path).resolve()
    document = yaml.safe_load(path.read_text())
    if "worker" not in document:
        document = resolve_job(document, path.parent)
    worker = document.get("worker", {})
    folder = Path(worker.get("directory", path.parent / "results"))
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "status.json").exists():
        raise FileExistsError("This worker directory has already run; resubmit with a new run ID")
    registry = JobRegistry(Path(worker["registry"])) if worker.get("registry") else None
    writer = MetricWriter(folder / "metrics.jsonl")
    status = {"state": "running", "started": utc_now(), "stages": []}
    status_path = folder / "status.json"
    artifacts = []
    previous = {}
    if registry:
        registry.update(worker["record_id"], application_state="running", started_at=utc_now())
    try:
        for index, original in enumerate(document["stages"]):
            stage = deepcopy(original)
            action = stage["action"]
            destination = folder / f"{index:02d}-{action}"
            destination.mkdir(exist_ok=True)
            entry = {"index": index, "action": action, "state": "running", "started": utc_now()}
            status["stages"].append(entry)
            write_json_atomic(status_path, status)
            writer.write("stage_started", stage=index, action=action)
            started = time.perf_counter()
            if stage.get("from_preprocessing"):
                prepared = load_document(previous["preprocess"])
                supplied = stage.get("config", {})
                supplied.setdefault("data", {}).update(prepared["data"])
                supplied.setdefault("mae", {}).setdefault(
                    "input_shape", prepared["mae"]["input_shape"]
                )
                stage["config"] = supplied
            if stage.get("from_training"):
                stage["checkpoint"] = previous["train"]
                stage.setdefault("config", previous["training_config"])
                stage.setdefault(
                    "training_cost",
                    {"seconds": previous["training_seconds"], "checkpoint": previous["train"]},
                )
            if stage.get("from_extraction"):
                stage["embedding"] = previous["extract"]
            if stage.get("from_embeddings"):
                stage.setdefault("embeddings", {}).update(
                    {name: previous["named_embeddings"][name] for name in stage["from_embeddings"]}
                )
            if action == "train":
                from morphofeatures.mae3d import train_from_config

                config = scope_training_outputs(
                    validate_training(stage["config"], path.parent), destination
                )
                checkpoint = destination / "checkpoint.pt"
                config_path = destination / "resolved_config.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False))
                previous["training_config"] = config
                result = train_from_config(config_path, checkpoint)
            elif action == "extract":
                from morphofeatures.representations import extract_representation

                result = extract_representation(stage, destination, writer=writer)
            elif action == "analyze":
                from morphofeatures.representation_analysis import analyze

                result = analyze(Path(stage["embedding"]), destination, stage)
            elif action == "compare":
                from morphofeatures.representation_analysis import compare

                result = compare(stage["embeddings"], destination, stage)
            else:
                from morphofeatures.data.preprocessing import preprocess

                result = preprocess(
                    stage, destination, progress=lambda **v: writer.write("preprocessing", **v)
                )
            previous[action] = str(result)
            if action == "extract" and stage.get("name"):
                previous.setdefault("named_embeddings", {})[stage["name"]] = str(result)
            artifacts.append(str(result))
            entry.update(
                state="completed", result=str(result), seconds=time.perf_counter() - started
            )
            if action == "train":
                previous["training_seconds"] = entry["seconds"]
            writer.write("stage_completed", **entry)
        status.update(state="completed", completed=utc_now(), artifacts=artifacts)
        if registry:
            registry.update(
                worker["record_id"],
                application_state="completed",
                completed_at=utc_now(),
                artifacts=tuple(artifacts),
                exit_status="0",
            )
    except BaseException as error:
        status.update(state="failed", error=str(error), completed=utc_now())
        if status["stages"]:
            status["stages"][-1].update(state="failed", error=str(error))
        writer.write("failed", error=str(error))
        if registry:
            registry.update(
                worker["record_id"],
                application_state="failed",
                completed_at=utc_now(),
                error_message=str(error),
                exit_status="1",
            )
        raise
    finally:
        write_json_atomic(status_path, status)
    return status_path
