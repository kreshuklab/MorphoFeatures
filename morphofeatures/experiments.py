"""Experiment planning and lifecycle orchestration shared by Streamlit and tests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import yaml

from morphofeatures.artifacts import existing_artifacts, write_json_atomic
from morphofeatures.metrics import METRICS_ENV, utc_now
from morphofeatures.registry import JobRecord, JobRegistry
from morphofeatures.slurm import (
    ClusterProfile,
    DryRunScheduler,
    SchedulerBackend,
    SlurmScheduler,
    apply_profile_command,
    render_slurm_script,
)
from morphofeatures.workflows import (
    WorkflowRequest,
    build_workflow_command,
    validate_workflow_request,
)

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
SNAPSHOT_FILES = (
    "train_config.yml",
    "data_config.yml",
    "test_config.yml",
    "test_config_patches.yml",
)


@dataclass(frozen=True)
class JobPlan:
    run_id: str
    request: WorkflowRequest
    profile: ClusterProfile
    working_directory: Path
    snapshot_path: Path
    script_path: Path
    stdout_path: Path
    stderr_path: Path
    metrics_path: Path
    metadata_path: Path
    command: Tuple[str, ...]
    script: str
    submission_key: str
    checkpoint_path: Optional[Path]
    embedding_path: Optional[Path]
    dependency_job_id: Optional[str] = None
    parent_job_id: Optional[str] = None
    warnings: Tuple[str, ...] = ()


def _absolute(path: Path, base: Path) -> str:
    value = Path(path).expanduser()
    return str(value.resolve() if value.is_absolute() else (base / value).resolve())


def _snapshot_yaml(source: Path, destination: Path, request: WorkflowRequest, working: Path) -> None:
    with source.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    base = source.resolve().parent
    if request.workflow.startswith("mae_"):
        from morphofeatures.configuration_editor import absolute_paths, load_document

        config = (load_document(source) if config.get("data", {}).get("source") == "n5_masked_patches"
                  else absolute_paths(config, base))
        config["device"] = request.device
    if request.workflow.startswith("shape_"):
        data = config.setdefault("data", {})
        for key in ("manifest", "root"):
            if data.get(key):
                data[key] = _absolute(Path(data[key]), base)
        if request.input_path is not None:
            input_path = Path(request.input_path).resolve()
            if input_path.is_dir():
                data.pop("manifest", None)
                data["root"] = str(input_path)
            else:
                data.pop("root", None)
                data["manifest"] = str(input_path)
        if request.workflow == "shape_train":
            config["experiment_dir"] = str(working)
        elif request.checkpoint is not None:
            config.setdefault("model", {})["checkpoint"] = str(Path(request.checkpoint).resolve())
    if request.workflow.startswith("mae_"):
        data = config.setdefault("data", {})
        crops = data.get("crops")
        if crops:
            data["crops"] = _absolute(Path(crops), base)
        if request.input_path is not None:
            data["crops"] = str(Path(request.input_path).resolve())
        if "paths" in config and config["paths"].get("repo_root"):
            config["paths"]["repo_root"] = _absolute(Path(config["paths"]["repo_root"]), base)
        if data.get("source") == "n5_masked_patches":
            # Keep the resolved real-data run and the registry's explicit
            # artifacts in the same immutable experiment directory.
            paths = config.setdefault("paths", {})
            paths.update(
                {
                    "run_dir": str(working),
                    "resolved_config": str(working / "resolved_config.yaml"),
                    "metrics": str(working / "metrics.jsonl"),
                    "checkpoint": str(
                        Path(request.checkpoint).resolve()
                        if request.checkpoint
                        else working / "checkpoints" / "checkpoint.pt"
                    ),
                    "best_checkpoint": str(working / "checkpoints" / "checkpoint-best.pt"),
                    "embedding": str(
                        Path(request.output).resolve()
                        if request.output
                        else working / "embeddings" / "nucleus_texture_mae.npy"
                    ),
                    "split_manifest": str(working / "split_manifest.tsv"),
                    "run_metadata": str(working / "run_metadata.json"),
                }
            )
            config.setdefault("runtime", {})["metrics_path"] = paths["metrics"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    os.replace(str(temporary), str(destination))


def _snapshot_texture(source: Path, destination: Path, request: WorkflowRequest) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    found = False
    for name in SNAPSHOT_FILES:
        source_path = source / name
        if not source_path.is_file():
            continue
        found = True
        with source_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        if name != "train_config.yml":
            data = config.setdefault("data_config", {})
            if data.get("data_root"):
                data["data_root"] = _absolute(Path(data["data_root"]), source_path.parent)
            if request.input_path is not None:
                data["data_root"] = str(Path(request.input_path).resolve())
        destination_path = destination / name
        temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
        temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        os.replace(str(temporary), str(destination_path))
    if not found:
        raise ValueError("Texture experiment contains no recognized configuration files")


def collect_provenance(repository: Path) -> Dict[str, Any]:
    provenance: Dict[str, Any] = {
        "timestamp": utc_now(),
        "python": sys.version.splitlines()[0],
        "platform": platform.platform(),
    }
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            provenance["git_commit"] = result.stdout.strip()
        dirty = subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
        if dirty.returncode == 0:
            provenance["git_dirty"] = bool(dirty.stdout.strip())
    except OSError:
        pass
    return provenance


def _job_paths(output_root: Path, run_id: str, request: WorkflowRequest) -> Dict[str, Path]:
    working = Path(output_root).resolve() / "experiments" / run_id / request.workflow
    snapshot = working / ("experiment" if request.definition.configuration_kind == "directory" else "config.yaml")
    checkpoint = Path(request.checkpoint).resolve() if request.checkpoint else None
    embedding = Path(request.output).resolve() if request.output else None
    if request.workflow == "mae_train" and checkpoint is None:
        checkpoint = working / "checkpoints" / "checkpoint.pt"
    elif request.workflow in {"shape_train", "texture_train"}:
        checkpoint = working / "checkpoints" / "best.pt"
    if request.definition.stage == "encoding" and embedding is None:
        if request.workflow == "texture_encode" and request.save_patches and not request.aggregate_patches:
            embedding = working / "encoded_patches.npz"
        else:
            embedding = working / "embeddings.npy"
    return {
        "working": working,
        "snapshot": snapshot,
        "script": working / "job.slurm",
        "stdout": working / "stdout.log",
        "stderr": working / "stderr.log",
        "metrics": working / "metrics.jsonl",
        "metadata": working / "run.json",
        "checkpoint": checkpoint,
        "embedding": embedding,
    }


def _configuration_digest(path: Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    paths = [source] if source.is_file() else [
        source / name for name in SNAPSHOT_FILES if (source / name).is_file()
    ]
    for item in paths:
        digest.update(item.name.encode("utf-8"))
        digest.update(item.read_bytes())
    return digest.hexdigest()


def plan_job(
    request: WorkflowRequest,
    profile: ClusterProfile,
    *,
    run_id: str,
    output_root: Path,
    protected_output_roots: Iterable[Path] = (),
    dependency_job_id: Optional[str] = None,
    parent_job_id: Optional[str] = None,
) -> JobPlan:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("Run ID must contain 1-80 letters, numbers, dots, underscores, or dashes")
    paths = _job_paths(output_root, run_id, request)
    effective = request
    if request.workflow == "mae_train":
        effective = replace(request, checkpoint=paths["checkpoint"])
    elif request.definition.stage == "encoding" and request.output is None:
        effective = replace(request, output=paths["embedding"])
    allow_missing = dependency_job_id is not None
    report = validate_workflow_request(
        effective,
        protected_output_roots=tuple(protected_output_roots),
        allow_missing_checkpoint=allow_missing,
    )
    if not report.valid:
        raise ValueError("; ".join(report.errors))
    warnings = list(report.warnings)
    if request.workflow.startswith("mae_") and profile.gpus > 1:
        warnings.append(
            "The maintained MAE currently uses one CUDA device; additional requested GPUs "
            "will remain idle unless a distributed training implementation is added."
        )
    command = tuple(
        apply_profile_command(build_workflow_command(effective, paths["snapshot"]), profile)
    )
    environment = {
        METRICS_ENV: str(paths["metrics"]),
        "MORPHOFEATURES_RUN_METADATA": str(paths["metadata"]),
    }
    script = render_slurm_script(
        command,
        profile,
        job_name="mf-{}".format(request.workflow.replace("_", "-")),
        working_directory=paths["working"],
        stdout_path=paths["stdout"],
        stderr_path=paths["stderr"],
        environment=environment,
    )
    key_payload = {
        "run_id": run_id,
        "request": {
            **request.__dict__,
            "configuration": str(Path(request.configuration).resolve()),
            "input_path": str(request.input_path) if request.input_path else None,
            "checkpoint": str(request.checkpoint) if request.checkpoint else None,
            "output": str(request.output) if request.output else None,
        },
        "profile": profile.__dict__,
        "dependency": dependency_job_id,
        "configuration_sha256": _configuration_digest(request.configuration),
    }
    key = hashlib.sha256(json.dumps(key_payload, sort_keys=True, default=list).encode()).hexdigest()
    return JobPlan(
        run_id,
        effective,
        profile,
        paths["working"],
        paths["snapshot"],
        paths["script"],
        paths["stdout"],
        paths["stderr"],
        paths["metrics"],
        paths["metadata"],
        command,
        script,
        key,
        paths["checkpoint"],
        paths["embedding"],
        dependency_job_id,
        parent_job_id,
        tuple(warnings),
    )


def materialize_plan(
    plan: JobPlan,
    repository: Path,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    if plan.snapshot_path.exists() or plan.script_path.exists():
        raise FileExistsError(
            "Run snapshot already exists; choose a new run ID instead of overwriting it"
        )
    plan.working_directory.mkdir(parents=True, exist_ok=True)
    source = Path(plan.request.configuration).resolve()
    if plan.request.definition.configuration_kind == "directory":
        _snapshot_texture(source, plan.snapshot_path, plan.request)
    else:
        _snapshot_yaml(source, plan.snapshot_path, plan.request, plan.working_directory)
    plan.script_path.write_text(plan.script, encoding="utf-8")
    provenance = dict(provenance or collect_provenance(repository))
    metadata = {
        "run_id": plan.run_id,
        "workflow": plan.request.workflow,
        "stage": plan.request.definition.stage,
        "command": list(plan.command),
        "configuration_snapshot": str(plan.snapshot_path),
        "profile": plan.profile.__dict__,
        "dependency_job_id": plan.dependency_job_id,
        "provenance": provenance,
        "artifacts": {
            "checkpoint": str(plan.checkpoint_path) if plan.checkpoint_path else None,
            "embedding": str(plan.embedding_path) if plan.embedding_path else None,
            "metrics": str(plan.metrics_path),
            "stdout": str(plan.stdout_path),
            "stderr": str(plan.stderr_path),
        },
    }
    write_json_atomic(plan.metadata_path, metadata)
    return provenance


def submit_plan(
    plan: JobPlan,
    registry: JobRegistry,
    scheduler: SchedulerBackend,
    *,
    repository: Path,
) -> JobRecord:
    provenance = collect_provenance(repository)
    submission_command: Tuple[str, ...] = ()
    if isinstance(scheduler, SlurmScheduler):
        submission_command = tuple(
            scheduler.submission_command(plan.script_path, plan.dependency_job_id)
        )
    record = JobRecord(
        run_id=plan.run_id,
        workflow=plan.request.workflow,
        stage=plan.request.definition.stage,
        command=plan.command,
        submission_command=submission_command,
        submission_key=plan.submission_key,
        working_directory=str(plan.working_directory),
        config_snapshot=str(plan.snapshot_path),
        stdout_path=str(plan.stdout_path),
        stderr_path=str(plan.stderr_path),
        metrics_path=str(plan.metrics_path),
        checkpoint_path=str(plan.checkpoint_path) if plan.checkpoint_path else None,
        embedding_path=str(plan.embedding_path) if plan.embedding_path else None,
        parent_job_id=plan.parent_job_id,
        dependency_job_id=plan.dependency_job_id,
        application_state="saving" if isinstance(scheduler, DryRunScheduler) else "submitting",
        provenance=provenance,
    )
    registry.create(record)
    try:
        materialize_plan(plan, repository, provenance=provenance)
    except Exception as error:
        return registry.update(
            record.id,
            application_state="failed",
            raw_scheduler_state="SNAPSHOT_FAILED",
            completed_at=utc_now(),
            error_message=str(error),
        )
    try:
        scheduler_job_id = scheduler.submit(plan.script_path, plan.dependency_job_id)
    except Exception as error:
        return registry.update(
            record.id,
            application_state="failed",
            raw_scheduler_state="SUBMISSION_FAILED",
            completed_at=utc_now(),
            error_message=str(error),
        )
    if scheduler_job_id is None:
        return registry.update(record.id, application_state="dry_run", raw_scheduler_state="DRY_RUN")
    return registry.update(
        record.id,
        slurm_job_id=scheduler_job_id,
        submitted_at=utc_now(),
        raw_scheduler_state="PENDING",
        application_state="queued",
    )


def refresh_jobs(registry: JobRegistry, scheduler: SchedulerBackend) -> Tuple[int, Tuple[str, ...]]:
    active = registry.list(states=("queued", "running", "unknown"))
    slurm_ids = [record.slurm_job_id for record in active if record.slurm_job_id]
    result = scheduler.query(slurm_ids)
    updated = 0
    for record in active:
        if not record.slurm_job_id:
            continue
        status = result.statuses.get(record.slurm_job_id)
        if status is None:
            registry.update(
                record.id,
                application_state="unknown",
                error_message="No squeue/sacct record; accounting may be delayed or purged",
            )
            continue
        changes: Dict[str, Any] = {
            "application_state": status.state,
            "raw_scheduler_state": status.raw_state,
            "exit_status": status.exit_status,
            "error_message": None,
        }
        if status.started_at:
            changes["started_at"] = status.started_at
        if status.completed_at:
            changes["completed_at"] = status.completed_at
        if status.state in {"completed", "failed", "cancelled"} and not status.completed_at:
            changes["completed_at"] = utc_now()
        artifacts = existing_artifacts(
            Path(path) if path else None
            for path in (
                record.config_snapshot,
                record.metrics_path,
                record.checkpoint_path,
                record.embedding_path,
                record.stdout_path,
                record.stderr_path,
            )
        )
        if record.workflow == "workspace_run":
            status_path = Path(record.working_directory) / "status.json"
            if status_path.exists():
                worker_status = json.loads(status_path.read_text())
                # A scheduler refresh can race the worker's final registry update.
                # Retain scientific outputs rather than replacing them with log paths.
                artifacts = tuple(worker_status.get("artifacts", record.artifacts))
                if worker_status.get("state") in {"completed", "failed"}:
                    changes["application_state"] = worker_status["state"]
                    changes["error_message"] = worker_status.get("error")
        changes["artifacts"] = artifacts
        registry.update(record.id, **changes)
        updated += 1
    return updated, result.warnings


def cancel_job(record: JobRecord, registry: JobRegistry, scheduler: SchedulerBackend) -> JobRecord:
    if not record.slurm_job_id:
        raise ValueError("This job has no submitted SLURM job ID")
    if record.application_state not in {"queued", "running", "unknown"}:
        raise ValueError("Only active jobs can be cancelled")
    scheduler.cancel(record.slurm_job_id)
    return registry.update(
        record.id,
        raw_scheduler_state="CANCEL_REQUESTED",
        application_state="cancelled",
        completed_at=utc_now(),
    )
