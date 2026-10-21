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
from morphofeatures.data.remote_n5 import is_remote_url
from morphofeatures.experiments import collect_provenance
from morphofeatures.metrics import METRICS_ENV, MetricWriter, utc_now
from morphofeatures.registry import JobRecord, JobRegistry
from morphofeatures.slurm import ClusterProfile, SlurmScheduler, render_slurm_script
from morphofeatures.workflow_validation import (
    FieldError,
    WorkflowValidationError,
    at_fields,
    prefix_fields,
    stage_issue,
)


def embedding_names(stage):
    if stage.get("action") == "distill":
        from morphofeatures.analysis.distillation import output_names

        return output_names(stage)
    return [stage["name"]] if stage.get("name") and stage.get("action") in {"extract", "fuse", "context"} else []


def resolve_job(
    document, base, *, validate=True, issues=None, stage_indices=None, allow_synthetic=True
):
    document = deepcopy(document)
    from morphofeatures.data.object_selection import resolve_selection_paths

    stages = document.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("Job requires a nonempty stages list")
    available = set()
    named_embeddings = set()
    prepared = None
    for index, stage in enumerate(stages):
        if stage_indices is not None and index not in stage_indices:
            if isinstance(stage, dict):
                available.add(stage.get("action"))
                if stage.get("action") == "preprocess":
                    prepared = stage
                named_embeddings.update(embedding_names(stage))
            continue
        try:
            if not isinstance(stage, dict):
                raise ValueError("Each pipeline stage must be a YAML mapping")
            resolve_selection_paths(stage, base)
            for link, required in (
                ("from_preprocessing", "preprocess"),
                ("from_training", "train"),
                ("from_extraction", "extract"),
                ("from_bilateral", "bilateral"),
            ):
                if validate and stage.get(link) and required not in available:
                    raise ValueError(f"{link} requires an earlier {required} stage")
            if validate and stage.get("action") not in {
                "train",
                "extract",
                "analyze",
                "compare",
                "joint_analyze",
                "preprocess",
                "export_labels",
                "export_mobie",
                "dino_experiment",
                "fuse",
                "transfer",
                "distill",
                "context",
                "bilateral",
                "export_bilateral",
            }:
                raise ValueError(f"Unsupported stage action: {stage.get('action')}")
            if "config" in stage:
                with at_fields("config"):
                    stage["config"] = (
                        load_document(Path(base) / stage["config"])
                        if isinstance(stage["config"], str)
                        else absolute_paths(stage["config"], Path(base))
                    )
            if (
                stage.get("action") in {"dino_experiment", "transfer", "fuse"}
                or stage.get("views", {}).get("mode", "stored") != "stored"
            ):
                from morphofeatures.dino_experiments import resolve_paths

                stage.update(resolve_paths(stage, base))
            for key in (
                "checkpoint",
                "embedding",
                "annotations",
                "raw",
                "segmentation",
                "crops",
                "label_ids",
                "masks",
                "labels",
                "id_mapping",
                "input_config",
                "model_repository",
                "cache",
                "object_index",
                "analysis",
                "candidates",
                "sides",
                "adjacency_table",
                "positions",
                "graph_cache",
            ):
                if key == "analysis" and stage.get("action") == "distill":
                    continue  # Distillation's analysis is a settings block, not a report path.
                if stage.get(key) and not is_remote_url(stage[key]):
                    stage[key] = str((Path(base) / Path(stage[key]).expanduser()).resolve())
            for options in [stage.get("bilateral", {})] + list(stage.get("identities", {}).values()) + list(stage.get("bilateral", {}).get("identities", {}).values()):
                for key in ("candidates", "sides", "id_mapping"):
                    if options.get(key):
                        options[key] = str((Path(base) / Path(options[key]).expanduser()).resolve())
            for options in (stage, stage.get("bilateral", {})):
                if options.get("comparison_reports"):
                    options["comparison_reports"] = {name: str((Path(base) / Path(value).expanduser()).resolve()) for name, value in options["comparison_reports"].items()}
            for options in (stage.get("platybrowser", {}), stage.get("remote_options", {})):
                if options.get("cache_directory"):
                    options["cache_directory"] = str(
                        (Path(base) / Path(options["cache_directory"]).expanduser()).resolve()
                    )
            if (
                validate
                and stage.get("action") == "preprocess"
                and stage.get("source") == "platybrowser"
                and stage.get("layout") != "distributed_patches"
            ):
                from morphofeatures.analysis.platybrowser import resolve_platybrowser_preprocessing

                with at_fields("platybrowser"):
                    stage.update(resolve_platybrowser_preprocessing(stage))
            if (
                validate
                and stage.get("action") == "preprocess"
                and stage.get("source") == "ome_zarr"
                and stage.get("layout") != "distributed_patches"
            ):
                from morphofeatures.data.ome_zarr import resolve_ome_zarr_preprocessing

                with at_fields("ome_zarr"):
                    stage.update(resolve_ome_zarr_preprocessing(stage))
            if stage.get("embeddings"):
                stage["embeddings"] = {
                    k: str((Path(base) / Path(v).expanduser()).resolve())
                    for k, v in stage["embeddings"].items()
                }
            if stage.get("action") == "joint_analyze":
                datasets = stage.get("datasets", [])
                if not isinstance(datasets, list) or any(not isinstance(d, dict) for d in datasets):
                    raise FieldError("Datasets must be a list of mappings.", "datasets")
                for dataset in datasets:
                    for key in ("embedding", "annotations", "input_config"):
                        if dataset.get(key):
                            dataset[key] = str(
                                (Path(base) / Path(dataset[key]).expanduser()).resolve()
                            )
            if stage.get("attachments"):
                stage["attachments"] = [
                    str((Path(base) / Path(v).expanduser()).resolve()) for v in stage["attachments"]
                ]
            if not validate:
                continue
            if stage.get("use_published_layout"):
                from morphofeatures.analysis.published import references

                reference = references().get(stage.get("published_reference"), {})
                if not reference.get("layout") or Path(stage.get("embedding", "")).resolve() != Path(reference.get("embedding", "")).resolve():
                    raise FieldError("Saved published coordinates require the matching published feature file. Use this published feature file, or turn off Preserve published UMAP coordinates and clusters to fit your own map.", "published_reference", "embedding", "use_published_layout")
            if stage["action"] == "context":
                from morphofeatures.analysis.context_sources import resolve_context_source
                from morphofeatures.analysis.spatial_context import validate_context

                stage.update(resolve_context_source(stage))
                validate_context(stage)
                if not (stage.get("from_embedding") or stage.get("from_extraction")) and not (stage.get("embedding") and Path(stage["embedding"]).is_file()):
                    raise FieldError("Choose a saved embedding or an earlier named output to build spatial context.", "embedding")
            if stage["action"] == "bilateral" or stage.get("bilateral", {}).get("enabled"):
                from morphofeatures.analysis.bilateral import defaults, validate_bilateral

                count = (
                    len(set(stage.get("embeddings", {})) | set(stage.get("from_embeddings", [])))
                    if stage["action"] in {"bilateral", "compare"} else 1
                )
                if stage["action"] == "bilateral":
                    validate_bilateral({**defaults(), **stage}, representation_count=count)
                else:
                    with prefix_fields("bilateral"):
                        validate_bilateral(
                            {**defaults(), **stage["bilateral"]}, representation_count=count
                        )
            if stage.get("from_embedding") and stage["from_embedding"] not in named_embeddings:
                raise FieldError("Choose an earlier named embedding output", "from_embedding")
            if stage["action"] == "distill":
                from morphofeatures.analysis.distillation import validate_distillation

                validate_distillation(stage)
                if not (stage.get("from_embedding") or stage.get("from_extraction")) and not (
                    stage.get("embedding") and Path(stage["embedding"]).is_file()
                ):
                    raise FieldError("Choose a saved embedding or an earlier named output.", "embedding")
            if (
                stage["action"] in {"train", "extract"}
                and not allow_synthetic
                and not (stage.get("from_preprocessing") or stage.get("from_training"))
            ):
                data = stage.get("config", {}).get("data", {})
                if not data.get("crops") and data.get("source") not in {
                    "n5_masked_patches",
                    "physical_volume",
                }:
                    raise FieldError(
                        "Please select prepared crops, or load a grouped N5 model/data YAML. No real input data is configured.",
                        "config.data.crops",
                    )
            if stage["action"] == "export_bilateral":
                from morphofeatures.analysis.bilateral_export import validate_bilateral_export

                validate_bilateral_export(stage, require_report=not stage.get("from_bilateral"))
            if stage["action"] == "export_mobie":
                from morphofeatures.analysis.mobie_export import resolve_mobie_export

                linked = stage.get("from_analysis", False) or stage.get("from_bilateral", False)
                if stage.get("from_analysis") and not available.intersection({"analyze", "compare"}):
                    raise ValueError("from_analysis requires an earlier analyze or compare stage")
                # A preceding preparation can supply its spatial source before its
                # crops/config files exist. Analysis results are resolved by the worker.
                export_settings = stage
                if linked and prepared:
                    from morphofeatures.analysis.mobie_export import SOURCE_FIELDS

                    export_settings = {
                        **{k: v for k, v in prepared.items() if k in SOURCE_FIELDS},
                        **stage,
                    }
                stage.update(resolve_mobie_export(export_settings, require_analysis=not linked))
            if stage["action"] == "export_labels":
                from morphofeatures.analysis.label_volumes import validate_volume_export

                validate_volume_export(stage)
            if stage["action"] in {"train", "extract"}:
                grouped = (
                    stage.get("from_preprocessing")
                    and prepared
                    and prepared.get("layout") == "distributed_patches"
                ) or stage.get("config", {}).get("data", {}).get("source") == "grouped_crops"
                if grouped and (stage["action"] == "train" or stage.get("model", "mae") == "mae"):
                    raise FieldError(
                        "Distributed fine patches currently feed DINO. The crop MAE expects one crop per object; choose DINO here or select Single object crop on Prepare data. Existing grouped-N5 MAE models require their original N5 format.",
                        "config.data.source",
                        "model",
                    )
                if grouped and stage.get("from_preprocessing"):
                    views = stage.setdefault("views", {})
                    if views.get("normalization", "unit") != "unit":
                        raise FieldError(
                            "Distributed patches already share per-object contrast. Select unit under DINO intensity normalization to preserve it.",
                            "views.normalization",
                        )
                    views.setdefault("normalization", "unit")
                    views.setdefault("resize", "letterbox")
                    if views.get("mode", "stored") != "stored":
                        raise FieldError(
                            "Saved distributed patches need stored DINO views", "views.mode"
                        )
            if stage["action"] == "train":
                if "config" not in stage:
                    raise FieldError("Load an MAE model/data YAML before training.", "config")
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
                    stage["config"] = _validate_stage_training(config, base, require_data=False)
                else:
                    stage["config"] = _validate_stage_training(stage["config"], base)
            if stage["action"] == "preprocess":
                from morphofeatures.data.preprocessing import validate_preprocessing

                validate_preprocessing(stage)
                prepared = stage
            if stage["action"] == "extract":
                from morphofeatures.data.object_selection import validate_extraction_selection

                validate_extraction_selection(stage)
                if (
                    stage.get("from_preprocessing")
                    and stage.get("model", "mae") == "mae"
                    and stage.get("config", {}).get("data", {}).get("source") == "n5_masked_patches"
                ):
                    raise ValueError(
                        "Prepared whole-object crops need a crop MAE checkpoint/configuration; grouped N5 patch models use a different input layout"
                    )
                if stage.get("model", "mae") not in {"mae", "dinov2", "dinov3"}:
                    raise ValueError("Extraction model must be mae, dinov2, or dinov3")
                if not stage.get("checkpoint") and not stage.get("from_training"):
                    raise FieldError(
                        "Choose an existing checkpoint, or link this stage to earlier training.",
                        "checkpoint",
                    )
                if stage.get("checkpoint") and not Path(stage["checkpoint"]).is_file():
                    raise ValueError(f"Checkpoint does not exist: {stage['checkpoint']}")
                if (
                    not stage.get("config")
                    and not stage.get("from_preprocessing")
                    and not stage.get("from_training")
                ):
                    raise ValueError(
                        "Extraction requires the model/data config or from_preprocessing"
                    )
                if stage.get("model", "mae") != "mae":
                    from morphofeatures.dino import VARIANTS, validate_dino_settings

                    if stage.get("variant") not in VARIANTS[stage["model"]]:
                        raise ValueError(
                            f"Choose a supported {stage['model']} variant: {sorted(VARIANTS[stage['model']])}"
                        )
                    if not (Path(stage.get("model_repository") or "") / "hubconf.py").is_file():
                        raise ValueError(
                            "DINO extraction requires model_repository pointing to an official local checkout"
                        )
                    if stage.get("from_training"):
                        raise ValueError(
                            "DINO uses pretrained backbone weights, not an MAE training checkpoint"
                        )
                    validate_dino_settings(stage, require_data=not stage.get("from_preprocessing"))
            if stage["action"] in {
                "analyze", "compare", "joint_analyze", "dino_experiment", "transfer"
            }:
                from morphofeatures.analysis.projection import validate_projection_settings
                from morphofeatures.analysis.variants import validate_variants

                validate_projection_settings(stage)
                validate_variants(stage)
            if stage["action"] == "analyze" and not (
                stage.get("from_extraction") or stage.get("from_embedding")
            ):
                if not stage.get("embedding") or not Path(stage["embedding"]).is_file():
                    raise FieldError(
                        "Select an existing embedding file, or use embeddings from an earlier extraction stage.",
                        "embedding",
                    )
            if stage["action"] in {"compare", "fuse", "bilateral"}:
                references = stage.get("from_embeddings", [])
                if any(name not in named_embeddings for name in references):
                    raise ValueError("from_embeddings must name earlier extraction stages")
                count = len(set(stage.get("embeddings", {})) | set(references))
                minimum = 1 if stage["action"] == "bilateral" else 2
                if stage["action"] == "compare":
                    from morphofeatures.analysis.projection import analysis_normalizations

                    minimum = 1 if len(analysis_normalizations(stage)) > 1 else 2
                if count < minimum:
                    raise FieldError("Add named embedding files or earlier named outputs (at least one for the bilateral probe, two for comparison/fusion).", "embeddings", "from_embeddings")
                for name, path in stage.get("embeddings", {}).items():
                    if not Path(path).is_file():
                        raise ValueError(f"Embedding for {name} does not exist: {path}")
            if stage["action"] == "fuse":
                from morphofeatures.analysis.fusion import (
                    dataset_namespace,
                    fusion_identity,
                    validate_fusion_settings,
                )
                from morphofeatures.analysis.joint import extraction_metadata

                validate_fusion_settings(stage)
                metadata = {
                    name: extraction_metadata(path) for name, path in stage.get("embeddings", {}).items()
                }
                if metadata and all(dataset_namespace(m) for m in metadata.values()):
                    fusion_identity(stage, metadata)
            if stage["action"] == "joint_analyze":
                from morphofeatures.analysis.joint import validate_joint_datasets

                validate_joint_datasets(stage, available=named_embeddings)
            if stage["action"] == "dino_experiment":
                from morphofeatures.dino_experiments import validate_experiment

                validate_experiment(stage)
            if stage["action"] == "transfer":
                from morphofeatures.analysis.transfer import (
                    validate_transfer_inputs,
                    validate_transfer_settings,
                )

                validate_transfer_settings(stage)
                if not stage.get("reference_model") and not stage.get("reference", {}).get(
                    "annotations"
                ):
                    raise FieldError(
                        "Select reference annotations or a saved reference model",
                        "reference.annotations",
                    )
                if stage.get("reference_model") and not Path(stage["reference_model"]).is_file():
                    raise FieldError("Saved reference model does not exist", "reference_model")
                if not stage.get("targets"):
                    raise FieldError("Add at least one target", "targets")
                specs = (
                    [("reference", stage.get("reference", {}))]
                    if not stage.get("reference_model")
                    else []
                )
                specs += [("targets", v) for v in stage.get("targets", [])]
                for key, spec in specs:
                    if spec.get("from_embedding") not in named_embeddings and not (
                        spec.get("embedding") and Path(spec["embedding"]).is_file()
                    ):
                        raise FieldError("Choose a saved embedding or an earlier named output", key)
                validate_transfer_inputs(stage, allow_pending=True)
            if stage.get("annotations") and not Path(stage["annotations"]).is_file():
                raise FieldError(
                    "Annotation table does not exist. Select an existing TSV/CSV or clear the optional field.",
                    "annotations",
                )
            if set(embedding_names(stage)) & named_embeddings:
                raise FieldError("Embedding output names must be unique, including .base, .pca and .adapter outputs.", "name")
        except (ValueError, TypeError, KeyError, OSError, ImportError) as error:
            if not validate:
                raise
            issue = stage_issue(error, index, stage)
            if issues is None:
                raise WorkflowValidationError(issue) from error
            issues.append(issue)
        finally:
            if isinstance(stage, dict):
                available.add(stage.get("action"))
                if stage.get("action") == "preprocess":
                    prepared = stage
                named_embeddings.update(embedding_names(stage))
    return document


def _validate_stage_training(config, base, **kwargs):
    try:
        return validate_training(config, base, **kwargs)
    except FieldError as error:
        raise FieldError(str(error), *(("config",) + path for path in error.fields)) from error


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
    memory_plans: tuple = ()


def submission_fingerprint(document, *, output_root, run_id, execution, dependency=None):
    document = deepcopy(document)
    for stage in document.get("stages", []):
        if stage.get("action") == "fuse":
            from morphofeatures.analysis.fusion import freeze_component_order

            freeze_component_order(stage)
    payload = {
        "document": document,
        "output_root": str(Path(output_root).expanduser().resolve()),
        "run_id": run_id,
        "execution": execution,
        "dependency": dependency or None,
    }
    return hashlib.sha256(yaml.safe_dump(payload, sort_keys=True).encode()).hexdigest()


def plan_workspace_job(
    document,
    *,
    output_root,
    run_id,
    execution="local",
    base=None,
    dependency=None,
    issues=None,
    allow_synthetic=True,
):
    """Validate and render the exact submission without persisting anything."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise FieldError(
            "Run ID requires 1–80 letters, digits, dots, underscores or dashes", "run_id"
        )
    if execution not in {"local", "slurm", "dry-run"}:
        raise FieldError("execution must be local, slurm, or dry-run", "execution")
    if dependency and execution != "slurm":
        raise FieldError(
            "Job dependencies require SLURM; use stages for a local pipeline", "dependency"
        )
    if dependency and not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", dependency):
        raise FieldError("Dependency must be a Slurm job ID", "dependency")
    fingerprint = submission_fingerprint(
        document, output_root=output_root, run_id=run_id, execution=execution, dependency=dependency
    )
    document = resolve_job(
        document, base or repository_root(), issues=issues, allow_synthetic=allow_synthetic
    )
    if issues:
        raise WorkflowValidationError(issues[0])
    resources = document.get("slurm", {})
    if execution == "local":
        # Only interpreter and CPU threads apply locally. Incomplete cluster mail
        # or account settings must not prevent switching a draft to local execution.
        resources = {
            "partition": "local",
            "cpus": resources.get("cpus", 4),
            "python_executable": resources.get("python_executable"),
        }
    profile = ClusterProfile.from_mapping("workspace", resources)
    from morphofeatures.dino_memory import attach_allocation, review_plans

    attach_allocation(document, execution, {"memory": profile.memory, "cpus": profile.cpus})
    memory_plans = review_plans(document)
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
        memory_plans=tuple(memory_plans),
    )


def submit_workspace_plan(plan, *, expected_fingerprint, dry_run=False):
    """Persist a reviewed plan and launch once; reject stale reviews and collisions."""
    if expected_fingerprint != plan.fingerprint:
        raise ValueError("Settings changed after review. Review the run again before submitting.")
    from morphofeatures.run_deletion import run_lock

    with run_lock(Path(plan.record.working_directory).parents[2], plan.record.run_id):
        return _submit_workspace_plan(plan, dry_run=dry_run)


def _submit_workspace_plan(plan, *, dry_run):
    document = yaml.safe_load(plan.resolved_yaml)
    execution = "dry-run" if dry_run else plan.execution
    document["worker"]["execution"] = execution
    record = replace(
        plan.record, application_state="dry_run" if execution == "dry-run" else "queued"
    )
    folder = Path(record.working_directory)
    registry = JobRegistry(Path(document["worker"]["registry"]))
    from morphofeatures.run_deletion import DELETING

    if any(r.application_state in DELETING for r in registry.list(run_id=record.run_id, include_deleted=True)):
        raise ValueError("This run ID belongs to deleted work. Choose a new run ID before submitting.")
    # Atomic reservation also protects against double clicks and concurrent sessions.
    folder.mkdir(parents=True, exist_ok=False)
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
            from morphofeatures.job_control import process_metadata

            write_json_atomic(folder / "process.json", process_metadata(record, process.pid))
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
    # Older saved worker bundles also gain the same allocation-aware behavior.
    from morphofeatures.dino_memory import attach_allocation

    folder = Path(worker.get("directory", path.parent / "results"))
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "status.json").exists():
        raise FileExistsError("This worker directory has already run; resubmit with a new run ID")
    registry = JobRegistry(Path(worker["registry"])) if worker.get("registry") else None
    writer = MetricWriter(folder / "metrics.jsonl")
    status = {"state": "running", "started": utc_now(), "stages": [], "record_id": worker.get("record_id")}
    status_path = folder / "status.json"
    artifacts = []
    previous = {}
    if registry:
        registry.update(worker["record_id"], application_state="running", started_at=utc_now())
    try:
        attach_allocation(document, worker.get("execution", "slurm" if os.environ.get("SLURM_JOB_ID") else "local"), document.get("slurm", {}))
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
                if prepared.get("inspection"):
                    supplied["inspection"] = prepared["inspection"]
                if prepared.get("mae"):
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
            if stage.get("from_embedding"):
                stage["embedding"] = previous["named_embeddings"][stage["from_embedding"]]
            if stage.get("from_analysis"):
                stage["analysis"] = previous["analysis_result"]
            if stage.get("from_bilateral"):
                stage["analysis"] = previous["bilateral"]
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
            elif action == "bilateral":
                from morphofeatures.analysis.bilateral import run_bilateral

                result = run_bilateral(stage["embeddings"], destination, stage)
            elif action == "context":
                from morphofeatures.analysis.spatial_context import build_context

                result = build_context(stage, destination, progress=lambda **v: writer.write("spatial_graph", **v))
                artifacts.append(str(destination / "context.json"))
            elif action == "joint_analyze":
                from morphofeatures.analysis.joint import joint_analyze

                for dataset in stage["datasets"]:
                    if dataset.get("from_embedding"):
                        dataset["embedding"] = previous["named_embeddings"][
                            dataset.pop("from_embedding")
                        ]
                result = joint_analyze(stage, destination)
            elif action == "dino_experiment":
                from morphofeatures.dino_experiments import run_experiment

                result = run_experiment(stage, destination, writer=writer)
            elif action == "fuse":
                from morphofeatures.analysis.fusion import fuse_embeddings

                result = fuse_embeddings(stage, destination)
            elif action == "distill":
                from morphofeatures.analysis.distillation import distill_embeddings

                result = distill_embeddings(stage, destination)
                for method in ("base", "pca", "adapter"):
                    embedding = str(destination / method / "embeddings.npz")
                    artifacts.append(embedding)
                    if stage.get("name"):
                        previous.setdefault("named_embeddings", {})[stage["name"] + "." + method] = embedding
                if stage.get("name"):
                    previous["named_embeddings"][stage["name"]] = str(destination / "adapter/embeddings.npz")
            elif action == "transfer":
                from morphofeatures.analysis.transfer import transfer_labels

                for spec in [stage.get("reference", {})] + stage.get("targets", []):
                    if spec.get("from_embedding"):
                        spec["embedding"] = previous["named_embeddings"][spec.pop("from_embedding")]
                result = transfer_labels(stage, destination)
            elif action == "export_labels":
                from morphofeatures.analysis.label_volumes import export_label_volumes

                result = export_label_volumes(
                    stage, destination, progress=lambda **v: writer.write("volume_export", **v)
                )
            elif action == "export_bilateral":
                from morphofeatures.analysis.bilateral_export import export_bilateral

                result = export_bilateral(stage, destination, progress=lambda **v: writer.write("bilateral_export", **v))
            elif action == "export_mobie":
                from morphofeatures.analysis.mobie_export import export_mobie

                result = export_mobie(
                    stage, destination, progress=lambda **v: writer.write("mobie_export", **v)
                )
            else:
                from morphofeatures.data.preprocessing import preprocess

                result = preprocess(
                    stage, destination, progress=lambda **v: writer.write("preprocessing", **v)
                )
            previous[action] = str(result)
            if action in {"analyze", "compare"}:
                previous["analysis_result"] = str(result)
            if action in {"extract", "fuse", "context"} and stage.get("name"):
                previous.setdefault("named_embeddings", {})[stage["name"]] = str(result)
            artifacts.append(str(result))
            entry.update(
                state="completed", result=str(result), seconds=time.perf_counter() - started
            )
            if action == "train":
                previous["training_seconds"] = entry["seconds"]
            writer.write("stage_completed", **entry)
        status.update(state="completed", completed=utc_now(), artifacts=artifacts)
    except BaseException as error:
        status.update(state="failed", error=str(error), completed=utc_now())
        if status["stages"]:
            status["stages"][-1].update(state="failed", error=str(error))
            issue = stage_issue(error, status["stages"][-1]["index"], stage).to_dict()
            status["issue"] = issue
            status["stages"][-1]["issue"] = issue
        writer.write("failed", error=str(error))
        raise
    finally:
        status["artifacts"] = artifacts
        try:
            write_json_atomic(status_path, status)
        except Exception as error:
            status.update(
                state="failed",
                completed=utc_now(),
                error=f"{status.get('error', '')} Could not save final worker status: {error}".strip(),
            )
            raise
        finally:
            if registry:
                from morphofeatures.job_status import reconcile_job_states

                # Keep a terminal registry outcome even if status.json cannot be
                # saved (e.g. a full output disk). Successful writes precede this.
                registry.update(
                    worker["record_id"],
                    application_state=status["state"],
                    completed_at=status["completed"],
                    exit_status="1" if status["state"] == "failed" else "0",
                    error_message=status.get("error"),
                )
                registry.append_artifacts(worker["record_id"], artifacts)
                reconcile_job_states(registry, record_ids=(worker["record_id"],))
    return status_path
