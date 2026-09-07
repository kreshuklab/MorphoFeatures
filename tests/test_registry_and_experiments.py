from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import yaml

from morphofeatures.experiments import cancel_job, plan_job, refresh_jobs, submit_plan
from morphofeatures.registry import DuplicateSubmissionError, JobRegistry
from morphofeatures.slurm import ClusterProfile, DryRunScheduler, FakeScheduler
from morphofeatures.workflows import WorkflowRequest


def _profile():
    return ClusterProfile("test", "compute", time="00:05:00", memory="2G", cpus=1)


def test_mae_plan_warns_when_multiple_gpus_would_be_idle(tmp_path, repo_root):
    plan = plan_job(
        WorkflowRequest("mae_train", repo_root / "configs" / "smoke.yaml"),
        ClusterProfile("multi-gpu", "gpu", gpus=2),
        run_id="multi-gpu-warning",
        output_root=tmp_path,
    )
    assert any("one CUDA device" in warning for warning in plan.warnings)


def test_registry_round_trip_duplicate_protection_and_concurrent_updates(tmp_path, repo_root):
    registry = JobRegistry(tmp_path / "runtime" / "registry.sqlite3")
    plan = plan_job(
        WorkflowRequest("mae_train", repo_root / "configs" / "smoke.yaml"),
        _profile(),
        run_id="round-trip",
        output_root=tmp_path,
    )
    record = submit_plan(plan, registry, DryRunScheduler(), repository=repo_root)
    assert registry.get(record.id).command == plan.command
    assert registry.get(record.id).application_state == "dry_run"
    assert Path(record.config_snapshot).is_file()
    with pytest.raises(DuplicateSubmissionError):
        submit_plan(plan, registry, DryRunScheduler(), repository=repo_root)

    def update(index):
        registry.update(record.id, raw_scheduler_state="UPDATE_{}".format(index))

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(update, range(12)))
    assert registry.get(record.id).raw_scheduler_state.startswith("UPDATE_")


def test_workflow_specific_input_override_is_snapshotted_without_overwrite(tmp_path, repo_root):
    registry = JobRegistry(tmp_path / "registry.sqlite3")
    source_config = tmp_path / "source.yaml"
    source_config.write_text(
        (repo_root / "configs" / "smoke.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    crops = tmp_path / "crops.npy"
    np.save(crops, np.zeros((4, 8, 8, 8), dtype=np.float32))
    plan = plan_job(
        WorkflowRequest("mae_train", source_config, input_path=crops),
        _profile(),
        run_id="snapshot-input",
        output_root=tmp_path,
    )
    record = submit_plan(plan, registry, DryRunScheduler(), repository=repo_root)
    snapshot = Path(record.config_snapshot)
    saved = snapshot.read_text(encoding="utf-8")
    assert yaml.safe_load(saved)["data"]["crops"] == str(crops.resolve())

    source_config.write_text("seed: 999\n", encoding="utf-8")
    changed_plan = plan_job(
        WorkflowRequest("mae_train", source_config),
        _profile(),
        run_id="snapshot-input",
        output_root=tmp_path,
    )
    with pytest.raises(DuplicateSubmissionError):
        submit_plan(changed_plan, registry, DryRunScheduler(), repository=repo_root)
    assert snapshot.read_text(encoding="utf-8") == saved


def test_fake_scheduler_submission_refresh_dependency_failure_and_cancellation(tmp_path, repo_root):
    registry = JobRegistry(tmp_path / "registry.sqlite3")
    scheduler = FakeScheduler(next_job_id=700)
    training_plan = plan_job(
        WorkflowRequest("mae_train", repo_root / "configs" / "smoke.yaml"),
        _profile(),
        run_id="linked-training",
        output_root=tmp_path,
    )
    training = submit_plan(training_plan, registry, scheduler, repository=repo_root)
    assert training.slurm_job_id == "700"
    encoding_plan = plan_job(
        WorkflowRequest(
            "mae_encode",
            repo_root / "configs" / "smoke.yaml",
            checkpoint=Path(training.checkpoint_path),
            output=tmp_path / "encoded.npy",
        ),
        _profile(),
        run_id="linked-encoding",
        output_root=tmp_path,
        dependency_job_id=training.slurm_job_id,
        parent_job_id=training.id,
    )
    encoding = submit_plan(encoding_plan, registry, scheduler, repository=repo_root)
    assert scheduler.submissions[-1][1] == "700"
    assert encoding.parent_job_id == training.id

    scheduler.set_state("700", "RUNNING")
    updated, warnings = refresh_jobs(registry, scheduler)
    assert updated == 2
    assert not warnings
    assert registry.get(training.id).application_state == "running"
    cancelled = cancel_job(registry.get(training.id), registry, scheduler)
    assert cancelled.application_state == "cancelled"

    failure_scheduler = FakeScheduler(fail_submission=True)
    failure_plan = plan_job(
        WorkflowRequest("mae_train", repo_root / "configs" / "smoke.yaml"),
        _profile(),
        run_id="submission-failure",
        output_root=tmp_path,
    )
    failed = submit_plan(failure_plan, registry, failure_scheduler, repository=repo_root)
    assert failed.application_state == "failed"
    assert "fake submission failure" in failed.error_message


def test_unknown_accounting_state_is_retained(tmp_path, repo_root):
    registry = JobRegistry(tmp_path / "registry.sqlite3")
    scheduler = FakeScheduler(next_job_id=800)
    plan = plan_job(
        WorkflowRequest("mae_train", repo_root / "configs" / "smoke.yaml"),
        _profile(),
        run_id="purged-job",
        output_root=tmp_path,
    )
    record = submit_plan(plan, registry, scheduler, repository=repo_root)
    scheduler.states.clear()
    refresh_jobs(registry, scheduler)
    refreshed = registry.get(record.id)
    assert refreshed.application_state == "unknown"
    assert "delayed or purged" in refreshed.error_message
