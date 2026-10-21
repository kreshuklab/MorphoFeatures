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

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda i: registry.append_artifacts(record.id, [f"result-{i}.json"]), range(12)))
    assert all(f"result-{i}.json" in registry.get(record.id).artifacts for i in range(12))
    renamed = registry.rename_run(record.run_id, "A readable experiment")[0]
    assert renamed.display_name == "A readable experiment"
    assert renamed.run_id == record.run_id and renamed.working_directory == record.working_directory
    assert renamed.provenance["name_history"][0]["previous"] == record.run_id
    registry.update(record.id, application_state="running")
    with pytest.raises(ValueError, match="finished"):
        registry.rename_run(record.run_id, "Too early")
    assert registry.get(record.id).display_name == "A readable experiment"


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


@pytest.mark.parametrize("accounting", ["FAILED", "purged", "unavailable"])
def test_refresh_preserves_terminal_worker_error_and_artifacts(tmp_path, accounting):
    from test_run_deletion import _worker_status
    from test_workspace_usability import make_run

    from morphofeatures.slurm import SchedulerQuery

    record = make_run(tmp_path, "stale-failure", [])
    registry = JobRegistry.under_output_root(tmp_path)
    status = _worker_status(record)
    registry.update(
        record.id, application_state="failed", slurm_job_id="90", raw_scheduler_state="RUNNING"
    )
    scheduler = FakeScheduler()
    if accounting == "FAILED":
        scheduler.set_state("90", "FAILED", "1:0")
    if accounting == "unavailable":
        scheduler.query = lambda ids: SchedulerQuery({}, ("sacct unavailable",))
    refresh_jobs(registry, scheduler)
    repaired = registry.get(record.id)
    assert repaired.application_state == "failed"
    assert repaired.error_message == status["error"]
    assert set(record.artifacts) <= set(repaired.artifacts)
    assert repaired.raw_scheduler_state == ("FAILED" if accounting == "FAILED" else None)


@pytest.mark.parametrize("failure", [False, True])
def test_worker_persists_final_status_before_finalizing_registry(tmp_path, monkeypatch, failure):
    from test_workspace_usability import make_run

    import morphofeatures.dino_memory as memory
    from morphofeatures.workspace_jobs import run_job

    record = make_run(tmp_path, "initialization", [])
    registry = JobRegistry.under_output_root(tmp_path)
    registry.update(
        record.id, application_state="queued", raw_scheduler_state="PENDING", slurm_job_id="91"
    )
    Path(record.config_snapshot).write_text(
        yaml.safe_dump(
            {
                "worker": {
                    "record_id": record.id,
                    "registry": str(registry.path),
                    "directory": record.working_directory,
                },
                "stages": [],
            }
        )
    )
    if failure:

        def fail(*args, **kwargs):
            raise ValueError("Invalid memory setup")

        monkeypatch.setattr(memory, "attach_allocation", fail)
        with pytest.raises(ValueError, match="Invalid memory setup"):
            run_job(Path(record.config_snapshot))
    else:
        run_job(Path(record.config_snapshot))
    final = registry.get(record.id)
    assert final.application_state == ("failed" if failure else "completed")
    assert final.exit_status == ("1" if failure else "0")
    assert final.raw_scheduler_state is None
    assert (Path(record.working_directory) / "status.json").is_file()


def test_worker_status_write_failure_still_marks_registry_failed(tmp_path, monkeypatch):
    from test_workspace_usability import make_run

    import morphofeatures.workspace_jobs as jobs

    record = make_run(tmp_path, "full-disk", [])
    registry = JobRegistry.under_output_root(tmp_path)
    registry.update(
        record.id, application_state="queued", slurm_job_id="92", raw_scheduler_state="PENDING"
    )
    Path(record.config_snapshot).write_text(
        yaml.safe_dump(
            {
                "worker": {
                    "record_id": record.id,
                    "registry": str(registry.path),
                    "directory": record.working_directory,
                },
                "stages": [],
            }
        )
    )

    def disk_full(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(jobs, "write_json_atomic", disk_full)
    with pytest.raises(OSError, match="No space left"):
        jobs.run_job(Path(record.config_snapshot))
    final = registry.get(record.id)
    assert final.application_state == "failed" and final.exit_status == "1"
    assert final.raw_scheduler_state is None
    assert "Could not save final worker status" in final.error_message


def test_worker_outcome_uses_identity_and_aware_times(tmp_path):
    import json
    from dataclasses import replace

    from test_run_deletion import _worker_status
    from test_workspace_usability import make_run

    from morphofeatures.job_status import worker_outcome

    record = make_run(tmp_path, "identity", [])
    status = _worker_status(record)
    # Naive SLURM local time cannot be compared with a UTC worker timestamp.
    assert worker_outcome(replace(record, started_at="2099-01-01T00:00:00"))
    assert not worker_outcome(replace(record, started_at="2099-01-01T00:00:00+00:00"))
    status["record_id"] = "another-job"
    (Path(record.working_directory) / "status.json").write_text(json.dumps(status))
    assert not worker_outcome(record)
