"""Stop only disposable workers launched by these tests, never workspace jobs."""

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from test_workspace_usability import app_at, make_run, widget

from morphofeatures.artifacts import write_json_atomic
from morphofeatures.experiments import cancel_job, refresh_jobs
from morphofeatures.job_control import local_target, process_metadata, stop_local_job
from morphofeatures.job_status import reconcile_job_states
from morphofeatures.registry import JobRecord, JobRegistry
from morphofeatures.run_deletion import prepare_run_deletion
from morphofeatures.slurm import FakeScheduler, SlurmScheduler


@contextmanager
def local_worker(root, *, legacy=False, stubborn_child=False):
    folder = root / "experiments/worker/workspace"
    folder.mkdir(parents=True)
    # A tiny disposable worker with the real command shape, so the UI can
    # validate/display it without launching scientific code in this test.
    script = folder / "morphofeatures" / "__main__.py"
    script.parent.mkdir()
    (script.parent / "__init__.py").touch()
    script.write_text(
        "import pathlib, signal, subprocess, sys, time\n"
        + (
            "child = subprocess.Popen([sys.executable, '-c', \"import pathlib, signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path('child-ready').touch(); signal.pause()\"])\n"
            "while not pathlib.Path('child-ready').exists(): time.sleep(0.01)\n"
            if stubborn_child
            else ""
        )
        + "pathlib.Path('ready').touch()\nsignal.pause()\n"
    )
    config = folder / "job.yaml"
    config.write_text("stages: []\n")
    result = folder / "result.txt"
    result.write_text("partial result to keep")
    command = (sys.executable, "-m", "morphofeatures", "workspace-run", "--config", str(config))
    record = JobRecord(
        run_id="worker",
        workflow="workspace_run",
        stage="pipeline",
        command=command,
        working_directory=str(folder),
        config_snapshot=str(config),
        submission_key="worker",
        stdout_path=str(folder / "stdout.log"),
        stderr_path=str(folder / "stderr.log"),
        metrics_path=str(folder / "metrics.jsonl"),
        application_state="running",
        artifacts=(str(result),),
    )
    registry = JobRegistry.under_output_root(root)
    registry.create(record)
    real_killpg = os.killpg
    process = subprocess.Popen(command, cwd=folder, start_new_session=True)
    try:
        data = process_metadata(record, process.pid)
        if legacy:
            data = {key: data[key] for key in ("pid", "started")}
        write_json_atomic(folder / "process.json", data)
        deadline = time.monotonic() + 5
        while not (folder / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (folder / "ready").exists()
        yield registry, record, process
    finally:
        try:
            real_killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def wait_cancelled(registry, record):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        reconcile_job_states(registry)
        if registry.get(record.id).application_state == "cancelled":
            return registry.get(record.id)
        time.sleep(0.02)
    pytest.fail("Test worker group did not stop")


@pytest.mark.parametrize("legacy", [False, True])
def test_stop_local_worker_and_keep_outputs(tmp_path, legacy):
    with local_worker(tmp_path, legacy=legacy) as (registry, record, process):
        stop_local_job(record, registry)
        finished = wait_cancelled(registry, record)
        assert process.wait(timeout=5) == -signal.SIGTERM
        assert finished.provenance["local_stop"]["confirmed_at"]
        assert finished.raw_scheduler_state == "LOCAL_CANCELLED"
        assert Path(record.artifacts[0]).read_text() == "partial result to keep"
        # A leftover running status marker cannot resurrect a confirmed stop.
        write_json_atomic(Path(record.working_directory) / "status.json", {"state": "running"})
        reconcile_job_states(registry)
        assert registry.get(record.id).application_state == "cancelled"
        assert prepare_run_deletion(registry, tmp_path, record.run_id)


def test_force_kill_stubborn_child_after_leader_exits(tmp_path):
    with local_worker(tmp_path, stubborn_child=True) as (registry, record, process):
        stop_local_job(record, registry)
        process.wait(timeout=5)
        reconcile_job_states(registry)
        assert registry.get(record.id).application_state == "cancelling"
        assert local_target(registry.get(record.id))[1]
        with pytest.raises(ValueError, match="finished"):
            prepare_run_deletion(registry, tmp_path, record.run_id)
        stop_local_job(record, registry, force=True)
        finished = wait_cancelled(registry, record)
        assert finished.provenance["local_stop"]["signal"] == "SIGKILL"


@pytest.mark.parametrize("change", ["hostname", "create_time", "command"])
def test_local_identity_mismatch_never_signals(tmp_path, monkeypatch, change):
    with local_worker(tmp_path) as (registry, record, process):
        path = Path(record.working_directory) / "process.json"
        data = json.loads(path.read_text())
        if change == "hostname":
            data["hostname"] = "another-host"
        elif change == "create_time":
            data["create_time"] -= 100
        else:
            with registry._connect() as connection:
                connection.execute(
                    "UPDATE jobs SET command = ? WHERE id = ?",
                    (json.dumps([sys.executable, "wrong.py"]), record.id),
                )
        write_json_atomic(path, data)
        signals = []
        monkeypatch.setattr(os, "killpg", lambda *args: signals.append(args))
        with pytest.raises(ValueError, match="host|reused|command"):
            stop_local_job(record, registry)
        assert not signals and process.poll() is None
        assert registry.get(record.id).application_state == "running"


def test_slurm_can_cancel_live_allocation_after_worker_failure(tmp_path):
    from morphofeatures.metrics import utc_now

    record = make_run(tmp_path, "failed-but-live", [])
    registry = JobRegistry.under_output_root(tmp_path)
    registry.update(
        record.id,
        application_state="failed",
        slurm_job_id="56",
        raw_scheduler_state="RUNNING",
        completed_at=utc_now(),
        exit_status="1",
    )
    scheduler = FakeScheduler()
    scheduler.set_state("56", "RUNNING")
    refresh_jobs(registry, scheduler)
    cancelled = cancel_job(record, registry, scheduler)
    assert cancelled.raw_scheduler_state == "CANCEL_REQUESTED"
    assert cancelled.completed_at is None
    refresh_jobs(registry, scheduler)
    assert registry.get(record.id).application_state == "cancelled"


def test_slurm_request_failure_preserves_live_state(tmp_path):
    record = make_run(tmp_path, "live", [])
    registry = JobRegistry.under_output_root(tmp_path)
    registry.update(
        record.id, application_state="running", slurm_job_id="57", raw_scheduler_state="RUNNING"
    )
    with pytest.raises(RuntimeError, match="unknown fake job"):
        cancel_job(record, registry, FakeScheduler())
    assert registry.get(record.id).application_state == "running"


def test_slurm_stop_is_visible_and_requires_confirmation(tmp_path, monkeypatch):
    record = make_run(tmp_path, "cluster", [])
    registry = JobRegistry.under_output_root(tmp_path)
    registry.update(
        record.id, application_state="running", slurm_job_id="58", raw_scheduler_state="RUNNING"
    )
    cancelled = []
    monkeypatch.setattr(SlurmScheduler, "cancel", lambda self, job_id: cancelled.append(job_id))
    app = app_at(tmp_path)
    app.session_state["runs:selected"] = record.id
    app.radio(key="workspace_navigation").set_value("Runs").run()
    assert not app.exception
    assert widget(app, "button", "Cancel SLURM job").disabled
    details = next(e for e in app.expander if e.label == "Execution details and configuration")
    assert not any(b.label == "Cancel SLURM job" for b in details.button)
    widget(app, "checkbox", "Confirm cancellation of this SLURM job").check().run()
    widget(app, "button", "Cancel SLURM job").click().run()
    assert not app.exception and cancelled == ["58"]
    assert registry.get(record.id).raw_scheduler_state == "CANCEL_REQUESTED"
    assert any("Cancellation requested" in message.value for message in app.success)


def test_local_stop_ui_and_force_option(tmp_path):
    with local_worker(tmp_path) as (registry, record, process):
        app = app_at(tmp_path)
        app.session_state["runs:selected"] = record.id
        app.radio(key="workspace_navigation").set_value("Runs").run()
        assert not app.exception
        assert widget(app, "button", "Stop local job").disabled
        assert widget(app, "checkbox", "Force kill local worker (SIGKILL)")
        widget(app, "checkbox", "Confirm stopping this local job").check().run()
        widget(app, "button", "Stop local job").click().run()
        assert not app.exception
        wait_cancelled(registry, record)
        assert process.poll() == -signal.SIGTERM
        app.run()
        assert app.dataframe[0].value.iloc[0]["state"] == "cancelled"
