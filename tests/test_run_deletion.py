"""Deletion tests only use synthetic runs and files under pytest's temporary root."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_workspace_usability import app_at, make_run, widget

from morphofeatures.registry import DuplicateSubmissionError, JobRegistry
from morphofeatures.run_deletion import (
    delete_run,
    delete_runs,
    prepare_run_deletion,
    prepare_runs_deletion,
    run_write_guard,
)


def remove(registry, root, run_id):
    plan = prepare_run_deletion(registry, root, run_id)
    return delete_run(registry, root, run_id, confirmation_token=plan["token"])


def test_whole_run_removed_cache_inputs_and_dependency_identity_preserved(tmp_path):
    record = make_run(tmp_path, "old", [{"action": "analyze"}])
    registry = JobRegistry.under_output_root(tmp_path)
    root = Path(record.working_directory).parent
    cache = tmp_path / "embedding_cache"
    cache.mkdir()
    external = cache / "shared.npz"
    external.write_bytes(b"shared vectors")
    (root / "linked-cache").symlink_to(cache, target_is_directory=True)
    registry.append_artifacts(record.id, [str(external)])
    other = make_run(tmp_path, "keep", [{"action": "analyze", "embedding": str(external)}])
    sibling = replace(
        record,
        id="sibling",
        workflow="mae_encode",
        submission_key="sibling",
        parent_job_id=record.id,
        working_directory=str(root / "mae_encode"),
    )
    Path(sibling.working_directory).mkdir()
    (Path(sibling.working_directory) / "vectors.npz").write_bytes(b"owned")
    registry.create(sibling)
    plan = remove(registry, tmp_path, record.run_id)
    assert len(plan["jobs"]) == 2 and plan["inventory"]["symlinks"] == 1
    assert str(external) in plan["preserved_paths"]
    assert not root.exists() and external.read_bytes() == b"shared vectors"
    assert [r.id for r in registry.list()] == [other.id]
    deleted = registry.get(record.id)
    assert deleted.application_state == "deleted" and not deleted.artifacts
    assert deleted.provenance["deletion"]["deleted_at"]
    assert registry.get(sibling.id).parent_job_id == record.id
    with pytest.raises(KeyError):
        registry.append_artifacts(record.id, ["late-result"])
    with pytest.raises(KeyError):
        registry.update(record.id, application_state="completed")
    with pytest.raises(DuplicateSubmissionError, match="new run ID"):
        registry.create(replace(record, id="new", workflow="other", submission_key="new"))
    with pytest.raises(ValueError, match="deleted"):
        with run_write_guard(Path(record.working_directory) / "analysis.json"):
            pytest.fail("Deleted outputs must not launch interactive analysis")


@pytest.mark.parametrize("state", ["created", "queued", "running", "unknown"])
def test_all_jobs_must_finish(tmp_path, state):
    record = make_run(tmp_path, "run", [])
    registry = JobRegistry.under_output_root(tmp_path)
    registry.create(
        replace(
            record, id="active", workflow="other", submission_key="active", application_state=state
        )
    )
    with pytest.raises(ValueError, match="finished"):
        prepare_run_deletion(registry, tmp_path, record.run_id)
    assert Path(record.working_directory).exists()


def test_cancel_request_requires_scheduler_confirmation(tmp_path):
    from morphofeatures.experiments import refresh_jobs
    from morphofeatures.slurm import FakeScheduler

    record = make_run(tmp_path, "cancelled", [])
    registry = JobRegistry.under_output_root(tmp_path)
    registry.update(
        record.id,
        application_state="cancelled",
        slurm_job_id="50",
        raw_scheduler_state="CANCEL_REQUESTED",
    )
    with pytest.raises(ValueError, match="cancellation is still pending"):
        prepare_run_deletion(registry, tmp_path, record.run_id)
    scheduler = FakeScheduler()
    scheduler.set_state("50", "CANCELLED", "0:0")
    refresh_jobs(registry, scheduler)
    assert registry.get(record.id).raw_scheduler_state == "CANCELLED"
    remove(registry, tmp_path, record.run_id)


def test_changed_confirmation_cannot_delete(tmp_path):
    record = make_run(tmp_path, "changing", [])
    registry = JobRegistry.under_output_root(tmp_path)
    plan = prepare_run_deletion(registry, tmp_path, record.run_id)
    (Path(record.working_directory) / "new-result.tsv").write_text("new output")
    with pytest.raises(ValueError, match="changed after review"):
        delete_run(registry, tmp_path, record.run_id, confirmation_token=plan["token"])
    assert Path(record.working_directory).is_dir()
    plan = prepare_run_deletion(registry, tmp_path, record.run_id)
    registry.update(record.id, application_state="running")
    with pytest.raises(ValueError, match="finished"):
        delete_run(registry, tmp_path, record.run_id, confirmation_token=plan["token"])


def test_dependencies_warn_or_block_without_deleting_other_runs(tmp_path):
    record = make_run(tmp_path, "upstream", [])
    dependent = make_run(
        tmp_path, "downstream", [{"action": "analyze", "embedding": record.artifacts[0]}]
    )
    registry = JobRegistry.under_output_root(tmp_path)
    plan = prepare_run_deletion(registry, tmp_path, record.run_id)
    assert plan["dependent_runs"][0]["run_id"] == dependent.run_id
    registry.update(dependent.id, application_state="queued")
    with pytest.raises(ValueError, match="depends on this run"):
        delete_run(registry, tmp_path, record.run_id, confirmation_token=plan["token"])
    registry.update(dependent.id, application_state="completed")
    remove(registry, tmp_path, record.run_id)
    assert Path(dependent.working_directory).exists()


def test_live_interactive_analysis_blocks_deletion(tmp_path):
    record = make_run(tmp_path, "interactive", [])
    registry = JobRegistry.under_output_root(tmp_path)
    index = Path(record.working_directory) / "analysis_variants.json"
    index.write_text(json.dumps({"variants": [{"state": "running", "origin": "results"}]}))
    with pytest.raises(ValueError, match="interactive analysis"):
        prepare_run_deletion(registry, tmp_path, record.run_id)
    index.write_text(json.dumps({"variants": [{"state": "completed", "origin": "results"}]}))
    remove(registry, tmp_path, record.run_id)


def test_dependency_through_owned_symlink_is_not_hidden_by_resolution(tmp_path):
    record = make_run(tmp_path, "links", [])
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "vectors.npz").write_bytes(b"shared")
    link = Path(record.working_directory) / "shared"
    link.symlink_to(cache, target_is_directory=True)
    dependent = make_run(
        tmp_path, "uses-link", [{"action": "analyze", "embedding": str(link / "vectors.npz")}]
    )
    registry = JobRegistry.under_output_root(tmp_path)
    registry.update(dependent.id, application_state="running")
    with pytest.raises(ValueError, match="depends on this run"):
        prepare_run_deletion(registry, tmp_path, record.run_id)
    assert (cache / "vectors.npz").read_bytes() == b"shared"


def test_cleanup_failure_remains_retryable(tmp_path, monkeypatch):
    import morphofeatures.run_deletion as module

    record = make_run(tmp_path, "retry", [])
    registry = JobRegistry.under_output_root(tmp_path)
    original = module.shutil.rmtree

    def fail(path):
        raise PermissionError("test failure")

    fail.avoids_symlink_attacks = True
    monkeypatch.setattr(module.shutil, "rmtree", fail)
    with pytest.raises(ValueError, match="cleanup is incomplete"):
        remove(registry, tmp_path, record.run_id)
    assert registry.get(record.id).application_state == "deletion_failed"
    assert len(registry.list()) == 1
    cleanup = Path(registry.get(record.id).provenance["deletion"]["cleanup_directory"])
    assert cleanup.is_dir() and not Path(record.working_directory).exists()
    monkeypatch.setattr(module.shutil, "rmtree", original)
    remove(registry, tmp_path, record.run_id)
    assert not cleanup.exists() and not registry.list()


def test_paths_cannot_escape_run_and_root_symlinks_are_refused(tmp_path):
    record = make_run(tmp_path, "unsafe", [])
    registry = JobRegistry.under_output_root(tmp_path)
    root = Path(record.working_directory).parent
    elsewhere = tmp_path / "source-data"
    root.rename(elsewhere)
    root.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        prepare_run_deletion(registry, tmp_path, record.run_id)
    with pytest.raises(ValueError, match="Invalid run ID"):
        prepare_run_deletion(registry, tmp_path, "../source-data")
    assert (elsewhere / "workspace/result.json").is_file()


def test_missing_run_folder_and_invalid_snapshot_can_be_removed(tmp_path):
    import shutil

    record = make_run(tmp_path, "already-missing", [])
    registry = JobRegistry.under_output_root(tmp_path)
    Path(record.config_snapshot).write_text("invalid: [yaml")
    plan = prepare_run_deletion(registry, tmp_path, record.run_id)
    assert plan["warnings"]
    shutil.rmtree(Path(record.working_directory).parent)
    remove(registry, tmp_path, record.run_id)
    assert not registry.list()


def test_deleted_id_cannot_be_resubmitted_or_recreated_by_result_rename(tmp_path):
    from types import SimpleNamespace

    from morphofeatures.analysis.variants import rename_result
    from morphofeatures.workspace_jobs import submit_workspace_plan

    record = make_run(tmp_path, "old-id", [])
    registry = JobRegistry.under_output_root(tmp_path)
    remove(registry, tmp_path, record.run_id)
    plan = SimpleNamespace(
        record=record,
        fingerprint="same",
        execution="dry-run",
        resolved_yaml=json.dumps({"worker": {"registry": str(registry.path)}}),
    )
    with pytest.raises(ValueError, match="deleted work"):
        submit_workspace_plan(plan, expected_fingerprint="same")
    with pytest.raises(ValueError, match="deleted"):
        rename_result(Path(record.artifacts[0]), "New name")
    assert not Path(record.working_directory).parent.exists()


def test_ui_requires_confirmation_cancel_preserves_and_delete_clears_navigation(tmp_path):
    record = make_run(tmp_path, "delete-me", [])
    other = make_run(tmp_path, "keep-me", [])
    app = app_at(tmp_path)
    app.session_state["runs:selected"] = record.id
    app.radio(key="workspace_navigation").set_value("Runs").run()
    widget(app, "button", "Review deletion").click().run()
    assert not app.exception
    assert any("Are you sure" in entry.value for entry in app.warning)
    assert widget(app, "button", "Permanently delete run").disabled
    widget(app, "button", "Keep run").click().run()
    assert Path(record.working_directory).exists()
    widget(app, "button", "Review deletion").click().run()
    checkbox = next(
        w for w in app.checkbox if w.label.startswith("I understand that this permanently")
    )
    checkbox.check().run()
    widget(app, "button", "Permanently delete run").click().run()
    assert not app.exception
    assert not Path(record.working_directory).exists()
    assert Path(other.working_directory).exists()
    assert any("Deleted delete-me" in entry.value for entry in app.success)
    assert app.dataframe[0].value.run.tolist() == [other.run_id]
    assert app.session_state.filtered_state.get("runs:selected") != record.id
    assert all(record.id not in str(screen) for screen in app.session_state["navigation:history"])
    app.radio(key="workspace_navigation").set_value("Results").run()
    assert not app.exception
    assert not any(record.run_id in str(w.options) for w in app.selectbox)


def test_batch_deduplicates_runs_and_can_delete_connected_runs_together(tmp_path):
    source = make_run(tmp_path, "source", [])
    dependent = make_run(tmp_path, "dependent", [{"embedding": source.artifacts[0]}])
    other = make_run(tmp_path, "keep", [])
    registry = JobRegistry.under_output_root(tmp_path)
    plan = prepare_runs_deletion(
        registry, tmp_path, [source.run_id, dependent.run_id, source.run_id]
    )
    assert len(plan["runs"]) == 2
    assert all(Path(r.working_directory).exists() for r in (source, dependent, other))
    result = delete_runs(
        registry, tmp_path, [dependent.run_id, source.run_id], confirmation_token=plan["token"]
    )
    assert {p["run_id"] for p in result["deleted"]} == {source.run_id, dependent.run_id}
    assert not result["failed"]
    assert [r.id for r in registry.list()] == [other.id]
    assert Path(other.working_directory).exists()
    assert all(not Path(r.working_directory).exists() for r in (source, dependent))


@pytest.mark.parametrize("change", ["active", "file", "selection"])
def test_batch_rechecks_every_member_before_removing_anything(tmp_path, change):
    first = make_run(tmp_path, "a-first", [])
    last = make_run(tmp_path, "z-last", [])
    registry = JobRegistry.under_output_root(tmp_path)
    ids = [first.run_id, last.run_id]
    plan = prepare_runs_deletion(registry, tmp_path, ids)
    if change == "active":
        registry.update(last.id, application_state="running")
    elif change == "file":
        (Path(last.working_directory) / "extra.tsv").write_text("changed")
    else:
        ids = [first.run_id]
    with pytest.raises(ValueError, match="finished|changed after review"):
        delete_runs(registry, tmp_path, ids, confirmation_token=plan["token"])
    assert all(Path(r.working_directory).exists() for r in (first, last))
    assert registry.get(first.id).application_state == "completed"


def test_batch_staging_failure_restores_all_folders_and_records(tmp_path, monkeypatch):
    first = make_run(tmp_path, "a-first", [])
    last = make_run(tmp_path, "z-last", [])
    registry = JobRegistry.under_output_root(tmp_path)
    ids = [first.run_id, last.run_id]
    plan = prepare_runs_deletion(registry, tmp_path, ids)
    rename = Path.rename

    def fail_last(path, destination):
        if path == Path(last.working_directory).parent:
            raise PermissionError("cannot stage last folder")
        return rename(path, destination)

    monkeypatch.setattr(Path, "rename", fail_last)
    with pytest.raises(PermissionError, match="cannot stage"):
        delete_runs(registry, tmp_path, ids, confirmation_token=plan["token"])
    assert all(Path(r.artifacts[0]).is_file() for r in (first, last))
    assert all(registry.get(r.id).application_state == "completed" for r in (first, last))


def test_batch_cleanup_failure_reports_successes_and_remaining_run_can_retry(tmp_path, monkeypatch):
    import morphofeatures.run_deletion as module

    first = make_run(tmp_path, "a-fails", [])
    last = make_run(tmp_path, "z-succeeds", [])
    registry = JobRegistry.under_output_root(tmp_path)
    ids = [first.run_id, last.run_id]
    plan = prepare_runs_deletion(registry, tmp_path, ids)
    rmtree = module.shutil.rmtree

    def fail_first(path):
        if path.name.startswith(first.run_id + "-"):
            raise PermissionError("fixture permission error")
        return rmtree(path)

    fail_first.avoids_symlink_attacks = True
    monkeypatch.setattr(module.shutil, "rmtree", fail_first)
    result = delete_runs(registry, tmp_path, ids, confirmation_token=plan["token"])
    assert [p["run_id"] for p in result["deleted"]] == [last.run_id]
    assert [p["run_id"] for p in result["failed"]] == [first.run_id]
    assert "cleanup is incomplete" in result["failed"][0]["error"]
    assert registry.get(first.id).application_state == "deletion_failed"
    assert [r.id for r in registry.list()] == [first.id]
    monkeypatch.setattr(module.shutil, "rmtree", rmtree)
    remove(registry, tmp_path, first.run_id)
    assert not registry.list()


def _select_rows(app, records):
    from morphofeatures.run_presentation import select_table_run

    event = {"table": {"selection": {"rows": list(range(len(records)))}}}
    select_table_run(event, "table", [r.id for r in records])
    app.session_state["runs:selected"] = event["runs:selected"]
    app.session_state["runs:selected_ids"] = event["runs:selected_ids"]
    app.run()


def _worker_status(record, state="failed"):
    from morphofeatures.metrics import utc_now

    status = {
        "state": state,
        "record_id": record.id,
        "completed": utc_now(),
        "stages": [],
        "artifacts": list(record.artifacts),
    }
    if state == "failed":
        status["error"] = "Original scientific failure"
    (Path(record.working_directory) / "status.json").write_text(json.dumps(status))
    return status


@pytest.mark.parametrize("application", ["running", "queued", "unknown", "failed", "completed"])
def test_final_worker_outcome_repairs_stale_scheduler_before_deletion(tmp_path, application):
    record = make_run(tmp_path, "stale", [])
    registry = JobRegistry.under_output_root(tmp_path)
    status = _worker_status(record)
    registry.update(
        record.id, application_state=application, slurm_job_id="50", raw_scheduler_state="RUNNING"
    )
    plan = prepare_run_deletion(registry, tmp_path, record.run_id)
    repaired = registry.get(record.id)
    assert repaired.application_state == "failed"
    assert repaired.raw_scheduler_state is None
    assert repaired.error_message == status["error"]
    assert repaired.exit_status == "1"
    assert repaired.provenance["status_resolution"]["previous_scheduler_state"] == "RUNNING"
    delete_run(registry, tmp_path, record.run_id, confirmation_token=plan["token"])
    assert not Path(record.working_directory).exists()


def test_failed_downstream_worker_does_not_block_upstream_deletion(tmp_path):
    source = make_run(tmp_path, "source", [])
    dependent = make_run(tmp_path, "failed-dependent", [{"embedding": source.artifacts[0]}])
    registry = JobRegistry.under_output_root(tmp_path)
    registry.update(
        dependent.id, application_state="running", raw_scheduler_state="RUNNING", slurm_job_id="51"
    )
    _worker_status(dependent)
    plan = prepare_run_deletion(registry, tmp_path, source.run_id)
    assert plan["dependent_runs"][0]["state"] == "failed"
    delete_run(registry, tmp_path, source.run_id, confirmation_token=plan["token"])
    assert Path(dependent.working_directory).exists()


def test_confirmed_live_scheduler_after_worker_failure_still_blocks(tmp_path):
    from morphofeatures.experiments import refresh_jobs
    from morphofeatures.slurm import FakeScheduler

    source = make_run(tmp_path, "source", [])
    dependent = make_run(tmp_path, "requeued", [{"embedding": source.artifacts[0]}])
    registry = JobRegistry.under_output_root(tmp_path)
    registry.update(
        dependent.id, application_state="failed", raw_scheduler_state="RUNNING", slurm_job_id="51"
    )
    _worker_status(dependent)
    scheduler = FakeScheduler()
    scheduler.set_state("51", "RUNNING")
    refresh_jobs(registry, scheduler)
    with pytest.raises(ValueError, match="SLURM still records an active"):
        prepare_run_deletion(registry, tmp_path, dependent.run_id)
    with pytest.raises(ValueError, match="depends on this run"):
        prepare_run_deletion(registry, tmp_path, source.run_id)
    # A later query failure must not erase the last confirmed live allocation.
    scheduler.states.clear()
    refresh_jobs(registry, scheduler)
    with pytest.raises(ValueError, match="SLURM still records an active"):
        prepare_run_deletion(registry, tmp_path, dependent.run_id)
    scheduler.set_state("51", "FAILED", "1:0")
    refresh_jobs(registry, scheduler)
    remove(registry, tmp_path, source.run_id)
    remove(registry, tmp_path, dependent.run_id)


def test_old_job_or_error_text_alone_cannot_enable_deletion(tmp_path):
    record = make_run(tmp_path, "uncertain", [])
    registry = JobRegistry.under_output_root(tmp_path)
    registry.update(
        record.id,
        application_state="running",
        error_message="An error was printed",
        started_at="2020-01-01T00:00:00+00:00",
    )
    (Path(record.working_directory) / "status.json").write_text('{"state": "failed"')
    with pytest.raises(ValueError, match="finished"):
        prepare_run_deletion(registry, tmp_path, record.run_id)
    assert Path(record.working_directory).exists()


def test_runs_ui_shows_worker_failure_and_enables_review_without_slurm(tmp_path):
    record = make_run(tmp_path, "stale-ui", [])
    registry = JobRegistry.under_output_root(tmp_path)
    registry.update(
        record.id, application_state="running", raw_scheduler_state="PENDING", slurm_job_id="52"
    )
    _worker_status(record)
    app = app_at(tmp_path)
    app.session_state["runs:selected"] = record.id
    app.radio(key="workspace_navigation").set_value("Runs").run()
    assert not app.exception
    assert app.dataframe[0].value.iloc[0]["state"] == "failed"
    assert "outdated" in app.dataframe[0].value.iloc[0]["raw state"]
    assert not widget(app, "button", "Review deletion").disabled
    widget(app, "button", "Review deletion").click().run()
    assert any("Are you sure" in w.value for w in app.warning)
    assert Path(record.working_directory).exists()


def test_ui_batch_confirmation_cancel_and_filter_selection(tmp_path):
    first = make_run(tmp_path, "first", [])
    second = make_run(tmp_path, "second", [])
    keep = make_run(tmp_path, "keep", [])
    app = app_at(tmp_path)
    app.radio(key="workspace_navigation").set_value("Runs").run()
    _select_rows(app, [first, second])
    widget(app, "button", "Review deletion of selected runs").click().run()
    assert not app.exception
    assert any("these 2 runs" in w.value for w in app.warning)
    assert widget(app, "button", "Permanently delete 2 runs").disabled
    next(w for w in app.checkbox if w.label.startswith("I understand")).check().run()
    widget(app, "button", "Keep runs").click().run()
    assert not app.exception
    assert all(Path(r.working_directory).exists() for r in (first, second))
    widget(app, "button", "Review deletion of selected runs").click().run()
    assert widget(app, "button", "Permanently delete 2 runs").disabled
    # Hidden rows must not be silently retained for a later batch deletion.
    widget(app, "text_input", "Filter run ID, name or workflow").set_value("first").run()
    assert app.session_state["runs:selected_ids"] == [first.id]
    widget(app, "text_input", "Filter run ID, name or workflow").set_value("").run()
    assert app.session_state["runs:selected_ids"] == [first.id]
    _select_rows(app, [first, second])
    widget(app, "button", "Review deletion of selected runs").click().run()
    assert widget(app, "button", "Permanently delete 2 runs").disabled
    next(w for w in app.checkbox if w.label.startswith("I understand")).check().run()
    widget(app, "button", "Permanently delete 2 runs").click().run()
    assert not app.exception
    assert any("Deleted 2 run(s)" in w.value for w in app.success)
    assert app.dataframe[0].value.run.tolist() == [keep.run_id]
    assert app.session_state["runs:selected_ids"] == []
    assert all(not Path(r.working_directory).exists() for r in (first, second))
    assert Path(keep.working_directory).exists()
