"""Workflow interaction and reviewed-submission contracts."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from morphofeatures.data.io import export_embeddings
from morphofeatures.registry import JobRegistry
from morphofeatures.workspace_jobs import (
    plan_workspace_job,
    submission_fingerprint,
    submit_workspace_plan,
)
from morphofeatures.workspace_state import (
    import_workflow,
    load_draft,
    new_draft,
    save_draft,
    validate_draft_inputs,
    workflow_document,
)


def test_review_has_no_persistent_side_effects_and_submission_matches(tmp_path):
    draft = new_draft("demo")
    root = tmp_path / "workspace"
    plan = plan_workspace_job(
        draft["document"], output_root=root, run_id="reviewed", execution="slurm"
    )
    assert not root.exists()
    job = yaml.safe_load(plan.resolved_yaml)
    assert job["stages"][0]["config"]["paths"]["checkpoint"] == str(
        root / "experiments/reviewed/workspace/00-train/checkpoint.pt"
    )
    record = submit_workspace_plan(plan, expected_fingerprint=plan.fingerprint, dry_run=True)
    folder = Path(record.working_directory)
    assert (folder / "job.slurm").read_text() == plan.script
    saved = yaml.safe_load((folder / "job.yaml").read_text())
    assert saved["stages"] == job["stages"]
    assert saved["worker"]["execution"] == "dry-run"
    assert record.application_state == "dry_run"
    assert not (folder / "process.json").exists()


def test_stale_review_and_duplicate_submission_never_launch(tmp_path, monkeypatch):
    document = new_draft("demo")["document"]
    plan = plan_workspace_job(document, output_root=tmp_path, run_id="once", execution="slurm")
    changed = deepcopy(document)
    changed["stages"][0]["config"]["training"]["epochs"] = 7
    fingerprint = submission_fingerprint(
        changed, output_root=tmp_path, run_id="once", execution="slurm"
    )
    calls = []
    monkeypatch.setattr(
        "morphofeatures.workspace_jobs.SlurmScheduler.submit",
        lambda self, path, dependency: calls.append(path) or "12345",
    )
    with pytest.raises(ValueError, match="changed after review"):
        submit_workspace_plan(plan, expected_fingerprint=fingerprint)
    assert not (tmp_path / "experiments").exists()
    record = submit_workspace_plan(plan, expected_fingerprint=plan.fingerprint)
    assert record.slurm_job_id == "12345"
    assert JobRegistry.under_output_root(tmp_path).get(record.id).slurm_job_id == "12345"
    with pytest.raises(FileExistsError):
        submit_workspace_plan(plan, expected_fingerprint=plan.fingerprint)
    assert len(calls) == 1


def test_scheduler_failure_is_retained(tmp_path, monkeypatch):
    plan = plan_workspace_job(
        new_draft("demo")["document"], output_root=tmp_path, run_id="rejected", execution="slurm"
    )

    def fail(*args):
        raise RuntimeError("Invalid account")

    monkeypatch.setattr("morphofeatures.workspace_jobs.SlurmScheduler.submit", fail)
    with pytest.raises(RuntimeError, match="Invalid account"):
        submit_workspace_plan(plan, expected_fingerprint=plan.fingerprint)
    record = JobRegistry.under_output_root(tmp_path).get(plan.record.id)
    assert record.application_state == "failed"
    assert record.error_message == "Invalid account"


def test_draft_persistence_and_conflicting_sessions(tmp_path):
    draft = new_draft("crops")
    draft["document"]["future_extension"] = {"values": [None, "unchanged"]}
    saved = save_draft(tmp_path, draft)
    second_session = load_draft(tmp_path, saved["id"])
    saved["revision"] += 1
    saved["document"]["stages"][0]["config"]["training"]["epochs"] = 17
    latest = save_draft(tmp_path, saved)
    assert latest["saved_revision"] == latest["revision"]
    with pytest.raises(ValueError, match="another session"):
        save_draft(tmp_path, second_session)
    reopened = load_draft(tmp_path, saved["id"])
    assert reopened["document"]["future_extension"] == draft["document"]["future_extension"]
    assert reopened["document"]["stages"][0]["config"]["training"]["epochs"] == 17


def test_import_allows_incomplete_inputs_and_resolves_relative_paths(tmp_path):
    model = tmp_path / "model.yaml"
    model.write_text(
        yaml.safe_dump({"mae": {}, "data": {"crops": "missing.npy"}, "future": [1, 2]})
    )
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(yaml.safe_dump({"stages": [{"action": "train", "config": "model.yaml"}]}))
    loaded = import_workflow(pipeline)
    config = loaded["stages"][0]["config"]
    assert config["data"]["crops"] == str(tmp_path / "missing.npy")
    assert config["future"] == [1, 2]


def test_real_data_entry_points_require_explicit_data_and_linked_shape():
    for route in ("crops", "checkpoint"):
        draft = new_draft(route)
        with pytest.raises(ValueError, match="select prepared crops"):
            validate_draft_inputs(draft["document"])
    demo = new_draft("demo")
    validate_draft_inputs(demo["document"], allow_synthetic=demo["allow_synthetic"])
    raw = new_draft("raw")
    raw["document"]["stages"][0]["crop_shape"] = [16, 32, 16]
    assert workflow_document(raw)["stages"][1]["config"]["mae"]["input_shape"] == [16, 32, 16]


def _app(tmp_path):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "morphofeatures/dashboard.py"), default_timeout=30
    )
    app.session_state["workspace_root"] = str(tmp_path)
    return app.run()


def _button(app, label):
    return next(widget for widget in app.button if widget.label == label)


def _widget(app, kind, label):
    return next(widget for widget in getattr(app, kind) if widget.label == label)


@pytest.mark.parametrize("route", ["raw", "crops", "checkpoint", "embeddings", "demo"])
def test_all_starting_points_and_stages_render(tmp_path, route):
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value(route).run()
    _button(app, "Create workflow").click().run()
    for step in ("Prepare data", "Train", "Analyze", "Review & run"):
        app.radio(key="step_widget").set_value(step).run()
        assert not app.exception
    if route == "crops":
        _button(app, "Review run").click().run()
        assert any("select prepared crops" in error.value for error in app.error)
        assert not (tmp_path / "experiments").exists()


def test_ui_retains_edits_and_invalidates_review(tmp_path):
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value("demo").run()
    _button(app, "Create workflow").click().run()
    assert type(_widget(app, "number_input", "Training epochs").value) is int
    assert _widget(app, "number_input", "Learning rate").value == 0.001
    _widget(app, "number_input", "Training epochs").set_value(13).run()
    _button(app, "Save draft").click().run()
    for page in ("Results", "Runs", "Tools", "Workspace settings", "Help", "Workflow"):
        app.radio(key="workspace_navigation").set_value(page).run()
        assert not app.exception
        assert not app.get("doc_string")
    assert _widget(app, "number_input", "Training epochs").value == 13
    app.radio(key="step_widget").set_value("Review & run").run()
    _button(app, "Review run").click().run()
    assert not app.error
    assert not (tmp_path / "experiments").exists()
    app.radio(key="step_widget").set_value("Train").run()
    _widget(app, "number_input", "Training epochs").set_value(17).run()
    app.radio(key="step_widget").set_value("Review & run").run()
    assert any("Settings changed after review" in warning.value for warning in app.warning)
    assert not any(b.label == "Run locally" for b in app.button)
    _button(app, "Review run").click().run()
    _button(app, "Save dry-run bundle").click().run()
    assert not app.exception and not app.error
    assert app.radio(key="workspace_navigation").value == "Runs"
    files = list(tmp_path.glob("experiments/*/workspace/job.yaml"))
    assert len(files) == 1
    assert yaml.safe_load(files[0].read_text())["stages"][0]["config"]["training"]["epochs"] == 17
    reopened = _app(tmp_path)
    _button(reopened, "Open saved draft").click().run()
    assert reopened.radio(key="step_widget").value == "Train"
    assert _widget(reopened, "number_input", "Training epochs").value == 13


def test_yaml_buffer_cannot_overwrite_newer_form_edits(tmp_path):
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value("demo").run()
    _button(app, "Create workflow").click().run()
    _widget(app, "checkbox", "Edit pipeline YAML").check().run()
    _widget(app, "number_input", "Training epochs").set_value(9).run()
    assert _button(app, "Apply YAML to draft").disabled
    _button(app, "Reload YAML from current form settings").click().run()
    assert not _button(app, "Apply YAML to draft").disabled
    value = yaml.safe_load(_widget(app, "text_area", "Pipeline YAML").value)
    assert value["stages"][0]["config"]["training"]["epochs"] == 9
    _widget(app, "text_area", "Pipeline YAML").set_value("invalid: [").run()
    _button(app, "Apply YAML to draft").click().run()
    assert app.error
    assert _widget(app, "number_input", "Training epochs").value == 9


def test_reopen_saved_result_does_not_submit(tmp_path, monkeypatch):
    result = tmp_path / "saved-result"
    result.mkdir()
    export_embeddings(
        result / "embeddings.npz",
        np.array([10, 20, 30]),
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )
    (result / "coordinates.tsv").write_text(
        "label_id\tcluster\tpca_1\tpca_2\n10\t0\t1\t2\n20\t1\t3\t4\n30\t1\t5\t6\n"
    )
    source = result / "analysis.json"
    source.write_text(
        json.dumps(
            {
                "schema": "morphofeatures.analysis.v1",
                "interpretation": "Exploratory analysis.",
                "embedding": "embeddings.npz",
            }
        )
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Reopening a result must not submit a job")

    monkeypatch.setattr("morphofeatures.workflow_ui.submit_workspace_plan", forbidden)
    app = _app(tmp_path)
    app.radio(key="workspace_navigation").set_value("Results").run()
    _widget(app, "text_input", "Saved analysis.json, comparison.json, or embedding file").set_value(
        str(source)
    ).run()
    assert not app.exception and not app.error
    assert _widget(app, "selectbox", "Object for nearest-neighbor inspection").options == [
        "10",
        "20",
        "30",
    ]
    assert not (tmp_path / "experiments").exists()
