"""Intent routing, partial results, scientific availability and readable progress."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from morphofeatures.artifact_availability import probe, selected_artifact
from morphofeatures.data.io import export_embeddings
from morphofeatures.metrics import progress_events
from morphofeatures.result_catalog import run_results
from morphofeatures.workflow_catalog import (
    PRESET_FAMILIES,
    guided_draft,
    restore_draft,
    retain_draft,
    workflow_steps,
)
from morphofeatures.workspace_state import STARTING_POINTS, new_draft


def app_at(root):
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).parents[1] / "morphofeatures/dashboard.py"), default_timeout=30)
    app.session_state["workspace_root"] = str(root)
    return app.run()


def widget(app, kind, label):
    return next(w for w in getattr(app, kind) if w.label == label)


def test_all_presets_have_an_intent_and_existing_documents_stay_compatible():
    assert set(PRESET_FAMILIES) == set(STARTING_POINTS)
    for start in STARTING_POINTS:
        draft = new_draft(start)
        steps = workflow_steps(draft["document"]["stages"])
        assert steps[0] == "Start" and steps[-1] == "Review & run"
    for inputs, recipe, actions in (
        ("images", "preprocess", ["preprocess"]),
        ("images", "dino", ["preprocess", "extract", "analyze"]),
        ("images", "checkpoint", ["preprocess", "extract", "analyze"]),
        ("prepared", "dino", ["extract", "analyze"]),
        ("checkpoint", "checkpoint", ["extract", "analyze"]),
    ):
        draft = guided_draft(family="create", inputs=inputs, recipe=recipe)
        assert [s["action"] for s in draft["document"]["stages"]] == actions
    for task in ("distill", "transfer"):
        stages = guided_draft(family="learn", inputs="images", task=task)["document"]["stages"]
        assert [s["action"] for s in stages] == ["preprocess", "extract", task]
        target = stages[-1]["reference"] if task == "transfer" else stages[-1]
        assert target["from_embedding"] == stages[1]["name"]
    stages = guided_draft(family="explore", inputs="checkpoint", task="compare")["document"]["stages"]
    assert stages[-1]["from_embeddings"] == [stages[0]["name"]]
    assert list(stages[-1]["embeddings"].values()) == [""]


def test_switching_and_back_keep_live_draft_edits():
    from morphofeatures.workspace_navigation import go_back, remember_screen
    state = {"workspace_root": "/example", "workspace_navigation": "Workflow"}
    a, b = new_draft("crops"), new_draft("embeddings")
    retain_draft(state, "/example", a, "Train")
    remember_screen(state)
    state["workflow:/example"]["document"]["stages"][0]["config"]["training"]["epochs"] = 17
    retain_draft(state, "/example", b, "Analyze")
    remember_screen(state)
    go_back(state)
    assert state["workflow:/example"]["id"] == a["id"]
    assert state["workflow:/example"]["document"]["stages"][0]["config"]["training"]["epochs"] == 17
    assert restore_draft(state, "/example", b["id"])
    assert state["workflow:/example"]["id"] == b["id"]


@pytest.mark.parametrize("failure", ["missing", "empty", "duplicate", "nan", "wrong_shape", "truncated"])
def test_vectors_are_usable_only_after_content_checks(tmp_path, failure):
    path = tmp_path / "vectors.npz"
    if failure == "empty":
        path.touch()
    elif failure != "missing":
        ids = np.array([7, 7 if failure == "duplicate" else 2**60 + 3], dtype=np.int64)
        features = np.array([[1., 2.], [np.nan if failure == "nan" else 3., 4.]])
        np.savez(path, label_ids=ids, features=features[:1] if failure == "wrong_shape" else features)
        if failure == "truncated":
            path.write_bytes(path.read_bytes()[:50])
    assert not selected_artifact(path).ready


def test_projection_tables_are_not_vectors_and_cached_checks_invalidate(tmp_path):
    table = tmp_path / "saved.tsv"
    table.write_text("label_id\tpca_1\tpca_2\tcluster\tknown_label\n1152921504606846979\t1\t2\t0\tNeuron\n")
    assert selected_artifact(table).kind == "coordinates"
    assert selected_artifact(table).ready
    vectors = export_embeddings(tmp_path / "vectors.npz", [2**60 + 3, 7], np.ones((2, 3)))
    assert probe(vectors).state == "unchecked"
    assert selected_artifact(vectors).ready
    np.savez(vectors, label_ids=np.array([7, 7]), features=np.ones((2, 3)))
    assert not selected_artifact(vectors).ready


def test_images_archives_and_identity_tables_are_not_false_result_views(tmp_path):
    for filename in ("plot.png", "export.zip"):
        file = tmp_path / filename
        file.write_bytes(b"not a usable artifact")
        assert not probe(file).ready
    mapping = tmp_path / "id_mapping.tsv"
    mapping.write_text("label_id\tnative_id\n1\t9\n2\t8\n")
    assert selected_artifact(mapping).kind == "table"
    config = tmp_path / "broken.yaml"
    config.write_text("data: [")
    assert not probe(config).ready


def test_relocation_is_explicit_workspace_scoped_and_preserves_report(tmp_path):
    from morphofeatures.artifact_locations import configure_locations, record_location
    from morphofeatures.result_catalog import vector_artifacts

    workspace = tmp_path / "workspace"
    original = tmp_path / "lost.npz"
    moved = export_embeddings(tmp_path / "moved.npz", [2, 7], np.ones((2, 3)))
    report = tmp_path / "analysis.json"
    report.write_text(json.dumps({"schema": "morphofeatures.analysis.v1", "embedding": str(original)}))
    content = report.read_bytes()
    try:
        configure_locations(workspace)
        assert not selected_artifact(original).ready
        record_location(workspace, original, moved, reason="Same file moved")
        assert selected_artifact(original).ready
        assert list(vector_artifacts(probe(report).metadata, report).values()) == [str(moved)]
        assert report.read_bytes() == content
        configure_locations(tmp_path / "other_workspace")
        assert not selected_artifact(original).ready
    finally:
        configure_locations(tmp_path / "other_workspace")


def test_draft_switching_retains_page_invalid_fields_and_unapplied_yaml(tmp_path):
    app = app_at(tmp_path)
    widget(app, "selectbox", "Starting point").set_value("crops").run()
    widget(app, "button", "Create workflow").click().run()
    first = app.session_state["workflow:" + str(tmp_path)]["id"]
    widget(app, "text_input", "Patch size").set_value("[4,").run()
    widget(app, "checkbox", "Edit pipeline YAML").check().run()
    widget(app, "text_area", "Pipeline YAML").set_value("stages: [").run()
    widget(app, "button", "Start / load another workflow").click().run()
    widget(app, "selectbox", "Starting point").set_value("embeddings").run()
    widget(app, "button", "Create a new workflow").click().run()
    widget(app, "selectbox", "Session drafts").set_value(first).run()
    widget(app, "button", "Resume selected draft").click().run()
    # Starting another workflow is itself the last page; the stage edits survive.
    app.radio(key="step_widget").set_value("Train").run()
    assert widget(app, "text_input", "Patch size").value == "[4,"
    assert widget(app, "text_area", "Pipeline YAML").value == "stages: ["
    assert not app.exception
    app.radio(key="step_widget").set_value("Review & run").run()
    assert widget(app, "button", "Review run").disabled


def test_runs_structured_events_hide_resource_columns_and_keep_raw_records(tmp_path):
    from morphofeatures.registry import JobRecord, JobRegistry

    metrics = tmp_path / "metrics.jsonl"
    events = [
        {"timestamp": "now", "event": "dino_memory_plan", "rss_mb": 100},
        {"timestamp": "now", "event": "stage_started", "stage": 0, "resource_usage": 30, "modulation": "reduce"},
        {"timestamp": "later", "event": "failed", "error": "Missing input", "gpu_reserved_mb": 200},
    ]
    metrics.write_text("".join(json.dumps(e) + "\n" for e in events))
    record = JobRecord(run_id="events", workflow="workspace_run", stage="pipeline", command=("python", "-m", "morphofeatures", "workspace-run"),
                       submission_key="events", working_directory=str(tmp_path), config_snapshot=str(tmp_path / "job.yaml"),
                       stdout_path=str(tmp_path / "out.log"), stderr_path=str(tmp_path / "err.log"),
                       metrics_path=str(metrics), application_state="failed")
    JobRegistry.under_output_root(tmp_path).create(record)
    app = app_at(tmp_path)
    app.radio(key="workspace_navigation").set_value("Runs").run()
    app.session_state["runs:selected"] = record.id
    app.run()
    panel = next(e for e in app.expander if e.label == "Structured event timeline (latest 100)")
    table = panel.dataframe[0].value
    assert list(table.event) == ["stage_started", "failed"]
    assert not set(table.columns) & {"resource_usage", "modulation", "rss_mb", "gpu_reserved_mb"}
    assert "error" in table
    diagnostics = next(e for e in app.expander if e.label == "Resource and runtime diagnostics")
    assert "dino_memory_plan" in diagnostics.json[0].value
    assert [json.loads(line) for line in metrics.read_text().splitlines()] == events
    assert not app.exception


def test_partial_dino_results_offer_vectors_without_missing_map_selectors(tmp_path):
    vector = export_embeddings(tmp_path / "original.npz", [2, 7], np.ones((2, 3)))
    report = tmp_path / "experiment.json"
    report.write_text(json.dumps({"schema": "morphofeatures.dino_experiment.v1", "execution_state": "partial",
                                 "original_embeddings": [{"dataset": "Cells", "representation": "Fine", "path": str(vector)}],
                                 "comparisons": [{"path": "missing/comparison.json"}]}))
    app = app_at(tmp_path)
    app.radio(key="workspace_navigation").set_value("Results").run()
    widget(app, "text_input", "Saved analysis.json, comparison.json, joint_analysis.json, or embedding file").set_value(str(report)).run()
    assert not app.exception and not app.error
    assert any("Partial experiment" in message.value for message in app.info)
    assert not any(w.label in {"Vector artifact", "Analysis to display"} for w in app.selectbox)
    assert any(w.label == "Download NPZ vectors and exact object IDs" for w in app.get("download_button"))
    widget(app, "radio", "Result view").set_value("Next steps").run()
    assert widget(app, "multiselect", "Input representations for follow-up").value


def test_catalog_uses_completed_stage_outputs_of_failed_runs_without_loading_arrays(tmp_path, monkeypatch):
    vectors = export_embeddings(tmp_path / "vectors.npz", [4, 9], np.ones((2, 3)))
    (tmp_path / "status.json").write_text(json.dumps({"state": "failed", "stages": [
        {"state": "completed", "result": str(vectors)}, {"state": "failed", "action": "analyze"}
    ]}))
    record = SimpleNamespace(artifacts=(), checkpoint_path=None, embedding_path=None, working_directory=str(tmp_path))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Listing must not load arrays"))
    results, unavailable = run_results(record)
    assert [r["path"] for r in results] == [str(vectors)]
    assert not unavailable


def test_structured_progress_excludes_resource_columns_and_events_without_mutation():
    events = [
        {"timestamp": "now", "event": "dino_memory_adjustment", "old_workers": 8, "new_workers": 2},
        {"timestamp": "now", "event": "batch_progress", "stage": 1, "loss": .2,
         "resource_usage": {"rss_mb": 100}, "modulation": "reduced", "gpu_reserved_mb": 50},
        {"timestamp": "later", "event": "failed", "error": "No retained objects"},
    ]
    original = deepcopy(events)
    assert progress_events(events) == [
        {"timestamp": "now", "event": "batch_progress", "stage": 1, "loss": .2},
        {"timestamp": "later", "event": "failed", "error": "No retained objects"},
    ]
    assert events == original


def test_partial_analysis_hides_missing_views_but_retains_vector_inspection(tmp_path, monkeypatch):
    export_embeddings(tmp_path / "embeddings.npz", [2, 5, 8], np.ones((3, 4)))
    report = tmp_path / "analysis.json"
    report.write_text(json.dumps({"schema": "morphofeatures.analysis.v1", "embedding": "embeddings.npz"}))
    (tmp_path / "analysis_variants.json").write_text("interrupted index {")
    monkeypatch.setattr("morphofeatures.workflow_ui.submit_workspace_plan", lambda *a, **k: pytest.fail("Navigation cannot submit"))
    app = app_at(tmp_path)
    app.radio(key="workspace_navigation").set_value("Results").run()
    widget(app, "text_input", "Saved analysis.json, comparison.json, joint_analysis.json, or embedding file").set_value(str(report)).run()
    assert not app.exception and not app.error
    assert widget(app, "radio", "Analysis section").options == ["Objects & meshes", "Exports"]
    widget(app, "radio", "Result view").set_value("Next steps").run()
    widget(app, "multiselect", "Follow-up analyses").set_value(["compare"]).run()
    widget(app, "button", "Create follow-up workflow draft").click().run()
    assert not app.exception
    draft = app.session_state["workflow:" + str(tmp_path)]
    assert draft["document"]["stages"][0]["action"] == "compare"
    assert draft["document"]["followup_of"]["analysis"] == str(report)
    assert not (tmp_path / "experiments").exists()


def test_custom_prepared_config_name_continues_to_training_without_submission(tmp_path):
    crops = tmp_path / "crops.npy"
    ids = tmp_path / "ids.npy"
    np.save(crops, np.zeros((2, 8, 8, 8), dtype=np.float32))
    np.save(ids, np.array([2, 9]))
    config = tmp_path / "my_prepared_objects.yaml"
    config.write_text(yaml.safe_dump({"data": {"crops": str(crops), "label_ids": str(ids)},
                                     "mae": {"input_shape": [8, 8, 8], "patch_size": [4, 4, 4]}}))
    app = app_at(tmp_path)
    app.radio(key="workspace_navigation").set_value("Results").run()
    widget(app, "text_input", "Saved analysis.json, comparison.json, joint_analysis.json, or embedding file").set_value(str(config)).run()
    widget(app, "button", "Create workflow from this input").click().run()
    assert not app.exception
    assert app.radio(key="step_widget").value == "Train"
    stage = app.session_state["workflow:" + str(tmp_path)]["document"]["stages"][0]
    assert stage["action"] == "train" and stage["config"]["data"]["crops"] == str(crops)
    assert stage["config"]["mae"]["input_shape"] == [8, 8, 8]
    assert not (tmp_path / "experiments").exists()


def test_followup_copies_applicable_settings_without_overwriting_identity(tmp_path):
    from morphofeatures.followup_ui import followup_document
    settings = {"seed": 81, "clusters": 3, "classifier_models": ["mlp"], "group_column": "specimen",
                "embedding": "/wrong.npz", "from_extraction": True, "use_published_layout": True}
    document = followup_document({"Chosen": "/chosen.npz"}, ["analyze"], source=tmp_path / "analysis.json", settings=settings)
    stage = document["stages"][0]
    assert stage["seed"] == 81 and stage["classifier_models"] == ["mlp"]
    assert stage["embedding"] == "/chosen.npz"
    assert "from_extraction" not in stage and "use_published_layout" not in stage


def test_nested_distillation_focus_resolves_heldout_relative_to_parent(tmp_path, monkeypatch):
    from morphofeatures import result_browser_ui

    state = {"results:root": str(tmp_path / "distillation.json"), "results:parent": {
        "schema": "morphofeatures.distillation.v1",
        "representations": {"pca": {"name": "PCA", "held_out": "pca/held_out.npz"}},
    }}
    monkeypatch.setattr(result_browser_ui.st, "session_state", state)
    result_browser_ui.focus(tmp_path / "comparison/comparison.json", {}, vectors={"PCA": "training_objects.npz"})
    assert state["results:focus"]["vectors"] == {"PCA · held-out population": str(tmp_path / "pca/held_out.npz")}


def test_reviewed_variant_uses_existing_scientific_engine_in_a_new_directory(tmp_path):
    from morphofeatures.analysis.variants import run_saved_variant
    from morphofeatures.representation_analysis import analyze
    from morphofeatures.workspace_jobs import plan_workspace_job

    ids = np.arange(12, dtype=np.int64) + 2**60
    vectors = export_embeddings(tmp_path / "vectors.npz", ids, np.random.default_rng(9).normal(size=(12, 4)))
    parent = analyze(vectors, tmp_path / "parent", {"umap": False, "clusters": 2})
    original = parent.read_bytes()
    stage = {"action": "analysis_variant", "analysis": str(parent), "variant": {"name": "K3", "clusters": 3, "umap": False}}
    for execution in ("local", "slurm"):
        plan = plan_workspace_job({"stages": [stage]}, output_root=tmp_path / "review", run_id="variant", execution=execution)
        assert yaml.safe_load(plan.resolved_yaml)["stages"][0]["action"] == "analysis_variant"
    assert not (tmp_path / "review" / "experiments").exists()
    destination = tmp_path / "new"
    destination.mkdir()
    result = run_saved_variant(stage, destination)
    metadata = json.loads(result.read_text())
    assert metadata["reanalysis"]["classification_reused"]
    assert metadata["reanalysis"]["parent"] == str(parent)
    assert parent.read_bytes() == original
    assert (result.parent / "embeddings.npz").read_bytes() == (parent.parent / "embeddings.npz").read_bytes()


def test_worker_publishes_completed_stage_before_later_failure(tmp_path, monkeypatch):
    from morphofeatures.registry import JobRegistry
    from morphofeatures.workspace_jobs import run_job, submit_job

    vector = export_embeddings(tmp_path / "vectors.npz", [1, 2, 3], np.ones((3, 2)))
    stage = {"action": "analyze", "embedding": str(vector), "umap": False, "clusters": 2}
    record = submit_job({"stages": [stage, deepcopy(stage)]}, output_root=tmp_path, run_id="partial", execution="dry-run")
    registry = JobRegistry.under_output_root(tmp_path)
    calls = []

    def analyze(embedding, destination, settings):
        if calls:
            saved = registry.get(record.id)
            assert saved.application_state == "running"
            assert str(calls[0]) in saved.artifacts
            assert run_results(saved)[0][0]["path"] == str(calls[0])
            raise ValueError("Second stage failed")
        report = destination / "analysis.json"
        report.write_text(json.dumps({"schema": "morphofeatures.analysis.v1", "embedding": str(vector)}))
        calls.append(report)
        return report

    monkeypatch.setattr("morphofeatures.representation_analysis.analyze", analyze)
    with pytest.raises(ValueError, match="Second stage failed"):
        run_job(record.config_snapshot)
    saved = registry.get(record.id)
    assert saved.application_state == "failed"
    assert run_results(saved)[0][0]["path"] == str(calls[0])
