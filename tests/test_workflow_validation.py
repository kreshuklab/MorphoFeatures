"""Pointed workflow validation without jobs, training or remote pixel reads."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from morphofeatures.workflow_validation import WorkflowValidationError, check_workflow
from morphofeatures.workspace_jobs import plan_workspace_job, resolve_job
from morphofeatures.workspace_state import new_draft


def test_multiple_stages_report_locations_and_do_not_mutate_or_allocate(tmp_path):
    draft = new_draft("demo")
    document = draft["document"]
    document["stages"][0]["config"]["training"]["epochs"] = 0
    document["stages"].append(
        {"action": "analyze", "embedding": "missing.npz", "name": "Other model"}
    )
    original = deepcopy(document)
    issues = check_workflow(document, tmp_path, allow_synthetic=True)
    assert [(i.page, i.stage_index) for i in issues] == [("Train", 0), ("Analyze", 3)]
    assert issues[0].fields == (("stages", 0, "config", "training", "epochs"),)
    assert issues[1].fields == (("stages", 3, "embedding"),)
    assert "Other model" in issues[1].stage_name
    assert document == original
    collected = []
    with pytest.raises(WorkflowValidationError, match="Train.*Stage 1"):
        plan_workspace_job(document, output_root=tmp_path, run_id="bad", issues=collected)
    assert len(collected) == 2
    assert not (tmp_path / "experiments").exists()


@pytest.mark.parametrize("linked", [False, True])
def test_attached_bilateral_confirmation_is_checked_before_submission(tmp_path, linked):
    from morphofeatures.data.io import export_embeddings

    source = str(export_embeddings(tmp_path / "source.npz", [1, 2, 3], np.ones((3, 2))))
    stages = []
    compare = {
        "action": "compare", "analysis_normalizations": ["none", "l2", "standardize"],
        "bilateral": {"enabled": True, "same_specimen": False},
    }
    if linked:
        stages.extend([
            {"action": "fuse", "name": name, "combination": mode,
             "embeddings": {"fine": source, "coarse": source}}
            for name, mode in [("Raw", "raw"), ("Balanced", "balanced")]
        ])
        compare["from_embeddings"] = ["Raw", "Balanced"]
    else:
        compare["embeddings"] = {"Raw": source, "Balanced": source}
    stages.append(compare)
    document = {"stages": stages}
    issues = check_workflow(document, tmp_path)
    assert len(issues) == 1
    assert issues[0].fields == (("stages", len(stages) - 1, "bilateral", "same_specimen"),)
    assert issues[0].page == "Analyze"
    assert "Evaluate bilateral-partner retrieval" in issues[0].message
    for execution in ("local", "slurm"):
        with pytest.raises(WorkflowValidationError, match="Bilateral-partner retrieval"):
            plan_workspace_job(document, output_root=tmp_path, run_id="bad", execution=execution)
    assert not (tmp_path / "experiments").exists()
    # Either explicit confirmation or disabling this optional probe resolves it.
    compare["bilateral"]["same_specimen"] = True
    assert not check_workflow(document, tmp_path)
    compare["bilateral"].update(same_specimen=False, enabled=False)
    assert not check_workflow(document, tmp_path)


def test_attached_bilateral_file_error_keeps_nested_field(tmp_path):
    document = {"stages": [{"action": "analyze", "embedding": "later.npz", "bilateral": {
        "enabled": True, "candidates": str(tmp_path / "missing.tsv"),
    }}]}
    issues = check_workflow(document, tmp_path)
    assert issues[0].fields == (("stages", 0, "bilateral", "candidates"),)


@pytest.mark.parametrize(
    "action", ["analyze", "compare", "joint_analyze", "transfer", "dino_experiment"]
)
def test_umap_pca_error_is_located_before_reading_analysis_inputs(tmp_path, action):
    document = {"stages": [{"action": action, "umap_pca_components": -2}]}
    issues = check_workflow(document, tmp_path)
    assert len(issues) == 1
    assert issues[0].page == "Analyze"
    assert issues[0].fields == (("stages", 0, "umap_pca_components"),)
    assert "positive whole number" in issues[0].message


def test_page_checks_skip_other_pages_io_and_keep_linked_inputs(tmp_path, monkeypatch):
    from morphofeatures.analysis import platybrowser

    draft = new_draft("raw")
    document = draft["document"]
    document["stages"][0].update(source="platybrowser", crop_shape=[32, 32, 32])
    document["stages"][1]["config"]["training"]["learning_rate"] = -1

    def no_remote(*args, **kwargs):
        pytest.fail("Checking Train must not reconnect the Prepare data source")

    monkeypatch.setattr(platybrowser, "resolve_platybrowser_preprocessing", no_remote)
    issues = check_workflow(document, tmp_path, indices={1})
    assert len(issues) == 1 and issues[0].page == "Train"
    assert issues[0].fields == (("stages", 1, "config", "training", "learning_rate"),)


@pytest.mark.parametrize(
    "path,value",
    [
        (("tile_sizes_um",), [-1]),
        (("max_objects",), -1),
        (("views", "fine_fractions"), [1.5]),
        (("views", "min_foreground_fraction"), 1.5),
        (("views", "axes"), [3]),
        (("views", "size"), 225),
        (("views", "percentiles"), [99, 1]),
    ],
)
def test_dino_preparation_checks_views_without_requiring_weights(monkeypatch, path, value):
    from morphofeatures import dino_experiments
    from morphofeatures.workflow_validation import FieldError, stage_issue

    monkeypatch.setattr(dino_experiments, "validate_source", lambda source: source)
    stage = dino_experiments.experiment_defaults()
    # Empty checkpoints and invalid execution settings must not block Prepare data.
    stage["views"]["batch_size"] = 0
    dino_experiments.validate_experiment(stage, sources_only=True)
    part = stage
    for key in path[:-1]:
        part = part[key]
    part[path[-1]] = value
    with pytest.raises(FieldError) as caught:
        dino_experiments.validate_experiment(stage, sources_only=True)
    issue = stage_issue(caught.value, 0, stage)
    assert issue.page == "Prepare data"
    assert ("stages", 0) + path in issue.fields
    for field in (("backbones", 0, "checkpoint"), ("views", "batch_size"), ("fusion_weights",)):
        assert stage_issue(FieldError("Fix this setting", field), 0, stage).page == "Analyze"


def test_reader_errors_point_to_preparation_volume_fields(tmp_path):
    h5py = pytest.importorskip("h5py")
    file = tmp_path / "volumes.h5"
    with h5py.File(file, "w") as output:
        output["seg"] = np.zeros((16, 16, 16), dtype=np.int64)
    stage = new_draft("preprocess")["document"]["stages"][0]
    stage.update(
        raw=str(file),
        segmentation=str(file),
        raw_key="missing",
        segmentation_key="seg",
        raw_axes="zyx",
        segmentation_axes="zyx",
    )
    issues = check_workflow({"stages": [stage]}, tmp_path)
    assert len(issues) == 1
    assert ("stages", 0, "raw_key") in issues[0].fields
    assert issues[0].page == "Prepare data"
    assert "missing" in issues[0].message


def test_conversion_and_decoder_errors_retain_exact_fields(tmp_path):
    document = new_draft("demo")["document"]
    document["stages"][0]["config"]["training"]["epochs"] = "abc"
    issues = check_workflow(document, tmp_path, allow_synthetic=True)
    assert issues[0].fields == (("stages", 0, "config", "training", "epochs"),)
    document["stages"][0]["config"]["training"]["epochs"] = 2
    document["stages"][0]["config"]["mae"]["decoder_dim"] = -1
    with pytest.raises(WorkflowValidationError) as caught:
        resolve_job(document, tmp_path)
    assert ("stages", 0, "config", "mae", "decoder_dim") in caught.value.issue.fields


def _app(tmp_path):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "morphofeatures/dashboard.py"), default_timeout=30
    )
    app.session_state["workspace_root"] = str(tmp_path)
    return app.run()


def _widget(app, kind, label):
    return next(w for w in getattr(app, kind) if w.label == label)


def test_continue_checks_current_page_and_corrected_errors_disappear(tmp_path):
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value("demo").run()
    _widget(app, "button", "Create workflow").click().run()
    _widget(app, "number_input", "Training epochs").set_value(0).run()
    _widget(app, "button", "Continue to Analyze").click().run()
    assert app.radio(key="step_widget").value == "Train"
    assert not app.exception
    assert any("Train → Stage 1" in e.value and "Training epochs" in e.value for e in app.error)
    # At least one error is directly adjacent to the widget, as well as the summary.
    assert any(e.value.startswith("Training epochs:") for e in app.error)
    _widget(app, "number_input", "Training epochs").set_value(2).run()
    assert not app.error
    _widget(app, "button", "Continue to Analyze").click().run()
    assert app.radio(key="step_widget").value == "Analyze"
    assert not app.exception
    assert not (tmp_path / "experiments").exists()


def test_review_links_to_hidden_decoder_and_expands_its_section(tmp_path):
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value("demo").run()
    _widget(app, "button", "Create workflow").click().run()
    _widget(app, "number_input", "Decoder dim").set_value(-1).run()
    # Direct navigation remains possible; final review catches the issue.
    app.radio(key="step_widget").set_value("Review & run").run()
    _widget(app, "button", "Review run").click().run()
    assert not app.exception
    assert any(
        "Train → Stage 1" in e.value and "MAE decoder and reconstruction → Decoder dim" in e.value
        for e in app.error
    )
    assert not any(w.label == "Run locally" for w in app.button)
    _widget(app, "button", "Go to Train").click().run()
    assert app.radio(key="step_widget").value == "Train"
    decoder = next(e for e in app.expander if e.label == "MAE decoder and reconstruction")
    assert decoder.proto.expanded
    assert any(e.value.startswith("Decoder dim:") for e in app.error)


def test_slurm_mail_error_names_mail_section_and_box(tmp_path):
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value("demo").run()
    _widget(app, "button", "Create workflow").click().run()
    app.radio(key="step_widget").set_value("Review & run").run()
    _widget(app, "selectbox", "Run on").set_value("slurm").run()
    _widget(app, "multiselect", "Notify me when").set_value(["FAIL"]).run()
    _widget(app, "button", "Review run").click().run()
    assert not app.exception
    assert any(
        "Review & run → Mail notifications → Mail address (optional)" in e.value for e in app.error
    )
    assert any(e.value.startswith("Mail address (optional):") for e in app.error)
    _widget(app, "text_input", "Mail address (optional)").set_value("scientist@example.org").run()
    assert not app.error
    _widget(app, "button", "Review run").click().run()
    assert not app.error
    assert any(w.label == "Save dry-run bundle" for w in app.button)
    assert not (tmp_path / "experiments").exists()


def test_invalid_field_yaml_keeps_text_and_location_across_navigation(tmp_path):
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value("demo").run()
    _widget(app, "button", "Create workflow").click().run()
    _widget(app, "text_input", "Patch size").set_value("[4,").run()
    assert not app.exception
    app.radio(key="step_widget").set_value("Review & run").run()
    assert _widget(app, "button", "Review run").disabled
    assert any("Train → Stage 1" in e.value and "Patch size" in e.value for e in app.error)
    _widget(app, "button", "Go to Train").click().run()
    assert _widget(app, "text_input", "Patch size").value == "[4,"
    _widget(app, "text_input", "Patch size").set_value("[4, 4, 4]").run()
    assert not app.error
    assert not (tmp_path / "experiments").exists()
