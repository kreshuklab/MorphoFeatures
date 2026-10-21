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


def test_distributed_patch_controls_preview_and_dino_handoff(tmp_path):
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value("preprocess").run()
    _button(app, "Create workflow").click().run()
    _widget(app, "selectbox", "Object preparation layout").set_value("distributed_patches").run()
    assert not app.exception
    assert any("What you should have for distributed" in text.value for text in app.markdown)
    assert _widget(app, "number_input", "Minimum object content per patch (0–1)").value == 0.5
    _widget(app, "number_input", "Minimum object content per patch (0–1)").set_value(0.25).run()
    _widget(app, "selectbox", "Patch grid placement").set_value("bbox_center").run()
    raw = np.arange(16**3, dtype=np.uint16).reshape((16,) * 3)
    labels = np.zeros_like(raw, dtype=np.int64)
    labels[4:9, 4:9, 4:9] = 77
    np.save(tmp_path / "raw.npy", raw)
    np.save(tmp_path / "labels.npy", labels)
    for label, value in (
        ("Raw image path", str(tmp_path / "raw.npy")),
        ("Instance segmentation path", str(tmp_path / "labels.npy")),
        ("Raw image axis order", "zyx"),
        ("Instance segmentation axis order", "zyx"),
    ):
        _widget(app, "text_input", label).set_value(value).run()
    _widget(app, "selectbox", "Coordinate unit").set_value("um").run()
    _button(app, "Inspect data dimensions").click().run()
    _button(app, "Use full volume ROI").click().run()
    _button(app, "Preview distributed fine patches").click().run()
    task = next(value[1] for key, value in app.session_state.filtered_state.items()
                if key.startswith("distributed-preview:") and key.endswith(":task"))
    assert task.future.result(timeout=30)["state"] == "complete"
    app.run()
    assert not app.exception and not app.error
    assert any("Object 77:" in c.value for c in app.caption)
    assert any("Whole-object context" in c.value for c in app.caption)
    _button(app, "Use preview bounds for preparation").click().run()
    assert _widget(app, "text_input", "Existing object bounds table (optional)").value.endswith("objects.tsv")
    assert _widget(app, "selectbox", "MAE workflow").options == ["No MAE"]
    _widget(app, "checkbox", "Include DINO embeddings").check().run()
    _button(app, "Apply training and embedding choices").click().run()
    assert not app.exception
    draft = next(
        value
        for key, value in app.session_state.filtered_state.items()
        if key.startswith("workflow:")
    )
    preparation, extraction = draft["document"]["stages"][:2]
    assert preparation["patches"]["min_foreground_fraction"] == 0.25
    assert preparation["patches"]["tile_anchor"] == "bbox_center"
    assert extraction["views"]["normalization"] == "unit"
    assert extraction["views"]["fractions"] == [0.5]


def test_native_patch_sampling_controls_roundtrip_and_ignore_physical_width(tmp_path):
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value("preprocess").run()
    _button(app, "Create workflow").click().run()
    _widget(app, "selectbox", "Object preparation layout").set_value("distributed_patches").run()
    _widget(app, "number_input", "Fine patch width (µm)").set_value(3.0).run()
    _widget(app, "selectbox", "Fine patch sampling").set_value("native_voxels").run()
    assert not any(widget.label == "Fine patch width (µm)" for widget in app.number_input)
    assert any("without spatial interpolation" in entry.value for entry in app.info)
    assert "Measured spacing" in _widget(app, "selectbox", "Fine patch sampling").proto.help
    _widget(app, "checkbox", "Edit pipeline YAML").check().run()
    stage = yaml.safe_load(_widget(app, "text_area", "Pipeline YAML").value)["stages"][0]
    assert stage["patches"]["sampling"] == "native_voxels"
    assert stage["patches"]["tile_size_um"] == 3.0  # Preserved for switching back, ignored in native mode.
    _widget(app, "selectbox", "Fine patch sampling").set_value("physical_resampled").run()
    assert _widget(app, "number_input", "Fine patch width (µm)").value == 3.0
    assert not app.exception


def test_inspection_example_loads_without_training_and_resolves_source_paths(tmp_path):
    from morphofeatures.analysis.mesh_inspection import mesh_source_defaults
    from morphofeatures.configuration_editor import load_document

    example = (
        Path(__file__).resolve().parents[1]
        / "configs/sites/inspection_platynereis_nuclei_embl.example.yaml"
    )
    document = load_document(example)
    assert document["data"]["source"] == "n5_masked_patches"
    assert "training" not in document and "mae" not in document
    defaults = mesh_source_defaults(document)
    assert defaults["label_kind"] == "instances"
    assert defaults["segmentation_key"] == "volumes/paintera/nuclei/data/s2"
    assert defaults["spacing_zyx"] == [0.4, 0.32, 0.32]
    assert document["data"]["foreground_mask_kind"] == "foreground_scores"

    document["data"]["patches_container"] = "patches.n5"
    document["data"]["foreground_mask_container"] = "scores.n5"
    document["inspection"]["mesh"]["segmentation"] = "instances.n5"
    document["inspection"]["mesh"]["id_mapping"] = "mapping.tsv"
    local = tmp_path / "inspection.yaml"
    local.write_text(yaml.safe_dump(document))
    loaded = load_document(local)
    assert loaded["data"]["patches_container"] == str(tmp_path / "patches.n5")
    assert loaded["data"]["foreground_mask_container"] == str(tmp_path / "scores.n5")
    assert mesh_source_defaults(loaded)["segmentation"] == str(tmp_path / "instances.n5")
    assert mesh_source_defaults(loaded)["id_mapping"] == str(tmp_path / "mapping.tsv")
    assert mesh_source_defaults(loaded)["segmentation_key"] == defaults["segmentation_key"]


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


def test_spatial_probe_workflow_controls_roundtrip(tmp_path):
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value("bilateral").run()
    _button(app, "Create workflow").click().run()
    assert not app.exception
    assert _widget(app, "checkbox", "These representations describe the same specimen and cells").value
    _button(app, "Add bilateral settings variant").click().run()
    searches = [w for w in app.selectbox if w.label == "Cells eligible as retrieved neighbors"]
    searches[-1].set_value("opposite").run()
    assert not app.exception
    _button(app, "Check this page").click().run()
    assert not app.error and not app.exception
    draft = next(v for k, v in app.session_state.filtered_state.items() if k.startswith("workflow:"))
    stage = workflow_document(draft)["stages"][0]
    assert len(stage["variants"]) == 2 and stage["variants"][1]["search"] == "opposite"
    assert len(stage["embeddings"]) == 3


def test_context_controls_and_missing_source_errors(tmp_path):
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value("context").run()
    _button(app, "Create workflow").click().run()
    _widget(app, "selectbox", "How to define physical neighbors").set_value("radius").run()
    assert not app.exception
    assert _widget(app, "number_input", "Neighborhood radius (µm)").value == 10
    assert _widget(app, "number_input", "Minimum fraction of neighbors with embeddings").value == 1
    _widget(app, "selectbox", "When a graph neighbor lacks an embedding").set_value("available").run()
    _widget(app, "number_input", "Minimum fraction of neighbors with embeddings").set_value(.75).run()
    _button(app, "Check this page").click().run()
    assert not app.exception
    assert any("positions" in value.value for value in app.error)
    draft = next(v for k, v in app.session_state.filtered_state.items() if k.startswith("workflow:"))
    stage = workflow_document(draft)["stages"][0]
    assert stage["minimum_coverage"] == .75 and stage["include_self"] is True


def test_bilateral_saved_run_and_export_controls_preserve_yaml(tmp_path):
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value("bilateral").run()
    _button(app, "Create workflow").click().run()
    _widget(app, "checkbox", "Compare with previous bilateral runs").check().run()
    _widget(app, "text_input", "Named saved bilateral reports (YAML)").set_value("Previous DINO: /example/bilateral.json").run()
    _button(app, "Add bilateral 3D export").click().run()
    assert not app.exception
    _widget(app, "checkbox", "Create a MoBIE project with 3D pair views").uncheck().run()
    _widget(app, "number_input", "Maximum pairs per representation / variant").set_value(7).run()
    _widget(app, "text_input", "Physical position table for non-MoBIE geometry (optional)").set_value("/example/default.tsv").run()
    assert not app.exception
    draft = next(v for k, v in app.session_state.filtered_state.items() if k.startswith("workflow:"))
    stages = workflow_document(draft)["stages"]
    assert stages[0]["compare_saved_runs"]
    assert stages[0]["comparison_reports"] == {"Previous DINO": "/example/bilateral.json"}
    assert stages[1]["action"] == "export_bilateral" and stages[1]["from_bilateral"]
    assert stages[1]["max_pairs"] == 7 and not stages[1]["make_mobie"]
    assert stages[1]["positions"] == "/example/default.tsv"


def test_published_workflow_explains_preserved_layout(tmp_path):
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value("published").run()
    _button(app, "Create workflow").click().run()
    assert not app.exception
    assert _widget(app, "checkbox", "Preserve published UMAP coordinates and clusters").value
    assert any("saved published UMAP" in value.value for value in app.info)
    _button(app, "Check this page").click().run()
    assert not app.error and not app.exception


def test_fine_coarse_eligibility_controls_check_bounds_and_persist_policies(tmp_path):
    from morphofeatures.data.dino_views import index_source, validate_source

    raw = np.random.default_rng(10).random((24, 24, 24)).astype(np.float32)
    labels = np.zeros(raw.shape, dtype=np.int64)
    labels[:6, 3:11, 3:11] = 1
    labels[10:16, 3:11, 3:11] = 2
    np.save(tmp_path / "raw.npy", raw)
    np.save(tmp_path / "labels.npy", labels)
    source = validate_source({
        "raw": str(tmp_path / "raw.npy"),
        "segmentation": str(tmp_path / "labels.npy"),
        "unit": "um", "spacing_zyx": [0.1, 0.1, 0.1],
    })
    index_source(source, tmp_path / "index")
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value("dino_experiment").run()
    _button(app, "Create workflow").click().run()
    app.radio(key="step_widget").set_value("Prepare data").run()
    for label, value in (
        ("Raw container / array", str(tmp_path / "raw.npy")),
        ("Segmentation container / array", str(tmp_path / "labels.npy")),
        ("Raw voxel spacing (Z, Y, X)", "[0.1, 0.1, 0.1]"),
        ("Segmentation voxel spacing (Z, Y, X)", "[0.1, 0.1, 0.1]"),
        ("Existing object bounds table", str(tmp_path / "index/objects.tsv")),
    ):
        _widget(app, "text_input", label).set_value(value).run()
    budget = "Maximum object-box voxels (including halo)"
    _widget(app, "number_input", budget).set_value(100).run()
    _button(app, "Check object eligibility (no DINO)").click().run()
    assert not app.exception
    assert any("0 of 2" in entry.value and "800" in entry.value for entry in app.error)
    assert any("SLURM" in entry.value for entry in app.caption)
    _widget(app, "number_input", budget).set_value(1000).run()
    assert not any("0 of 2" in entry.value for entry in app.error)
    _button(app, "Check object eligibility (no DINO)").click().run()
    assert any("1 of 2" in entry.value for entry in app.warning)
    _widget(app, "selectbox", "Boundary objects").set_value("allow_partial").run()
    _button(app, "Check object eligibility (no DINO)").click().run()
    assert any("2 of 2" in entry.value for entry in app.warning)
    _widget(app, "selectbox", "Objects above the memory limit").set_value("error").run()
    _widget(app, "selectbox", "Other object-reading or encoding errors").set_value("error").run()
    _widget(app, "checkbox", "Process all eligible tiles").check().run()
    assert any("Maximum fine tiles per nucleus is ignored" in w.value for w in app.warning)
    app.radio(key="step_widget").set_value("Analyze").run()
    assert not any(w.label == "Process all eligible tiles" for w in app.checkbox)
    _widget(app, "checkbox", "Edit pipeline YAML").check().run()
    stage = yaml.safe_load(_widget(app, "text_area", "Pipeline YAML").value)["stages"][0]
    source = stage["datasets"][0]["source"]
    assert source["max_object_voxels"] == 1000
    assert source["boundary_policy"] == "allow_partial"
    assert source["oversized_policy"] == source["on_object_error"] == "error"
    assert stage["views"]["exhaustive"] is True
    assert not app.exception


@pytest.mark.parametrize(
    "route", ["raw", "preprocess", "crops", "dino", "checkpoint", "embeddings", "demo", "dino_experiment", "fusion", "transfer", "platybrowser", "ome_zarr", "joint_embeddings"]
)
def test_all_starting_points_and_stages_render(tmp_path, route):
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value(route).run()
    assert any("What you should have for this workflow" in item.value for item in app.markdown)
    _button(app, "Create workflow").click().run()
    expected = {
        "raw": ["Start", "Prepare data", "Train", "Analyze", "Review & run"],
        "preprocess": ["Start", "Prepare data", "Review & run"],
        "platybrowser": ["Start", "Prepare data", "Review & run"],
        "ome_zarr": ["Start", "Prepare data", "Review & run"],
        "dino_experiment": ["Start", "Prepare data", "Analyze", "Review & run"],
        "crops": ["Start", "Train", "Analyze", "Review & run"],
        "demo": ["Start", "Train", "Analyze", "Review & run"],
    }.get(route, ["Start", "Analyze", "Review & run"])
    assert app.radio(key="step_widget").options == expected
    for step in expected[1:]:
        app.radio(key="step_widget").set_value(step).run()
        assert not app.exception
        assert any("What you should have for this workflow" in item.value for item in app.markdown)
    if route == "crops":
        _button(app, "Review run").click().run()
        assert any("select prepared crops" in error.value for error in app.error)
        assert not (tmp_path / "experiments").exists()


def test_dino_runtime_controls_round_trip_without_hiding_scientific_settings(tmp_path):
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value("dino_experiment").run()
    _button(app, "Create workflow").click().run()
    app.radio(key="step_widget").set_value("Analyze").run()
    for label, value in (
        ("CPU preparation workers", 2),
        ("Preparation RAM budget (MiB)", 6144),
        ("Queued image budget (MiB)", 64),
        ("Decoded chunk cache per volume / worker (MiB)", 32),
        ("Prepared image disk cache (MiB; 0 disables)", 512),
    ):
        _widget(app, "number_input", label).set_value(value)
    _widget(app, "selectbox", "Inference precision").set_value("bfloat16").run()
    _widget(app, "checkbox", "Resume completed objects").uncheck().run()
    app.radio(key="step_widget").set_value("Review & run").run()
    app.radio(key="step_widget").set_value("Analyze").run()
    assert _widget(app, "number_input", "CPU preparation workers").value == 2
    assert _widget(app, "selectbox", "Inference precision").value == "bfloat16"
    assert _widget(app, "checkbox", "Resume completed objects").value is False
    assert not any(widget.label == "Fine slice positions within foreground" for widget in app.text_input)
    assert any(widget.label == "Images per model batch" for widget in app.number_input)
    _widget(app, "checkbox", "Edit pipeline YAML").check().run()
    stage = yaml.safe_load(_widget(app, "text_area", "Pipeline YAML").value)["stages"][0]
    assert stage["runtime"]["prepare_workers"] == 2
    assert stage["runtime"]["prepare_memory_mb"] == 6144
    assert stage["runtime"]["prepared_cache_mb"] == 512
    assert stage["views"]["mode"] == "both"
    app.radio(key="step_widget").set_value("Prepare data").run()
    assert any(widget.label == "Fine slice positions within foreground" for widget in app.text_input)
    assert not app.exception


def test_new_workflow_help_uses_stage_and_nested_field_context():
    from morphofeatures.ui_settings import (
        WORKFLOW_REQUIREMENTS,
        setting_help,
        workflow_requirements,
    )
    from morphofeatures.workspace_state import STARTING_POINTS

    assert set(STARTING_POINTS) <= set(WORKFLOW_REQUIREMENTS)
    root = ["document", "stages", 0]
    source = root + ["datasets", 0, "source"]
    transfer = setting_help(root + ["transfer", "enabled"], action="dino_experiment")
    assert "Target annotations are optional" in transfer
    assert "early stopping" not in transfer
    assert "Stop grouped training" in setting_help(root + ["config", "early_stopping", "enabled"], action="train")
    assert "may differ" in setting_help(source + ["raw_spacing_zyx"], action="dino_experiment")
    assert "ID convention" in setting_help(source + ["platybrowser", "object_ids"], action="dino_experiment")
    assert "[1354, 1043, 13539]" in setting_help(source + ["object_ids"], action="dino_experiment")
    assert "unique name for this dataset" in setting_help(root + ["datasets", 0, "name"], action="dino_experiment")
    assert "editable workflow" in setting_help(["name"], action="dino_experiment")

    # Loaded/edited pipelines have no starting-point selection to consult.
    requirements = workflow_requirements([{"action": "fuse"}, {"action": "export_mobie"}])
    assert "at least two embedding files" in requirements[0]
    assert any("MoBIE export" in item for item in requirements)
    assert workflow_requirements([{"action": "preprocess"}]) == WORKFLOW_REQUIREMENTS["preprocess"]


@pytest.mark.parametrize("route", ["dino_experiment", "fusion", "transfer"])
def test_new_workflow_controls_have_specific_help(tmp_path, route):
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value(route).run()
    _button(app, "Create workflow").click().run()
    if route == "dino_experiment":
        assert app.radio(key="step_widget").value == "Prepare data"
        assert (
            "[0.5, 1.0, 2.0]"
            in _widget(app, "text_input", "Fine tile widths to compare (µm)").proto.help
        )
        assert (
            "0.50 for half"
            in _widget(app, "number_input", "Minimum nucleus content per 3D fine tile (0–1)").proto.help
        )
        assert "Padding counts as background" in _widget(app, "number_input", "Minimum nucleus content per 2D fine image (0–1)").proto.help
        assert "centroid" in _widget(app, "selectbox", "Position of the fine tile grid").proto.help
        app.radio(key="step_widget").set_value("Analyze").run()
        _widget(app, "checkbox", "Train a reference classifier and predict other datasets").check().run()
    if route in {"dino_experiment", "transfer"}:
        conditions = ["uncorrected", "coral", "coral_source_to_target", "adapter", "deep_coral"]
        _widget(app, "multiselect", "Alignment conditions").set_value(conditions).run()
        _widget(app, "number_input", "Adapter training epochs").set_value(3).run()
        _widget(app, "number_input", "Adapted embedding dimensions").set_value(8).run()
        assert (
            "source"
            in _widget(app, "number_input", "Adapter objects per training batch").proto.help
            or "domain"
            in _widget(app, "number_input", "Adapter objects per training batch").proto.help
        )
        assert "Target labels" in _widget(app, "number_input", "Adapter training epochs").proto.help
        _widget(app, "selectbox", "Reference classifier").set_value("mlp").run()
        _widget(app, "text_input", "MLP hidden-layer widths").set_value("[32, 16]").run()
        _widget(app, "selectbox", "Feature scaling before classifier projection").set_value(
            "standardize"
        ).run()
        pca = _widget(app, "number_input", "PCA components before UMAP (0 = off)")
        assert "DeepCORAL" in pca.proto.help
        pca.set_value(50).run()
        _widget(app, "checkbox", "Edit pipeline YAML").check().run()
        stage = yaml.safe_load(_widget(app, "text_area", "Pipeline YAML").value)["stages"][0]
        assert stage["umap_pca_components"] == 50
        settings = stage["transfer"] if route == "dino_experiment" else stage
        assert settings["classifier"] == "mlp"
        assert settings["hidden_dimensions"] == [32, 16]
        assert settings["classifier_normalization"] == "standardize"
        assert settings["conditions"] == conditions
        assert settings["adapter"]["epochs"] == 3
        assert settings["adapter"]["latent_dim"] == 8
    for kind in ("text_input", "number_input", "selectbox", "checkbox", "multiselect"):
        for widget in getattr(app, kind):
            if widget.key and str(widget.key).startswith("guided:"):
                assert widget.proto.help, widget.label
                assert "Additional option preserved" not in widget.proto.help, widget.label
                assert "Each width creates a separate" not in widget.proto.help, widget.label
    if route == "dino_experiment":
        app.radio(key="step_widget").set_value("Prepare data").run()
        for source in ("local", "platybrowser", "ome_zarr"):
            _widget(app, "selectbox", "Image source").set_value(source).run()
            for kind in ("text_input", "number_input", "selectbox"):
                for widget in getattr(app, kind):
                    if widget.key and str(widget.key).startswith("guided:"):
                        assert widget.proto.help, widget.label
                        assert "Additional option preserved" not in widget.proto.help, widget.label
    assert not app.exception


@pytest.mark.parametrize("route", ["transfer", "embeddings", "dino_experiment"])
def test_classifier_projection_controls_roundtrip(tmp_path, route):
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value(route).run()
    _button(app, "Create workflow").click().run()
    if route == "dino_experiment":
        app.radio(key="step_widget").set_value("Analyze").run()
    if route == "transfer":
        assert (
            "Linear classifier (logistic regression)"
            in _widget(app, "selectbox", "Reference classifier").options
        )
    else:
        label = "Within-dataset classifiers" if route == "dino_experiment" else "Classifiers"
        _widget(app, "multiselect", label).set_value(["logistic"]).run()
    for space in ("pca", "umap", "embeddings"):
        _widget(app, "selectbox", "Classification input").set_value(space).run()
        if space != "embeddings":
            _widget(app, "number_input", "Classifier dimensions").set_value(3).run()
        _widget(app, "checkbox", "Standardize selected classifier dimensions").check().run()
        _widget(app, "checkbox", "Edit pipeline YAML").check().run()
        _button(app, "Reload YAML from current form settings").click().run()
        stage = yaml.safe_load(_widget(app, "text_area", "Pipeline YAML").value)["stages"][0]
        assert stage["classifier_space"] == space
        assert stage["classifier_standardize"] is True
        if space != "embeddings":
            assert stage["classifier_dimensions"] == 3
        _widget(app, "checkbox", "Edit pipeline YAML").uncheck().run()
    assert not app.exception


@pytest.mark.parametrize("route", ["embeddings", "joint_embeddings"])
def test_umap_pca_plot_control_survives_page_navigation(tmp_path, route):
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value(route).run()
    _button(app, "Create workflow").click().run()
    _widget(app, "checkbox", "Include UMAP projection").check().run()
    label = "PCA components before UMAP (0 = off)"
    _widget(app, "number_input", label).set_value(50).run()
    app.radio(key="step_widget").set_value("Review & run").run()
    app.radio(key="step_widget").set_value("Analyze").run()
    assert _widget(app, "number_input", label).value == 50
    _widget(app, "checkbox", "Edit pipeline YAML").check().run()
    stage = yaml.safe_load(_widget(app, "text_area", "Pipeline YAML").value)["stages"][0]
    assert stage["umap_pca_components"] == 50
    _widget(app, "number_input", label).set_value(0).run()
    _button(app, "Reload YAML from current form settings").click().run()
    stage = yaml.safe_load(_widget(app, "text_area", "Pipeline YAML").value)["stages"][0]
    assert stage["umap_pca_components"] == 0
    assert not app.exception


@pytest.mark.parametrize("route", ["embeddings", "joint_embeddings", "transfer", "dino_experiment"])
def test_stacked_analysis_controls_roundtrip_and_remove_one_combination(tmp_path, route):
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value(route).run()
    _button(app, "Create workflow").click().run()
    if route == "dino_experiment":
        app.radio(key="step_widget").set_value("Analyze").run()
    _widget(app, "number_input", "PCA components before clustering (0 = off)").set_value(4).run()
    _button(app, "Add analysis combination").click().run()
    _widget(app, "text_input", "Combination name").set_value("PCA2 K3").run()
    item = next(w for w in app.number_input if str(w.key).endswith("analysis_variants.0.cluster_pca_components"))
    item.set_value(2).run()
    _button(app, "Add analysis combination").click().run()
    _widget(app, "checkbox", "Edit pipeline YAML").check().run()
    stage = yaml.safe_load(_widget(app, "text_area", "Pipeline YAML").value)["stages"][0]
    assert stage["cluster_pca_components"] == 4
    assert [v["cluster_pca_components"] for v in stage["analysis_variants"]] == [2, 4]
    assert stage["analysis_variants"][0]["normalization"] == {
        "transfer": "none", "dino_experiment": "l2"
    }.get(route, "standardize")
    _button(app, "Remove combination").click().run()
    _button(app, "Reload YAML from current form settings").click().run()
    stage = yaml.safe_load(_widget(app, "text_area", "Pipeline YAML").value)["stages"][0]
    assert len(stage["analysis_variants"]) == 1
    assert stage["analysis_variants"][0]["cluster_pca_components"] == 4
    assert not app.exception


@pytest.mark.parametrize("route", ["transfer", "dino_experiment"])
def test_primary_transfer_and_dino_map_controls_persist_without_variants(tmp_path, route):
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value(route).run()
    _button(app, "Create workflow").click().run()
    if route == "dino_experiment":
        app.radio(key="step_widget").set_value("Analyze").run()
    panel = next(w for w in app.expander if w.label == "Clustering, PCA and UMAP")
    assert panel.proto.expanded
    cluster_label = "Embedding clusters" if route == "dino_experiment" else "Number of clusters"
    umap_label = "Generate UMAP as well as PCA" if route == "dino_experiment" else "Generate UMAP"
    seed_label = "Cohort, split and projection seed" if route == "dino_experiment" else "Random seed"
    controls = {
        "clusters": (cluster_label, 13),
        "cluster_pca_components": ("PCA components before clustering (0 = off)", 5),
        "umap_pca_components": ("PCA components before UMAP (0 = off)", 7),
        "neighbors": ("UMAP / Leiden neighbors", 25),
        "min_dist": ("UMAP minimum distance", 0.35),
        "umap_epochs": ("UMAP epochs (null uses automatic)", 125),
        "seed": (seed_label, 17),
        "unlabeled_opacity": ("Unlabeled point opacity", 0.4),
    }
    for label, value in controls.values():
        _widget(app, "number_input", label).set_value(value)
    app.run()
    assert "not K-means or Leiden cluster assignments" in _widget(
        app, "number_input", "UMAP minimum distance"
    ).proto.help

    # Hiding a control must not discard its configured value.
    _widget(app, "checkbox", umap_label).uncheck().run()
    assert _widget(app, "number_input", cluster_label).value == 13
    assert not any(w.label == "UMAP minimum distance" for w in app.number_input)
    _widget(app, "selectbox", "Clustering method").set_value("leiden").run()
    _widget(app, "number_input", "Leiden resolution").set_value(0.02).run()
    assert _widget(app, "number_input", "UMAP / Leiden neighbors").value == 25
    assert not any(w.label == cluster_label for w in app.number_input)
    app.radio(key="step_widget").set_value("Review & run").run()
    app.radio(key="step_widget").set_value("Analyze").run()
    assert _widget(app, "number_input", "Leiden resolution").value == 0.02
    _widget(app, "selectbox", "Clustering method").set_value("kmeans").run()
    _widget(app, "checkbox", umap_label).check().run()
    for label, value in controls.values():
        assert _widget(app, "number_input", label).value == value

    _widget(app, "checkbox", "Edit pipeline YAML").check().run()
    document = yaml.safe_load(_widget(app, "text_area", "Pipeline YAML").value)
    stage = document["stages"][0]
    assert not stage.get("analysis_variants")
    assert stage["cluster_method"] == "kmeans"
    assert stage["resolution"] == 0.02
    for key, (_, value) in controls.items():
        assert stage[key] == value

    # A loaded configuration must populate these same main controls.
    stage.update(clusters=11, cluster_pca_components=None, umap_pca_components=None, umap_epochs=None)
    _widget(app, "text_area", "Pipeline YAML").set_value(yaml.safe_dump(document)).run()
    _button(app, "Apply YAML to draft").click().run()
    assert _widget(app, "number_input", cluster_label).value == 11
    assert _widget(app, "number_input", "PCA components before clustering (0 = off)").value == 0
    assert _widget(app, "number_input", "PCA components before UMAP (0 = off)").value == 0
    assert _widget(app, "text_input", "UMAP epochs (null uses automatic)").value == "null"
    assert not app.exception


def test_results_can_select_rename_and_submit_parameter_combinations(tmp_path, monkeypatch):
    from morphofeatures import analysis_variants_ui
    from morphofeatures.analysis import variants
    from morphofeatures.representation_analysis import analyze

    source = export_embeddings(
        tmp_path / "input.npz", np.arange(12), np.random.default_rng(5).normal(size=(12, 6))
    )
    path = analyze(source, tmp_path / "analysis", {
        "umap": False, "analysis_variants": [{"name": "PCA2", "cluster_pca_components": 2}],
    })
    child = variants.variant_history(path)[0]["result"]
    captured = []
    monkeypatch.setattr(analysis_variants_ui, "start_variants", lambda p, v: captured.append((p, v)))
    app = _app(tmp_path)
    app.radio(key="workspace_navigation").set_value("Results").run()
    _widget(app, "text_input", "Saved analysis.json, comparison.json, joint_analysis.json, or embedding file").set_value(str(path)).run()
    _widget(app, "selectbox", "Saved analysis combination").set_value(child).run()
    _widget(app, "text_input", "Result display name").set_value("My PCA map").run()
    _button(app, "Save result name").click().run()
    assert json.loads(Path(child).read_text())["display_name"] == "My PCA map"
    _widget(app, "radio", "Result view").set_value("Follow-up analysis").run()
    _widget(app, "number_input", "PCA components before clustering (0 = off)").set_value(3).run()
    _button(app, "Compute and save combinations").click().run()
    assert len(captured) == 1
    assert captured[0][0] == path
    assert captured[0][1][0]["cluster_pca_components"] == 3
    assert not app.exception


def test_dino_loads_grouped_mae_data_without_replacing_backbone_or_views(tmp_path):
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value("dino").run()
    _button(app, "Create workflow").click().run()
    source = Path(__file__).parents[1] / "configs/sites/mae_platynereis_nuclei_embl.yaml"
    data = yaml.safe_load(source.read_text())["data"]
    weights = str(tmp_path / "dinov2_vits14_pretrain.pth")
    repository = str(tmp_path / "dinov2")
    _widget(app, "text_input", "Pretrained DINO weights").set_value(weights).run()
    _widget(app, "text_input", "Official local model repository").set_value(repository).run()
    _widget(app, "selectbox", "Combine views per object").set_value("max").run()
    _widget(app, "text_input", "Prepared crops (.npy / .h5 / .n5)").set_value(
        data["patches_container"]
    ).run()
    assert any("positions/ID index" in warning.value for warning in app.warning)
    _widget(app, "text_input", "Model / data YAML path").set_value(str(source)).run()
    _widget(app, "text_input", "Data profile (optional)").set_value("full").run()
    _button(app, "Load settings into this stage").click().run()
    assert not app.exception and not app.error
    assert _widget(app, "text_input", "N5 patch container").value == data["patches_container"]
    assert _widget(app, "text_input", "N5 positions container").value == data["positions_container"]
    assert not any(widget.label == "Prepared crops (.npy / .h5 / .n5)" for widget in app.text_input)
    assert _widget(app, "text_input", "Pretrained DINO weights").value == weights
    assert _widget(app, "text_input", "Official local model repository").value == repository
    assert _widget(app, "selectbox", "Combine views per object").value == "max"
    _widget(app, "checkbox", "Edit pipeline YAML").check().run()
    exported = yaml.safe_load(_widget(app, "text_area", "Pipeline YAML").value)
    assert [stage["action"] for stage in exported["stages"]] == ["extract", "analyze"]
    stage = exported["stages"][0]
    assert stage["model"] == "dinov2"
    assert stage["config"]["data"]["source"] == "n5_masked_patches"
    assert stage["config"]["resolved_profile"] == "full"
    assert not (tmp_path / "experiments").exists()


@pytest.mark.parametrize("output_format", ["npy", "h5", "n5"])
def test_preprocessing_preview_and_standalone_review(tmp_path, output_format):
    if output_format != "npy":
        pytest.importorskip("h5py" if output_format == "h5" else "z5py")
    raw = np.arange(16**3, dtype=np.uint16).reshape((16,) * 3)
    labels = np.zeros_like(raw, dtype=np.int64)
    labels[4:8, 4:8, 4:8] = 2**60 + 1
    np.save(tmp_path / "raw.npy", raw)
    np.save(tmp_path / "labels.npy", labels)
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value("raw").run()
    _button(app, "Create workflow").click().run()
    for label, value in (
        ("Raw image path", str(tmp_path / "raw.npy")),
        ("Instance segmentation path", str(tmp_path / "labels.npy")),
        ("Raw image axis order", "zyx"),
        ("Instance segmentation axis order", "zyx"),
    ):
        _widget(app, "text_input", label).set_value(value).run()
    _button(app, "Inspect data dimensions").click().run()
    _button(app, "Use full volume ROI").click().run()
    assert not app.exception and not app.error
    assert any("Selected ROI dimensions (Z, Y, X): (16, 16, 16)" in c.value for c in app.caption)
    _widget(app, "selectbox", "Output format").set_value(output_format).run()
    _button(app, "Just Run Preprocessing").click().run()
    assert app.radio(key="step_widget").value == "Review & run"
    assert app.radio(key="step_widget").options == ["Start", "Prepare data", "Review & run"]
    _button(app, "Review run").click().run()
    assert not app.exception and not app.error
    _button(app, "Save dry-run bundle").click().run()
    saved = yaml.safe_load(next(tmp_path.glob("experiments/*/workspace/job.yaml")).read_text())
    assert [stage["action"] for stage in saved["stages"]] == ["preprocess"]
    assert saved["stages"][0]["roi"] == [[0, 0, 0], [16, 16, 16]]
    assert saved["stages"][0]["output_format"] == output_format
    from morphofeatures.workspace_jobs import run_job

    worker = next(tmp_path.glob("experiments/*/workspace/job.yaml"))
    status = json.loads(run_job(worker).read_text())
    prepared = status["stages"][0]["result"]
    app.radio(key="workspace_navigation").set_value("Workflow").run()
    app.radio(key="step_widget").set_value("Start").run()
    _widget(app, "selectbox", "Starting point").set_value("dino").run()
    _button(app, "Create a new workflow").click().run()
    _widget(app, "text_input", "Model / data YAML path").set_value(prepared).run()
    _button(app, "Load settings into this stage").click().run()
    assert not app.exception and not app.error
    assert _widget(app, "text_input", "Prepared crops (.npy / .h5 / .n5)").value.endswith(
        "crops." + output_format
    )
    if output_format != "npy":
        for label, key in (
            ("Crop dataset key", "crops"),
            ("Object ID dataset key", "label_ids"),
            ("Mask dataset key", "masks"),
        ):
            assert _widget(app, "text_input", label).value == key
        assert not app.warning
    # The same generated configuration must also render the MAE training form.
    app.radio(key="step_widget").set_value("Start").run()
    _widget(app, "text_input", "Training configuration or pipeline YAML").set_value(prepared).run()
    _button(app, "Preview settings to load").click().run()
    _button(app, "Replace current draft settings").click().run()
    assert not app.exception and not app.error
    assert app.radio(key="step_widget").value == "Train"


def test_slurm_mail_selection_persists_and_reaches_reviewed_script(tmp_path):
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value("demo").run()
    _button(app, "Create workflow").click().run()
    app.radio(key="step_widget").set_value("Review & run").run()
    _widget(app, "selectbox", "Run on").set_value("slurm").run()
    _widget(app, "multiselect", "Notify me when").set_value(["END", "FAIL"]).run()
    _widget(app, "text_input", "Mail address (optional)").set_value("scientist@example.org").run()
    app.radio(key="step_widget").set_value("Train").run()
    app.radio(key="step_widget").set_value("Review & run").run()
    assert _widget(app, "multiselect", "Notify me when").value == ["END", "FAIL"]
    _button(app, "Review run").click().run()
    assert not app.exception and not app.error
    scripts = [code.value for code in app.code if "#SBATCH" in code.value]
    assert any(
        "--mail-type=END,FAIL" in script and "--mail-user=scientist@example.org" in script
        for script in scripts
    )
    _widget(app, "multiselect", "Notify me when").set_value(["FAIL"]).run()
    assert any("Settings changed after review" in warning.value for warning in app.warning)
    assert not any(button.label == "Save dry-run bundle" for button in app.button)
    _widget(app, "text_input", "Mail address (optional)").set_value("").run()
    _widget(app, "selectbox", "Run on").set_value("local").run()
    _button(app, "Review run").click().run()
    assert not app.exception and not app.error


def test_classification_results_reopen_without_recomputation(tmp_path, monkeypatch):
    import pandas as pd

    from morphofeatures.representation_analysis import analyze

    ids = np.arange(12) + 100
    source = export_embeddings(
        tmp_path / "features.npz", ids, np.random.default_rng(7).normal(size=(12, 4))
    )
    annotations = tmp_path / "annotations.tsv"
    pd.DataFrame({"label_id": ids, "label": np.tile(["a", "b"], 6)}).to_csv(
        annotations, sep="\t", index=False
    )
    result = analyze(
        source,
        tmp_path / "analysis",
        {"umap": False, "clusters": 2, "annotations": str(annotations), "folds": 3},
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Opening classification results must only read saved artifacts")

    monkeypatch.setattr("morphofeatures.representation_analysis.analyze", forbidden)
    monkeypatch.setattr("morphofeatures.workflow_ui.submit_workspace_plan", forbidden)
    app = _app(tmp_path)
    app.radio(key="workspace_navigation").set_value("Results").run()
    _widget(
        app,
        "text_input",
        "Saved analysis.json, comparison.json, joint_analysis.json, or embedding file",
    ).set_value(str(result)).run()
    assert not app.exception and not app.error
    _widget(app, "radio", "Analysis section").set_value("Classification").run()
    assert any(header.value == "Classification check" for header in app.subheader)
    assert any("Confusion matrix" in caption.value for caption in app.caption)
    assert any(w.label == "Download confusion matrix SVG" for w in app.get("download_button"))


@pytest.mark.parametrize("kind", ["transfer", "comparison"])
def test_saved_evaluation_heatmaps_and_unlabeled_targets(tmp_path, monkeypatch, kind):
    from streamlit.testing.v1 import AppTest

    from morphofeatures import dino_experiment_ui

    def forbidden(*args, **kwargs):
        pytest.fail("Displaying saved evaluation counts must not fit a classifier")

    monkeypatch.setattr("morphofeatures.analysis.classification.configured_classifier", forbidden)
    monkeypatch.setattr(dino_experiment_ui, "render_analysis", lambda *args: None)
    metrics = {"confusion_labels": ["a", "b"], "confusion_matrix": [[8, 2], [1, 9]]}
    if kind == "transfer":
        metadata = {
            "schema": "morphofeatures.transfer.v1",
            "interpretation": "Saved evaluation",
            "source_validation": metrics,
            "settings": {"classifier": "knn"},
            "targets": [
                {
                    "name": "Annotated",
                    "condition": "uncorrected",
                    "directory": "target-0",
                    "joint_analysis": "target-0/joint.json",
                    "metrics": metrics,
                },
                {
                    "name": "Unlabeled",
                    "condition": "coral",
                    "directory": "target-1",
                    "joint_analysis": "target-1/joint.json",
                    "metrics": {"assigned_fraction": 0.8},
                },
            ],
        }
        renderer = "experiment_results(None, metadata, path)"
    else:
        evaluation = {
            "evaluated_objects": 20,
            "excluded_object_ids": [],
            "classifiers": {
                "mlp": {"classes": ["a", "b"], "confusion_matrix": metrics["confusion_matrix"]}
            },
        }
        metadata = {
            "representations": {
                name: {"classification": evaluation} for name in ("Fine", "Combined")
            }
        }
        renderer = "classification_results(metadata, path)"
    path = tmp_path / "saved.json"
    path.write_text(json.dumps(metadata))
    app = AppTest.from_string(
        "import json\nfrom pathlib import Path\n"
        "from morphofeatures.dino_experiment_ui import experiment_results\n"
        "from morphofeatures.analysis_ui import classification_results\n"
        f"path = Path({str(path)!r})\nmetadata = json.loads(path.read_text())\n{renderer}\n",
        default_timeout=30,
    ).run()
    assert not app.exception
    assert _widget(app, "metric", "Accuracy").value == "85.0%"
    assert any(w.label == "Download confusion matrix SVG" for w in app.get("download_button"))
    _widget(app, "selectbox", "Confusion matrix values").set_value("Object counts").run()
    if kind == "transfer":
        _widget(app, "selectbox", "Target dataset and correction").set_value(1).run()
        assert sum(w.label == "Accuracy" for w in app.metric) == 1
        assert any("No matched known labels" in w.value for w in app.info)
    else:
        _widget(app, "selectbox", "Classification representation").set_value("Combined").run()
        assert any("Combined · MLP" in w.value for w in app.subheader)
    assert not app.exception


def test_training_decoder_controls_and_descriptive_help(tmp_path):
    app = _app(tmp_path)
    _widget(app, "selectbox", "Starting point").set_value("demo").run()
    _button(app, "Create workflow").click().run()
    _widget(app, "number_input", "Decoder dim").set_value(17).run()
    _widget(app, "number_input", "Weight decay").set_value(0.03).run()
    help_text = _widget(app, "text_input", "Repo root").proto.help
    assert "Repository-relative" in help_text and "Configuration field:" not in help_text
    app.radio(key="step_widget").set_value("Review & run").run()
    _button(app, "Review run").click().run()
    assert not app.error
    _button(app, "Save dry-run bundle").click().run()
    saved = yaml.safe_load(next(tmp_path.glob("experiments/*/workspace/job.yaml")).read_text())
    assert saved["stages"][0]["config"]["mae"]["decoder_dim"] == 17
    assert saved["stages"][0]["config"]["training"]["weight_decay"] == 0.03


@pytest.mark.parametrize("pca_components", [0, 3])
def test_projection_tool_exports_shared_coordinates_and_annotations(tmp_path, pca_components):
    from streamlit.testing.v1 import AppTest

    from morphofeatures.config import load_config
    from morphofeatures.data.io import load_embeddings
    from morphofeatures.representation_analysis import project_table

    pytest.importorskip("umap")
    ids = np.arange(24, dtype=np.int64) + 2**53 + 3
    features = np.random.default_rng(3).normal(size=(24, 4))
    source = export_embeddings(tmp_path / "embeddings.npz", ids, features)
    labels = tmp_path / "labels.tsv"
    labels.write_text(f"label_id\tcell_type\n{ids[0]}\tneuron\n{ids[1]}\tepithelial\n")
    destination = tmp_path / "projection.npz"

    def projection_app():
        from morphofeatures.config import load_config
        from morphofeatures.dashboard import _projection

        _projection(load_config())

    app = AppTest.from_function(projection_app, default_timeout=45).run()
    _widget(app, "text_input", "Projection embedding").set_value(str(source))
    _widget(app, "text_input", "Projection output").set_value(str(destination))
    _widget(app, "text_input", "Biological metadata (optional, joined by label_id)").set_value(
        str(labels)
    )
    _widget(app, "number_input", "UMAP epochs").set_value(10)
    _widget(app, "number_input", "PCA components before UMAP (0 = off)").set_value(pca_components)
    _button(app, "Run projection").click().run()
    assert not app.exception and not app.error
    exported = load_embeddings(destination)
    expected, _ = project_table(ids, features, {
        "umap_epochs": 10, "seed": load_config().seed, "umap_pca_components": pca_components,
    })
    np.testing.assert_array_equal(exported.label_ids, ids)
    np.testing.assert_allclose(
        exported.features, expected[["cluster", "pca_1", "pca_2", "umap_1", "umap_2"]]
    )
    assert app.session_state["projection_result"].known_label.notna().sum() == 2
    assert destination.with_name("projection_labels.tsv").is_file()
    import json

    metadata = json.loads(destination.with_name("projection_projection.json").read_text())
    assert metadata["diagnostics"]["umap_input"]["effective_components"] == (pca_components or None)
    _widget(app, "slider", "Unlabeled point opacity").set_value(0.03).run()
    assert not app.exception and not app.error


def test_mesh_id_check_warns_for_full_embedding_and_builds_render_bounds(tmp_path):
    from streamlit.testing.v1 import AppTest

    segmentation = np.zeros((8, 8, 8), dtype=np.int64)
    segmentation[1:3, 1:3, 1:3] = 300
    segmentation[5:7, 5:7, 5:7] = 700
    np.save(tmp_path / "seg.npy", segmentation)
    (tmp_path / "mapping.tsv").write_text("label_id\tnucleus_id\n3.0\t300.0\n9.0\t900.0\n")

    def mesh_app(directory):
        from pathlib import Path

        from morphofeatures.analysis_ui import _mesh_comparison

        config = {
            "inspection": {
                "mesh": {
                    "source": "segmentation",
                    "segmentation": directory + "/seg.npy",
                    "id_mapping": directory + "/mapping.tsv",
                }
            }
        }
        _mesh_comparison(
            config, [3], "test", embedding_ids=[3, 9], check_root=Path(directory) / "checks"
        )

    app = AppTest.from_function(mesh_app, args=(str(tmp_path),), default_timeout=30).run()
    task = app.session_state["test:meshes:check_task"]
    report = task.future.result(timeout=10)
    assert report["embedded_objects"] == 2 and report["missing_embedding_ids"] == [9]
    app.run()
    assert not app.exception and not app.error
    assert any("1 embedded IDs are missing" in message.value for message in app.warning)
    assert _widget(app, "text_input", "Object bounding-box table").value == ""
    assert not _button(app, "Render selected meshes").disabled
    _button(app, "Render selected meshes").click().run()
    assert not app.exception and not app.error
    batch = app.session_state["test:meshes:batch"]
    assert not batch["failures"] and batch["meshes"][0]["segmentation_id"] == "300"

    _widget(app, "selectbox", "Segmentation contents").set_value("Foreground scores").run()
    task = app.session_state["test:meshes:check_task"]
    assert task.future.result(timeout=10)["state"] == "not_comparable"
    app.run()
    assert not app.exception and not app.error
    assert _button(app, "Render selected meshes").disabled
    assert any("no per-object IDs" in message.value for message in app.warning)


def test_result_selection_gallery_and_reviewed_volume_export(tmp_path, monkeypatch):
    import pandas as pd

    from morphofeatures.representation_analysis import analyze
    from morphofeatures.workspace_jobs import run_job

    ids = np.arange(12, dtype=np.int64) + 2**60 + 1
    features = np.random.default_rng(12).normal(size=(12, 4))
    features[6:] += 4
    embedding = export_embeddings(tmp_path / "embeddings.npz", ids, features)
    pd.DataFrame({"label_id": ids[:8], "cell_type": ["a"] * 4 + ["b"] * 4}).to_csv(
        tmp_path / "labels.tsv", sep="\t", index=False
    )
    np.save(tmp_path / "crops.npy", np.arange(12 * 8**3, dtype=np.float32).reshape(12, 8, 8, 8))
    np.save(tmp_path / "ids.npy", ids)
    segmentation = np.zeros((8, 8, 8), dtype=np.int64)
    segmentation[1:3, 1:3, 1:3] = ids[2]
    np.save(tmp_path / "seg.npy", segmentation)
    prep = tmp_path / "preprocessing.json"
    prep.write_text(
        json.dumps(
            {
                "settings": {
                    "segmentation": str(tmp_path / "seg.npy"),
                    "segmentation_axes": "zyx",
                    "unit": "voxel",
                    "spacing_zyx": [1, 1, 1],
                }
            }
        )
    )
    source = tmp_path / "mae_config.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "crops": str(tmp_path / "crops.npy"),
                    "label_ids": str(tmp_path / "ids.npy"),
                    "preprocessing": str(prep),
                }
            }
        )
    )
    result = analyze(
        embedding,
        tmp_path / "analysis",
        {
            "annotations": str(tmp_path / "labels.tsv"),
            "folds": 2,
            "clusters": 2,
            "umap": False,
            "input_config": str(source),
        },
    )
    # AppTest does not yet synthesize Plotly browser events. Test their decoder
    # separately, and inject the decoded event here to cover selection -> form.
    monkeypatch.setattr(
        "morphofeatures.analysis_ui.selected_object_ids", lambda event, allowed: [int(ids[2])]
    )
    app = _app(tmp_path)
    app.radio(key="workspace_navigation").set_value("Results").run()
    _widget(
        app,
        "text_input",
        "Saved analysis.json, comparison.json, joint_analysis.json, or embedding file",
    ).set_value(str(result)).run()
    assert not app.exception and not app.error
    _widget(app, "slider", "Unlabeled point opacity").set_value(0.05).run()
    _widget(app, "radio", "Analysis section").set_value("Objects & meshes").run()
    assert _widget(app, "selectbox", "Object for nearest-neighbor inspection").value == int(ids[2])
    _widget(app, "number_input", "Number of neighbors").set_value(2).run()
    _button(app, "Show selected object and neighbors").click().run()
    assert not app.exception and not app.error
    assert any(
        widget.label == "Download neighbor table SVG" for widget in app.get("download_button")
    )
    _widget(app, "radio", "Analysis section").set_value("Projection").run()
    _button(app, "Prepare projection SVG / PNG").click().run()
    assert not app.exception and not app.error
    _widget(app, "selectbox", "Graphs to export").set_value("Known cell types").run()
    _button(app, "Prepare projection SVG / PNG").click().run()
    exported = app.session_state[f"projection:{result}:figure:Embedding"]["svg"]
    assert b"Known cell types" in exported and b"Embedding clusters" not in exported
    _button(app, "Add selected points to comparison").click().run()
    monkeypatch.setattr(
        "morphofeatures.analysis_ui.selected_object_ids", lambda event, allowed: [int(ids[9])]
    )
    app.run()
    _button(app, "Add selected points to comparison").click().run()
    _widget(app, "radio", "Analysis section").set_value("Objects & meshes").run()
    _widget(app, "radio", "Comparison mode").set_value("Chosen objects").run()
    assert _widget(app, "multiselect", "Objects to compare").value == [int(ids[2]), int(ids[9])]
    _widget(app, "multiselect", "Objects to compare").set_value(
        [int(ids[2]), int(ids[9]), int(ids[11])]
    ).run()
    _widget(app, "radio", "Comparison mode").set_value("Nearest neighbors").run()
    _widget(app, "radio", "Comparison mode").set_value("Chosen objects").run()
    assert _widget(app, "multiselect", "Objects to compare").value == [
        int(ids[2]),
        int(ids[9]),
        int(ids[11]),
    ]
    _widget(app, "multiselect", "Objects to compare").set_value([int(ids[2]), int(ids[9])]).run()
    _button(app, "Show chosen objects").click().run()
    assert not app.exception and not app.error
    import trimesh

    mesh_folder = tmp_path / "meshes"
    mesh_folder.mkdir()
    for object_id in ids[[2, 9]]:
        trimesh.creation.icosphere(subdivisions=1).export(mesh_folder / f"{object_id}.ply")
    _widget(app, "selectbox", "Mesh source").set_value("Existing mesh files").run()
    _widget(app, "text_input", "Mesh folder (ID.ply / ID.obj / ID.glb / ID.stl)").set_value(
        str(mesh_folder)
    ).run()
    _widget(app, "number_input", "Maximum meshes rendered at once").set_value(1).run()
    assert len(_widget(app, "multiselect", "Meshes to display").value) == 1
    assert len(_widget(app, "multiselect", "Objects to compare").value) == 2
    _button(app, "Render selected meshes").click().run()
    assert not app.exception and not app.error
    batch = app.session_state[f"inspection:{result}:meshes:batch"]
    assert len(batch["meshes"]) == 1
    _button(app, "Prepare mesh comparison SVG / PNG").click().run()
    _button(app, "Prepare mesh files").click().run()
    assert not app.exception and not app.error
    assert {
        "Download mesh comparison SVG",
        "Download mesh comparison PNG",
        "Download displayed meshes ZIP",
    } <= {item.label for item in app.get("download_button")}
    _button(app, "Clear comparison list").click().run()
    assert _widget(app, "multiselect", "Objects to compare").value == []
    _widget(app, "radio", "Comparison mode").set_value("Nearest neighbors").run()
    assert not app.exception and not app.error
    _button(app, "Create volume export workflow").click().run()
    assert not app.exception and not app.error
    assert _widget(app, "text_input", "Original instance segmentation").value == str(
        tmp_path / "seg.npy"
    )
    app.radio(key="step_widget").set_value("Review & run").run()
    _button(app, "Review run").click().run()
    assert not app.exception and not app.error
    _button(app, "Save dry-run bundle").click().run()
    worker = next(tmp_path.glob("experiments/*/workspace/job.yaml"))
    saved = yaml.safe_load(worker.read_text())
    assert saved["stages"][0]["action"] == "export_labels"
    status = json.loads(run_job(worker).read_text())
    assert status["state"] == "completed"
    assert json.loads(Path(status["stages"][0]["result"]).read_text())["matched_objects"] == 1


def test_grouped_decoder_edits_survive_profile_export(tmp_path):
    app = _app(tmp_path)
    source = Path(__file__).parents[1] / "configs/sites/mae_platynereis_nuclei_embl.yaml"
    _widget(app, "text_input", "Training configuration or pipeline YAML").set_value(
        str(source)
    ).run()
    _widget(app, "text_input", "Training profile (optional)").set_value("full").run()
    _button(app, "Preview settings to load").click().run()
    _button(app, "Load settings into draft").click().run()
    for label, value in (("Decoder dim", 96), ("Decoder depth", 3), ("Decoder heads", 6)):
        _widget(app, "number_input", label).set_value(value).run()
    app.radio(key="step_widget").set_value("Analyze").run()
    app.radio(key="step_widget").set_value("Train").run()
    assert not app.exception
    _widget(app, "checkbox", "Edit pipeline YAML").check().run()
    exported = yaml.safe_load(_widget(app, "text_area", "Pipeline YAML").value)
    mae = exported["stages"][0]["config"]["mae"]
    assert (mae["decoder_dim"], mae["decoder_depth"], mae["decoder_heads"]) == (96, 3, 6)
    assert exported["stages"][0]["config"]["profiles"]["full"]["mae"]["decoder_depth"] == 8


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
    _widget(
        app,
        "text_input",
        "Saved analysis.json, comparison.json, joint_analysis.json, or embedding file",
    ).set_value(str(source)).run()
    assert not app.exception and not app.error
    _widget(app, "radio", "Analysis section").set_value("Objects & meshes").run()
    assert _widget(app, "selectbox", "Object for nearest-neighbor inspection").options == [
        "10",
        "20",
        "30",
    ]
    assert not (tmp_path / "experiments").exists()
