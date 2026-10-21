"""Run-table identity, navigation history and portable workspace preferences."""

import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
import yaml

from morphofeatures.config import config_from_mapping, load_config
from morphofeatures.registry import JobRecord, JobRegistry
from morphofeatures.run_presentation import select_table_run, workflow_label
from morphofeatures.slurm import cluster_profiles_from_mapping, load_cluster_profiles
from morphofeatures.workspace_navigation import begin_screen, go_back, remember_screen
from morphofeatures.workspace_preferences import save_settings, validate_settings


def make_run(root, name, stages):
    folder = root / "experiments" / name / "workspace"
    folder.mkdir(parents=True)
    config = folder / "job.yaml"
    config.write_text(yaml.safe_dump({"stages": stages}))
    result = folder / "result.json"
    result.write_text(json.dumps({"schema": "morphofeatures.analysis.v1", "run": name}))
    record = JobRecord(
        run_id=name,
        workflow="workspace_run",
        stage="pipeline",
        command=("python", "-m", "morphofeatures", "workspace-run", "--config", str(config)),
        submission_key=name,
        working_directory=str(folder),
        config_snapshot=str(config),
        stdout_path=str(folder / "stdout.log"),
        stderr_path=str(folder / "stderr.log"),
        metrics_path=str(folder / "metrics.jsonl"),
        application_state="completed",
        artifacts=(str(result),),
    )
    JobRegistry.under_output_root(root).create(record)
    return record


def app_at(root):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "morphofeatures/dashboard.py"), default_timeout=30
    )
    app.session_state["workspace_root"] = str(root)
    return app.run()


def widget(app, kind, label):
    return next(w for w in getattr(app, kind) if w.label == label)


def test_workflow_labels_use_snapshots_and_preserve_internal_keys(tmp_path):
    record = make_run(
        tmp_path,
        "pipeline",
        [
            {"action": "preprocess", "layout": "distributed_patches"},
            {"action": "extract", "model": "dinov3"},
            {"action": "analyze", "umap": True},
        ],
    )
    assert (
        workflow_label(record)
        == "Prepare distributed fine patches → DINOv3 embeddings → PCA + UMAP & clustering"
    )
    assert record.workflow == "workspace_run"
    Path(record.config_snapshot).write_text(
        yaml.safe_dump(
            {
                "stages": [
                    {
                        "action": "dino_experiment",
                        "backbones": [{"model": "dinov2"}, {"model": "dinov3"}],
                    }
                ]
            }
        )
    )
    assert workflow_label(record) == "DINOv2 / DINOv3 fine/coarse comparison"
    assert workflow_label(replace(record, workflow="mae_train")) == "MAE training"
    Path(record.config_snapshot).unlink()
    assert "unavailable" in workflow_label(record)


def test_row_selection_uses_the_rendered_filtered_order():
    state = {"table": {"selection": {"rows": [1]}}}
    select_table_run(state, "table", ["second-run", "first-run"])
    assert state["runs:selected"] == "first-run"
    state["table"]["selection"]["rows"] = [5]
    select_table_run(state, "table", ["filtered-run"])
    assert state["runs:selected"] == "first-run"


def test_multiple_row_selection_retains_inspector_and_tracks_checked_rows():
    state = {"table": {"selection": {"rows": [2, 0]}}}
    ids = ["third", "first", "second"]
    select_table_run(state, "table", ids)
    assert state["runs:selected_ids"] == ["second", "third"]
    assert state["runs:selected"] in state["runs:selected_ids"]
    state["table"]["selection"]["rows"] = [2]
    select_table_run(state, "table", ids)
    assert state["runs:selected_ids"] == ["second"]
    assert state["runs:selected"] == "second"
    state["table"]["selection"]["rows"] = []
    select_table_run(state, "table", ids)
    assert state["runs:selected_ids"] == []


def test_back_restores_nested_result_selection_without_undoing_edits():
    key = "transfer-artifact:/run/transfer.json"
    state = {
        "workspace_navigation": "Results",
        "results:registered": "/run/transfer.json",
        "navigation:selectors": [key],
        key: 0,
        "draft-edits": {"clusters": 12},
    }
    remember_screen(state)
    state[key] = 2
    remember_screen(state)
    go_back(state)
    assert state[key] == 0
    assert state["draft-edits"] == {"clusters": 12}
    begin_screen(state)
    state["navigation:selectors"] = [key]
    remember_screen(state)
    assert not state["navigation:history"]  # Restoring a screen must not add it again.


def test_runs_table_selects_without_dropdown_and_back_restores_run_and_result(tmp_path):
    a = make_run(tmp_path, "first", [{"action": "extract", "model": "dinov2"}])
    b = make_run(tmp_path, "second", [{"action": "transfer"}])
    app = app_at(tmp_path)
    app.radio(key="workspace_navigation").set_value("Runs").run()
    assert not app.exception
    assert not any(w.label == "Inspect run" for w in app.selectbox)
    assert set(app.dataframe[0].value.workflow) == {"DINOv2 embeddings", "Label transfer"}
    assert not widget(app, "button", "Back").disabled
    # AppTest does not yet expose dataframe click events; use the same row callback.
    table_key = next(k for k in app.session_state.filtered_state if k.startswith("runs:table:"))
    ids = JobRegistry.under_output_root(tmp_path).list(limit=500)
    event = {
        table_key: {"selection": {"rows": [next(i for i, r in enumerate(ids) if r.id == a.id)]}}
    }
    select_table_run(event, table_key, [r.id for r in ids])
    app.session_state["runs:selected"] = event["runs:selected"]
    app.run()
    assert not app.exception
    assert any(a.run_id in w.value for w in app.subheader)
    app.session_state["runs:selected"] = b.id
    app.run()
    assert not app.exception
    widget(app, "button", "Back").click().run()
    assert app.session_state["runs:selected"] == a.id
    widget(app, "button", "Open results · Embedding analysis").click().run()
    assert app.radio(key="workspace_navigation").value == "Results"
    manual = "Saved analysis.json, comparison.json, joint_analysis.json, or embedding file"
    assert widget(app, "text_input", manual).value == a.artifacts[0]
    widget(app, "text_input", manual).set_value(b.artifacts[0]).run()
    widget(app, "button", "Back").click().run()
    assert widget(app, "text_input", manual).value == a.artifacts[0]
    widget(app, "button", "Back").click().run()
    assert app.radio(key="workspace_navigation").value == "Runs"
    assert app.session_state["runs:selected"] == a.id
    widget(app, "text_input", "Filter run ID, name or workflow").set_value("Label transfer").run()
    assert app.dataframe[0].value.run.tolist() == ["second"]
    assert not app.exception


def test_completed_dino_report_opens_without_selecting_past_job_yaml(tmp_path, monkeypatch):
    """Historical workspace records register job.yaml/logs before experiment.json."""
    import numpy as np

    from morphofeatures.data.io import export_embeddings

    record = make_run(tmp_path, "fine-coarse", [{"action": "dino_experiment"}])
    folder = Path(record.working_directory) / "00-dino_experiment"
    folder.mkdir()
    vector = folder / "matched-0" / "embeddings.npz"
    vector.parent.mkdir()
    export_embeddings(vector, np.array([2, 2**53 + 7], dtype=np.int64), np.ones((2, 4)))
    vector.with_suffix(".metadata.json").write_text(json.dumps({"schema": "morphofeatures.embedding.v1", "model": "dinov2"}))
    report = folder / "experiment.json"
    report.write_text(json.dumps({
        "schema": "morphofeatures.dino_experiment.v1", "interpretation": "Matched methods.",
        "settings": {}, "cohorts": [{"dataset": "Cells", "selected": 3, "matched": 2}],
        "embeddings": [{"dataset": "Cells", "representation": "Fine CLS", "backbone": "dinov2-0", "scale_um": 1.0,
                        "path": "matched-0/embeddings.npz", "original_path": "extract/views/fixed_tiles.cls.npz"}],
        "comparisons": [], "joint_analyses": [], "transfers": [],
    }))
    # Sparse fixture: ensure browsing never loads an archive above the safeguard.
    archive = folder / "export.zip"
    import zipfile
    with archive.open("wb") as handle:
        handle.seek(201 * 1024**2)
        with zipfile.ZipFile(handle, "w") as bundle:
            bundle.writestr("report.txt", "Valid indexed archive following sparse padding")
    read_bytes = Path.read_bytes

    def bounded_read(path):
        assert path != archive, "Large report archives must not be loaded into browser memory"
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", bounded_read)
    JobRegistry.under_output_root(tmp_path).update(
        record.id, artifacts=(record.config_snapshot, record.stdout_path, str(report)),
        raw_scheduler_state="RUNNING",  # Worker completion, not stale sacct state, determines availability.
    )
    app = app_at(tmp_path)
    app.radio(key="workspace_navigation").set_value("Runs").run()
    app.session_state["runs:selected"] = record.id
    app.run()
    assert not app.exception
    assert widget(app, "selectbox", "Continue from an output").value == str(report)
    assert not any(w.label == "Open this output in Results" for w in app.button)
    widget(app, "button", "Open results · DINO fine/coarse experiment").click().run()
    assert not app.exception and not app.error
    assert app.radio(key="workspace_navigation").value == "Results"
    assert not any(w.label == "Vector artifact" for w in app.selectbox)
    assert any("Vector artifact:" in c.value for c in app.caption)
    labels = {item.label for item in app.get("download_button")}
    assert "Download NPZ vectors and exact object IDs" in labels
    assert "Download vector metadata (keep beside NPZ)" in labels
    assert "Download complete experiment report" not in labels
    assert any("201.0 MiB" in item.value and str(archive) in item.value for item in app.caption)
    widget(app, "radio", "Result view").set_value("Files & provenance").run()
    assert any("Outputs and how to use them" in item.value for item in app.markdown)
    assert any(item.label == "Fine/coarse output files explained" for item in app.expander)
    inventory = next(item.value for item in app.dataframe if "Matched vectors (reuse these)" in item.value.columns)
    assert inventory.iloc[0]["Matched vectors (reuse these)"] == "matched-0/embeddings.npz"
    widget(app, "button", "Back").click().run()
    assert widget(app, "radio", "Result view").value == "Explore results"
    widget(app, "button", "Back").click().run()
    assert app.radio(key="workspace_navigation").value == "Runs"
    assert app.session_state["runs:selected"] == record.id


def test_workflow_output_guides_follow_entry_point_and_actual_stages(tmp_path):
    from morphofeatures.workflow_outputs import expected_outputs
    from morphofeatures.workspace_state import STARTING_POINTS

    assert all(expected_outputs(start=start) for start in STARTING_POINTS)
    app = app_at(tmp_path)
    widget(app, "selectbox", "Starting point").set_value("dino_experiment").run()
    text = "\n".join(item.value for item in app.markdown)
    assert text.index("What you should have") < text.index("What this workflow will produce")
    assert "experiment.json" in text and "joint maps need multiple datasets" in text
    widget(app, "button", "Create workflow").click().run()
    assert not app.exception
    assert app.radio(key="step_widget").value == "Prepare data"
    assert any("What this workflow will produce" in item.value and "experiment.json" in item.value for item in app.markdown)
    # Imported/custom pipelines describe actual enabled stages, not the original preset.
    outputs = expected_outputs([{"action": "preprocess"}, {"action": "extract"}, {"action": "transfer"}])
    assert [title for title, _ in outputs] == ["Prepared data", "Embedding extraction", "Label transfer"]
    assert "Target accuracy" in outputs[-1][1]


def test_fusion_controls_stack_methods_and_scaling_without_losing_settings(tmp_path):
    from morphofeatures.workspace_state import with_fusion_comparison

    app = app_at(tmp_path)
    widget(app, "selectbox", "Starting point").set_value("dino_experiment").run()
    widget(app, "button", "Create workflow").click().run()
    app.radio(key="step_widget").set_value("Analyze").run()
    widget(app, "multiselect", "Feature combination methods").set_value(["balanced", "raw"]).run()
    widget(app, "multiselect", "Additional feature combinations").set_value(["all_four"]).run()
    widget(app, "multiselect", "Analysis scaling methods to compare").set_value(["none", "l2"]).run()
    assert not app.exception
    draft = next(v for k, v in app.session_state.filtered_state.items() if k.startswith("workflow:"))
    stage = draft["document"]["stages"][0]
    assert stage["combination_methods"] == ["balanced", "raw"]
    assert stage["additional_combinations"] == ["all_four"]
    assert stage["analysis_normalizations"] == ["none", "l2"]
    assert "same DINO extraction" in widget(app, "multiselect", "Feature combination methods").proto.help
    assert "held-out folds" in widget(app, "multiselect", "Analysis scaling methods to compare").proto.help
    app.radio(key="step_widget").set_value("Prepare data").run()
    app.radio(key="step_widget").set_value("Analyze").run()
    assert widget(app, "multiselect", "Feature combination methods").value == ["balanced", "raw"]
    # Creating a comparison retains classifier choices, annotations and upstream names.
    original = {"stages": [
        {"action": "fuse", "name": "Combined", "embeddings": {"fine": "fine.npz", "coarse": "coarse.npz"}, "weights": [0.8, 0.2]},
        {"action": "analyze", "from_embedding": "Combined", "annotations": "labels.tsv", "classifier_models": ["mlp"], "folds": 3},
    ]}
    revised = with_fusion_comparison(original, 0)
    a, b, analysis = revised["stages"]
    assert a == original["stages"][0]
    assert b["combination"] == "raw" and b["weights"] == [0.8, 0.2]
    assert analysis["action"] == "compare" and analysis["classifier_models"] == ["mlp"]
    assert analysis["from_embeddings"] == [a["name"], b["name"]]
    assert analysis["annotations"] == "labels.tsv" and analysis["folds"] == 3
    assert analysis["analysis_normalizations"] == ["none", "l2"]
    assert len(original["stages"]) == 2


def test_saved_dino_combinations_make_an_editable_worker_plan(tmp_path):
    import numpy as np

    from morphofeatures.data.io import export_embeddings
    from morphofeatures.dino_experiment_ui import saved_combination_document
    from morphofeatures.workspace_jobs import resolve_job

    entries = []
    for i, name in enumerate(["Fine CLS", "Fine foreground", "Coarse CLS", "Coarse foreground"]):
        source = export_embeddings(tmp_path / str(i) / "embeddings.npz", [1, 2, 3], np.ones((3, 2)))
        source.with_suffix(".metadata.json").write_text(json.dumps({"schema": "morphofeatures.embedding.v1", "dataset_identity": "shared"}))
        entries.append({"dataset": "Cells", "backbone": "dinov2-0", "scale_um": 1.0, "representation": name, "path": str(source.relative_to(tmp_path))})
    metadata = {"settings": {"datasets": [{"name": "Cells"}], "classify": False, "clusters": 2, "umap": False}}
    document = saved_combination_document(metadata, tmp_path / "experiment.json", entries)
    assert [s["action"] for s in document["stages"]] == ["fuse", "fuse", "compare"]
    assert [s["combination"] for s in document["stages"][:2]] == ["balanced", "raw"]
    assert document["stages"][0]["weights"] == [0.25] * 4
    assert document["stages"][-1]["analysis_normalizations"] == ["none", "l2"]
    resolved = resolve_job(document, tmp_path)
    assert len(resolved["stages"]) == 3


def test_cross_source_fusion_declaration_is_editable_and_preserved(tmp_path):
    app = app_at(tmp_path)
    widget(app, "selectbox", "Starting point").set_value("fusion").run()
    widget(app, "button", "Create workflow").click().run()
    app.radio(key="step_widget").set_value("Build features").run()
    widget(app, "checkbox", "I verified matching objects across different source files").check().run()
    widget(app, "text_input", "Shared specimen / dataset name").set_value("Specimen A").run()
    widget(app, "selectbox", "Common embedding ID namespace").set_value("cell").run()
    widget(app, "text_input", "How was object correspondence verified?").set_value("Verified mapping table").run()
    assert not app.exception
    draft = next(v for k, v in app.session_state.filtered_state.items() if k.startswith("workflow:"))
    assert draft["document"]["stages"][0]["identity_override"] == {
        "confirmed": True, "dataset": "Specimen A", "id_namespace": "cell", "reason": "Verified mapping table",
    }
    widget(app, "button", "Add raw/balanced comparison").click().run()
    draft = next(v for k, v in app.session_state.filtered_state.items() if k.startswith("workflow:"))
    assert draft["document"]["stages"][0]["identity_override"] == draft["document"]["stages"][1]["identity_override"]


def test_workspace_settings_resolve_paths_preserve_extras_and_save_portably(tmp_path, monkeypatch):
    monkeypatch.setenv("MORPHOFEATURES_OUTPUT_ROOT", str(tmp_path / "environment-output"))
    source = {
        "paths": {"repo_root": "../repository", "output_root": "results", "data_root": "images"},
        "seed": 17,
        "device": "cpu",
        "custom": {"retain": [1, 2]},
        "slurm": {"profiles": "configs/cluster.yaml"},
        "data": {"raw_volume": "data/raw.n5"},
    }
    document = validate_settings(source, tmp_path / "shared")
    assert document["paths"]["output_root"] == str(tmp_path / "repository/results")
    assert document["custom"] == source["custom"]
    assert document["slurm"]["profiles"] == str(tmp_path / "repository/configs/cluster.yaml")
    path = save_settings(tmp_path / "elsewhere/workspace.yaml", document)
    assert load_config(path).paths.output_root == tmp_path / "environment-output"
    monkeypatch.delenv("MORPHOFEATURES_OUTPUT_ROOT")
    imported = load_config(path)
    assert imported.paths.output_root == tmp_path / "repository/results"
    assert imported.seed == 17 and imported.device == "cpu"
    with pytest.raises(FileExistsError):
        save_settings(path, {**document, "seed": 19})
    save_settings(path, {**document, "seed": 19}, replace_existing=True)
    assert load_config(path).seed == 19
    assert config_from_mapping(source, base=tmp_path / "shared").raw == source
    with pytest.raises(ValueError, match="pipeline"):
        validate_settings({"stages": [{"action": "analyze"}]}, tmp_path)


def test_workspace_settings_editor_applies_saves_and_loads_yaml(tmp_path):
    app = app_at(tmp_path)
    app.radio(key="workspace_navigation").set_value("Workspace settings").run()
    assert not app.exception

    widget(app, "text_input", "Analysis tables and embeddings directory").set_value(
        str(tmp_path / "shared-tables")
    )
    widget(app, "number_input", "Default random seed").set_value(23)
    widget(app, "selectbox", "Default compute device").set_value("cpu").run()
    widget(app, "button", "Apply workspace settings").click().run()
    assert app.session_state["workspace_config_override"]["seed"] == 23
    assert not app.exception
    output = tmp_path / "shared/workspace.yaml"
    widget(app, "text_input", "Save workspace YAML to").set_value(str(output)).run()
    widget(app, "button", "Save workspace YAML").click().run()
    saved = yaml.safe_load(output.read_text())
    assert saved["paths"]["analysis_data"] == str(tmp_path / "shared-tables")
    assert saved["seed"] == 23 and saved["device"] == "cpu"
    saved["seed"] = 29
    output.write_text(yaml.safe_dump(saved))
    widget(app, "text_input", "Workspace defaults YAML").set_value(str(output)).run()
    widget(app, "button", "Load workspace YAML").click().run()
    assert widget(app, "number_input", "Default random seed").value == 29
    assert (
        app.session_state["workspace_config_override"]["seed"] == 23
    )  # Load edits, Apply activates.
    widget(app, "button", "Apply workspace settings").click().run()
    app.radio(key="workspace_navigation").set_value("Tools").run()
    widget(app, "selectbox", "Tool").set_value("Classification and projection tools").run()
    assert widget(app, "number_input", "Projection seed").value == 29
    assert not app.exception
    app.radio(key="workspace_navigation").set_value("Workspace settings").run()
    widget(app, "text_input", "Workspace output directory").set_value(
        str(tmp_path / "another-workspace")
    ).run()
    widget(app, "button", "Apply workspace settings").click().run()
    assert app.session_state["workspace_root"] == str(tmp_path / "another-workspace")
    assert not app.session_state.filtered_state.get("navigation:history")
    assert output.is_file()
    assert not app.exception


def test_dino_workspace_defaults_roundtrip_and_keep_explicit_workflows(tmp_path):
    from morphofeatures.dino_defaults import workspace_defaults
    from morphofeatures.workspace_state import new_draft, stages_after_preparation

    values = {
        "paths": {"repo_root": str(tmp_path)},
        "dino": {
            "dinov2": {
                "model_repository": "code/dinov2",
                "weights_directory": "weights",
                "variant": "dinov2_vitb14_reg",
                "checkpoints": {"dinov2_vitb14_reg": "base-reg.pth", "dinov2_vits14": "small.pth"},
            },
            "dinov3": {
                "model_repository": "code/dinov3",
                "variant": "dinov3_vitb16",
                "checkpoints": {"dinov3_vitb16": "/shared/v3.pth"},
            },
        },
    }
    path = save_settings(tmp_path / "shared.yaml", values)
    config = load_config(path)
    defaults = workspace_defaults(config)
    assert defaults["dinov2"]["checkpoints"]["dinov2_vitb14_reg"] == str(
        tmp_path / "weights/base-reg.pth"
    )
    for route in ("dino", "dino_experiment"):
        draft = new_draft(route, model_defaults=defaults)
        stage = draft["document"]["stages"][0]
        backbone = stage["backbones"][0] if route == "dino_experiment" else stage
        assert backbone["variant"] == "dinov2_vitb14_reg"
        assert backbone["model_repository"] == str(tmp_path / "code/dinov2")
        assert backbone["checkpoint"] == str(tmp_path / "weights/base-reg.pth")
    explicit = {
        "stages": [
            {
                "action": "extract",
                "model": "dinov2",
                "variant": "dinov2_vits14",
                "checkpoint": "/own/small.pth",
                "model_repository": "/own/code",
            }
        ]
    }
    assert new_draft(document=explicit, model_defaults=defaults)["document"] == explicit
    prepared = stages_after_preparation(
        {"action": "preprocess"}, dino=True, model_defaults=defaults
    )
    assert prepared[1]["variant"] == "dinov2_vitb14_reg"
    assert prepared[1]["checkpoint"] == str(tmp_path / "weights/base-reg.pth")
    with pytest.raises(ValueError, match="supported backbone names"):
        validate_settings(
            {"dino": {"dinov2": {"checkpoints": {"dinov3_vitb16": "wrong.pth"}}}}, tmp_path
        )


def test_dino_default_editor_and_new_workflow_selections(tmp_path):
    app = app_at(tmp_path)
    app.radio(key="workspace_navigation").set_value("Workspace settings").run()
    widget(app, "text_input", "DINOv2 backbone repository").set_value(str(tmp_path / "v2-code"))
    widget(app, "text_input", "DINOv2 weights directory (optional)").set_value(
        str(tmp_path / "weights")
    )
    widget(app, "selectbox", "Default DINOv2 backbone").set_value("dinov2_vitb14_reg").run()
    widget(app, "text_input", "DINOv2 weights for dinov2_vitb14_reg").set_value(
        "base-reg.pth"
    ).run()
    widget(app, "text_input", "DINOv3 backbone repository").set_value(str(tmp_path / "v3-code"))
    widget(app, "text_input", "DINOv3 weights for dinov3_vits16").set_value(
        str(tmp_path / "v3-small.pth")
    ).run()
    widget(app, "button", "Apply workspace settings").click().run()
    assert not app.exception
    saved = tmp_path / "defaults.yaml"
    widget(app, "text_input", "Save workspace YAML to").set_value(str(saved)).run()
    widget(app, "button", "Save workspace YAML").click().run()
    assert yaml.safe_load(saved.read_text())["dino"]["dinov2"]["checkpoints"][
        "dinov2_vitb14_reg"
    ] == str(tmp_path / "weights/base-reg.pth")
    app.radio(key="workspace_navigation").set_value("Workflow").run()
    widget(app, "selectbox", "Starting point").set_value("dino_experiment").run()
    widget(app, "button", "Create workflow").click().run()
    assert app.radio(key="step_widget").value == "Prepare data"
    widget(app, "text_input", "Fine slice positions within foreground").set_value(
        "[0.25, 0.5, 0.75]"
    ).run()
    widget(app, "number_input", "Minimum nucleus content per 3D fine tile (0–1)").set_value(
        0.5
    ).run()
    assert not any(w.label == "Backbone variant" for w in app.selectbox)
    app.radio(key="step_widget").set_value("Analyze").run()
    assert widget(app, "selectbox", "Backbone variant").value == "dinov2_vitb14_reg"
    assert widget(app, "text_input", "Local DINO weights").value == str(
        tmp_path / "weights/base-reg.pth"
    )
    assert not any(w.label == "Fine slice positions within foreground" for w in app.text_input)
    assert not any(
        w.label == "Minimum nucleus content per 3D fine tile (0–1)" for w in app.number_input
    )
    widget(app, "selectbox", "Backbone variant").set_value("dinov2_vitb14").run()
    assert widget(app, "text_input", "Local DINO weights").value == ""
    widget(app, "selectbox", "DINO family").set_value("dinov3").run()
    assert widget(app, "selectbox", "Backbone variant").value == "dinov3_vits16"
    assert widget(app, "text_input", "Official local backbone checkout").value == str(
        tmp_path / "v3-code"
    )
    assert widget(app, "text_input", "Local DINO weights").value == str(tmp_path / "v3-small.pth")
    widget(app, "text_input", "Local DINO weights").set_value(str(tmp_path / "custom.pth")).run()
    widget(app, "button", "Edit data and DINO views").click().run()
    assert app.radio(key="step_widget").value == "Prepare data"
    assert (
        widget(app, "text_input", "Fine slice positions within foreground").value
        == "[0.25, 0.5, 0.75]"
    )
    assert (
        widget(app, "number_input", "Minimum nucleus content per 3D fine tile (0–1)").value == 0.5
    )
    app.radio(key="step_widget").set_value("Analyze").run()
    assert widget(app, "text_input", "Local DINO weights").value == str(tmp_path / "custom.pth")
    assert not app.exception


@pytest.mark.parametrize("route", ["dino", "dino_experiment"])
def test_imported_dino_paths_and_landing_page_with_workspace_defaults(tmp_path, route):
    from morphofeatures.workspace_state import new_draft

    document = new_draft(route)["document"]
    stage = document["stages"][0]
    backbone = stage["backbones"][0] if route == "dino_experiment" else stage
    backbone.update(checkpoint="/explicit/model.pth", model_repository="/explicit/code")
    pipeline = tmp_path / "import.yaml"
    pipeline.write_text(yaml.safe_dump(document))
    app = app_at(tmp_path)
    app.session_state["workspace_config_override"] = validate_settings(
        {
            "paths": {"repo_root": str(tmp_path), "output_root": str(tmp_path)},
            "dino": {
                "dinov2": {
                    "model_repository": "/saved/code",
                    "checkpoints": {"dinov2_vits14": "/saved/model.pth"},
                }
            },
        },
        tmp_path,
    )
    app.run()
    widget(app, "text_input", "Training configuration or pipeline YAML").set_value(
        str(pipeline)
    ).run()
    widget(app, "button", "Preview settings to load").click().run()
    widget(app, "button", "Load settings into draft").click().run()
    assert app.radio(key="step_widget").value == (
        "Prepare data" if route == "dino_experiment" else "Extract"
    )
    app.radio(key="step_widget").set_value("Analyze" if route == "dino_experiment" else "Extract").run()
    label = "Local DINO weights" if route == "dino_experiment" else "Pretrained DINO weights"
    assert widget(app, "text_input", label).value == "/explicit/model.pth"
    widget(app, "button", "Use saved DINO locations for this backbone").click().run()
    assert widget(app, "text_input", label).value == "/saved/model.pth"
    assert not app.exception


def test_workspace_presets_round_trip_without_external_file(tmp_path):
    from morphofeatures.cluster_presets_ui import import_presets

    imported = import_presets(
        "profiles:\n  gpu:\n    cpus: '12'\n    mail_types: end,fail\n    mail_user: biologist@example.org\n    site_note: Keep this note\n"
    )
    document = validate_settings({"slurm": {"resource_presets": imported}}, tmp_path)
    path = save_settings(tmp_path / "workspace.yaml", document)
    saved = load_config(path).raw["slurm"]["resource_presets"]
    assert saved["gpu"]["site_note"] == "Keep this note"
    assert saved["gpu"]["cpus"] == 12
    assert saved["gpu"]["mail_types"] == ["END", "FAIL"]
    assert "profiles" not in document["slurm"]
    standalone = tmp_path / "presets.yaml"
    standalone.write_text(yaml.safe_dump({"profiles": saved}))
    assert load_cluster_profiles(standalone) == cluster_profiles_from_mapping(saved)
    with pytest.raises(ValueError, match="Cluster resource presets.*Preset 'gpu'.*memory"):
        validate_settings({"slurm": {"resource_presets": {"gpu": {"memory": "lots"}}}}, tmp_path)
    with pytest.raises(ValueError, match="preset_source"):
        validate_settings({"slurm": {"preset_source": "missing"}}, tmp_path)


def test_cluster_preset_editor_management_validation_and_navigation(tmp_path):
    app = app_at(tmp_path)
    app.radio(key="workspace_navigation").set_value("Workspace settings").run()
    widget(app, "radio", "Cluster preset source").set_value("workspace").run()
    widget(app, "text_input", "Partition").set_value("gpu")
    widget(app, "text_input", "Memory").set_value("64G")
    widget(app, "number_input", "CPUs per task").set_value(12)
    widget(app, "number_input", "GPUs").set_value(1)
    widget(app, "text_input", "New preset name").set_value("gpu-long").run()
    widget(app, "button", "Rename preset").click().run()
    assert not app.exception
    assert widget(app, "selectbox", "Preset to edit").value == "gpu-long"
    widget(app, "text_input", "New preset name").set_value("gpu-copy").run()
    widget(app, "button", "Duplicate preset").click().run()
    assert widget(app, "text_input", "Memory").value == "64G"
    widget(app, "text_input", "Memory").set_value("96G").run()
    widget(app, "selectbox", "Preset to edit").set_value("gpu-long").run()
    assert widget(app, "text_input", "Memory").value == "64G"
    widget(app, "text_area", "Module / environment setup").set_value("[[module, load").run()
    assert any("Module / environment setup" in error.value for error in app.error)
    widget(app, "selectbox", "Preset to edit").set_value("gpu-copy").run()
    assert not any(button.label == "Apply workspace settings" for button in app.button)
    assert any("gpu-long" in error.value and "setup" in error.value for error in app.error)
    widget(app, "selectbox", "Preset to edit").set_value("gpu-long").run()
    assert widget(app, "text_area", "Module / environment setup").value == "[[module, load"
    widget(app, "text_area", "Module / environment setup").set_value(
        "[[module, load, CUDA/12.8]]"
    ).run()
    widget(app, "multiselect", "Notify me when").set_value(["END", "FAIL"]).run()
    assert any("gpu-long" in error.value and "mail_user" in error.value for error in app.error)
    widget(app, "text_input", "Mail address (optional)").set_value("biologist@example.org").run()
    app.radio(key="workspace_navigation").set_value("Runs").run()
    app.radio(key="workspace_navigation").set_value("Workspace settings").run()
    assert widget(app, "multiselect", "Notify me when").value == ["END", "FAIL"]
    assert widget(app, "number_input", "CPUs per task").value == 12
    assert not app.exception and not app.error
    widget(app, "text_input", "New preset name").set_value("cpu-test").run()
    widget(app, "button", "Add preset").click().run()
    assert widget(app, "number_input", "GPUs").value == 0
    widget(app, "button", "Remove selected preset").click().run()
    assert "cpu-test" not in widget(app, "selectbox", "Preset to edit").options
    widget(app, "button", "Apply workspace settings").click().run()
    presets = app.session_state["workspace_config_override"]["slurm"]["resource_presets"]
    assert list(presets) == ["gpu-long", "gpu-copy"]
    assert presets["gpu-copy"]["memory"] == "96G"
    assert presets["gpu-long"]["mail_types"] == ["END", "FAIL"]
    assert not app.exception and not app.error
    # A rejected import keeps both pending and active presets intact.
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("profiles:\n  broken:\n    gpus: -1\n")
    widget(app, "text_input", "Cluster resource presets YAML").set_value(str(invalid)).run()
    widget(app, "button", "Import presets into editor").click().run()
    assert any("Import YAML" in error.value and "broken" in error.value for error in app.error)
    assert list(app.session_state["workspace:settings:document"]["slurm"]["resource_presets"]) == [
        "gpu-long",
        "gpu-copy",
    ]
    assert app.session_state["workspace_config_override"]["slurm"]["resource_presets"] == presets

    # Switching to an external file explicitly takes precedence over embedded presets.
    external = tmp_path / "external.yaml"
    external.write_text("profiles:\n  external-cpu:\n    memory: 16G\n")
    widget(app, "text_input", "Cluster resource presets YAML").set_value(str(external))
    widget(app, "radio", "Cluster preset source").set_value("file").run()
    widget(app, "button", "Apply workspace settings").click().run()
    app.radio(key="workspace_navigation").set_value("Workflow").run()
    widget(app, "selectbox", "Starting point").set_value("demo").run()
    widget(app, "button", "Create workflow").click().run()
    app.radio(key="step_widget").set_value("Review & run").run()
    widget(app, "selectbox", "Run on").set_value("slurm").run()
    assert widget(app, "selectbox", "Cluster resource preset").value == "external-cpu"
    widget(app, "button", "Use cluster resource preset").click().run()
    assert widget(app, "text_input", "Memory").value == "16G"
    assert not app.exception and not app.error


@pytest.mark.parametrize("upload", [False, True])
def test_cluster_presets_import_apply_export_and_reach_submission(tmp_path, monkeypatch, upload):
    import streamlit as st

    contents = yaml.safe_dump(
        {
            "profiles": {
                "lab-gpu": {
                    "partition": "gpu",
                    "account": "lab",
                    "qos": "normal",
                    "memory": "48G",
                    "cpus": 12,
                    "gpus": 1,
                    "time": "2-00:00:00",
                    "gpu_directive": "gres",
                    "mail_types": ["END", "FAIL"],
                    "mail_user": "biologist@example.org",
                    "setup": [["module", "load", "CUDA/12.8"]],
                    "python_executable": "/opt/env/bin/python",
                    "cpu_thread_env": True,
                    "local_runtime_directories": ["matplotlib"],
                    "log_job_context": True,
                }
            }
        }
    )
    source = tmp_path / "cluster.yaml"
    source.write_text(contents)
    if upload:
        uploader = st.file_uploader

        def fixture_upload(label, *args, **kwargs):
            if label == "Upload cluster resource presets YAML":
                return BytesIO(contents.encode())
            return uploader(label, *args, **kwargs)

        monkeypatch.setattr(st, "file_uploader", fixture_upload)
    app = app_at(tmp_path)
    app.radio(key="workspace_navigation").set_value("Workspace settings").run()
    widget(app, "text_input", "Cluster resource presets YAML").set_value(
        str(tmp_path / "missing.yaml") if upload else str(source)
    ).run()
    widget(app, "button", "Import presets into editor").click().run()
    assert not app.exception and not app.error
    assert widget(app, "radio", "Cluster preset source").value == "workspace"
    assert widget(app, "selectbox", "Preset to edit").value == "lab-gpu"
    assert widget(app, "multiselect", "Notify me when").value == ["END", "FAIL"]
    assert "workspace_config_override" not in app.session_state
    widget(app, "button", "Apply workspace settings").click().run()
    workspace_file = tmp_path / "shared.yaml"
    widget(app, "text_input", "Save workspace YAML to").set_value(str(workspace_file)).run()
    widget(app, "button", "Save workspace YAML").click().run()
    saved = yaml.safe_load(workspace_file.read_text())
    assert saved["slurm"]["preset_source"] == "workspace"
    assert saved["slurm"]["resource_presets"]["lab-gpu"]["memory"] == "48G"
    assert any(w.label == "Download cluster presets YAML" for w in app.get("download_button"))
    assert source.read_text() == contents
    source.unlink()  # Workspace presets must be self-contained after import.
    app.radio(key="workspace_navigation").set_value("Workflow").run()
    widget(app, "selectbox", "Starting point").set_value("demo").run()
    widget(app, "button", "Create workflow").click().run()
    app.radio(key="step_widget").set_value("Review & run").run()
    widget(app, "selectbox", "Run on").set_value("slurm").run()
    assert widget(app, "selectbox", "Preset source").value == "workspace"
    assert widget(app, "selectbox", "Cluster resource preset").value == "lab-gpu"
    assert widget(app, "text_input", "Memory").value == "8G"  # Explicit load required.
    widget(app, "button", "Use cluster resource preset").click().run()
    assert widget(app, "text_input", "Memory").value == "48G"
    widget(app, "button", "Review run").click().run()
    assert not app.exception and not app.error
    scripts = [code.value for code in app.code if "#SBATCH" in code.value]
    assert len(scripts) == 1
    for expected in (
        "--partition=gpu",
        "--account=lab",
        "--qos=normal",
        "--mem=48G",
        "--cpus-per-task=12",
        "--time=2-00:00:00",
        "--gres=gpu:1",
        "--mail-type=END,FAIL",
        "--mail-user=biologist@example.org",
        "module load CUDA/12.8",
        "/opt/env/bin/python",
        "OMP_NUM_THREADS",
        "MPLCONFIGDIR",
    ):
        assert expected in scripts[0]
    # Published workflow tools use the same active presets.
    app.radio(key="workspace_navigation").set_value("Tools").run()
    widget(app, "selectbox", "Tool").set_value(
        "Published shape / texture workflows and MAE sweeps"
    ).run()
    assert widget(app, "selectbox", "Cluster profile").options == ["lab-gpu"]
    assert not app.exception
