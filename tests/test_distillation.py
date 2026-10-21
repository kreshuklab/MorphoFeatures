"""Scientific split/identity contracts and the guided follow-up entry points."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from morphofeatures.analysis.distillation import (
    apply_pca,
    defaults,
    distill_embeddings,
    prepare_split,
    validate_distillation,
)
from morphofeatures.analysis.joint import _signature
from morphofeatures.analysis.transfer_adapter import NumericAdapter
from morphofeatures.data.io import export_embeddings, load_embeddings
from morphofeatures.followup_ui import embedding_artifacts, followup_document
from morphofeatures.workspace_jobs import plan_workspace_job, resolve_job, run_job
from morphofeatures.workspace_state import new_draft


def inputs(folder):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12)
    ids = 2**53 + 11 + np.arange(36, dtype=np.int64)
    labels = np.array(["A", "B"] * 18)
    x = rng.normal(size=(36, 8))
    x[:, 0] += (labels == "A") * 2
    embedding = export_embeddings(folder / "input.npz", ids[::-1], x[::-1])
    embedding.with_suffix(".metadata.json").write_text(json.dumps({
        "dataset_identity": "test-specimen", "model": "dinov2",
        "sources": {"checkpoint": {"sha256": "frozen"}},
        "inspection": {"example": "preserved"},
    }))
    pd.DataFrame({"label_id": ids[::-1], "cell_type": labels[::-1], "specimen": np.repeat(np.arange(6), 6)[::-1]}).to_csv(folder / "labels.tsv", sep="\t", index=False)
    return {**defaults(), "embedding": str(embedding), "annotations": str(folder / "labels.tsv"),
            "folds": 3, "pca_components": 4,
            "adapter": {"epochs": 2, "hidden_dim": 8, "latent_dim": 4, "batch_size": 16, "device": "cpu"},
            "analysis": {"umap": False, "clusters": 2, "normalization": "none", "seed": 42}}


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    folder = tmp_path_factory.mktemp("distillation")
    settings = inputs(folder)
    report_path = distill_embeddings(settings, folder / "result")
    return settings, report_path, json.loads(report_path.read_text())


def test_split_native_ids_grouping_and_shuffled_annotation_order(tmp_path):
    settings = inputs(tmp_path)
    table = load_embeddings(settings["embedding"]).sorted()
    settings["group_column"] = "specimen"
    annotations, rows, y, groups, train, test, folds = prepare_split(table, settings)
    assert not set(groups[train]) & set(groups[test])
    assert not set(table.label_ids[rows[train]]) & set(table.label_ids[rows[test]])
    assert set(y[train]) == set(y[test]) == {"A", "B"}
    assert folds == 3 and table.label_ids.min() > 2**53
    annotations.sample(frac=1, random_state=3).to_csv(settings["annotations"], sep="\t", index=False)
    repeated = prepare_split(table, settings)
    np.testing.assert_array_equal(repeated[4], train)


def test_exported_models_reproduce_vectors_and_preserve_identity(completed):
    settings, path, report = completed
    source = load_embeddings(settings["embedding"]).sorted()
    adapter = NumericAdapter.load(path.parent / "adapter_model/adapter.json")
    with np.load(path.parent / "pca_parameters.npz") as params:
        reconstructed = apply_pca(params, source.features)
    np.testing.assert_allclose(reconstructed, load_embeddings(report["representations"]["pca"]["embedding"]).features, atol=2e-6)
    np.testing.assert_allclose(adapter.transform(source.features), load_embeddings(report["representations"]["adapter"]["embedding"]).features, atol=2e-6)
    split = pd.read_csv(path.parent / "split.tsv", sep="\t")
    training_ids = set(split.loc[split.role == "train", "label_id"])
    for method, entry in report["representations"].items():
        full = load_embeddings(entry["embedding"])
        if method == "base":
            np.testing.assert_array_equal(full.features, source.features)
        holdout = load_embeddings(entry["held_out"])
        np.testing.assert_array_equal(full.label_ids, source.label_ids)
        assert not set(holdout.label_ids) & training_ids
        sidecar = json.loads(Path(entry["embedding"]).with_suffix(".metadata.json").read_text())
        assert sidecar["inspection"] == {"example": "preserved"}
        assert sidecar["dataset_identity"] == "test-specimen"
        if method != "base":
            assert _signature(sidecar)["checkpoint"]
        assert Path(entry["confusion"]).is_file()
    assert len(report["metrics"]) == 3
    assert not adapter.metadata["target_features_used"]
    comparison = json.loads(Path(report["comparison"]).read_text())
    assert not any("classification" in entry for entry in comparison["representations"].values())
    assert (path.parent / "export.zip").is_file()


def test_validation_features_do_not_fit_scalers_or_reducers(tmp_path, completed, monkeypatch):
    settings, _, _ = completed
    table = load_embeddings(settings["embedding"]).sorted()
    split = prepare_split(table, settings)
    rows, train, test = split[1], split[4], split[5]
    changed = table.features.copy()
    changed[rows[test]] += 100
    # Preserve the predeclared split: test labels are evaluation data only.
    changed_y = split[2].copy()
    changed_y[test] = changed_y[test][::-1]
    monkeypatch.setattr("morphofeatures.analysis.distillation.prepare_split",
                        lambda table, settings: (split[0], rows, changed_y, split[3], train, test, split[6]))
    new_input = export_embeddings(tmp_path / "changed.npz", table.label_ids, changed)
    new_report = distill_embeddings({**settings, "embedding": str(new_input)}, tmp_path / "changed")
    original_model = NumericAdapter.load(completed[1].parent / "adapter_model/adapter.json")
    new_model = NumericAdapter.load(new_report.parent / "adapter_model/adapter.json")
    for name in original_model.arrays:
        np.testing.assert_array_equal(original_model.arrays[name], new_model.arrays[name])
    with np.load(completed[1].parent / "pca_parameters.npz") as original, np.load(new_report.parent / "pca_parameters.npz") as other:
        for name in original.files:
            np.testing.assert_array_equal(original[name], other[name])


def test_grouped_bilateral_excludes_training_specimens_and_preserves_candidates(tmp_path):
    settings = inputs(tmp_path)
    settings["group_column"] = "specimen"
    annotation = pd.read_csv(settings["annotations"], sep="\t")
    # Keep some unlabeled objects in every group to exercise probe membership.
    annotation.loc[annotation.index % 6 == 0, "cell_type"] = None
    annotation.to_csv(settings["annotations"], sep="\t", index=False)
    ids = load_embeddings(settings["embedding"]).label_ids
    candidate_path = tmp_path / "candidates.tsv"
    # Broad synthetic candidate sets ensure each retained query has a partner.
    pd.DataFrame([(int(a), int(b)) for a in ids for b in ids if a != b],
                 columns=["source_id", "target_id"]).to_csv(candidate_path, sep="\t", index=False)
    settings["bilateral"] = {
        "enabled": True, "same_specimen": True, "id_namespace": "cell",
        "candidates": str(candidate_path), "workers": 1,
        "variants": [{"name": "raw", "normalization": "none", "metric": "euclidean", "search": "all", "pca": None}],
    }
    path = distill_embeddings(settings, tmp_path / "result")
    report = json.loads(path.read_text())
    split = pd.read_csv(path.parent / "split.tsv", sep="\t")
    training_groups = set(split.loc[split.role == "train", "group"])
    evaluation = load_embeddings(report["representations"]["adapter"]["held_out"])
    held_groups = set(annotation.set_index("label_id").loc[evaluation.label_ids, "specimen"])
    assert not training_groups & held_groups
    bilateral = json.loads(Path(report["bilateral"]).read_text())
    assert bilateral["matched_objects"] == len(evaluation.label_ids)
    assert len(bilateral["representations"]) == 3
    for value in bilateral["representations"].values():
        table = load_embeddings(Path(report["bilateral"]).parent / value["directory"] / "embeddings.npz")
        np.testing.assert_array_equal(table.label_ids, evaluation.label_ids)


def test_configuration_and_named_worker_handoff(tmp_path, monkeypatch):
    settings = inputs(tmp_path)
    calls = []
    def fake_distill(stage, destination):
        calls.append(stage)
        for method in ("base", "pca", "adapter"):
            export_embeddings(destination / method / "embeddings.npz", [10, 20, 30], np.eye(3))
        path = destination / "distillation.json"
        path.write_text("{}")
        return path
    monkeypatch.setattr("morphofeatures.analysis.distillation.distill_embeddings", fake_distill)
    stages = [settings, {"action": "compare", "from_embeddings": ["Distilled.base", "Distilled.pca", "Distilled.adapter"],
                         "normalization": "none", "umap": False, "clusters": 2}]
    document = {"stages": stages, "slurm": {"partition": "test", "cpus": 2, "memory": "4G", "time": "00:10:00"}}
    resolved = resolve_job(document, tmp_path)
    plan = plan_workspace_job(document, output_root=tmp_path, run_id="plan", execution="slurm", base=tmp_path)
    assert "#SBATCH" in plan.script and "distill" in plan.resolved_yaml
    job = tmp_path / "job.yaml"
    resolved["worker"] = {"directory": str(tmp_path / "worker")}
    job.write_text(yaml.safe_dump(resolved))
    run_job(job)
    status = json.loads((tmp_path / "worker/status.json").read_text())
    assert status["state"] == "completed"
    assert calls and any(Path(p).name == "distillation.json" for p in status["artifacts"])
    assert new_draft("distill")["document"]["stages"][0]["action"] == "distill"


def test_followup_drafts_link_distillation_to_heldout_bilateral(completed):
    settings, path, report = completed
    choices = embedding_artifacts(report, path)
    assert len(choices) == 6 and all(Path(v).is_file() for v in choices.values())
    document = followup_document({"DINO": settings["embedding"]}, ["distill", "bilateral"], source=path,
                                 settings=settings, distill_then_compare=True)
    stage = document["stages"][0]
    assert len(document["stages"]) == 1 and stage["bilateral_population"] == "held_out"
    assert stage["bilateral"]["enabled"] and stage["bilateral"]["same_specimen"]
    assert document["followup_of"]["analysis"] == str(path)
    with pytest.raises(ValueError, match="at least two"):
        followup_document({"one": settings["embedding"]}, ["fuse"], source=path)


def test_pointed_validation_errors(tmp_path):
    settings = inputs(tmp_path)
    with pytest.raises(ValueError, match="Validation fold"):
        validate_distillation({**settings, "validation_fold": 3})
    pd.DataFrame({"label_id": [1, 2], "label": ["A", "B"]}).to_csv(settings["annotations"], sep="\t", index=False)
    with pytest.raises(ValueError, match="ID namespace"):
        prepare_split(load_embeddings(settings["embedding"]), settings)


def test_distillation_workflow_controls_and_followup_sections(tmp_path, completed):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).parents[1] / "morphofeatures/dashboard.py"), default_timeout=45)
    app.session_state["workspace_root"] = str(tmp_path)
    app.run()
    def widget(kind, label):
        return next(w for w in getattr(app, kind) if w.label == label)
    widget("selectbox", "Starting point").set_value("distill").run()
    widget("button", "Create workflow").click().run()
    assert not app.exception
    assert widget("number_input", "PCA dimensions").value == 64
    assert widget("number_input", "Exported adapter dimensions").value == 64
    assert widget("text_input", "Known cell-type table").proto.help
    app.session_state["navigate_to"] = "Results"
    app.session_state["value:results:path"] = str(completed[1])
    app.session_state["widget:results:path"] = str(completed[1])
    app.run()
    assert not app.exception and not app.error
    widget("radio", "Distillation results").set_value("Maps & inspection").run()
    assert not app.exception and not app.error
    assert len(widget("multiselect", "Representations to display").value) == 1
    widget("radio", "Analysis section").set_value("Objects & meshes").run()
    assert not app.exception and not app.error
    assert widget("radio", "Comparison mode").value == "Nearest neighbors"
    widget("radio", "Result view").set_value("Next steps").run()
    widget("radio", "Follow-up scope").set_value("Whole result").run()
    assert widget("multiselect", "Input representations for follow-up").value
    widget("multiselect", "Follow-up analyses").set_value(["analyze"]).run()
    widget("button", "Create follow-up workflow draft").click().run()
    assert not app.exception
    draft = next(v for k, v in app.session_state.filtered_state.items() if k.startswith("workflow:"))
    assert len(draft["document"]["stages"]) == 3
    assert all(s["embedding"].endswith("held_out.npz") for s in draft["document"]["stages"])
