"""Analysis batches and Results revisions preserve scientific identity and lineage."""

import json
import time
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from morphofeatures.analysis import projection, variants
from morphofeatures.analysis.joint import joint_analyze
from morphofeatures.data.io import export_embeddings, load_embeddings
from morphofeatures.provenance import build_provenance, workflow_overview
from morphofeatures.registry import JobRecord, JobRegistry
from morphofeatures.representation_analysis import analyze, compare


def inputs(root, *, dimensions=6):
    root.mkdir(parents=True, exist_ok=True)
    ids = np.arange(20, dtype=np.int64) + 2**60 + 3
    features = np.random.default_rng(dimensions).normal(size=(20, dimensions))
    features[10:] += 2
    source = export_embeddings(root / "input.npz", ids, features)
    annotation = root / "labels.tsv"
    pd.DataFrame({"label_id": ids, "cell_type": ["a"] * 10 + ["b"] * 10}).to_csv(
        annotation, sep="\t", index=False
    )
    return source, annotation


def test_cluster_pca_is_independent_of_umap_input_and_reports_its_metric_space(monkeypatch):
    ids = np.arange(30)
    features = np.random.default_rng(10).normal(size=(30, 7))
    umap_inputs = []
    monkeypatch.setattr(
        projection, "compute_umap", lambda x, **kw: umap_inputs.append(x.copy()) or x[:, :2]
    )
    with patch.object(
        projection, "cluster_embeddings", wraps=projection.cluster_embeddings
    ) as cluster:
        frame, report = projection.project_embeddings(
            ids,
            features,
            {
                "cluster_pca_components": 3,
                "umap_pca_components": 2,
                "normalization": "none",
                "clusters": 4,
            },
        )
    expected = PCA(3, random_state=42).fit_transform(features)
    np.testing.assert_allclose(cluster.call_args.args[0], expected)
    np.testing.assert_allclose(umap_inputs[0], PCA(2, random_state=42).fit_transform(features))
    assert report["cluster_input"]["effective_components"] == 3
    assert report["silhouette"] == pytest.approx(silhouette_score(expected, frame.cluster))
    unchanged, _ = projection.project_embeddings(
        ids,
        features,
        {
            "cluster_pca_components": 3,
            "umap": False,
            "normalization": "none",
            "clusters": 4,
        },
    )
    pd.testing.assert_frame_equal(frame.drop(columns=["umap_1", "umap_2"]), unchanged)


def test_stacked_analysis_preserves_predictions_vectors_and_classification_settings(tmp_path):
    source, labels = inputs(tmp_path)
    parent = analyze(
        source,
        tmp_path / "analysis",
        {
            "umap": False,
            "normalization": "none",
            "clusters": 2,
            "annotations": str(labels),
            "classifier_models": ["logistic"],
            "folds": 2,
            "analysis_variants": [
                {
                    "name": "PCA2 K3",
                    "normalization": "standardize",
                    "cluster_pca_components": 2,
                    "clusters": 3,
                },
                {"name": "Full K4", "cluster_pca_components": 0, "clusters": 4},
            ],
        },
    )
    original = json.loads(parent.read_text())
    original_vectors = (parent.parent / "embeddings.npz").read_bytes()
    rows = variants.variant_history(parent)
    assert [row["state"] for row in rows] == ["completed", "completed"]
    for row in rows:
        result = Path(row["result"])
        report = json.loads(result.read_text())
        assert report["classification"] == original["classification"]
        assert report["classification_settings"]["normalization"] == "none"
        assert (result.parent / "embeddings.npz").read_bytes() == original_vectors
        for name in ("predictions.tsv", "splits.tsv"):
            assert (result.parent / name).read_bytes() == (parent.parent / name).read_bytes()
        a = pd.read_csv(parent.parent / "object_labels.tsv", sep="\t").drop(columns="cluster")
        b = pd.read_csv(result.parent / "object_labels.tsv", sep="\t").drop(columns="cluster")
        pd.testing.assert_frame_equal(a, b)
        assert (result.parent / "projection.svg").is_file()
        assert (result.parent / "projection.png").is_file()
        assert (result.parent / "export.zip").is_file()
        with zipfile.ZipFile(result.parent / "export.zip") as archive:
            assert json.loads(archive.read("variant.json"))["state"] == "completed"
    assert json.loads(parent.read_text()) == original
    graph = build_provenance(artifact=parent)
    assert sum(e["label"] == "analysis combination" for e in graph["edges"]) == 2


def test_comparison_variants_use_the_matched_plotted_cohort(tmp_path):
    a, labels = inputs(tmp_path / "a", dimensions=5)
    b, _ = inputs(tmp_path / "b", dimensions=7)
    parent = compare(
        {"Fine": str(a), "Coarse": str(b)},
        tmp_path / "comparison",
        {
            "umap": False,
            "classify": False,
            "annotations": str(labels),
            "subset": 12,
            "analysis_variants": [
                {"name": "PCA3", "cluster_pca_components": 3, "clusters": 3, "seed": 19}
            ],
        },
    )
    row = variants.variant_history(parent)[0]
    assert row["state"] == "completed", row
    new = Path(row["result"])
    old_frame = pd.read_csv(parent.parent / "coordinates.tsv", sep="\t")
    frame = pd.read_csv(new.parent / "coordinates.tsv", sep="\t")
    assert frame.groupby("representation").size().tolist() == [12, 12]
    for name in frame.representation.unique():
        np.testing.assert_array_equal(
            frame.loc[frame.representation == name, "label_id"],
            old_frame.loc[old_frame.representation == name, "label_id"],
        )


def test_joint_revision_preserves_native_identity_predictions_and_inspection(tmp_path):
    a, labels = inputs(tmp_path / "a")
    b, labels_b = inputs(tmp_path / "b")
    for name in ("a", "b"):
        (tmp_path / f"{name}.yaml").write_text("data: {}\n")
    parent = joint_analyze(
        {
            "datasets": [
                {
                    "name": "A",
                    "embedding": str(a),
                    "annotations": str(labels),
                    "input_config": str(tmp_path / "a.yaml"),
                },
                {
                    "name": "B",
                    "embedding": str(b),
                    "annotations": str(labels_b),
                    "input_config": str(tmp_path / "b.yaml"),
                },
            ],
            "assume_same_model": True,
            "max_objects_per_dataset": 12,
            "umap": False,
            "clusters": 2,
        },
        tmp_path / "joint",
    )
    frame = pd.read_csv(parent.parent / "coordinates.tsv", sep="\t")
    frame["predicted_label"] = "a"
    frame.to_csv(parent.parent / "coordinates.tsv", sep="\t", index=False)
    variants.run_configured_variants(
        parent,
        {"analysis_variants": [{"name": "K3 PCA2", "cluster_pca_components": 2, "clusters": 3}]},
    )
    row = variants.variant_history(parent)[0]
    assert row["state"] == "completed", row
    result = Path(row["result"])
    revised = pd.read_csv(result.parent / "coordinates.tsv", sep="\t")
    pd.testing.assert_frame_equal(
        frame[["label_id", "source_label_id", "dataset", "predicted_label"]],
        revised[["label_id", "source_label_id", "dataset", "predicted_label"]],
    )
    native = result.parent / "dataset-1"
    coordinates = pd.read_csv(native / "coordinates.tsv", sep="\t")
    assert coordinates.predicted_label.eq("a").all()
    assert coordinates.label_id.isin(load_embeddings(b).label_ids).all()
    meta = json.loads((native / "analysis.json").read_text())
    assert meta["settings"]["input_config"] == str(tmp_path / "b.yaml")
    assert meta["diagnostics"]["cluster_input"]["fitted_objects"] == 24
    assert len(pd.read_csv(native / "object_labels.tsv", sep="\t")) == 20


def test_failed_combination_keeps_other_results_and_parent_unchanged(tmp_path, monkeypatch):
    source, _ = inputs(tmp_path)
    parent = analyze(source, tmp_path / "analysis", {"umap": False})
    before = parent.read_bytes()
    original = variants.project_embeddings

    def fail_one(ids, features, settings):
        if settings["clusters"] == 7:
            raise RuntimeError("deliberate scientific failure")
        return original(ids, features, settings)

    monkeypatch.setattr(variants, "project_embeddings", fail_one)
    variants.run_configured_variants(
        parent,
        {
            "analysis_variants": [
                {"name": "bad", "clusters": 7},
                {"name": "good", "clusters": 3},
            ]
        },
    )
    history = variants.variant_history(parent)
    assert [r["state"] for r in history] == ["failed", "completed"]
    assert "deliberate scientific failure" in history[0]["error"]
    assert parent.read_bytes() == before


def test_background_results_attach_to_existing_run_and_rename_provenance(tmp_path):
    source, _ = inputs(tmp_path)
    workspace = tmp_path / "workspace"
    parent = analyze(source, workspace / "00-analyze", {"umap": False, "clusters": 2})
    registry = JobRegistry(tmp_path / "registry.sqlite3")
    record = registry.create(
        JobRecord(
            run_id="stable-id",
            workflow="workspace_run",
            stage="analysis",
            command=(),
            submission_key="first",
            working_directory=str(workspace),
            config_snapshot=str(workspace / "job.yaml"),
            stdout_path="stdout",
            stderr_path="stderr",
            metrics_path="metrics",
            application_state="completed",
            artifacts=(str(parent),),
        )
    )
    (workspace / "job.yaml").write_text(
        yaml.safe_dump(
            {
                "worker": {"registry": str(registry.path), "record_id": record.id},
                "stages": [{"action": "analyze", "embedding": str(source)}],
            }
        )
    )
    (workspace / "status.json").write_text(
        json.dumps(
            {
                "state": "completed",
                "stages": [
                    {"index": 0, "state": "completed", "result": str(parent)},
                ],
            }
        )
    )
    variants.start_variants(
        parent, [{"name": "New K3", "clusters": 3}, {"name": "PCA2", "cluster_pca_components": 2}]
    )
    with pytest.raises(ValueError, match="already active"):
        variants.start_variants(parent, [{"name": "duplicate"}])
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        history = variants.variant_history(parent)
        if all(row["state"] in {"completed", "failed"} for row in history):
            break
        time.sleep(0.1)
    assert [row["state"] for row in history] == ["completed", "completed"], history
    updated = registry.get(record.id)
    assert updated.application_state == "completed"
    assert updated.run_id == "stable-id"
    assert set(updated.artifacts) == {str(parent), *(row["result"] for row in history)}
    registry.rename_run("stable-id", "Nuclei · comparison")
    assert registry.get(record.id).display_name == "Nuclei · comparison"
    result = Path(history[0]["result"])
    variants.rename_result(result, "PCA trial")
    assert variants.variant_history(parent)[0]["name"] == "PCA trial"
    graph = build_provenance(artifact=workspace)
    assert any(n["label"] == "PCA trial" for n in graph["nodes"])
    assert any(n["label"] == "PCA trial" for n in workflow_overview(graph)["nodes"])
    assert any(n["label"].startswith("Nuclei · comparison") for n in graph["nodes"])
    assert json.loads(result.read_text())["name_history"][0]["previous"] == "New K3"
    assert workspace.exists() and result.exists()


@pytest.mark.parametrize(
    "variant,field",
    [
        ({"name": "bad", "cluster_pca_components": -1}, "cluster_pca_components"),
        ({"name": "bad", "classifier_space": "pca"}, "classifier_space"),
        ({"name": "bad", "neighbors": 1}, "neighbors"),
    ],
)
def test_variant_validation_identifies_nested_field(variant, field):
    from morphofeatures.workflow_validation import FieldError

    with pytest.raises(FieldError) as caught:
        variants.validate_variants({"analysis_variants": [variant]})
    assert caught.value.fields == (("analysis_variants", 0, field),)
