"""Joint fitting, exact dataset identities, worker handoffs and guided controls."""

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from morphofeatures.analysis import joint
from morphofeatures.analysis.visualization import interactive_projection, projection_panels
from morphofeatures.data.io import export_embeddings, load_embeddings
from morphofeatures.workflow_validation import FieldError, WorkflowValidationError
from morphofeatures.workspace_jobs import resolve_job, run_job
from morphofeatures.workspace_state import add_dataset_extraction, new_draft, training_defaults


def inputs(tmp_path):
    datasets = []
    for i, name in enumerate(("Specimen A", "Specimen B")):
        path = export_embeddings(
            tmp_path / f"sample-{i}.npz",
            np.arange(6, dtype=np.int64) + 2**60 + 1,
            np.random.default_rng(i).normal(size=(6, 4)) + i * 4,
        )
        path.with_suffix(".metadata.json").write_text(
            json.dumps(
                {
                    "model": "mae",
                    "sources": {
                        "checkpoint": {"sha256": "same-weights", "path": f"/copy-{i}/checkpoint.pt"}
                    },
                    "settings": {"model": "mae", "config": {"data": {"crops": f"sample-{i}.npy"}}},
                }
            )
        )
        annotation = tmp_path / f"labels-{i}.tsv"
        annotation.write_text(f"label_id\tcell_type\n{2**60 + 1}\t{'neuron' if i else 'muscle'}\n")
        datasets.append({"name": name, "embedding": str(path), "annotations": str(annotation)})
    return {
        "action": "joint_analyze",
        "datasets": datasets,
        "umap": False,
        "clusters": 3,
        "seed": 3,
    }


def test_joint_fit_preserves_overlapping_large_ids_and_native_reports(tmp_path):
    settings = inputs(tmp_path)
    with patch.object(joint, "project_embeddings", wraps=joint.project_embeddings) as fit:
        path = joint.joint_analyze(settings, tmp_path / "joint")
    assert fit.call_count == 1
    report = json.loads(path.read_text())
    assert report["model_verification"]["verified"]
    frame = pd.read_csv(path.parent / "coordinates.tsv", sep="\t")
    assert len(frame) == 12 and frame.label_id.is_unique
    assert frame.source_label_id.nunique() == 6
    assert frame.loc[frame.source_label_id == 2**60 + 1, "known_label"].tolist() == [
        "muscle",
        "neuron",
    ]
    features = np.concatenate(
        [load_embeddings(d["embedding"]).sorted().features for d in settings["datasets"]]
    )
    expected = PCA(2, random_state=3).fit_transform(StandardScaler().fit_transform(features))
    np.testing.assert_allclose(frame[["pca_1", "pca_2"]], expected, atol=1e-12)
    assert [v[0] for v in projection_panels(frame)] == ["dataset", "cluster", "known_label"]
    chart = interactive_projection(frame, "pca")
    assert len({trace.legend for trace in chart.data}) == 3
    custom = [row for trace in chart.data for row in trace.customdata]
    assert {row[2] for row in custom} == {"Specimen A", "Specimen B"}
    assert any(row[3] == str(2**60 + 1) for row in custom)
    for name, entry in report["datasets"].items():
        folder = path.parent / entry["directory"]
        native = pd.read_csv(folder / "coordinates.tsv", sep="\t")
        parent = frame[frame.dataset == name]
        np.testing.assert_array_equal(native.label_id, parent.source_label_id)
        np.testing.assert_allclose(
            native[["pca_1", "pca_2", "cluster"]], parent[["pca_1", "pca_2", "cluster"]]
        )
        assert joint.extraction_metadata(folder / "embeddings.npz")["model"] == "mae"
        assert (
            json.loads((folder / "analysis.json").read_text())["shared_projection"]["dataset"]
            == name
        )
    assert (path.parent / "export.zip").is_file()


def test_adapted_spaces_require_matching_fitted_models(tmp_path):
    settings = inputs(tmp_path)
    sidecars = [Path(d["embedding"]).with_suffix(".metadata.json") for d in settings["datasets"]]
    metadata = [json.loads(path.read_text()) for path in sidecars]
    adaptation = {"condition": "deep_coral", "model_id": "fitted-on-reference-and-target"}
    metadata[0]["adaptation"] = adaptation
    sidecars[0].write_text(json.dumps(metadata[0]))
    # Equal dimensions/checkpoints do not make raw and learned coordinates equivalent.
    with pytest.raises(FieldError, match="different adaptation"):
        joint.validate_joint_datasets(settings)
    metadata[1]["adaptation"] = adaptation
    sidecars[1].write_text(json.dumps(metadata[1]))
    assert joint.validate_joint_datasets(settings)["verified"]
    metadata[1]["adaptation"] = {**adaptation, "model_id": "a-different-fitted-adapter"}
    sidecars[1].write_text(json.dumps(metadata[1]))
    with pytest.raises(FieldError, match="different adaptation"):
        joint.validate_joint_datasets({**settings, "assume_same_model": True})


@pytest.mark.parametrize("pca_components", [None, 2, 50])
def test_umap_receives_the_same_pooled_scaling_and_cap_is_reproducible(tmp_path, pca_components):
    settings = {
        **inputs(tmp_path), "umap": True, "max_objects_per_dataset": 4,
        "umap_pca_components": pca_components,
    }
    pytest.importorskip("umap")
    from morphofeatures.analysis import projection

    with patch.object(projection, "compute_umap", wraps=projection.compute_umap) as umap:
        path = joint.joint_analyze(settings, tmp_path / "joint")
    assert umap.call_count == 1
    pooled = load_embeddings(path.parent / "embeddings.npz")
    expected = StandardScaler().fit_transform(pooled.features)
    if pca_components:
        expected = PCA(n_components=min(pca_components, 4), random_state=3).fit_transform(expected)
    np.testing.assert_allclose(umap.call_args.args[0], expected)
    report = json.loads(path.read_text())
    assert report["diagnostics"]["umap_input"]["fitted_objects"] == 8
    child = json.loads((path.parent / "dataset-0/analysis.json").read_text())
    assert child["diagnostics"]["umap_input"] == report["diagnostics"]["umap_input"]
    frame = pd.read_csv(path.parent / "coordinates.tsv", sep="\t")
    assert frame.groupby("dataset").size().tolist() == [4, 4]
    assert np.isfinite(frame[["umap_1", "umap_2"]]).all().all()
    index = pd.read_csv(path.parent / "object_index.tsv", sep="\t")
    assert len(index) == 12 and index.projected.sum() == 8
    with patch.object(joint, "save_projection_figure"):
        again = joint.joint_analyze({**settings, "umap": False}, tmp_path / "again")
    assert (
        load_embeddings(again.parent / "embeddings.npz").label_ids.tolist()
        == pooled.label_ids.tolist()
    )


def test_model_checks_reject_known_mismatches_even_with_confirmation(tmp_path):
    settings = inputs(tmp_path)
    sidecar = Path(settings["datasets"][1]["embedding"]).with_suffix(".metadata.json")
    metadata = json.loads(sidecar.read_text())
    metadata["sources"]["checkpoint"]["sha256"] = "different-weights"
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(FieldError, match="different checkpoint"):
        joint.validate_joint_datasets({**settings, "assume_same_model": True})
    sidecar.unlink()
    with pytest.raises(FieldError, match="provenance is unavailable"):
        joint.validate_joint_datasets(settings)
    assert joint.validate_joint_datasets({**settings, "assume_same_model": True}) == {
        "verified": False,
        "unverified_datasets": ["Specimen B"],
    }
    settings["assume_same_model"] = True
    export_embeddings(settings["datasets"][1]["embedding"], [1, 2, 3], np.ones((3, 7)))
    with pytest.raises(FieldError, match="feature dimension"):
        joint.joint_analyze(settings, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_dino_feature_settings_checked_but_batch_size_can_differ(tmp_path):
    settings = inputs(tmp_path)
    for i, dataset in enumerate(settings["datasets"]):
        path = Path(dataset["embedding"]).with_suffix(".metadata.json")
        metadata = json.loads(path.read_text())
        metadata.update(
            model="dinov2", settings={"variant": "dinov2_vits14", "views": {"batch_size": i + 1}}
        )
        path.write_text(json.dumps(metadata))
    assert joint.validate_joint_datasets(settings)["verified"]
    metadata["settings"]["views"]["feature"] = "patch_mean"
    path.write_text(json.dumps(metadata))
    with pytest.raises(FieldError, match="different views"):
        joint.validate_joint_datasets(settings)


def test_reanalysis_preserves_and_recovers_extraction_provenance(tmp_path):
    from morphofeatures.representation_analysis import analyze
    from morphofeatures.representations import fingerprint_file

    settings = inputs(tmp_path)
    source = Path(settings["datasets"][0]["embedding"])
    original = joint.extraction_metadata(source)
    first = analyze(source, tmp_path / "first", {"umap": False, "clusters": 2})
    second = analyze(
        first.parent / "embeddings.npz", tmp_path / "second", {"umap": False, "clusters": 2}
    )
    assert joint.extraction_metadata(second.parent / "embeddings.npz") == original
    assert (second.parent / "embeddings.metadata.json").is_file()
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    copy = export_embeddings(legacy / "embeddings.npz", [1, 2], [[1, 2], [3, 4]])
    (legacy / "analysis.json").write_text(
        json.dumps({"embedding": copy.name, "source": fingerprint_file(source)})
    )
    assert joint.extraction_metadata(copy) == original
    source.write_bytes(b"changed file")
    assert joint.extraction_metadata(copy) == {}


def test_fusion_compares_component_recipe_not_old_dataset_digest():
    from morphofeatures.analysis.recipe_compatibility import compare_recipes

    component = {
        "model": "dinov2",
        "checkpoint": "same",
        "variant": "dinov2_vitb14_reg",
        "input layout": "grouped patches",
        "patch sampling": {"group_size": 200},
    }
    reference = {
        "model": "fusion",
        "sources": {"checkpoint": {"sha256": "old-dataset-dependent-digest"}},
        "fusion_recipe": {
            "combination": "balanced",
            "blocks": [{"start": 0, "stop": 768, "weight": 0.5, "recipe": component}],
        },
    }
    target = deepcopy(reference)
    target["sources"]["checkpoint"]["sha256"] = "new-dataset-dependent-digest"
    recipe = target["fusion_recipe"]["blocks"][0]["recipe"]
    recipe["input layout"] = "object crops"
    recipe.pop("patch sampling")
    assert compare_recipes(reference, target)["compatible"]
    recipe["checkpoint"] = "other-weights"
    mismatch = compare_recipes(reference, target)
    assert not mismatch["compatible"]
    assert mismatch["differences"][0]["parameter"] == "fusion.blocks[0].recipe.checkpoint"


def test_coarse_recipe_ignores_fine_controls_and_fine_defaults_are_canonical():
    from morphofeatures.analysis.recipe_compatibility import compare_recipes

    reference = {
        "model": "dinov2",
        "sources": {"checkpoint": {"sha256": "same"}},
        "settings": {"views": {"mode": "whole_label", "tile_size_um": 1.0}},
    }
    target = deepcopy(reference)
    target["settings"]["views"].update(
        tile_size_um=2.0, tile_anchor="centroid", min_foreground_fraction=0.75
    )
    assert compare_recipes(reference, target)["compatible"]
    reference["settings"]["views"] = {"mode": "fixed_tiles"}
    target["settings"]["views"] = {
        "mode": "fixed_tiles",
        "tile_anchor": "bbox_min",
        "min_slice_foreground_fraction": 0.0,
    }
    assert compare_recipes(reference, target)["compatible"]
    target["settings"]["views"]["tile_anchor"] = "centroid"
    assert not compare_recipes(reference, target)["compatible"]


def test_dataset_validation_points_to_the_field_and_resolves_relative_paths(tmp_path):
    settings = inputs(tmp_path)
    settings["datasets"][0]["embedding"] = "sample-0.npz"
    settings["datasets"][0]["annotations"] = "labels-0.tsv"
    resolved = resolve_job({"stages": [settings]}, tmp_path)["stages"][0]
    assert resolved["datasets"][0]["embedding"] == str(tmp_path / "sample-0.npz")
    assert settings["datasets"][0]["embedding"] == "sample-0.npz"
    settings["datasets"][1]["annotations"] = "missing.tsv"
    with pytest.raises(WorkflowValidationError) as error:
        resolve_job({"stages": [settings]}, tmp_path)
    assert error.value.issue.fields == (("stages", 0, "datasets", 1, "annotations"),)
    settings["datasets"][1] = {"name": "Specimen B", "from_embedding": "missing-stage"}
    with pytest.raises(WorkflowValidationError, match="earlier named extraction"):
        resolve_job({"stages": [settings]}, tmp_path)


def test_worker_embeds_two_datasets_using_one_trained_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    config = training_defaults(demo=True)
    configs = []
    for i in range(2):
        crops = tmp_path / f"crops-{i}.npy"
        ids = tmp_path / f"ids-{i}.npy"
        np.save(crops, np.random.default_rng(i).random((6, 1, 8, 8, 8)).astype("float32"))
        np.save(ids, np.arange(6, dtype=np.int64) + 1)
        configs.append({**deepcopy(config), "data": {"crops": str(crops), "label_ids": str(ids)}})
    document = {
        "stages": [
            {"action": "train", "config": configs[0]},
            {"action": "extract", "model": "mae", "name": "A", "from_training": True},
            {
                "action": "extract",
                "model": "mae",
                "name": "B",
                "from_training": True,
                "config": configs[1],
            },
            {
                "action": "joint_analyze",
                "datasets": [
                    {"name": "A", "from_embedding": "A"},
                    {"name": "B", "from_embedding": "B"},
                ],
                "umap": False,
                "clusters": 3,
            },
        ]
    }
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(document))
    status = json.loads(run_job(path).read_text())
    assert status["state"] == "completed"
    report = json.loads(Path(status["artifacts"][-1]).read_text())
    assert report["model_verification"]["verified"]
    assert [d["objects"] for d in report["datasets"].values()] == [6, 6]
    assert status["artifacts"][1] != status["artifacts"][2]


def test_guided_joint_workflow_checks_datasets_and_copies_model_only(tmp_path):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "morphofeatures/dashboard.py"), default_timeout=30
    )
    app.session_state["workspace_root"] = str(tmp_path)
    app.run()

    def widget(kind, label):
        return next(w for w in getattr(app, kind) if w.label == label)

    widget("selectbox", "Starting point").set_value("joint_embeddings").run()
    widget("button", "Create workflow").click().run()
    assert app.radio(key="step_widget").value == "Analyze"
    widget("button", "Check this page").click().run()
    assert any("Saved embedding file" in e.value for e in app.error)
    assert not app.exception
    widget("button", "Add another dataset").click().run()
    assert len([w for w in app.text_input if w.label == "Dataset name"]) == 3
    widget("button", "Add extraction for this dataset").click().run()
    assert not app.exception
    assert any(w.label == "Model checkpoint" for w in app.text_input)
    assert not (tmp_path / "experiments").exists()
    document = new_draft("joint_embeddings")["document"]
    document = add_dataset_extraction(document, 0, 0)
    document["stages"][0].update(checkpoint="same.pt", model="dinov2", variant="dinov2_vits14")
    document["stages"][0]["config"]["data"]["crops"] = "first.npy"
    other = add_dataset_extraction(document, 1, 1)
    assert other["stages"][1]["checkpoint"] == "same.pt"
    assert other["stages"][1]["model"] == "dinov2"
    assert other["stages"][1]["config"]["data"]["crops"] is None
    assert document["stages"][0]["config"]["data"]["crops"] == "first.npy"


def test_results_route_shared_selections_to_native_dataset_ids(tmp_path):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    settings = inputs(tmp_path)
    # Names that pandas normally interprets as missing must remain dataset identities.
    settings["datasets"][1]["name"] = "NA"
    path = joint.joint_analyze(settings, tmp_path / "joint")
    app = AppTest.from_string(
        "import json\nfrom pathlib import Path\n"
        "from morphofeatures.config import load_config\n"
        "from morphofeatures.analysis_ui import joint_projection_results\n"
        f"path = Path({str(path)!r})\n"
        "joint_projection_results(load_config(), json.loads(path.read_text()), path)\n",
        default_timeout=30,
    )
    # Pooled IDs 1 and 8 refer to separate specimens; the original IDs overlap.
    app.session_state["comparison:" + str(path) + ":Embedding"] = [1, 8]
    app.session_state["picked:" + str(path) + ":Embedding"] = [1, 8]
    app.run()
    assert not app.exception
    child_a = path.parent / "dataset-0/analysis.json"
    child_b = path.parent / "dataset-1/analysis.json"
    assert app.session_state["comparison:" + str(child_a) + ":Embedding"] == [2**60 + 1]
    next(w for w in app.selectbox if w.label == "Dataset to inspect / export").set_value("NA").run()
    assert not app.exception
    assert app.session_state["comparison:" + str(child_b) + ":Embedding"] == [2**60 + 2]
    next(w for w in app.radio if w.label == "Comparison mode").set_value("Chosen objects").run()
    assert next(w for w in app.multiselect if w.label == "Objects to compare").value == [2**60 + 2]
    assert not app.exception
    next(w for w in app.button if w.label == "Prepare projection SVG / PNG").click().run()
    assert not app.exception
    assert any(w.label == "Download projection SVG" for w in app.get("download_button"))
