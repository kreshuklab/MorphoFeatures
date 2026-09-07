import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from morphofeatures.configuration_editor import load_document
from morphofeatures.data.io import export_embeddings, load_embeddings
from morphofeatures.representation_analysis import compare
from morphofeatures.workspace_jobs import run_job, submit_job


def test_site_config_roundtrip_preserves_extensions_and_edits(tmp_path, repo_root):
    values = load_document(repo_root / "configs/sites/mae_platynereis_nuclei_embl.yaml")
    values["training"]["epochs"] = 17
    values["new_extension"] = {"some_future_option": [1, None, "hello"]}
    path = tmp_path / "export.yaml"
    path.write_text(yaml.safe_dump(values))
    loaded = load_document(path)
    assert loaded["training"]["epochs"] == 17
    assert loaded["new_extension"] == values["new_extension"]
    assert loaded["profiles"] == values["profiles"]
    assert loaded["data"] == values["data"]


def test_config_relocation_preserves_all_data_paths(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    values = {
        "data": {"crops": "crops.npy", "label_ids": "ids.npy", "loss_masks": "masks.npy"},
        "training": {"resume_from": "checkpoint.pt"},
    }
    path = source / "config.yaml"
    path.write_text(yaml.safe_dump(values))
    loaded = load_document(path)
    assert loaded["data"] == {k: str(source / v) for k, v in values["data"].items()}
    assert loaded["training"]["resume_from"] == str(source / "checkpoint.pt")


def test_invalid_training_rejected_before_submission(tmp_path):
    with pytest.raises(ValueError, match="epochs"):
        submit_job(
            {"stages": [{"action": "train", "config": {"mae": {}, "training": {"epochs": 0}}}]},
            output_root=tmp_path,
            run_id="invalid",
            execution="dry-run",
        )
    assert not (tmp_path / "experiments").exists()


def test_large_object_ids_roundtrip(tmp_path):
    ids = np.array([2**60 + 1, 2**60 + 2], dtype=np.int64)
    path = export_embeddings(
        tmp_path / "embedding.npz", ids, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    )
    table = load_embeddings(path)
    np.testing.assert_array_equal(ids, table.label_ids)
    assert table.features.dtype == np.float32
    text_path = export_embeddings(tmp_path / "embedding.tsv", ids, table.features)
    np.testing.assert_array_equal(load_embeddings(text_path).label_ids, ids)
    with pytest.raises(ValueError, match="cannot preserve"):
        export_embeddings(tmp_path / "unsafe.npy", ids, table.features)


def test_train_extract_analyze_pipeline(tmp_path, repo_root, monkeypatch):
    pytest.importorskip("torch").set_num_threads(1)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    values = load_document(repo_root / "configs/smoke.yaml")
    crops = np.random.default_rng(7).normal(size=(8, 8, 8, 8)).astype(np.float32)
    ids = np.array([3, 9, 14, 25, 80, 101, 150, 901])
    np.save(tmp_path / "crops.npy", crops)
    np.save(tmp_path / "ids.npy", ids)
    values["data"] = {"crops": str(tmp_path / "crops.npy"), "label_ids": str(tmp_path / "ids.npy")}
    record = submit_job(
        {
            "stages": [
                {"action": "train", "config": values},
                {"action": "extract", "model": "mae", "from_training": True},
                {"action": "analyze", "from_extraction": True, "umap": False, "clusters": 2},
            ]
        },
        output_root=tmp_path,
        run_id="pipeline",
        execution="dry-run",
    )
    status = json.loads(run_job(record.config_snapshot).read_text())
    assert status["state"] == "completed"
    table = load_embeddings(status["stages"][1]["result"])
    np.testing.assert_array_equal(ids, table.label_ids)
    result = Path(status["stages"][2]["result"])
    np.testing.assert_array_equal(
        pd.read_csv(result.parent / "coordinates.tsv", sep="\t").label_id, ids
    )
    assert (result.parent / "projection.svg").is_file()
    import zipfile

    with zipfile.ZipFile(result.parent / "export.zip") as archive:
        assert {"analysis.json", "coordinates.tsv", "embeddings.npz", "projection.svg"} <= set(
            archive.namelist()
        )
    assert "workspace-run" in (Path(record.working_directory) / "job.slurm").read_text()


def test_comparison_identical_ids_and_group_splits(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    ids = np.arange(1, 25)
    rng = np.random.default_rng(7)
    first = export_embeddings(tmp_path / "first.npz", ids, rng.normal(size=(24, 4)))
    second = export_embeddings(tmp_path / "second.npz", ids[::-1], rng.normal(size=(24, 8)))
    annotations = pd.DataFrame(
        {"label_id": ids, "label": np.tile(["a", "b"], 12), "specimen": np.repeat(np.arange(6), 4)}
    )
    annotations.to_csv(tmp_path / "labels.tsv", sep="\t", index=False)
    settings = {
        "annotations": str(tmp_path / "labels.tsv"),
        "group_column": "specimen",
        "folds": 3,
        "clusters": 2,
        "umap": False,
    }
    output = compare({"MF": str(first), "DINO": str(second)}, tmp_path / "comparison", settings)
    report = json.loads(output.read_text())
    assert report["matched_objects"] == 24
    splits = pd.read_csv(output.parent / "splits.tsv", sep="\t")
    for _, fold in splits.groupby("fold"):
        assert not set(fold[fold.role == "train"].group) & set(fold[fold.role == "test"].group)
    assert len(report["representations"]["MF"]["evaluation"]) == 3
    assert len(report["representations"]["DINO"]["evaluation"]) == 3


def test_dino_views_and_object_failures(tmp_path):
    torch = pytest.importorskip("torch")
    from morphofeatures.dino import extract_dino, microscopy_views

    class TinyBackbone(torch.nn.Module):
        def forward_features(self, batch):
            vector = batch.mean((2, 3))
            return {"x_norm_clstoken": vector, "x_norm_patchtokens": vector[:, None]}

    crops = np.ones((3, 8, 8, 8), dtype=np.uint8) * 10
    crops[1] = 0
    np.save(tmp_path / "crops.npy", crops)
    np.save(tmp_path / "ids.npy", [100, 900, 300])
    ids, features, excluded = extract_dino(
        {
            "model": "dinov2",
            "views": {"size": 28},
            "config": {
                "device": "cpu",
                "data": {
                    "crops": str(tmp_path / "crops.npy"),
                    "label_ids": str(tmp_path / "ids.npy"),
                },
            },
        },
        backbone=TinyBackbone(),
    )
    assert ids.tolist() == [100, 300]
    assert features.shape == (2, 3)
    assert excluded[0]["label_id"] == 900
    views = list(microscopy_views(crops[0], crops[0] > 0, {"size": 28}))
    assert len(views) == 9 and views[0].shape == (3, 28, 28)
    gradient = np.arange(512, dtype=np.float32).reshape(8, 8, 8)
    assert next(microscopy_views(gradient, gradient > 0, {"size": 28})).dtype == torch.float32


def test_training_ui_retains_navigation_state(tmp_path, monkeypatch, repo_root):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    script = tmp_path / "app.py"
    script.write_text("""import streamlit as st
from morphofeatures.config import load_config
from morphofeatures.workspace_ui import training_page
page = st.radio("Page", ["Training", "Elsewhere"])
if page == "Training":
    training_page(load_config())
""")
    app = AppTest.from_file(str(script)).run(timeout=30)
    assert not app.exception
    epochs = next(widget for widget in app.number_input if widget.label == "training.epochs")
    epochs.set_value(13).run()
    next(w for w in app.text_input if w.label == "Output root").set_value(
        str(tmp_path / "runs")
    ).run()
    next(w for w in app.button if w.label == "Save dry run").click().run()
    assert not app.error
    first = tmp_path / "runs/experiments/training-001/workspace/job.yaml"
    assert yaml.safe_load(first.read_text())["stages"][0]["config"]["training"]["epochs"] == 13
    app.radio[0].set_value("Elsewhere").run()
    app.radio[0].set_value("Training").run()
    assert not app.exception
    assert (
        next(widget for widget in app.number_input if widget.label == "training.epochs").value == 13
    )
    next(w for w in app.text_input if w.label.startswith("Experiment / run ID")).set_value(
        "variant-002"
    ).run()
    next(w for w in app.button if w.label == "Save dry run").click().run()
    assert not app.error
    second = tmp_path / "runs/experiments/variant-002/workspace/job.yaml"
    assert yaml.safe_load(second.read_text())["stages"][0]["config"]["training"]["epochs"] == 13


def test_preprocessing_alignment_masks_and_training_inputs(tmp_path):
    h5py = pytest.importorskip("h5py")
    from morphofeatures.data.preprocessing import preprocess
    from morphofeatures.mae3d import _load_crops, _load_label_ids, _load_loss_masks

    shape = (16, 20, 24)
    raw = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    labels = np.zeros(shape, dtype=np.uint64)
    labels[1:5, 2:7, 3:8] = 100
    labels[10:13, 11:14, 17:20] = 2**53 + 17
    raw[2, 3, 4] = 0  # Real zero intensity remains inside the instance mask.
    for name, array in (("raw", raw), ("segmentation", labels)):
        with h5py.File(tmp_path / (name + ".h5"), "w") as f:
            f.create_dataset("data", data=array[..., None], chunks=(4, 5, 6, 1))
    settings = {
        "raw": str(tmp_path / "raw.h5"),
        "segmentation": str(tmp_path / "segmentation.h5"),
        "raw_key": "data",
        "segmentation_key": "data",
        "raw_axes": "zyxc",
        "segmentation_axes": "zyxc",
        "spacing_zyx": [2.0, 3.0, 4.0],
        "origin_zyx": [10.0, 20.0, 30.0],
        "unit": "nm",
        "crop_shape": [8, 8, 8],
        "block_shape": [3, 5, 7],
        "normalization": "none",
    }
    path = preprocess(settings, tmp_path / "prepared")
    config = load_document(path)
    crops = _load_crops(config, 42)
    ids = _load_label_ids(config, len(crops))
    masks = _load_loss_masks(config, len(crops), crops.shape[-3:])
    assert ids.tolist() == [100, 2**53 + 17]
    assert crops.shape == (2, 1, 8, 8, 8)
    assert masks[0].sum() == 100
    assert (crops[0].reshape(-1)[masks[0].reshape(-1).astype(bool)] == 0).sum() == 1
    assert np.all(crops.reshape(2, -1)[~masks.reshape(2, -1).astype(bool)] == 0)
    manifest = pd.read_csv(path.parent / "objects.tsv", sep="\t")
    first = manifest.iloc[0]
    assert first.bbox_min_z == 1 and first.bbox_max_z == 5
    assert first.crop_center_coordinate_z == 16
    with pytest.raises(ValueError, match="spacing differ"):
        preprocess({**settings, "raw_spacing_zyx": [1, 1, 1]}, tmp_path / "bad")


def test_preprocessing_boundary_and_missing_ids_are_reported(tmp_path):
    from morphofeatures.data.preprocessing import preprocess

    labels = np.zeros((12, 12, 12), dtype=np.int64)
    labels[:2, :2, :2] = 1
    labels[5:7, 5:7, 5:7] = 2
    np.save(tmp_path / "raw.npy", np.ones_like(labels, dtype=np.uint8))
    np.save(tmp_path / "labels.npy", labels)
    path = preprocess(
        {
            "raw": str(tmp_path / "raw.npy"),
            "segmentation": str(tmp_path / "labels.npy"),
            "spacing_zyx": [1, 1, 1],
            "unit": "voxel",
            "crop_shape": [8, 8, 8],
            "object_ids": [1, 2, 999],
        },
        tmp_path / "out",
    )
    rows = pd.read_csv(path.parent / "objects.tsv", sep="\t").set_index("label_id")
    assert rows.loc[1, "status"] == "skipped"
    assert rows.loc[2, "status"] == "processed"
    assert rows.loc[999, "reason"] == "ID absent from ROI"


def test_mesh_sampling_transform_and_export(tmp_path):
    pytest.importorskip("trimesh")
    from morphofeatures.mesh import Surface, export_surface, load_surface, sample_surface

    z, y, x = np.indices((8, 9, 10))
    raw = (100 * z + 10 * y + x).astype(np.float32)
    np.save(tmp_path / "raw.npy", raw)
    # Vertices in XYZ already use physical units; test anisotropic spacing.
    vertices = np.array([[8.0, 9.0, 4.0], [10.0, 9.0, 4.0], [8.0, 12.0, 5.0], [-1.0, 0.0, 0.0]])
    surface = Surface(vertices, np.array([[0, 1, 2], [0, 2, 3]]), np.array([77] * 4))
    result = sample_surface(
        surface,
        {
            "raw": str(tmp_path / "raw.npy"),
            "spacing_zyx": [2, 3, 4],
            "unit": "nm",
            "block_shape": [2, 2, 2],
            "interpolation": "linear",
        },
    )
    np.testing.assert_allclose(result.intensity[:3], [232.0, 232.5, 292.0])
    assert np.isnan(result.intensity[3])
    path = export_surface(result, tmp_path / "surface.ply")
    restored = load_surface(path)
    np.testing.assert_allclose(restored.vertices, surface.vertices)
    np.testing.assert_array_equal(restored.object_ids, surface.object_ids)
    np.testing.assert_allclose(restored.intensity, result.intensity, equal_nan=True)


def test_local_worker_updates_registry_without_ui(tmp_path, repo_root):
    import time

    from morphofeatures.registry import JobRegistry

    embedding = export_embeddings(
        tmp_path / "input.npz", np.arange(8), np.random.default_rng(42).normal(size=(8, 4))
    )
    record = submit_job(
        {
            "stages": [
                {"action": "analyze", "embedding": str(embedding), "umap": False, "clusters": 2}
            ]
        },
        output_root=tmp_path,
        run_id="local",
        execution="local",
    )
    status = Path(record.working_directory) / "status.json"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if status.exists() and json.loads(status.read_text()).get("state") in {
            "completed",
            "failed",
        }:
            break
        time.sleep(0.1)
    assert json.loads(status.read_text())["state"] == "completed"
    saved = next(r for r in JobRegistry.under_output_root(tmp_path).list() if r.id == record.id)
    assert saved.application_state == "completed" and len(saved.artifacts) == 1


def test_slurm_pipeline_preserves_dependency_and_outputs(tmp_path, monkeypatch):
    from morphofeatures.registry import JobRegistry
    from morphofeatures.slurm import SlurmScheduler

    embedding = export_embeddings(tmp_path / "input.npz", np.arange(4), np.eye(4))
    captured = []

    def submit(self, path, dependency=None):
        captured.append((path, dependency))
        return "12345"

    monkeypatch.setattr(SlurmScheduler, "submit", submit)
    record = submit_job(
        {
            "stages": [{"action": "analyze", "embedding": str(embedding), "umap": False}],
            "slurm": {"partition": "gpu", "gpus": 1, "memory": "16G"},
        },
        output_root=tmp_path,
        run_id="cluster",
        execution="slurm",
        dependency="12344",
    )
    assert captured[0][1] == "12344"
    assert "--gpus=1" in captured[0][0].read_text()
    saved = JobRegistry.under_output_root(tmp_path).list()[0]
    assert saved.slurm_job_id == "12345"
    run_job(record.config_snapshot)
    completed = JobRegistry.under_output_root(tmp_path).list()[0]
    assert (
        completed.application_state == "completed"
        and Path(completed.artifacts[0]).name == "analysis.json"
    )


def test_cached_embeddings_do_not_repeat_extraction(tmp_path, monkeypatch):
    from morphofeatures import mae3d
    from morphofeatures.representations import extract_representation

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint identity fixture")
    calls = []

    def encode(config, checkpoint, output):
        calls.append(output)
        return export_embeddings(output, [3, 7], np.eye(2))

    monkeypatch.setattr(mae3d, "encode_from_config", encode)
    settings = {
        "model": "mae",
        "checkpoint": str(checkpoint),
        "config": {"mae": {}},
        "cache": str(tmp_path / "cache"),
    }
    first = extract_representation(settings, tmp_path / "first")
    second = extract_representation(settings, tmp_path / "second")
    assert first == second and len(calls) == 1
    assert json.loads((tmp_path / "second/result.json").read_text())["cache_hit"]
