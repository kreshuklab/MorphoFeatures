import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from morphofeatures.analysis.fusion import fuse_embeddings
from morphofeatures.analysis.transfer import (
    coral_transform,
    load_reference,
    transfer_labels,
)
from morphofeatures.data.dino_views import (
    PhysicalObjects,
    distributed_indices,
    index_source,
    sample_volume,
    validate_source,
)
from morphofeatures.data.io import export_embeddings, load_embeddings
from morphofeatures.data.volumes import SpatialVolume
from morphofeatures.dino import pool_tokens
from morphofeatures.dino_experiments import experiment_defaults, run_experiment
from morphofeatures.dino_physical import extract_physical


class Backbone(torch.nn.Module):
    def __init__(self, stride=14):
        super().__init__()
        self.calls = 0
        self.stride = stride

    def forward_features(self, images):
        self.calls += 1
        tokens = (
            torch.nn.functional.avg_pool2d(images, self.stride, self.stride)
            .flatten(2)
            .transpose(1, 2)
        )
        return {"x_norm_patchtokens": tokens, "x_norm_clstoken": tokens.mean(1)}


def source_files(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw = np.random.default_rng(2).random((24, 24, 24)).astype(np.float32)
    segmentation = np.zeros(raw.shape, dtype=np.int64)
    for i, center in enumerate(np.ndindex(2, 2, 2)):
        start = np.array(center) * 10 + 3
        segmentation[tuple(slice(a, a + 5) for a in start)] = i + 1
    np.save(tmp_path / "raw.npy", raw)
    np.save(tmp_path / "seg.npy", segmentation)
    return {
        "raw": str(tmp_path / "raw.npy"),
        "segmentation": str(tmp_path / "seg.npy"),
        "unit": "um",
        "spacing_zyx": [0.1, 0.1, 0.1],
    }


def settings(tmp_path):
    path = tmp_path / "weights.pth"
    path.write_bytes(b"test checkpoint identity")
    return {
        "model": "dinov2",
        "variant": "dinov2_vits14",
        "checkpoint": str(path),
        "views": {
            "mode": "both",
            "size": 28,
            "axes": [0],
            "fine_fractions": [0.5],
            "coarse_fractions": [0.5],
            "tile_size_um": 0.3,
            "max_tiles": 3,
            "features": ["cls", "foreground_mean"],
            "batch_size": 16,
        },
        "config": {"device": "cpu", "data": {"volume": source_files(tmp_path / "data")}},
    }


def test_physical_sampling_and_coverage(tmp_path):
    coordinates = np.indices((30, 30, 30)).transpose(1, 2, 3, 0)
    origin = np.array([1.0, -2.0, 4.0])
    points = np.array([[1.6, -1.2, 4.7], [2.0, -1.0, 5.0]])
    for spacing in (0.05, 0.1):
        data = (coordinates * spacing + origin).sum(3).astype(np.float32)
        sampled = sample_volume(
            SpatialVolume(data, "zyx"), points, np.full(3, spacing), origin, budget=100
        )
        np.testing.assert_allclose(sampled, points.sum(1), atol=1e-6)
    s = settings(tmp_path)
    source = validate_source(s["config"]["data"]["volume"])
    objects = index_source(source, tmp_path / "index")
    with PhysicalObjects(source, objects, s["views"]) as reader:
        obj = reader.prepare(objects.iloc[0].to_dict())
        bounded = list(reader.planes(obj, "fixed_tiles"))
        reader.views["exhaustive"] = True
        exhaustive = list(reader.planes(obj, "fixed_tiles"))
    assert bounded[0][2]["selected_tiles"] == 3
    assert len(exhaustive) > len(bounded)
    assert all(v[1].any() for v in exhaustive)
    centers = np.arange(9)[:, None] * np.ones((1, 3))
    assert set(distributed_indices(centers, 3)) == {0, 4, 8}


def test_tile_placement_centers_without_losing_peripheral_tiles():
    from morphofeatures.data.dino_views import eligible_tiles, tile_grid_origin

    spacing = np.array([0.2, 0.1, 0.1])
    points = np.array(
        [[z, y, x] for z in range(5) for y in range(3) for x in range(3)] + [[5, 3, 3]]
    ) * spacing + [2, -4, 1]
    lower, upper = points.min(0) - spacing / 2, points.max(0) + spacing / 2
    width = np.array([0.3, 0.3, 0.3])
    for anchor in ("bbox_min", "bbox_center", "centroid"):
        origin, center = tile_grid_origin(lower, upper, width, points, anchor)
        cells = eligible_tiles(points, spacing, origin, upper, width, 0)
        assert np.all(origin <= lower + 1e-9)
        # Every foreground voxel center belongs to an eligible tile.
        expected = np.floor((points - origin) / width).astype(int)
        assert set(map(tuple, expected)) <= set(map(tuple, cells))
        if anchor == "bbox_min":
            np.testing.assert_array_equal(origin, lower)
        else:
            expected_center = points.mean(0) if anchor == "centroid" else (lower + upper) / 2
            np.testing.assert_allclose(center, expected_center)
            indices = (center - origin) / width - 0.5
            np.testing.assert_allclose(indices, indices.round(), atol=1e-10)
            centers = origin + (cells + 0.5) * width
            chosen = distributed_indices(centers, 1, center=center)
            assert chosen[0] == np.argmin(np.square(centers - center).sum(1))


@pytest.mark.parametrize("origin", [0, 1, 10, 100])
def test_foreground_threshold_includes_full_tiles_after_translation(origin):
    from morphofeatures.data.dino_views import eligible_tiles

    spacing = np.full(3, 0.1)
    points = np.indices((2, 2, 2)).reshape(3, -1).T * spacing + origin
    lower, upper = points.min(0) - spacing / 2, points.max(0) + spacing / 2
    cells = eligible_tiles(points, spacing, lower, upper, np.full(3, 0.2), 1.0)
    np.testing.assert_array_equal(cells, [[0, 0, 0]])
    # Tolerance must not admit a tile with a genuinely missing foreground voxel.
    assert not len(eligible_tiles(points[:-1], spacing, lower, upper, np.full(3, 0.2), 1.0))


def test_volume_and_image_foreground_cutoffs_are_independent(tmp_path):
    s = settings(tmp_path)
    s["views"].update(
        tile_size_um=1.0, min_foreground_fraction=0.1, min_slice_foreground_fraction=0.0
    )
    source = validate_source(s["config"]["data"]["volume"])
    objects = index_source(source, tmp_path / "index")
    with PhysicalObjects(source, objects, s["views"]) as reader:
        obj = reader.prepare(objects.iloc[0].to_dict())
        fine = list(reader.planes(obj, "fixed_tiles"))
        assert len(fine) == 1  # 0.5³ / 1³ = 12.5% occupied.
        assert fine[0][1].mean() < 0.5
        reader.views["min_foreground_fraction"] = 0.25
        assert list(reader.planes(obj, "fixed_tiles")) == []
        assert "3D nucleus-content cutoff" in reader.excluded_views[-1]["reason"]
        assert reader.sampling_summary["eligible_tiles"] == 0
        reader.views.update(min_foreground_fraction=0.1, min_slice_foreground_fraction=0.5)
        assert list(reader.planes(obj, "fixed_tiles")) == []
        assert "2D cutoff" in reader.excluded_views[-1]["reason"]
        assert list(reader.planes(obj, "whole_label"))  # The fine cutoff never filters coarse.


def test_invalid_physical_cutoffs_report_the_setting():
    from morphofeatures.dino_physical import validate_physical_settings
    from morphofeatures.workflow_validation import FieldError

    for key, value in (
        ("min_foreground_fraction", 25),
        ("min_slice_foreground_fraction", float("nan")),
        ("tile_anchor", "unknown"),
    ):
        with pytest.raises(FieldError) as caught:
            validate_physical_settings({"views": {"mode": "both", key: value}}, check_source=False)
        assert caught.value.fields == (("views", key),)


def test_transfer_preflight_separates_dino_preparation_from_recipe(tmp_path, monkeypatch):
    from copy import deepcopy

    from morphofeatures.analysis.transfer import validate_transfer_inputs
    from morphofeatures.workflow_validation import FieldError

    reference, target, labels = transfer_data(tmp_path)
    inputs = {
        "reference": {"name": "Platy", "embedding": str(reference), "annotations": str(labels)},
        "targets": [{"name": "Parhyale", "embedding": str(target)}],
    }
    left = json.loads(reference.with_suffix(".metadata.json").read_text())
    left["settings"].update(
        variant="dinov2_vitb14_reg",
        config={
            "data": {"source": "n5_masked_patches", "group_size": 200, "patch_shape_zyx": [32] * 3}
        },
    )
    right = deepcopy(left)
    right["settings"]["config"]["data"] = {"source": "masked_crops"}
    reference.with_suffix(".metadata.json").write_text(json.dumps(left))
    target.with_suffix(".metadata.json").write_text(json.dumps(right))
    result = validate_transfer_inputs(inputs)[0]
    assert result["verified"] and result["compatible"]
    assert any(r["parameter"] == "input layout" for r in result["preparation_differences"])
    right["settings"]["views"] = {"feature": "foreground_mean", "size": 448}
    target.with_suffix(".metadata.json").write_text(json.dumps(right))
    monkeypatch.setattr(
        "morphofeatures.analysis.transfer.fit_reference",
        lambda *args: pytest.fail("must check compatibility before training"),
    )
    with pytest.raises(FieldError) as caught:
        transfer_labels({**inputs, "assume_same_model": True}, tmp_path / "blocked")
    assert "views.feature" in str(caught.value) and "views.size" in str(caught.value)
    assert "foreground_mean" in str(caught.value) and "448" in str(caught.value)
    assert ("targets", 0, "embedding") in caught.value.fields
    assert not (tmp_path / "blocked").exists()


@pytest.mark.parametrize("stride", [14, 16])
def test_foreground_token_pooling(stride):
    mask = torch.zeros((1, stride * 2, stride * 2))
    mask[:, :stride, :stride] = 1
    mask[:, :stride, stride:] = 0.5
    result = {
        "x_norm_clstoken": torch.tensor([[99.0]]),
        "x_norm_patchtokens": torch.tensor([[[2.0], [8.0], [100.0], [200.0]]]),
    }
    assert pool_tokens(result, mask, "foreground_mean", stride).item() == pytest.approx(4.0)
    assert pool_tokens(result, mask, "cls", stride).item() == 99
    with pytest.raises(ValueError, match="empty"):
        pool_tokens(result, mask * 0, "foreground_mean", stride)


@pytest.mark.parametrize("family,stride", [("dinov2", 14), ("dinov3", 16)])
def test_multi_feature_extraction_fusion_cache(tmp_path, family, stride):
    s, model = settings(tmp_path), Backbone(stride)
    s.update(model=family, variant="dinov2_vits14" if family == "dinov2" else "dinov3_vits16")
    s["views"]["size"] = stride * 2
    paths = extract_physical(s, tmp_path / "extract", backbone=model)
    calls = model.calls
    assert set(paths) == {
        f"{mode}.{feature}"
        for mode in ("fixed_tiles", "whole_label")
        for feature in ("cls", "foreground_mean")
    }
    assert len(load_embeddings(paths["fixed_tiles.cls"]).label_ids) == 8
    extract_physical(s, tmp_path / "extract", backbone=model)
    assert model.calls == calls
    fused = fuse_embeddings(
        {
            "embeddings": {
                "fine": str(paths["fixed_tiles.foreground_mean"]),
                "coarse": str(paths["whole_label.cls"]),
            }
        },
        tmp_path / "fusion",
    )
    values = load_embeddings(fused).features
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(np.square(values[:, :3]).sum(1), 0.5, atol=1e-6)


@pytest.mark.parametrize("workers", [0, 2])
def test_parallel_views_and_resumption_preserve_objects(tmp_path, workers):
    from copy import deepcopy

    s = settings(tmp_path)
    s["runtime"] = {"prepare_workers": workers, "prefetch_mb": 1, "decoded_cache_mb": 1}
    s["cache"] = str(tmp_path / "cache")
    s["views"]["batch_size"] = 5

    class Interrupted(Backbone):
        def forward_features(self, images):
            if self.calls == 3:
                raise RuntimeError("deliberate interruption")
            return super().forward_features(images)

    with pytest.raises(RuntimeError, match="deliberate interruption"):
        extract_physical(s, tmp_path / "interrupted", backbone=Interrupted())
    cache = next((tmp_path / "cache").glob("views-*"))
    committed = list((cache / "object-checkpoints").iterdir())
    assert committed
    timestamps = {p.name: (p / "vectors.npz").stat().st_mtime_ns for p in committed}
    # A crash after writing uncommitted manifest bytes must not duplicate views.
    with (cache / "views.jsonl").open("ab") as stream:
        stream.write(b"uncommitted tail\n")
    resumed_model = Backbone()
    resumed = extract_physical(s, tmp_path / "resumed", backbone=resumed_model)
    assert all((cache / "object-checkpoints" / name / "vectors.npz").stat().st_mtime_ns == stamp for name, stamp in timestamps.items())
    performance = json.loads((cache / "performance.json").read_text())
    assert performance["resumed_objects"] == len(committed)
    assert performance["completed_objects"] == 8
    assert performance["reused_object_index"]
    if workers:
        assert performance["worker_peak_rss_mb"] > 0
    assert b"uncommitted" not in (cache / "views.jsonl").read_bytes()
    baseline = deepcopy(s)
    baseline["runtime"] = {"prepare_workers": 0, "decoded_cache_mb": 0, "reuse_object_mask": False, "group_normalization_reads": False}
    baseline["cache"] = str(tmp_path / "baseline-cache")
    baseline["views"]["batch_size"] = 1
    baseline_model = Backbone()
    expected = extract_physical(baseline, tmp_path / "baseline", backbone=baseline_model)
    for name in expected:
        left, right = load_embeddings(resumed[name]), load_embeddings(expected[name])
        np.testing.assert_array_equal(left.label_ids, right.label_ids)
        np.testing.assert_allclose(left.features, right.features, rtol=1e-6, atol=1e-6)
    assert resumed_model.calls < baseline_model.calls
    assert sum(1 for _ in (cache / "views.jsonl").open()) == sum(1 for _ in next((tmp_path / "baseline-cache").glob("views-*/views.jsonl")).open())


def test_prepared_views_reused_across_checkpoints(tmp_path, monkeypatch):
    s = settings(tmp_path)
    s["runtime"] = {"prepared_cache_mb": 4, "scratch_directory": str(tmp_path / "scratch")}
    s["cache"] = str(tmp_path / "embeddings")
    first = extract_physical(s, tmp_path / "one", backbone=Backbone())
    s["checkpoint"] = str(tmp_path / "other.pth")
    Path(s["checkpoint"]).write_bytes(b"different checkpoint")
    monkeypatch.setattr(PhysicalObjects, "prepare", lambda *_: pytest.fail("prepared pixels should be reused"))
    second = extract_physical(s, tmp_path / "two", backbone=Backbone())
    np.testing.assert_array_equal(load_embeddings(first["fixed_tiles.cls"]).features, load_embeddings(second["fixed_tiles.cls"]).features)


def test_preparation_memory_limit_never_silently_excludes_objects(tmp_path):
    s = settings(tmp_path)
    s["runtime"] = {"memory_mode": "manual", "auto_adjust_memory": False, "prepare_workers": 2, "prepare_memory_mb": 1}
    with pytest.raises(ValueError, match="RAM budget"):
        extract_physical(s, tmp_path / "fail", backbone=Backbone())


def test_memory_correction_reaches_workers_and_is_recorded(tmp_path):
    from morphofeatures.metrics import MetricWriter, read_metric_events

    s = settings(tmp_path)
    s["runtime"] = {
        "memory_mode": "manual", "auto_adjust_memory": True,
        "prepare_workers": 2, "prepare_memory_mb": 1,
        "prefetch_mb": 10000, "decoded_cache_mb": 10000,
    }
    writer = MetricWriter(tmp_path / "metrics.jsonl")
    paths = extract_physical(s, tmp_path / "corrected", backbone=Backbone(), writer=writer)
    table = load_embeddings(paths["fixed_tiles.cls"])
    np.testing.assert_array_equal(table.label_ids, np.arange(1, 9))
    assert np.isfinite(table.features).all()
    root = paths["fixed_tiles.cls"].parent
    plan = json.loads((root / "memory_plan.json").read_text())
    performance = json.loads((root / "performance.json").read_text())
    assert plan["requested"]["prepare_memory_mb"] == 1
    assert plan["runtime"]["prepare_memory_mb"] > 1
    assert performance["runtime"]["decoded_cache_mb"] < 10000
    assert performance["completed_objects"] == 8
    assert performance["worker_peak_rss_mb"] > 0
    assert any(e["event"] == "dino_memory_adjustment" for e in read_metric_events(writer.path))
    assert s["runtime"]["prepare_memory_mb"] == 1


def test_benchmark_measures_cold_warm_and_feature_drift(tmp_path):
    from morphofeatures.dino_benchmark import run_benchmark

    s = settings(tmp_path)
    result = run_benchmark(s, tmp_path / "benchmark", workers=[0], batch_sizes=[8], max_objects=3, backbone=Backbone())
    report = json.loads(result.read_text())
    assert len(report["trials"]) == 4
    assert {trial["cache"] for trial in report["trials"]} == {"cold", "warm"}
    assert all("error" not in trial for trial in report["trials"])
    for trial in report["trials"]:
        assert trial["performance"]["images"] > 0
        for values in trial["feature_drift"].values():
            assert values["objects"] == 3
            assert values["max_absolute_difference"] < 1e-6
            assert values["neighbor_overlap"] == 1
            assert values["nearest_neighbor_agreement"] == 1


def test_benchmark_neighbor_overlap_handles_duplicate_vectors(tmp_path):
    from morphofeatures.data.io import export_embeddings
    from morphofeatures.dino_benchmark import feature_drift

    path = export_embeddings(tmp_path / "duplicates.npz", np.arange(9), np.ones((9, 4)))
    result = feature_drift(path, path, k=3)
    assert result["neighbor_k"] == 3
    assert result["neighbor_overlap"] == 1
    assert result["nearest_neighbor_agreement"] == 1


def test_precision_fallback_and_tuning_identity(tmp_path):
    from copy import deepcopy

    from morphofeatures.dino_runtime import identity_settings, numerical_settings
    from morphofeatures.representations import extraction_identity

    s = settings(tmp_path)
    s["runtime"] = {"precision": "bfloat16"}
    numerical = numerical_settings(s, torch.device("cpu"))
    assert numerical["precision"] == "float32" and numerical["fallback"]
    before = extraction_identity(identity_settings(s, numerical))[0]
    adjusted = deepcopy(s)
    adjusted["runtime"].update(prepare_workers=2, decoded_cache_mb=64, scratch_directory="/tmp/another-cache")
    adjusted["views"]["batch_size"] = 32
    assert before == extraction_identity(identity_settings(adjusted, numerical))[0]
    assert before != extraction_identity(identity_settings(s, {**numerical, "precision": "bfloat16"}))[0]


def test_gpu_memory_error_identifies_the_batch_control(tmp_path):
    from morphofeatures.workflow_validation import FieldError

    class FullGPU(Backbone):
        def forward_features(self, images):
            raise torch.cuda.OutOfMemoryError("simulated GPU allocation failure")

    with pytest.raises(FieldError, match="Images per model batch") as caught:
        extract_physical(settings(tmp_path), tmp_path / "failed", backbone=FullGPU())
    assert ("views", "batch_size") in caught.value.fields


def test_journal_rejects_incompatible_or_corrupt_commits(tmp_path):
    from morphofeatures.dino_runtime import ObjectJournal, extraction_lock

    root = tmp_path / "checkpoint"
    with extraction_lock(root):
        with pytest.raises(RuntimeError, match="Another worker"):
            with extraction_lock(root):
                pytest.fail("must not acquire a live lock")
        journal = ObjectJournal(root, "identity")
        pending = journal.start(2**60 + 1)
        (pending / "views.jsonl").write_text('{"label_id": 1152921504606846977}\n')
        journal.commit(2**60 + 1, {}, {"fixed_tiles.cls": np.ones(3, dtype=np.float32)})
        journal.close()
        with pytest.raises(ValueError, match="signature differs"):
            ObjectJournal(root, "another-recipe")
        path = root / "views.jsonl"
        path.write_bytes(path.read_bytes().replace(b"label_id", b"label_ix"))
        with pytest.raises(ValueError, match="Corrupt committed view manifest"):
            ObjectJournal(root, "identity")


def test_journal_resumes_empty_exclusions_in_any_directory_order(tmp_path, monkeypatch):
    from pathlib import Path

    from morphofeatures.dino_runtime import ObjectJournal

    root = tmp_path / "checkpoint"
    journal = ObjectJournal(root, "identity")
    (journal.start(2) / "views.jsonl").write_text("")
    journal.commit(2, {"excluded": [{"reason": "No eligible views"}]}, {})
    (journal.start(1) / "views.jsonl").write_text('{"label_id": 1}\n')
    journal.commit(1, {}, {"fixed_tiles.cls": np.ones(3, dtype=np.float32)})
    journal.close()
    # Force the nonempty commit to appear first, as a filesystem is free to do.
    iterdir = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda path: iter(sorted(iterdir(path))))
    restored = ObjectJournal(root, "identity")
    assert restored.resumed == 2
    assert restored.records[2]["excluded"][0]["reason"] == "No eligible views"
    restored.close()


def test_fusion_exact_ids_missing_and_namespace(tmp_path):
    large = 2**60 + 9
    a = export_embeddings(tmp_path / "a.npz", [large, 2], [[3, 4], [2, 1]])
    b = export_embeddings(tmp_path / "b.npz", [7, large], [[4, 3], [0, 2]])
    result = fuse_embeddings(
        {"embeddings": {"a": a, "b": b}, "assume_same_dataset": True}, tmp_path / "fused"
    )
    assert load_embeddings(result).label_ids.tolist() == [large]
    a.with_suffix(".metadata.json").write_text(json.dumps({"dataset_identity": "A"}))
    b.with_suffix(".metadata.json").write_text(json.dumps({"dataset_identity": "B"}))
    with pytest.raises(ValueError, match="different datasets"):
        fuse_embeddings({"embeddings": {"a": a, "b": b}}, tmp_path / "bad")


def test_fusion_cross_source_declaration_preserves_values_and_provenance(tmp_path):
    from morphofeatures.workflow_validation import FieldError
    from morphofeatures.workspace_jobs import resolve_job

    high = 2**60 + 9
    a = export_embeddings(tmp_path / "old.npz", [high, 2, 8], [[3, 4], [2, 1], [8, 1]])
    b = export_embeddings(tmp_path / "new.npz", [2, high, 7], [[9, 3], [0, 2], [1, 1]])
    a.with_suffix(".metadata.json").write_text(json.dumps({"dataset_identity": "old-container"}))
    b.with_suffix(".metadata.json").write_text(json.dumps({"dataset_identity": "new-volume"}))
    settings = {"action": "fuse", "embeddings": {"old fine": str(a), "coarse": str(b)},
                "assume_same_dataset": True}
    # The existing missing-provenance checkbox must not bypass conflicting records.
    with pytest.raises(FieldError, match="source records"):
        fuse_embeddings(settings, tmp_path / "blocked")
    declaration = {"confirmed": True, "dataset": "specimen A", "id_namespace": "cell",
                   "reason": "The same verified cell-to-nucleus table was used by both exports."}
    settings["identity_override"] = declaration
    resolve_job({"stages": [settings]}, tmp_path)
    identities = []
    for mode in ("raw", "balanced"):
        path = fuse_embeddings({**settings, "combination": mode}, tmp_path / mode)
        values = load_embeddings(path)
        assert values.label_ids.tolist() == [2, high]
        if mode == "raw":
            np.testing.assert_array_equal(values.features, [[2, 1, 9, 3], [3, 4, 0, 2]])
        report = json.loads(path.with_suffix(".metadata.json").read_text())
        identities.append(report["dataset_identity"])
        assert report["identity_check"]["declaration"] == declaration
        assert report["identity_check"]["source_namespaces"] == {"old fine": "old-container", "coarse": "new-volume"}
        assert pd.read_csv(path.parent / "excluded.tsv", sep="\t").label_id.tolist() == [7, 8]
    assert identities[0] == identities[1] and identities[0].startswith("declared:")
    assert json.loads(a.with_suffix(".metadata.json").read_text())["dataset_identity"] == "old-container"
    for missing in ("dataset", "reason", "id_namespace"):
        invalid = {**declaration, missing: ""}
        with pytest.raises(FieldError) as caught:
            fuse_embeddings({**settings, "identity_override": invalid}, tmp_path / "bad")
        assert ("identity_override", missing) in caught.value.fields


def transfer_data(tmp_path):
    rng = np.random.default_rng(6)
    features = rng.normal(scale=0.08, size=(24, 4)) + np.repeat(
        [[1, 0, 0, 0.2], [0, 1, 0.2, 0]], 12, axis=0
    )
    source = export_embeddings(tmp_path / "source.npz", np.arange(1, 25), features)
    target = export_embeddings(tmp_path / "target.npz", np.arange(1, 25), features + 0.04)
    labels = tmp_path / "labels.tsv"
    pd.DataFrame({"label_id": np.arange(1, 25), "label": ["a"] * 12 + ["b"] * 12}).to_csv(
        labels, sep="\t", index=False
    )
    for p in (source, target):
        p.with_suffix(".metadata.json").write_text(
            json.dumps(
                {
                    "model": "dinov2",
                    "sources": {"checkpoint": {"sha256": "same"}},
                    "settings": {"model": "dinov2"},
                }
            )
        )
    return source, target, labels


def test_deep_coral_loss_and_adapter_control(tmp_path):
    torch = pytest.importorskip("torch")
    from morphofeatures.analysis.transfer_adapter import NumericAdapter, coral_loss, fit_adapter

    a = torch.tensor([[1.0, 2.0], [3.0, 1.0], [0.0, 4.0]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor(
        [[2.0, 1.0], [0.0, 2.0], [4.0, 3.0], [1.0, 1.0]], dtype=torch.float64, requires_grad=True
    )
    expected = np.square(np.cov(a.detach().numpy().T) - np.cov(b.detach().numpy().T)).sum() / 16
    assert coral_loss(a, b).item() == pytest.approx(expected)
    assert torch.autograd.gradcheck(coral_loss, (a, b))
    assert coral_loss(a, a).item() == 0
    with pytest.raises(ValueError, match="two objects"):
        coral_loss(a[:1], b)
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20, 4)).astype(np.float32)
    y = np.array(["a", "b"] * 10)
    options = {"epochs": 3, "batch_size": 8, "hidden_dim": 8, "latent_dim": 3, "coral_weight": 0.0}
    baseline, _ = fit_adapter(x, y, x, x + 0.1, options, seed=7, with_coral=False)
    control, _ = fit_adapter(x, y, x, x * 3 + 0.2, options, seed=7, with_coral=True)
    for key in baseline.arrays:
        np.testing.assert_array_equal(baseline.arrays[key], control.arrays[key])
    adapted, history = fit_adapter(
        x, y, x, x * 3 + 0.2, {**options, "coral_weight": 2.0}, seed=7, with_coral=True
    )
    assert not np.allclose(adapted.arrays["latent_weight"], baseline.arrays["latent_weight"])
    assert history[-1]["coral_loss"] > 0
    adapted.save(tmp_path, history)
    restored = NumericAdapter.load(tmp_path / "adapter.json")
    np.testing.assert_array_equal(restored.predict_proba(x), adapted.predict_proba(x))


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 1),
        ("latent_dim", 0),
        ("epochs", 2.5),
        ("learning_rate", float("nan")),
        ("coral_weight", -1),
        ("device", "invalid"),
    ],
)
def test_adapter_validation_points_to_setting(field, value):
    from morphofeatures.analysis.transfer import validate_transfer_settings
    from morphofeatures.workflow_validation import FieldError

    with pytest.raises(FieldError) as error:
        validate_transfer_settings({"conditions": ["deep_coral"], "adapter": {field: value}})
    assert error.value.fields == (("adapter", field),)


def test_saved_reference_rejection_quantile_can_change_without_refitting(tmp_path):
    from morphofeatures.analysis.transfer import _reference_thresholds, fit_reference

    source, _, labels = transfer_data(tmp_path)
    info, _, _ = fit_reference(
        {"embedding": str(source), "annotations": str(labels)},
        {"folds": 2, "distance_quantile": 1.0},
        tmp_path / "reference",
    )
    old = dict(info["thresholds"])
    revised = _reference_thresholds(info, tmp_path / "reference", 0.5)
    assert all(revised[label] < old[label] for label in old)
    assert info["thresholds"] == old
    assert _reference_thresholds(info, tmp_path / "reference", 1.0) == old


def test_five_transfer_conditions_share_cohorts_folds_and_reserve_target_labels(
    tmp_path, monkeypatch
):
    import yaml

    from morphofeatures.analysis import transfer as transfer_module
    from morphofeatures.analysis import transfer_adapter
    from morphofeatures.analysis.joint import (
        _signature,
        extraction_metadata,
        validate_joint_datasets,
    )
    from morphofeatures.analysis.transfer import CONDITION_NAMES, predict_transfer_condition
    from morphofeatures.workspace_jobs import plan_workspace_job

    source, target, labels = transfer_data(tmp_path)
    annotations = pd.read_csv(labels, sep="\t")
    annotations["specimen"] = np.tile(np.repeat(["first", "second"], 6), 2)
    annotations.to_csv(labels, sep="\t", index=False)
    options = {
        "reference": {
            "name": "reference",
            "embedding": str(source),
            "annotations": str(labels),
            "group_column": "specimen",
        },
        "targets": [{"name": "target", "embedding": str(target)}],
        "conditions": list(CONDITION_NAMES),
        "folds": 2,
        "classifier": "logistic",
        "classifier_space": "pca",
        "classifier_dimensions": 2,
        "classifier_standardize": True,
        "umap": False,
        "clusters": 2,
        "adapter": {"epochs": 3, "batch_size": 8, "hidden_dim": 8, "latent_dim": 3},
    }
    for execution in ("local", "slurm"):
        plan = plan_workspace_job(
            {"stages": [{"action": "transfer", **options}]},
            base=tmp_path,
            output_root=tmp_path / "planned",
            run_id=execution,
            execution=execution,
        )
        stage = yaml.safe_load(plan.resolved_yaml)["stages"][0]
        assert stage["conditions"] == list(CONDITION_NAMES)
        assert stage["adapter"] == options["adapter"]
    assert not (tmp_path / "planned").exists()
    calls = []
    original_fit = transfer_adapter.fit_adapter

    def record_fit(x, y, pool, target_values, *args, **kwargs):
        calls.append((x.copy(), pool.copy()))
        return original_fit(x, y, pool, target_values, *args, **kwargs)

    monkeypatch.setattr(transfer_adapter, "fit_adapter", record_fit)
    first = transfer_labels(options, tmp_path / "first-all")
    assert [len(x) for x, _ in calls] == [12, 12, 24, 12, 12, 24]
    for x, pool in calls:
        np.testing.assert_array_equal(
            x, pool
        )  # No held-out reference objects enter fold alignment.
    options["reference_model"] = str(first.parent / "reference/reference.json")
    options["targets"][0]["annotations"] = str(labels)
    # Plot preprocessing must reach every condition, including DeepCORAL,
    # without changing predictions or the adapted high-dimensional vectors.
    from morphofeatures.analysis import projection

    monkeypatch.setattr(projection, "compute_umap", lambda x, **kwargs: np.asarray(x[:, :2]))
    map_settings = {
        "umap": True, "umap_pca_components": 2, "cluster_pca_components": 2,
        "cluster_method": "kmeans", "clusters": 3, "neighbors": 7,
        "min_dist": 0.35, "umap_epochs": 12, "resolution": 0.02,
        "unlabeled_opacity": 0.4,
    }
    options.update(map_settings)
    options["analysis_variants"] = [
        {"name": "PCA2 clustering", "cluster_pca_components": 2, "umap": False}
    ]
    second = transfer_labels(options, tmp_path / "second-all")
    report = json.loads(second.read_text())
    assert len(report["targets"]) == 5
    assert (second.parent / "condition_comparison.tsv").is_file()
    shared_splits = (first.parent / "reference/splits.tsv").read_bytes()
    for condition in CONDITION_NAMES:
        folder = first.parent / "target-0" / condition
        evaluated = second.parent / "target-0" / condition
        joint_report = json.loads((evaluated / "joint/joint_analysis.json").read_text())
        for key, value in map_settings.items():
            assert joint_report["settings"][key] == value
        assert joint_report["diagnostics"]["cluster_input"]["effective_components"] == 2
        assert joint_report["diagnostics"]["clusters_observed"] == 3
        assert joint_report["diagnostics"]["umap_input"]["effective_components"] == 2
        assert joint_report["diagnostics"]["umap_input"]["fitted_objects"] == 48
        from morphofeatures.analysis.variants import variant_history

        revision = variant_history(evaluated / "joint/joint_analysis.json")[0]
        assert revision["state"] == "completed", revision
        revised = pd.read_csv(Path(revision["result"]).parent / "dataset-1/coordinates.tsv", sep="\t")
        assert revised.predicted_label.tolist() == pd.read_csv(
            evaluated / "joint/dataset-1/coordinates.tsv", sep="\t"
        ).predicted_label.tolist()
        frame = pd.read_csv(folder / "predictions.tsv", sep="\t")
        changed = pd.read_csv(evaluated / "predictions.tsv", sep="\t")
        pd.testing.assert_frame_equal(frame, changed[frame.columns])
        info = json.loads((folder / "condition.json").read_text())
        assert (
            info["adaptation"]
            == json.loads((evaluated / "condition.json").read_text())["adaptation"]
        )
        assert info["source_validation"]["grouping"] == "specimen"
        assert _signature(extraction_metadata(folder / "embeddings.npz")) == _signature(
            extraction_metadata(folder / "reference_embeddings.npz")
        )
        if condition == "uncorrected":
            assert _signature(extraction_metadata(folder / "embeddings.npz")) == _signature(
                extraction_metadata(target)
            )
        assert info["target_cohort"] == list(range(1, 25))
        assert info["reference_cohort"] == list(range(1, 25))
        assert not info["target_labels_used"]
        assert info["target_features_used"] == (
            condition in {"coral", "coral_source_to_target", "deep_coral"}
        )
        np.testing.assert_array_equal(
            load_embeddings(folder / "embeddings.npz").label_ids, range(1, 25)
        )
        reference_vectors = load_embeddings(folder / "reference_embeddings.npz").features
        assert reference_vectors.shape[1] == (3 if condition in {"adapter", "deep_coral"} else 4)
        splits = folder / (
            "model/splits.tsv" if condition == "coral_source_to_target" else "splits.tsv"
        )
        if splits.exists():
            assert splits.read_bytes() == shared_splits
        if condition == "coral_source_to_target":
            normalized_target = load_embeddings(folder / "embeddings.npz").features
            np.testing.assert_allclose(
                reference_vectors.mean(0), normalized_target.mean(0), atol=1e-6
            )
            with np.load(folder / "coral.npz") as alignment:
                np.testing.assert_allclose(alignment["input_mean"], alignment["reference_mean"])
                np.testing.assert_allclose(alignment["output_mean"], alignment["target_mean"])
        assert (evaluated / "confusion_counts.png").is_file()
        assert (folder / "source_validation/confusion_counts.svg").is_file()

    with pytest.raises(ValueError, match="different|incompatible|same|differ"):
        validate_joint_datasets(
            {
                "datasets": [
                    {
                        "name": "a",
                        "embedding": str(
                            first.parent / "target-0/adapter/reference_embeddings.npz"
                        ),
                    },
                    {
                        "name": "b",
                        "embedding": str(first.parent / "target-0/deep_coral/embeddings.npz"),
                    },
                ]
            }
        )

    def forbid(*args, **kwargs):
        raise AssertionError("Inference must not refit any condition")

    monkeypatch.setattr(transfer_module, "fit_reference", forbid)
    monkeypatch.setattr(transfer_adapter, "fit_adapter", forbid)
    for condition in CONDITION_NAMES:
        folder = first.parent / "target-0" / condition
        features, scores = predict_transfer_condition(
            folder / "condition.json", load_embeddings(target).features
        )
        np.testing.assert_allclose(
            features, load_embeddings(folder / "embeddings.npz").features, atol=1e-6
        )
        frame = pd.read_csv(folder / "predictions.tsv", sep="\t")
        np.testing.assert_allclose(scores, frame[["score:a", "score:b"]].to_numpy(), atol=1e-6)


@pytest.mark.parametrize(
    "classifier,space",
    [
        ("logistic", "embeddings"),
        ("mlp", "embeddings"),
        ("knn", "embeddings"),
        ("logistic", "umap"),
    ],
)
def test_transfer_no_target_label_leakage_and_restore(tmp_path, classifier, space):
    source, target, labels = transfer_data(tmp_path)
    options = {
        "reference": {"name": "reference", "embedding": str(source), "annotations": str(labels)},
        "targets": [{"name": "target", "embedding": str(target)}],
        "umap": False,
        "clusters": 2,
        "classifier": classifier,
        "classifier_space": space,
        "classifier_dimensions": 2,
        "classifier_standardize": space == "umap",
        "classifier_umap_neighbors": 4,
        "classifier_umap_epochs": 12,
    }
    first = transfer_labels(options, tmp_path / "first")
    options["reference_model"] = str(first.parent / "reference/reference.json")
    target_labels = tmp_path / "target-labels.tsv"
    pd.DataFrame(
        {"label_id": range(24, 0, -1), "label": ["novel lineage"] * 8 + ["a"] * 8 + ["b"] * 8}
    ).to_csv(target_labels, sep="\t", index=False)
    options["targets"][0]["annotations"] = str(target_labels)
    second = transfer_labels(options, tmp_path / "second")
    for condition in ("uncorrected", "coral"):
        a = pd.read_csv(first.parent / f"target-0/{condition}/predictions.tsv", sep="\t")
        b = pd.read_csv(second.parent / f"target-0/{condition}/predictions.tsv", sep="\t")
        pd.testing.assert_frame_equal(a, b[a.columns])
    report = json.loads(first.read_text())
    assert all("balanced_accuracy" not in v["metrics"] for v in report["targets"])
    assert report["reference_classifier"] == classifier
    assert "accuracy" in report["source_validation"]
    assert (first.parent / "reference/confusion_normalized.svg").is_file()
    assert not (first.parent / "target-0/uncorrected/confusion_counts.png").exists()
    assert (second.parent / "target-0/uncorrected/confusion_counts.png").is_file()
    annotated = json.loads(second.read_text())["targets"][0]["metrics"]
    counts = pd.read_csv(
        second.parent / "target-0/uncorrected/confusion_matrix.tsv", sep="\t", index_col=0
    )
    np.testing.assert_array_equal(counts.to_numpy(), annotated["confusion_matrix"])
    assert counts.to_numpy().sum() == 24
    assert "unknown" in counts.columns
    info, params, model = load_reference(first.parent / "reference/reference.json")
    assert set(model.predict(params["features"])) == set(info["classes"])
    rng = np.random.default_rng(9)
    reference = rng.normal(size=(100, 3))
    shifted = reference * [1, 2, 3] + [2, -4, 6]
    corrected, transform = coral_transform(reference, shifted)
    assert np.linalg.norm(corrected.mean(0) - reference.mean(0)) < 1e-6
    assert np.linalg.norm(np.cov(corrected.T) - np.cov(reference.T)) < np.linalg.norm(
        np.cov(shifted.T) - np.cov(reference.T)
    )
    assert np.isfinite(coral_transform(np.ones((4, 3)), np.ones((5, 3)))[0]).all()


@pytest.mark.parametrize("classes", [2, 3])
def test_mlp_numeric_reference_matches_fitted_network_without_retraining(
    tmp_path, monkeypatch, classes
):
    from sklearn.neural_network import MLPClassifier

    from morphofeatures.analysis.transfer import (
        _estimator,
        fit_reference,
        validate_transfer_settings,
    )

    source, _, labels = transfer_data(tmp_path)
    if classes == 3:
        frame = pd.read_csv(labels, sep="\t")
        frame.loc[:7, "label"] = "c"
        frame.to_csv(labels, sep="\t", index=False)
    settings = {
        "classifier": "mlp",
        "hidden_dimensions": [8, 4],
        "evaluation_pca": 2,
        "classifier_normalization": "standardize",
        "folds": 2,
        "max_iter": 2000,
    }
    validate_transfer_settings(settings)
    info, params, model = fit_reference(
        {"embedding": str(source), "annotations": str(labels)}, settings, tmp_path / "reference"
    )
    expected = _estimator(settings, len(params["labels"])).fit(params["features"], params["labels"])
    np.testing.assert_allclose(
        model.predict_proba(params["all_features"]), expected.predict_proba(params["all_features"])
    )
    assert info["estimator"]["layers"] == 3

    def no_fit(*args, **kwargs):
        raise AssertionError("Loading an MLP reference must not fit a new classifier")

    monkeypatch.setattr(MLPClassifier, "fit", no_fit)
    restored = load_reference(tmp_path / "reference/reference.json")[2]
    np.testing.assert_allclose(
        restored.predict_proba(params["all_features"]), model.predict_proba(params["all_features"])
    )
    np.testing.assert_array_equal(
        restored.predict(params["all_features"]), model.predict(params["all_features"])
    )
    with pytest.raises(ValueError, match="hidden_dimensions"):
        validate_transfer_settings({"classifier": "mlp", "hidden_dimensions": [0]})


@pytest.mark.parametrize(
    "space,classifier",
    [("pca", "linear"), ("pca", "knn"), ("umap", "logistic"), ("umap", "mlp"), ("umap", "knn")],
)
def test_projected_reference_restore_without_refitting(tmp_path, monkeypatch, space, classifier):
    from sklearn.decomposition import PCA
    from umap import UMAP

    from morphofeatures.analysis.transfer import fit_reference

    source, target, labels = transfer_data(tmp_path)
    settings = {
        "classifier": classifier,
        "classifier_space": space,
        "classifier_dimensions": 2,
        "classifier_normalization": "standardize",
        "classifier_standardize": True,
        "classifier_umap_neighbors": 4,
        "classifier_umap_epochs": 12,
        "folds": 2,
    }
    info, params, model = fit_reference(
        {"embedding": str(source), "annotations": str(labels)}, settings, tmp_path / "ref"
    )
    from morphofeatures.analysis.projection import normalize_features

    values = normalize_features(load_embeddings(target).features, "l2").astype(np.float64)
    expected = model.predict_proba(values)

    def forbidden(*args, **kwargs):
        raise AssertionError("Restoring a saved reference must not refit PCA or UMAP")

    monkeypatch.setattr(PCA, "fit", forbidden)
    monkeypatch.setattr(PCA, "fit_transform", forbidden)
    monkeypatch.setattr(UMAP, "fit", forbidden)
    restored = load_reference(tmp_path / "ref/reference.json")[2]
    np.testing.assert_allclose(restored.predict_proba(values), expected, atol=1e-7)
    assert model.steps[-1][1].n_features_in_ == 2
    assert bool(info["umap_preprocessing"]) == (space == "umap")


@pytest.mark.parametrize("pca_components", [None, 3])
def test_experiment_end_to_end(tmp_path, pca_components, monkeypatch):
    from morphofeatures import dino_experiments
    from morphofeatures.analysis import projection

    compare = dino_experiments.compare
    partials = []

    def compare_after_publication(*args, **kwargs):
        partial = json.loads((tmp_path / "experiment" / "experiment.json").read_text())
        assert partial["execution_state"] == "partial"
        assert partial["original_embeddings"] and partial["embeddings"]
        assert all(Path(entry["path"]).is_file() for entry in partial["embeddings"])
        partials.append(partial)
        return compare(*args, **kwargs)

    monkeypatch.setattr(dino_experiments, "compare", compare_after_publication)

    # The numerical UMAP fit is covered by projection tests; exercise the full
    # extraction/fusion/comparison/shared-map handoffs here with a cheap reducer.
    monkeypatch.setattr(projection, "compute_umap", lambda x, **kwargs: np.asarray(x[:, :2]))
    s = settings(tmp_path)
    options = experiment_defaults()
    options.update(
        datasets=[
            {"name": "one", "source": s["config"]["data"]["volume"]},
            {"name": "two", "source": source_files(tmp_path / "other")},
        ],
        backbones=[{k: s[k] for k in ("model", "variant", "checkpoint")}],
        views=s["views"],
        tile_sizes_um=[0.3],
        umap=bool(pca_components),
        umap_pca_components=pca_components,
        clusters=2,
        classify=False,
    )
    if pca_components:
        options["analysis_variants"] = [
            {"name": "PCA clustering", "cluster_pca_components": 2, "umap": False}
        ]
    baseline = export_embeddings(
        tmp_path / "baseline.npz", range(2, 9), np.random.default_rng(8).normal(size=(7, 2))
    )
    options["datasets"][0]["baselines"] = [
        {"name": "Saved baseline", "embedding": str(baseline), "assume_same_dataset": True}
    ]
    options["extra_pairings"] = [
        {"name": "Both foreground", "fine": "foreground_mean", "coarse": "foreground_mean"}
    ]
    labels = tmp_path / "experiment-labels.tsv"
    pd.DataFrame({"label_id": range(1, 9), "label": ["a"] * 4 + ["b"] * 4}).to_csv(
        labels, sep="\t", index=False
    )
    options.update(classify=True, folds=2, classifier_models=["logistic"], export_tsv=True)
    for dataset in options["datasets"]:
        dataset["annotations"] = str(labels)
    path = run_experiment(options, tmp_path / "experiment", backbones=[Backbone()])
    report = json.loads(path.read_text())
    assert report["execution_state"] == "completed" and len(partials) == 2
    assert len(report["embeddings"]) == 13
    assert len(report["joint_analyses"]) == 6
    assert len(report["comparisons"]) == 2
    assert (path.parent / "export.zip").exists()
    assert [c["matched"] for c in report["cohorts"]] == [7, 8]
    assert (path.parent / "metrics_summary.tsv").is_file()
    for entry in report["embeddings"]:
        assert Path(entry["path"]).with_suffix(".tsv").is_file()
    for comparison in report["comparisons"]:
        root = path.parent / comparison["path"]
        entries = json.loads(root.read_text())["representations"]
        if pca_components:
            assert all(
                entry["diagnostics"]["umap_input"]["requested_components"] == pca_components
                for entry in entries.values()
            )
        splits = [
            (root.parent / entry["directory"] / "splits.tsv").read_text()
            for entry in entries.values()
        ]
        assert len(set(splits)) == 1
    if pca_components:
        from morphofeatures.analysis.variants import variant_history

        for entry in report["joint_analyses"]:
            joint_report = json.loads((path.parent / entry["path"]).read_text())
            assert (
                joint_report["diagnostics"]["umap_input"]["requested_components"] == pca_components
            )
            revision = variant_history(path.parent / entry["path"])[0]
            assert revision["state"] == "completed", revision
    from morphofeatures.analysis.object_inspection import input_config_for_result, read_object_crops

    metadata = json.loads((path.parent / report["comparisons"][0]["path"]).read_text())
    config = input_config_for_result(metadata, "Fine + coarse")
    images = list(read_object_crops(config, [2], max_side=8))
    assert images[0][0] == 2 and images[0][1].shape == (8, 8, 8)


def test_views_match_anisotropic_registered_grids(tmp_path):
    images, masks = [], []
    for i, spacing in enumerate(([0.1, 0.1, 0.1], [0.05, 0.1, 0.2])):
        spacing = np.asarray(spacing)
        # Identical physical support with different voxel centers/origins.
        origin = np.array([3.0, -2.0, 7.0]) + spacing / 2
        shape = np.rint(2.4 / spacing).astype(int)
        coordinates = np.indices(shape).transpose(1, 2, 3, 0) * spacing + origin
        lower, upper = np.array([3.4, -1.6, 7.4]), np.array([4.2, -0.8, 8.2])
        labels = np.all((coordinates >= lower) & (coordinates < upper), axis=3).astype(np.int64)
        raw = coordinates.sum(3).astype(np.float32)
        np.save(tmp_path / f"raw-{i}.npy", raw)
        np.save(tmp_path / f"seg-{i}.npy", labels)
        source = validate_source(
            {
                "raw": str(tmp_path / f"raw-{i}.npy"),
                "segmentation": str(tmp_path / f"seg-{i}.npy"),
                "unit": "um",
                "spacing_zyx": spacing.tolist(),
                "origin_zyx": origin.tolist(),
            }
        )
        objects = index_source(source, tmp_path / f"index-{i}")
        with PhysicalObjects(source, objects, {"size": 28}) as reader:
            image, mask, pixel = reader.plane(lower, upper, 0, 3.8, 1)
            images.append(image)
            masks.append(mask)
    np.testing.assert_allclose(images[0], images[1], atol=2e-6)
    np.testing.assert_array_equal(masks[0], masks[1])
    assert pixel == pytest.approx(0.8 / 28)


def test_truncation_bounds_and_budget_are_never_silently_clipped(tmp_path):
    s = settings(tmp_path)
    source = validate_source(s["config"]["data"]["volume"])
    objects = index_source(source, tmp_path / "index")
    with PhysicalObjects(source, objects, s["views"]) as reader:
        row = objects.iloc[0].to_dict()
        reader.source["roi"] = [[3, 0, 0], [24, 24, 24]]
        with pytest.raises(ValueError, match="boundary"):
            reader.prepare(row)
        reader.source.pop("roi")
        reader.source["max_object_voxels"] = 10
        with pytest.raises(ValueError, match="max_object_voxels"):
            reader.prepare(row)
        reader.source["max_object_voxels"] = 1000
        row["bbox_max_z"] -= 2
        with pytest.raises(ValueError, match="refresh the object index"):
            reader.prepare(row)


def test_fusion_excludes_nonfinite_and_zero_rows_individually(tmp_path):
    a = export_embeddings(
        tmp_path / "a.npz", [1, 2, 3, 4], [[3, 4], [np.nan, 1], [0, 0], [np.inf, 2]]
    )
    b = export_embeddings(tmp_path / "b.npz", [4, 3, 2, 1], np.ones((4, 3)))
    with pytest.raises(ValueError, match="NaN"):
        load_embeddings(a)
    fused = fuse_embeddings(
        {"embeddings": {"a": a, "b": b}, "weights": [0.2, 0.8], "assume_same_dataset": True},
        tmp_path / "fused",
    )
    table = load_embeddings(fused)
    assert table.label_ids.tolist() == [1]
    np.testing.assert_allclose(np.square(table.features[0, :2]).sum(), 0.2)
    assert set(pd.read_csv(fused.parent / "excluded.tsv", sep="\t").label_id) == {2, 3, 4}


def test_reference_portability_pca_and_native_prediction_exports(tmp_path):
    source, target, labels = transfer_data(tmp_path)
    options = {
        "reference": {"embedding": str(source), "annotations": str(labels)},
        "targets": [{"name": "Reference", "embedding": str(target)}],
        "conditions": ["uncorrected"],
        "umap": False,
        "clusters": 2,
        "evaluation_pca": 2,
        "export_tsv": True,
    }
    first = transfer_labels(options, tmp_path / "first")
    labels.unlink()
    source.unlink()
    options["reference_model"] = str(first.parent / "reference/reference.json")
    second = transfer_labels(options, tmp_path / "second")
    a = pd.read_csv(first.parent / "target-0/uncorrected/predictions.tsv", sep="\t")
    b = pd.read_csv(second.parent / "target-0/uncorrected/predictions.tsv", sep="\t")
    pd.testing.assert_frame_equal(a, b)
    native = second.parent / "target-0/uncorrected/joint/dataset-1"
    frame = pd.read_csv(native / "coordinates.tsv", sep="\t")
    assert frame.label_id.tolist() == list(range(1, 25))
    assert frame.predicted_label.tolist() == b.predicted_label.tolist()
    assert (native / "projection.png").is_file()
    assert (native.parent.parent / "embeddings.tsv").is_file()
    assert (native.parent.parent.parent / "original_embeddings.npz").is_file()


def test_subvoxel_tiles_cover_foreground_and_enforce_candidate_budget():
    from morphofeatures.data.dino_views import eligible_tiles

    points = np.array([[1.0, 1.0, 1.0]])
    spacing, lower, upper, width = (np.full(3, v) for v in (1.0, 0.5, 1.5, 0.25))
    tiles = eligible_tiles(points, spacing, lower, upper, width, 0.01)
    assert len(tiles) == 64  # Every quarter-voxel tile has foreground, not just the center tile.
    with pytest.raises(ValueError, match="max_candidate_tiles"):
        eligible_tiles(points, spacing, lower, upper, width, 0.01, budget=63)


def test_fusion_worker_named_handoff_and_slurm_plan(tmp_path):
    import yaml

    from morphofeatures.workspace_jobs import plan_workspace_job, resolve_job, run_job

    rng = np.random.default_rng(3)
    for name in ("fine", "coarse"):
        path = export_embeddings(tmp_path / (name + ".npz"), [1, 2, 3, 4], rng.normal(size=(4, 3)))
        path.with_suffix(".metadata.json").write_text(
            json.dumps({"dataset_identity": "same objects"})
        )
    document = {
        "stages": [
            {
                "action": "fuse",
                "name": "Combined",
                "embeddings": {"fine": "fine.npz", "coarse": "coarse.npz"},
                "weights": [0.3, 0.7],
            },
            {
                "action": "analyze",
                "from_embedding": "Combined",
                "normalization": "none",
                "umap": True,
                "umap_pca_components": 3,
                "umap_epochs": 10,
                "clusters": 2,
            },
        ]
    }
    resolved = resolve_job(document, tmp_path)
    assert resolved["stages"][0]["embeddings"]["fine"] == str(tmp_path / "fine.npz")
    for execution in ("local", "slurm"):
        plan = plan_workspace_job(
            document,
            base=tmp_path,
            output_root=tmp_path / "workspace",
            run_id=execution,
            execution=execution,
        )
        assert "workspace-run" in " ".join(plan.record.command)
        assert yaml.safe_load(plan.resolved_yaml)["stages"][1]["umap_pca_components"] == 3
    path = tmp_path / "worker.yaml"
    path.write_text(yaml.safe_dump(document))
    run_job(path)
    result = tmp_path / "results/01-analyze"
    assert load_embeddings(result / "embeddings.npz").features.shape == (4, 6)
    report = json.loads((result / "analysis.json").read_text())
    assert report["diagnostics"]["umap_input"]["effective_components"] == 3
    assert (result / "projection.png").is_file()


def test_fusion_order_survives_sorted_serialization_and_invalidates_review(tmp_path):
    from copy import deepcopy

    from morphofeatures.analysis.fusion import freeze_component_order
    from morphofeatures.workspace_jobs import submission_fingerprint

    fine = export_embeddings(tmp_path / "fine.npz", [1], [[3.0, 4.0]])
    coarse = export_embeddings(tmp_path / "coarse.npz", [1], [[2.0, 0.0, 0.0]])
    stage = {
        "action": "fuse",
        "embeddings": {"fine": str(fine), "coarse": str(coarse)},
        "weights": [0.2, 0.8],
        "assume_same_dataset": True,
    }
    document = {"stages": [deepcopy(stage)]}
    kwargs = dict(output_root=tmp_path, run_id="order", execution="local")
    before = submission_fingerprint(document, **kwargs)
    document["stages"][0]["embeddings"] = dict(reversed(list(stage["embeddings"].items())))
    assert before != submission_fingerprint(document, **kwargs)
    freeze_component_order(stage)
    restored = json.loads(json.dumps(stage, sort_keys=True))
    result = fuse_embeddings(restored, tmp_path / "restored")
    table = load_embeddings(result)
    np.testing.assert_allclose(np.square(table.features[0, :2]).sum(), 0.2)
    assert (
        json.loads(result.with_suffix(".metadata.json").read_text())["components"][0]["name"]
        == "fine"
    )


def test_all_excluded_objects_leave_an_exclusion_table(tmp_path):
    s = settings(tmp_path)
    s["config"]["data"]["volume"]["max_object_voxels"] = 2
    with pytest.raises(ValueError, match="excluded.tsv"):
        extract_physical(s, tmp_path / "failed", backbone=Backbone())
    table = pd.read_csv(next((tmp_path / "failed").rglob("excluded.tsv")), sep="\t")
    assert set(table.label_id) == set(range(1, 9))
    assert table.reason.str.contains("max_object_voxels").all()


def test_raw_and_balanced_all_four_recipes_preserve_original_values(tmp_path):
    from morphofeatures.dino_experiments import FEATURE_BLOCKS, fusion_recipes

    recipes = fusion_recipes({"combination_methods": ["raw", "balanced"], "additional_combinations": ["all_four"]})
    assert len(recipes) == 4
    assert len(fusion_recipes({})) == 1  # Legacy YAML keeps the original single fusion.
    blocks = {}
    vectors = {}
    for i, block in enumerate(FEATURE_BLOCKS):
        vectors[block] = np.array([[i + 1.0, 3 * (i + 1.0)]])
        blocks[block] = str(export_embeddings(tmp_path / (block + ".npz"), [2**60 + 1], vectors[block]))
    for i, recipe in enumerate(recipes):
        path = fuse_embeddings({
            **recipe, "embeddings": {b: blocks[b] for b in recipe["blocks"]}, "assume_same_dataset": True,
        }, tmp_path / str(i))
        table = load_embeddings(path)
        assert table.label_ids.tolist() == [2**60 + 1]
        if recipe["combination"] == "raw":
            expected = np.concatenate([vectors[b] for b in recipe["blocks"]], axis=1)
            np.testing.assert_allclose(table.features, expected)
        else:
            weights = np.asarray(recipe["weights"]) / sum(recipe["weights"])
            for j, weight in enumerate(weights):
                np.testing.assert_allclose(np.square(table.features[:, j*2:j*2+2]).sum(), weight, rtol=1e-6)
    assert {r["combination"] for r in recipes if len(r["blocks"]) == 4} == {"raw", "balanced"}


def test_dino_sweep_reuses_extraction_and_applies_scaling_to_joint_maps(tmp_path, monkeypatch):
    s = settings(tmp_path)
    options = experiment_defaults()
    options.update(
        datasets=[{"name": "a", "source": s["config"]["data"]["volume"]},
                  {"name": "b", "source": source_files(tmp_path / "other")}],
        backbones=[{k: s[k] for k in ("model", "variant", "checkpoint")}],
        views=s["views"], tile_sizes_um=[0.3], clusters=2, umap=False, classify=False,
        combination_methods=["balanced", "raw"], additional_combinations=["all_four"],
        analysis_normalizations=["none", "l2"],
    )
    calls = []
    original = extract_physical

    def extraction(*args, **kwargs):
        calls.append(kwargs["source"])
        return original(*args, **kwargs)

    monkeypatch.setattr("morphofeatures.dino_experiments.extract_physical", extraction)
    monkeypatch.setattr("morphofeatures.representation_analysis.save_figure", lambda *a, **k: None)
    monkeypatch.setattr("morphofeatures.analysis.joint.save_projection_figure", lambda *a, **k: None)
    path = run_experiment(options, tmp_path / "sweep", backbones=[Backbone()])
    report = json.loads(path.read_text())
    assert len(calls) == 2  # Once per dataset, independent of fusion/scaling counts.
    assert len(report["embeddings"]) == 16  # Four original + four fusions, per dataset.
    assert len(report["joint_analyses"]) == 16
    for item in report["joint_analyses"]:
        saved = json.loads((path.parent / item["path"]).read_text())
        assert saved["settings"]["normalization"] == item["normalization"]
    for item in report["comparisons"]:
        saved = json.loads((path.parent / item["path"]).read_text())
        assert len(saved["representations"]) == 16
        assert {v["normalization"] for v in saved["representations"].values()} == {"none", "l2"}
