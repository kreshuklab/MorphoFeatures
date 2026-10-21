"""Visible object policies, bounded preflight, and explicit partial coverage."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from morphofeatures.data.dino_views import PhysicalObjects, index_source, validate_source
from morphofeatures.data.object_eligibility import assess_objects, write_eligibility
from morphofeatures.dino_experiments import run_experiment
from morphofeatures.dino_physical import extract_physical
from morphofeatures.workflow_validation import FieldError, stage_issue


@pytest.fixture
def source(tmp_path):
    raw = np.random.default_rng(5).random((24, 24, 24)).astype(np.float32)
    labels = np.zeros(raw.shape, dtype=np.int64)
    labels[:6, 3:11, 3:11] = 2**60 + 1
    labels[10:16, 3:11, 3:11] = 2**60 + 2
    np.save(tmp_path / "raw.npy", raw)
    np.save(tmp_path / "seg.npy", labels)
    return validate_source(
        {
            "raw": str(tmp_path / "raw.npy"),
            "segmentation": str(tmp_path / "seg.npy"),
            "unit": "um",
            "spacing_zyx": [0.1, 0.1, 0.1],
        }
    )


class Backbone(torch.nn.Module):
    def forward_features(self, images):
        tokens = torch.nn.functional.avg_pool2d(images, 14, 14).flatten(2).transpose(1, 2)
        return {"x_norm_patchtokens": tokens, "x_norm_clstoken": tokens.mean(1)}


def extraction(source, tmp_path):
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"test backbone identity")
    return {
        "model": "dinov2",
        "variant": "dinov2_vits14",
        "checkpoint": str(weights),
        "config": {"device": "cpu", "data": {"volume": source}},
        "views": {
            "mode": "both",
            "size": 28,
            "axes": [0, 1, 2],
            "features": ["cls", "foreground_mean"],
            "tile_size_um": 0.3,
            "max_tiles": 2,
            "coarse_fractions": [0.5],
        },
    }


def test_bounds_check_explains_all_causes_and_required_budget(source, tmp_path):
    objects = index_source(source, tmp_path / "index")
    table, summary = assess_objects({**source, "max_object_voxels": 100}, objects)
    assert summary["reasons"] == {"boundary": 1, "mask_budget": 1}
    assert summary["eligible_objects"] == 0
    assert table.label_id.tolist() == [2**60 + 1, 2**60 + 2]
    assert table.iloc[1].required_voxels == 8 * 10 * 10
    assert "800" in table.iloc[1].reason and "100" in table.iloc[1].reason
    assert table.iloc[0].volume_faces == "Z minimum" and table.iloc[0].roi_faces == ""
    with pytest.raises(FieldError, match="0 of 2") as caught:
        write_eligibility({**source, "max_object_voxels": 100}, objects, tmp_path / "report")
    assert ("max_object_voxels",) in caught.value.fields
    assert ("boundary_policy",) in caught.value.fields
    assert (tmp_path / "report/eligibility.json").is_file()
    assert len(pd.read_csv(tmp_path / "report/excluded.tsv", sep="\t")) == 2


def test_reusing_worker_bounds_preserves_mapped_ids(source, tmp_path):
    mapping = tmp_path / "mapping.tsv"
    pd.DataFrame({"label_id": [101, 102], "segmentation_id": [2**60 + 1, 2**60 + 2]}).to_csv(
        mapping, sep="\t", index=False
    )
    source = {**source, "id_mapping": str(mapping)}
    first = index_source(source, tmp_path / "first")
    reused = index_source(
        {**source, "object_index": str(tmp_path / "first/objects.tsv")}, tmp_path / "reused"
    )
    pd.testing.assert_frame_equal(first.reset_index(drop=True), reused.reset_index(drop=True))
    assert reused.label_id.tolist() == [101, 102]
    assert reused.segmentation_id.tolist() == [2**60 + 1, 2**60 + 2]


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_object_voxels", 0),
        ("max_object_voxels", 12.5),
        ("max_object_voxels", True),
        ("boundary_policy", "clip"),
        ("oversized_policy", "allow"),
        ("on_object_error", "ignore"),
    ],
)
def test_source_policy_validation_is_field_specific(source, key, value):
    with pytest.raises(FieldError) as caught:
        validate_source({**source, key: value})
    assert caught.value.fields == ((key,),)


def test_roi_partial_sampling_never_leaks_label_outside_roi(source, tmp_path):
    objects = index_source(source, tmp_path / "index")
    source = {**source, "roi": [[11, 0, 0], [15, 24, 24]], "boundary_policy": "allow_partial"}
    with PhysicalObjects(source, objects, {"size": 28, "tile_size_um": 0.5, "axes": [1]}) as reader:
        obj = reader.prepare(objects.iloc[1].to_dict())
        assert obj["coverage_scope"] == "partial_object"
        assert obj["bounds"][0][0] == pytest.approx(1.05)
        assert obj["bounds"][1][0] == pytest.approx(1.45)
        image, mask, _ = reader.plane(
            np.array([0.85, 0.3, 0.3]), np.array([1.65, 1.1, 1.1]), 1, 0.7, 2**60 + 2
        )
        assert not mask[:7].any() and not mask[21:].any()
        assert np.all(image[:7] == 0) and np.all(image[21:] == 0)
        for mode in ("fixed_tiles", "whole_label"):
            views = list(reader.planes(obj, mode))
            assert views and all(info["coverage_scope"] == "partial_object" for _, _, info in views)
            assert all("Z" in info["roi_faces"] for _, _, info in views)


def test_partial_policy_does_not_relax_budget_or_stale_bounds(source, tmp_path):
    objects = index_source(source, tmp_path / "index")
    source.update(boundary_policy="allow_partial", max_object_voxels=1)
    with PhysicalObjects(source, objects, {}) as reader:
        with pytest.raises(ValueError, match="max_object_voxels"):
            reader.prepare(objects.iloc[0].to_dict())
        source["max_object_voxels"] = 1000
        row = objects.iloc[0].to_dict()
        row["bbox_max_y"] -= 2
        with pytest.raises(ValueError, match="refresh the object index"):
            reader.prepare(row)


def test_experiment_stops_before_loading_backbone_and_points_to_prepare_fields(
    source, tmp_path, monkeypatch
):
    index_source(source, tmp_path / "index")
    source.update(max_object_voxels=1, object_index=str(tmp_path / "index/objects.tsv"))
    monkeypatch.setattr(
        "morphofeatures.dino.load_backbone",
        lambda settings: pytest.fail("Loaded a model despite no eligible objects"),
    )
    stage = {
        "action": "dino_experiment",
        "datasets": [{"name": "Parhyale", "source": source}],
        "backbones": [{}],
    }
    with pytest.raises(FieldError, match="Dataset Parhyale: 0 of 2") as caught:
        run_experiment(stage, tmp_path / "experiment")
    assert ("datasets", 0, "source", "max_object_voxels") in caught.value.fields
    assert stage_issue(caught.value, 0, stage).page == "Prepare data"


@pytest.mark.parametrize("policy", ["skip", "allow_partial"])
def test_extracted_vectors_keep_coverage_scope_and_native_ids(source, tmp_path, policy):
    from morphofeatures.data.io import load_embeddings

    source["boundary_policy"] = policy
    paths = extract_physical(extraction(source, tmp_path), tmp_path / "output", backbone=Backbone())
    expected = [2**60 + 2] if policy == "skip" else [2**60 + 1, 2**60 + 2]
    for path in paths.values():
        assert load_embeddings(path).label_ids.tolist() == expected
    parent = Path(next(iter(paths.values()))).parent
    views = [json.loads(line) for line in (parent / "views.jsonl").read_text().splitlines()]
    first = [v for v in views if v["label_id"] == 2**60 + 1]
    if policy == "allow_partial":
        assert first and all(v["coverage_scope"] == "partial_object" for v in first)
        coverage = pd.read_csv(parent / "coverage.tsv", sep="\t")
        assert set(coverage[coverage.label_id == 2**60 + 1].coverage_scope) == {"partial_object"}


def test_boundary_stop_policy_does_not_load_backbone(source, tmp_path, monkeypatch):
    source["boundary_policy"] = "error"
    settings = extraction(source, tmp_path)
    monkeypatch.setattr(
        "morphofeatures.dino.load_backbone",
        lambda settings: pytest.fail("Loaded backbone before eligibility error"),
    )
    with pytest.raises(FieldError, match="No DINO inference was started"):
        extract_physical(settings, tmp_path / "stopped")


def test_runtime_error_policy_records_failure_then_stops(source, tmp_path):
    source.update(on_object_error="error", boundary_policy="allow_partial")
    np.save(source["raw"], np.zeros((24, 24, 24), dtype=np.float32))
    with pytest.raises(FieldError, match="Stopped at object") as caught:
        extract_physical(extraction(source, tmp_path), tmp_path / "stopped", backbone=Backbone())
    assert ("on_object_error",) in caught.value.fields
    report = pd.read_csv(next((tmp_path / "stopped").rglob("excluded.tsv")), sep="\t")
    assert len(report) == 1 and "contrast" in report.iloc[0].reason
