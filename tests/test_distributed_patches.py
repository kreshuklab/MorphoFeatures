"""Reusable physical patch preparation, DINO grouping and workflow handoffs."""

import json
from contextlib import ExitStack
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from morphofeatures.configuration_editor import load_document
from morphofeatures.data.distributed_patches import (
    grouped_arrays,
    preview_patches,
    validate_patch_preparation,
)
from morphofeatures.data.preprocessing import preprocess
from morphofeatures.dino import extract_dino, validate_dino_settings
from morphofeatures.workspace_jobs import resolve_job, run_job
from morphofeatures.workspace_state import stages_after_preparation


def preparation(tmp_path):
    raw = np.random.default_rng(5).uniform(1, 100, (24, 24, 24)).astype(np.float32)
    seg = np.zeros(raw.shape, dtype=np.int64)
    for i, offset in enumerate((3, 13)):
        seg[offset : offset + 6, 3:9, 3:9] = 2**60 + i + 1
    np.save(tmp_path / "raw.npy", raw)
    np.save(tmp_path / "seg.npy", seg)
    return {
        "action": "preprocess",
        "layout": "distributed_patches",
        "raw": str(tmp_path / "raw.npy"),
        "segmentation": str(tmp_path / "seg.npy"),
        "unit": "um",
        "spacing_zyx": [0.1, 0.1, 0.1],
        "origin_zyx": [2.0, -3.0, 7.0],
        "roi": [[0, 0, 0], [24, 24, 24]],
        "crop_shape": [8, 8, 8],
        "max_objects": 0,
        "object_ids": [],
        "patches": {
            "tile_size_um": 0.2,
            "min_foreground_fraction": 0.5,
            "max_tiles": 4,
            "tile_anchor": "centroid",
        },
    }


class Backbone(torch.nn.Module):
    def forward_features(self, images):
        tokens = torch.nn.functional.avg_pool2d(images, 14, 14).flatten(2).transpose(1, 2)
        return {"x_norm_clstoken": tokens.mean(1), "x_norm_patchtokens": tokens}


@pytest.mark.parametrize("storage", ["npy", "h5", "n5"])
def test_formats_keep_exact_ids_groups_positions_and_masks(tmp_path, storage):
    if storage != "npy":
        pytest.importorskip("h5py" if storage == "h5" else "z5py")
    prep = {**preparation(tmp_path), "output_format": storage}
    config = load_document(preprocess(prep, tmp_path / "prepared"))
    assert "mae" not in config
    assert config["data"]["source"] == "grouped_crops"
    with ExitStack() as stack:
        crops, masks, ids, offsets = grouped_arrays(stack, config["data"])
        assert ids.tolist() == [2**60 + 1, 2**60 + 2]
        assert offsets.tolist() == [0, 4, 8]
        assert crops.shape == (8, 8, 8, 8)
        assert np.all(crops[:][~masks[:].astype(bool)] == 0)
        assert np.min(crops[:]) >= 0 and np.max(crops[:]) <= 1
    options = {
        "model": "dinov2",
        "config": config,
        "views": {
            "size": 28,
            "axes": [0],
            "fractions": [0.5],
            "feature": "foreground_mean",
            "batch_size": 3,
        },
    }
    validate_dino_settings(options)
    ids, vectors, excluded = extract_dino(options, backbone=Backbone())
    assert ids.tolist() == [2**60 + 1, 2**60 + 2]
    assert vectors.shape == (2, 3) and not excluded
    coverage = pd.read_csv(tmp_path / "prepared/coverage.tsv", sep="\t")
    assert (coverage.saved_patches == 4).all()
    manifest = pd.read_csv(tmp_path / "prepared/patches.tsv", sep="\t")
    assert manifest.groupby("label_id").normalization_low.nunique().max() == 1
    assert len(manifest) == 8


def test_deterministic_selection_exhaustive_and_preview(tmp_path):
    prep = preparation(tmp_path)
    previews, info = preview_patches(prep)
    assert len(previews) == 4 and info["selected"] == 4
    results = []
    for run in ("a", "b", "all"):
        settings = deepcopy(prep)
        settings["patches"]["exhaustive"] = run == "all"
        config = load_document(preprocess(settings, tmp_path / run))
        results.append(np.load(config["data"]["positions_um"]))
    np.testing.assert_array_equal(results[0], results[1])
    assert len(results[2]) > len(results[0])
    assert set(map(tuple, results[0])) <= set(map(tuple, results[2]))
    config = load_document(tmp_path / "a/data_config.yaml")
    ids = np.load(config["data"]["label_ids"])
    first = 0 if ids[0] == info["label_id"] else 4
    crops = np.load(config["data"]["crops"])
    np.testing.assert_array_equal(previews[0][0], crops[first])


@pytest.mark.parametrize("anchor", ["centroid", "bbox_center", "bbox_min"])
def test_native_anisotropic_patches_are_exact_source_cuts(tmp_path, anchor):
    settings = preparation(tmp_path)
    settings.update(crop_shape=[2, 3, 4], spacing_zyx=[0.2, 0.1, 0.05])
    settings["patches"].update(sampling="native_voxels", tile_anchor=anchor, tile_size_um=999)
    config = load_document(preprocess(settings, tmp_path / "native"))
    recipe = config["data"]["patch_recipe"]
    assert recipe["sampling"] == "native_voxels"
    assert recipe["algorithm"] == "distributed-native-patches-v1"
    np.testing.assert_allclose(recipe["tile_size_um"], [0.4, 0.3, 0.2])
    np.testing.assert_array_equal(recipe["saved_voxel_size_um"], settings["spacing_zyx"])
    raw, seg = np.load(settings["raw"]), np.load(settings["segmentation"])
    saved, masks = np.load(config["data"]["crops"]), np.load(config["data"]["loss_masks"])
    manifest = pd.read_csv(tmp_path / "native/patches.tsv", sep="\t")
    starts = []
    for row in manifest.to_dict("records"):
        lo = np.array([row["lower_um_" + axis] for axis in "zyx"])
        start = (lo - settings["origin_zyx"]) / settings["spacing_zyx"] + 0.5
        np.testing.assert_allclose(start, np.rint(start), atol=1e-10)
        start = np.rint(start).astype(int)
        starts.append(start)
        box = tuple(slice(int(a), int(a + b)) for a, b in zip(start, settings["crop_shape"]))
        expected = np.clip(
            (raw[box] - row["normalization_low"])
            / (row["normalization_high"] - row["normalization_low"]),
            0,
            1,
        )
        mask = seg[box] == row["segmentation_id"]
        expected[~mask] = 0
        np.testing.assert_array_equal(masks[row["row"]].astype(bool), mask)
        np.testing.assert_allclose(saved[row["row"]], expected, atol=1e-6)
    first_object = starts[:4]
    assert np.all(
        (np.array(first_object) - first_object[0]) % np.array(settings["crop_shape"]) == 0
    )
    ids, features, excluded = extract_dino(
        {
            "model": "dinov2",
            "config": config,
            "views": {"size": 28, "axes": [0, 1, 2], "fractions": [0.5]},
        },
        backbone=Backbone(),
    )
    assert not excluded and ids.tolist() == [2**60 + 1, 2**60 + 2]
    assert np.isfinite(features).all()


def test_native_patches_align_other_segmentation_grid_without_raw_interpolation(
    tmp_path, monkeypatch
):
    from morphofeatures.data import distributed_patches as patches
    from morphofeatures.data.dino_views import PhysicalObjects, index_source

    settings = preparation(tmp_path)
    settings.update(
        crop_shape=[3, 4, 5],
        segmentation_spacing_zyx=[0.2, 0.2, 0.2],
        segmentation_origin_zyx=[2.04, -2.96, 7.04],
        roi=None,
    )
    seg = np.zeros((12, 12, 12), dtype=np.int64)
    seg[3:8, 3:8, 3:8] = 2**60 + 1
    np.save(settings["segmentation"], seg)
    settings["patches"].update(sampling="native_voxels", max_tiles=2)
    source, options, shape = validate_patch_preparation(settings)
    objects = index_source(source, tmp_path / "index")
    with PhysicalObjects(source, objects, options) as reader:
        obj = reader.prepare(objects.iloc[0].to_dict())
        original_sample = patches.sample_volume

        def only_labels(volume, *args, **kwargs):
            assert volume is not reader.raw, "Native raw intensities must not be interpolated"
            return original_sample(volume, *args, **kwargs)

        monkeypatch.setattr(patches, "sample_volume", only_labels)
        regions, _, _ = patches.patch_regions(reader, obj, options, shape)
        for lo, hi, _ in regions:
            _, mask = patches.read_patch(reader, obj, lo, hi, shape)
            coords = (
                lo
                + (np.indices(tuple(shape)).transpose(1, 2, 3, 0) + 0.5) * source["raw_spacing_zyx"]
            )
            indices = np.floor(
                (coords - source["segmentation_origin_zyx"]) / source["segmentation_spacing_zyx"]
                + 0.5
            ).astype(int)
            expected = seg[tuple(indices[..., d] for d in range(3))] == 2**60 + 1
            np.testing.assert_array_equal(mask, expected)


def test_native_dino_inputs_preserve_physical_aspect_ratio():
    from morphofeatures.dino import microscopy_views

    raw = np.full((2, 4, 8), 0.5, dtype=np.float32)
    masks = [
        mask
        for _, mask in microscopy_views(
            raw,
            np.ones_like(raw, dtype=bool),
            {
                "normalization": "unit",
                "resize": "letterbox",
                "spacing_zyx": [0.4, 0.1, 0.1],
                "axes": [1, 0],
                "fractions": [0.5],
                "size": 28,
            },
            with_mask=True,
        )
    ]
    assert masks[0].all()  # XZ is physically square although its array is 2 × 8.
    assert masks[1].sum() == 14 * 28  # XY retains its 1:2 physical ratio.


def test_large_preview_background_lookup_is_reused_and_reads_one_raw_object(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from morphofeatures.data import dino_views, patch_preview

    opened = dino_views.open_source
    raw_reads = []

    def tracked(stack, source, kind):
        volume = opened(stack, source, kind)
        if kind != "raw":
            return volume

        def read(slices):
            raw_reads.append(slices)
            return volume.read(slices)

        return SimpleNamespace(shape=volume.shape, dtype=volume.dtype, read=read)

    monkeypatch.setattr(dino_views, "open_source", tracked)

    settings = preparation(tmp_path)
    labels = np.zeros((140, 140, 140), dtype=np.int64)
    labels[20:26, 20:26, 20:26] = 2**60 + 1
    labels[100:106, 100:106, 100:106] = 2**60 + 2
    np.save(settings["segmentation"], labels)
    np.save(settings["raw"], np.indices(labels.shape)[0].astype(np.float32))
    settings.update(roi=None, crop_shape=[4, 4, 4])
    settings["patches"].update(sampling="native_voxels")
    first = patch_preview.start_patch_preview(settings, tmp_path / "cache", object_id=2**60 + 2)
    result = first.future.result(timeout=30)
    assert result["state"] == "complete", result
    assert result["summary"]["label_id"] == 2**60 + 2
    assert len(result["images"]) <= 6 and len(result["summary"]["overview"]) == 3
    bounds = result["summary"]["object_index"]
    assert len(pd.read_csv(bounds, sep="\t")) == 2
    assert raw_reads and all(slices[0].start >= 90 for slices in raw_reads)
    raw_reads.clear()
    monkeypatch.setattr(
        patch_preview,
        "index_source",
        lambda *args, **kwargs: pytest.fail("Repeated the segmentation scan"),
    )
    settings["patches"]["max_tiles"] = 1
    second = patch_preview.start_patch_preview(settings, tmp_path / "cache", object_id=2**60 + 1)
    result = second.future.result(timeout=30)
    assert result["state"] == "complete", result
    assert result["summary"]["label_id"] == 2**60 + 1
    assert result["summary"]["object_index"] == bounds
    assert len(result["images"]) == 1
    assert raw_reads and all(slices[0].stop < 35 for slices in raw_reads)


def test_preview_bounds_cache_invalidates_roi_and_source_version(tmp_path):
    from morphofeatures.data.patch_preview import cached_bounds

    source, _, _ = validate_patch_preparation(preparation(tmp_path))
    a = cached_bounds(source, tmp_path / "cache")
    b = cached_bounds({**source, "roi": [[10, 0, 0], [24, 24, 24]]}, tmp_path / "cache")
    c = cached_bounds({**source, "data_version": "updated"}, tmp_path / "cache")
    assert len({a, b, c}) == 3
    assert pd.read_csv(b, sep="\t").label_id.tolist() == [2**60 + 2]


def test_cancelled_preview_scan_is_not_cached_as_complete(tmp_path):
    import threading

    from morphofeatures.data.patch_preview import PreviewCancelled, cached_bounds

    source, _, _ = validate_patch_preparation(preparation(tmp_path))
    cancel = threading.Event()
    with pytest.raises(PreviewCancelled):
        cached_bounds(
            source,
            tmp_path / "cache",
            cancel=cancel,
            progress=lambda **status: cancel.set() if status["blocks"] else None,
        )
    assert not list((tmp_path / "cache").rglob("index.json"))
    assert cached_bounds(source, tmp_path / "cache").is_file()


def test_different_raw_segmentation_grids_and_translated_origins(tmp_path):
    from morphofeatures.data.dino_views import PhysicalObjects, index_source
    from morphofeatures.data.distributed_patches import resample_patch

    prep = preparation(tmp_path)
    seg = np.zeros((16, 16, 16), dtype=np.int64)
    seg[3:10, 3:10, 3:10] = 1
    np.save(tmp_path / "seg.npy", seg)
    origin = np.array([2.0, -3.0, 7.0])
    prep.update(segmentation_spacing_zyx=[0.2, 0.1, 0.1], roi=[[0, 0, 0], [16, 16, 16]])
    images = []
    for run, spacing, shape in (
        ("fine", [0.1, 0.05, 0.05], (32, 32, 32)),
        ("coarse", [0.2, 0.1, 0.1], (16, 16, 16)),
    ):
        coordinates = np.indices(shape).transpose(1, 2, 3, 0) * spacing + origin
        np.save(tmp_path / "raw.npy", coordinates.sum(3).astype(np.float32))
        prep["raw_spacing_zyx"] = spacing
        source, options, saved_shape = validate_patch_preparation(prep)
        objects = index_source(source, tmp_path / run)
        with PhysicalObjects(source, objects, options) as reader:
            obj = reader.prepare(objects.iloc[0].to_dict())
            image, mask = resample_patch(
                reader, obj, origin + [1, 0.5, 0.5], origin + [1.2, 0.7, 0.7], saved_shape
            )
            images.append(image)
            assert mask.all()
    np.testing.assert_allclose(images[0], images[1], atol=1e-5)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"unit": "voxel"}, "physical units|unit=um"),
        ({"crop_shape": [32, 16, 32]}, "cubic"),
        ({"patches": {"min_foreground_fraction": 1.1}}, "fraction"),
        ({"patches": {"tile_size_um": 0}}, "positive"),
        ({"patches": {"max_tiles": 0}}, "integer"),
        ({"patches": {"exhaustive": "false"}}, "true or false"),
    ],
)
def test_validation_identifies_settings(tmp_path, change, message):
    from morphofeatures.workflow_validation import FieldError

    with pytest.raises(FieldError, match=message) as error:
        validate_patch_preparation({**preparation(tmp_path), **change})
    assert error.value.fields


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"max_object_voxels": 1}, "max_object_voxels"),
        ({"roi": [[3, 0, 0], [24, 24, 24]]}, "boundary"),
        ({"patches": {"tile_size_um": 50, "min_foreground_fraction": 1}}, "foreground cutoff"),
        ({"patches": {"tile_size_um": 0.01, "max_candidate_tiles": 2}}, "max_candidate_tiles"),
    ],
)
def test_exclusions_are_written_even_when_nothing_saved(tmp_path, change, reason):
    settings = {**preparation(tmp_path), "object_ids": [2**60 + 1], **change}
    with pytest.raises(ValueError, match="No distributed patches"):
        preprocess(settings, tmp_path / "failed")
    report = pd.read_csv(tmp_path / "failed/coverage.tsv", sep="\t")
    assert reason in report.iloc[0].reason
    assert (
        json.loads((tmp_path / "failed/preprocessing.json").read_text())["summary"]["saved_objects"]
        == 0
    )


def test_group_integrity_and_normalization_fail_before_encoding(tmp_path):
    config = load_document(preprocess(preparation(tmp_path), tmp_path / "output"))
    stage = {
        "model": "dinov2",
        "config": config,
        "views": {"normalization": "foreground_percentile"},
    }
    with pytest.raises(ValueError, match="Select unit"):
        validate_dino_settings(stage)
    stage["views"]["normalization"] = "unit"
    ids = np.load(config["data"]["patch_label_ids"])
    ids[0] += 1
    np.save(config["data"]["patch_label_ids"], ids)
    with pytest.raises(ValueError, match="disagree"):
        validate_dino_settings(stage)


def test_inspection_and_recipe_preserve_source_identity(tmp_path):
    from morphofeatures.analysis.joint import _signature
    from morphofeatures.analysis.mesh_inspection import mesh_source_defaults
    from morphofeatures.analysis.mobie_export import export_source_defaults
    from morphofeatures.analysis.object_inspection import read_object_crops

    prep = preparation(tmp_path)
    config = load_document(preprocess(prep, tmp_path / "out"))
    for source in ("prepared", "original"):
        crops = list(read_object_crops(config, [2**60 + 2], source=source, max_side=16))
        assert crops[0][0] == 2**60 + 2 and crops[0][1].size
    mesh = mesh_source_defaults(config)
    assert mesh["segmentation"] == prep["segmentation"]
    assert mesh["index_ids"] == "embedding"
    export = export_source_defaults(config)
    assert export["raw"] == prep["raw"] and export["segmentation"] == prep["segmentation"]
    assert export["resample_raw"] is True
    signature = _signature({"model": "dinov2", "settings": {"config": config}})
    assert signature["views"]["normalization"] == "unit"
    assert signature["distributed patch recipe"]["tile_size_um"] == 0.2


def test_saved_artifact_starts_dino_with_inspection_and_shared_contrast(tmp_path, monkeypatch):
    from morphofeatures.config import config_from_mapping
    from morphofeatures.workflow_ui import continue_from_artifact

    path = preprocess(preparation(tmp_path), tmp_path / "prepared")
    captured = []
    monkeypatch.setattr(
        "morphofeatures.workflow_ui.use_draft",
        lambda config, draft, step: captured.append((draft, step)),
    )
    monkeypatch.setattr("morphofeatures.workflow_ui.navigate", lambda page: None)
    # Artifact handoffs now inherit the workspace's DINO defaults.
    config = config_from_mapping(
        {"dino": {"dinov2": {"variant": "dinov2_vitb14_reg"}}},
        base=tmp_path,
        apply_environment=False,
    )
    continue_from_artifact(config, path)
    draft, step = captured[0]
    stage = draft["document"]["stages"][0]
    assert step == "Analyze" and stage["model"] == "dinov2"
    assert stage["variant"] == "dinov2_vitb14_reg"
    assert stage["views"]["normalization"] == "unit"
    assert stage["config"]["data"]["source"] == "grouped_crops"
    assert stage["config"]["inspection"]["mesh"]["index_ids"] == "embedding"


def test_linked_worker_prepares_groups_and_embeds_once_per_native_id(tmp_path, monkeypatch):
    prep = preparation(tmp_path)
    stages = stages_after_preparation(prep, dino=True)
    stage = stages[1]
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"mock checkpoint")
    (tmp_path / "hubconf.py").write_text("# controlled mock backend\n")
    stage.update(checkpoint=str(checkpoint), model_repository=str(tmp_path))
    stage["views"].update(size=28, batch_size=3)
    stage["config"]["device"] = "cpu"
    stage.pop("cache", None)
    document = {"stages": stages[:2]}
    monkeypatch.setattr("morphofeatures.dino.load_backbone", lambda settings: Backbone())
    job = tmp_path / "job.yaml"
    job.write_text(yaml.safe_dump(document))
    status = json.loads(run_job(job).read_text())
    assert status["state"] == "completed", status
    from morphofeatures.data.io import load_embeddings

    table = load_embeddings(status["stages"][1]["result"])
    assert table.label_ids.tolist() == [2**60 + 1, 2**60 + 2]
    assert table.features.shape[0] == 2
    from morphofeatures.analysis.fusion import dataset_namespace, fuse_embeddings
    from morphofeatures.analysis.joint import extraction_metadata
    from morphofeatures.dino_physical import extract_physical

    config = load_document(status["stages"][0]["result"])
    preparation_report = json.loads(open(config["data"]["preprocessing"]).read())
    metadata = extraction_metadata(status["stages"][1]["result"])
    assert dataset_namespace(metadata) == preparation_report["dataset_identity"]
    coarse = extract_physical(
        {
            **stage,
            "config": {"device": "cpu", "data": {"volume": config["data"]["volume"]}},
            "views": {
                "mode": "whole_label",
                "size": 28,
                "axes": [0],
                "coarse_fractions": [0.5],
                "features": ["cls"],
            },
        },
        tmp_path / "coarse",
        backbone=Backbone(),
    )
    combined = fuse_embeddings(
        {
            "embeddings": {
                "fine": status["stages"][1]["result"],
                "coarse": str(coarse["whole_label.cls"]),
            }
        },
        tmp_path / "combined",
    )
    assert load_embeddings(combined).label_ids.tolist() == [2**60 + 1, 2**60 + 2]
    assert load_embeddings(combined).features.shape == (2, 6)
    incompatible = deepcopy(document)
    incompatible["stages"][1]["model"] = "mae"
    with pytest.raises(ValueError, match="Distributed fine patches currently feed DINO"):
        resolve_job(incompatible, tmp_path)
    with pytest.raises(ValueError, match="Distributed patches currently feed DINO"):
        stages_after_preparation(prep, mae="train")
