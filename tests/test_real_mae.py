from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from morphofeatures.data.io import load_embeddings
from morphofeatures.data.n5 import (
    N5GroupedPatchDataset,
    N5MaskedPatchDataset,
    deterministic_label_split,
    discover_n5_metadata,
    load_patch_index,
    preprocess_masked_patch,
)
from morphofeatures.metrics import read_metric_events
from morphofeatures.real_mae import (
    build_learning_rate_scheduler,
    encode_real_mae,
    prepare_real_mae_data,
    read_patch_qc_views,
    reconstruct_real_samples,
    resolve_real_mae_config,
    train_real_mae,
)


def test_real_mae_learning_rate_scheduler_vocab_and_progression():
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    scheduler = build_learning_rate_scheduler(
        optimizer,
        {
            "scheduler": "cosine",
            "epochs": 4,
            "warmup_epochs": 0,
            "min_learning_rate": 0.01,
        },
    )
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[0] == pytest.approx(0.1)
    assert rates[-1] == pytest.approx(0.01)
    assert all(left >= right for left, right in zip(rates, rates[1:]))

    with pytest.raises(ValueError, match="Unsupported"):
        build_learning_rate_scheduler(
            optimizer, {"scheduler": "invented", "epochs": 4, "warmup_epochs": 0}
        )


def test_n5_metadata_discovery_stops_before_chunks(tmp_path):
    root = tmp_path / "data.n5"
    dataset = root / "group" / "patches"
    chunk = dataset / "0" / "0"
    chunk.mkdir(parents=True)
    (root / "attributes.json").write_text('{"n5": "2.0.0"}', encoding="utf-8")
    (root / "group" / "attributes.json").write_text("{}", encoding="utf-8")
    (dataset / "attributes.json").write_text(
        json.dumps(
            {
                "dimensions": [8, 8, 8, 12],
                "blockSize": [8, 8, 8, 1],
                "dataType": "uint8",
                "compression": {"type": "gzip", "level": 5},
            }
        ),
        encoding="utf-8",
    )
    # Invalid JSON below the dataset would fail the test if discovery entered chunks.
    (chunk / "attributes.json").write_text("not json", encoding="utf-8")

    rows = discover_n5_metadata(root, {"group/patches": "nzyx"})
    assert len(rows) == 1
    assert rows[0].shape == (12, 8, 8, 8)
    assert rows[0].chunks == (1, 8, 8, 8)
    assert rows[0].axes == "nzyx"


def test_stable_group_split_and_mask_preprocessing():
    ids = np.arange(1, 101, dtype=np.int64)
    first = deterministic_label_split(ids, (0.8, 0.1, 0.1), seed=7)
    second = deterministic_label_split(ids, (0.8, 0.1, 0.1), seed=7)
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert not np.intersect1d(first["train"], first["validation"]).size
    assert not np.intersect1d(first["train"], first["test"]).size
    assert sum(len(values) for values in first.values()) == len(ids)

    patch = np.zeros((4, 4, 4), dtype=np.uint8)
    patch[1:3, 1:3, 1:3] = 128
    values, mask, qc = preprocess_masked_patch(patch, normalization="dtype")
    assert values.dtype == np.float32
    assert values.max() == pytest.approx(128 / 255)
    assert mask.sum() == 8
    assert qc.foreground_fraction == pytest.approx(8 / 64)


def _write_n5_fixture(tmp_path: Path):
    z5py = pytest.importorskip("z5py")
    patch_path = tmp_path / "patches.n5"
    position_path = tmp_path / "positions.n5"
    labels = np.repeat(np.arange(1, 7, dtype=np.int64), 3)
    patches = np.zeros((len(labels), 8, 8, 8), dtype=np.uint8)
    for index, label_id in enumerate(labels):
        patches[index, 1:7, 1:7, 1:7] = 30 + label_id * 20
    patches[0] = 0  # exercises bounded same-label replacement
    positions = np.column_stack(
        (labels, np.arange(len(labels)), np.arange(len(labels)) + 1, np.arange(len(labels)) + 2)
    )
    with z5py.File(str(patch_path), "w") as store:
        store.create_dataset("patches", data=patches, chunks=(1, 8, 8, 8), compression="gzip")
    with z5py.File(str(position_path), "w") as store:
        store.create_dataset("positions", data=positions, chunks=(4, 4), compression="gzip")
        store.create_dataset("ids", data=np.arange(1, 7, dtype=np.int64), compression="gzip")
    return patch_path, position_path


def _write_config(tmp_path: Path, patch_path: Path, position_path: Path):
    config = {
        "seed": 11,
        "device": "cpu",
        "active_profile": "quick",
        "data": {
            "source": "n5_masked_patches",
            "patches_container": str(patch_path),
            "patches_key": "patches",
            "positions_container": str(position_path),
            "positions_key": "positions",
            "ids_key": "ids",
            "modality": "EM",
            "biological_unit": "nucleus",
            "axes": "zyx",
            "resolution_zyx_um": [0.04, 0.02, 0.02],
            "position_resolution_zyx_um": [0.08, 0.04, 0.04],
            "patch_shape_zyx": [8, 8, 8],
            "normalization": "dtype",
            "mask_mode": "nonzero",
            "min_foreground_fraction": 0.01,
            "group_size": 2,
            "position_stride_zyx": [1, 1, 1],
        },
        "split": {
            "train_fraction": 0.5,
            "validation_fraction": 1 / 3,
            "test_fraction": 1 / 6,
        },
        "mae": {
            "input_shape": [8, 8, 8],
            "architecture_version": "grouped-nucleus-patches-v3",
            "reconstruction_shape": [4, 4, 4],
            "embedding_dim": 8,
            "encoder_depth": 1,
            "encoder_heads": 2,
            "decoder_dim": 8,
            "decoder_depth": 1,
            "decoder_heads": 2,
            "norm_pix_loss": True,
            "mask_ratio": 0.5,
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "workers": 0,
            "learning_rate": 0.001,
            "mixed_precision": False,
            "checkpoint_interval": 1,
        },
        "inference": {"batch_size": 2, "workers": 0, "split": "all"},
        "augmentation": {"random_flip": False, "random_rot90": False},
        "paths": {"run_dir": str(tmp_path / "run")},
        "profiles": {
            "quick": {"training": {"epochs": 1}},
            "full": {"training": {"epochs": 2}},
        },
    }
    path = tmp_path / "real.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.optional
def test_dino_uses_grouped_n5_data_and_preserves_parent_ids(tmp_path):
    torch = pytest.importorskip("torch")
    from morphofeatures.configuration_editor import load_document
    from morphofeatures.dino import extract_dino, validate_dino_settings

    class TinyBackbone(torch.nn.Module):
        def forward_features(self, batch):
            return {"x_norm_clstoken": batch.mean((2, 3))}

    patch_path, position_path = _write_n5_fixture(tmp_path)
    config_path = _write_config(tmp_path, patch_path, position_path)
    settings = {
        "model": "dinov2",
        "config": load_document(config_path),
        "views": {"size": 28, "axes": [0], "fractions": [0.5], "normalization": "dtype"},
    }
    validate_dino_settings(settings)
    ids, features, excluded = extract_dino(settings, backbone=TinyBackbone())
    # Eighteen patch rows become six nucleus embeddings, including replacement
    # of the empty patch without dropping its parent or mixing nuclei.
    np.testing.assert_array_equal(ids, np.arange(1, 7))
    assert features.shape == (6, 3)
    assert np.isfinite(features).all() and not excluded
    assert (np.diff(features, axis=0) > 0).all()


@pytest.mark.optional
def test_grouped_object_inspection_reads_bounded_raw_volume(tmp_path):
    from morphofeatures.analysis.object_inspection import read_object_crops

    patch_path, position_path = _write_n5_fixture(tmp_path)
    raw = np.arange(16**3, dtype=np.uint16).reshape(16, 16, 16)
    np.save(tmp_path / "raw.npy", raw)
    config = {"data": {"source": "n5_masked_patches", "patches_container": str(patch_path),
                       "positions_container": str(position_path), "qc_raw_container": str(tmp_path / "raw.npy"),
                       "position_to_raw_scale_zyx": [2, 2, 2]}}
    images = list(read_object_crops(config, [2], source="original", max_side=8))
    np.testing.assert_array_equal(images[0][1], raw[4:12, 6:14, 8:16])
    assert "Raw volume" in images[0][2]
    prepared = list(read_object_crops(config, [2], max_side=8))
    assert prepared[0][1].shape == (8, 8, 8)
    assert "Representative" in prepared[0][2]


@pytest.mark.optional
def test_real_mae_resnet_train_checkpoint_reload_and_encode_cycle(tmp_path):
    pytest.importorskip("torch")
    patch_path, position_path = _write_n5_fixture(tmp_path)
    config_path = _write_config(tmp_path, patch_path, position_path)
    resolved = resolve_real_mae_config(
        config_path,
        profile="quick",
        overrides={
            "mae": {
                "patch_encoder": "resnet3d",
                "resnet_channels": [4],
                "resnet_blocks": [1],
            },
            "paths": {"run_dir": str(tmp_path / "resnet-run")},
        },
    )
    checkpoint = train_real_mae(resolved)
    embedding_path = encode_real_mae(resolved, checkpoint)
    embedding = load_embeddings(embedding_path)
    assert embedding.features.shape == (6, 8)
    assert np.all(np.isfinite(embedding.features))
    metadata = json.loads(
        embedding_path.with_suffix(".npy.metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["patch_encoder"] == "resnet3d"


@pytest.mark.optional
def test_lazy_loading_config_merge_and_label_first_cycle(tmp_path):
    pytest.importorskip("torch")
    patch_path, position_path = _write_n5_fixture(tmp_path)
    config_path = _write_config(tmp_path, patch_path, position_path)

    resolved = resolve_real_mae_config(
        config_path,
        profile="quick",
        overrides={"training": {"batch_size": 1}},
    )
    assert resolved.values["training"]["epochs"] == 1
    assert resolved.values["training"]["batch_size"] == 1
    assert resolved.values["training"]["progress_interval_batches"] == 25
    assert resolved.values["training"]["early_stopping"] == {
        "enabled": False,
        "patience": 10,
        "min_delta": 0.0,
        "restore_best": True,
    }
    assert resolved.values["paths"]["best_checkpoint"].endswith("checkpoint-best.pt")
    first_snapshot = resolved.save()
    changed = resolve_real_mae_config(
        config_path,
        profile="quick",
        overrides={"training": {"batch_size": 2}},
    )
    changed_snapshot = changed.save()
    assert changed_snapshot != first_snapshot
    assert first_snapshot.read_text(encoding="utf-8") != changed_snapshot.read_text(
        encoding="utf-8"
    )
    preparation_events = []
    prepared = prepare_real_mae_data(
        resolved,
        progress=lambda event, **values: preparation_events.append(
            {"event": event, **values}
        ),
    )
    assert [event["event"] for event in preparation_events] == [
        "metadata_validated",
        "index_loading_started",
        "index_loaded",
        "splits_created",
        "patch_selection_completed",
    ]
    assert sum(len(values) for values in prepared.label_splits.values()) == 6
    assert all(len(values) for values in prepared.label_splits.values())
    training_dataset = prepared.dataset("train", "train", augment=True)
    assert isinstance(training_dataset, N5GroupedPatchDataset)
    group = training_dataset.read_group(0)
    assert group["patches"].shape == (2, 1, 8, 8, 8)
    assert group["valid"].sum() == 2
    assert group["label_id"] in prepared.label_splits["train"]

    index = load_patch_index(position_path, expected_patch_count=18)
    dataset = N5MaskedPatchDataset(
        patch_path,
        "patches",
        index,
        [0],
        min_foreground_fraction=0.01,
        replacement_attempts=2,
        mode="inspect",
    )
    qc, _, _ = dataset[0]
    assert qc.patch_index == 1
    assert qc.label_id == 1

    checkpoint = train_real_mae(resolved)
    resumed = resolve_real_mae_config(
        config_path,
        profile="quick",
        overrides={
            "training": {
                "epochs": 2,
                "batch_size": 1,
                "resume_from": str(checkpoint),
            }
        },
    )
    checkpoint = train_real_mae(resumed)
    embedding_path = encode_real_mae(resumed, checkpoint)
    table = load_embeddings(embedding_path)
    assert table.as_array().shape == (6, 9)
    assert np.array_equal(table.label_ids, np.arange(1, 7))
    assert np.all(np.isfinite(table.features))
    reconstruction = reconstruct_real_samples(resumed, checkpoint, count=1)
    assert reconstruction["foreground_masks"].shape == reconstruction["inputs"].shape
    assert np.isfinite(reconstruction["masked_foreground_mse"])
    assert np.isfinite(reconstruction["visible_mean_baseline_mse"])
    events = read_metric_events(resumed.metrics_path)
    assert [event["epoch"] for event in events if event["event"] == "epoch"] == [1, 2]
    epoch_events = [event for event in events if event["event"] == "epoch"]
    assert all("validation_visible_mean_baseline_loss" in event for event in epoch_events)
    assert all("validation_improvement_over_visible_mean" in event for event in epoch_events)
    event_names = {event["event"] for event in events}
    assert {
        "preprocessing_started",
        "metadata_validated",
        "index_loaded",
        "splits_created",
        "patch_selection_completed",
        "split_manifest_written",
        "preprocessing_completed",
        "dataloader_ready",
        "model_ready",
        "batch_progress",
    }.issubset(event_names)
    progress = [event for event in events if event["event"] == "batch_progress"]
    assert all(
        {"phase", "batch", "total_batches", "progress_fraction", "eta_seconds"}
        <= event.keys()
        for event in progress
    )
    assert any(event["event"] == "completed" for event in events)


@pytest.mark.optional
def test_real_mae_validation_loss_early_stopping_restores_best_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    patch_path, position_path = _write_n5_fixture(tmp_path)
    config_path = _write_config(tmp_path, patch_path, position_path)
    resolved = resolve_real_mae_config(
        config_path,
        profile="quick",
        overrides={
            "training": {
                "epochs": 5,
                "progress_interval_batches": 1,
                "checkpoint_interval": 0,
                "early_stopping": {
                    "enabled": True,
                    "patience": 1,
                    "min_delta": 1_000_000.0,
                    "restore_best": True,
                },
            },
            "paths": {"run_dir": str(tmp_path / "early-stop-run")},
        },
    )

    checkpoint = train_real_mae(resolved)
    events = read_metric_events(resolved.metrics_path)
    completed = [event for event in events if event["event"] == "completed"][-1]
    assert completed["stopped_early"] is True
    assert completed["epochs_completed"] == 2
    assert any(event["event"] == "early_stopping" for event in events)
    assert any(event["event"] == "best_checkpoint_restored" for event in events)
    best_checkpoint = Path(resolved.values["paths"]["best_checkpoint"])
    assert best_checkpoint.is_file()
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    assert payload["epoch"] == 1
    assert payload["metrics"]["best_epoch"] == 1


@pytest.mark.parametrize(
    ("training_override", "message"),
    (
        ({"progress_interval_batches": -1}, "progress_interval_batches"),
        ({"early_stopping": {"enabled": True, "patience": 0}}, "patience"),
        ({"early_stopping": {"enabled": True, "min_delta": -0.1}}, "min_delta"),
        ({"early_stopping": {"enabled": "yes"}}, "enabled"),
    ),
)
def test_real_mae_monitoring_and_early_stopping_config_validation(
    tmp_path, training_override, message
):
    patch_path, position_path = _write_n5_fixture(tmp_path)
    config_path = _write_config(tmp_path, patch_path, position_path)
    with pytest.raises(ValueError, match=message):
        resolve_real_mae_config(
            config_path,
            profile="quick",
            overrides={"training": training_override},
        )


@pytest.mark.optional
def test_site_real_config_when_external_data_are_available():
    pytest.importorskip("z5py")
    config_path = Path("configs/sites/mae_platynereis_nuclei_embl.yaml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    external_paths = [
        raw["data"]["patches_container"],
        raw["data"]["positions_container"],
        raw["data"].get("qc_raw_container"),
    ]
    if any(path and not Path(path).exists() for path in external_paths):
        pytest.skip("External EMBL Platynereis N5 data are not mounted")
    resolved = resolve_real_mae_config(config_path, profile="quick")
    prepared = prepare_real_mae_data(resolved)
    assert prepared.patch_metadata["shape"] == [2_506_460, 32, 32, 32]
    assert len(prepared.index.unique_label_ids) == 11_382
    views = read_patch_qc_views(prepared, [0, 100])
    assert views["raw"].shape == (2, 32, 32, 32)
    assert np.all(views["masked_nonzero_matches_raw"])
