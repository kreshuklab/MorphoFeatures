from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from morphofeatures.analysis.classification import evaluate_embedding_classifier
from morphofeatures.data.io import export_embeddings
from morphofeatures.mae_sweep import (
    compare_soft_grid_classifiers,
    expand_soft_grid,
    plan_soft_grid,
    prepare_soft_grid,
    submit_soft_grid,
    summarize_soft_grid,
)
from morphofeatures.metrics import MetricWriter
from morphofeatures.registry import DuplicateSubmissionError, JobRegistry
from morphofeatures.slurm import ClusterProfile, FakeScheduler


def _real_config(tmp_path: Path) -> Path:
    patches = tmp_path / "patches.n5"
    positions = tmp_path / "positions.n5"
    patches.mkdir()
    positions.mkdir()
    values = {
        "seed": 7,
        "device": "cpu",
        "active_profile": "quick",
        "data": {
            "source": "n5_masked_patches",
            "patches_container": str(patches),
            "positions_container": str(positions),
            "patch_shape_zyx": [8, 8, 8],
            "resolution_zyx_um": [0.04, 0.02, 0.02],
            "position_resolution_zyx_um": [0.08, 0.04, 0.04],
            "modality": "EM",
            "biological_unit": "nucleus",
            "axes": "zyx",
            "group_size": 4,
            "position_stride_zyx": [1, 1, 1],
            "normalization": "dtype",
        },
        "split": {
            "train_fraction": 0.8,
            "validation_fraction": 0.1,
            "test_fraction": 0.1,
        },
        "mae": {
            "architecture_version": "grouped-nucleus-patches-v3",
            "input_shape": [8, 8, 8],
            "reconstruction_shape": [4, 4, 4],
            "embedding_dim": 8,
            "encoder_depth": 1,
            "encoder_heads": 2,
            "decoder_dim": 8,
            "decoder_depth": 1,
            "decoder_heads": 2,
            "patch_encoder": "linear",
            "mask_ratio": 0.5,
        },
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "workers": 0,
            "learning_rate": 0.001,
            "scheduler": "constant",
        },
        "inference": {"batch_size": 2, "workers": 0, "split": "all"},
        "paths": {"run_dir": str(tmp_path / "unused")},
        "profiles": {"quick": {}, "full": {"training": {"epochs": 3}}},
    }
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def test_soft_grid_is_bounded_and_submits_registered_afterok_jobs(tmp_path, repo_root):
    base = _real_config(tmp_path)
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(
        yaml.safe_dump(
            {
                "sweep": {
                    "name": "local-bias",
                    "mode": "one_at_a_time",
                    "max_runs": 4,
                    "parameters": {
                        "training.learning_rate": [0.001, 0.0003],
                        "mae.patch_encoder": ["linear", "resnet3d"],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs"
    prepared = prepare_soft_grid(base, sweep_path, output_root=output_root)
    assert {variant.name for variant in prepared.variants} == {
        "baseline",
        "learning_rate-0-0003",
        "patch_encoder-resnet3d",
    }
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "morphofeatures.mae_soft_grid.v1"
    assert all(Path(item["config_path"]).is_file() for item in manifest["variants"])

    plans = plan_soft_grid(
        prepared,
        ClusterProfile("test", partition="gpu", gpus=1),
        output_root=output_root,
    )
    assert len(plans) == 3
    assert all("morphofeatures mae-train" in plan.script for plan in plans)
    registry = JobRegistry.under_output_root(output_root)
    scheduler = FakeScheduler(next_job_id=900)
    records = submit_soft_grid(
        plans,
        registry,
        scheduler,
        repository=repo_root,
        encode_after_training=True,
    )
    assert len(records) == 6
    assert [record.workflow for record in records] == ["mae_train", "mae_encode"] * 3
    assert records[1].dependency_job_id == records[0].slurm_job_id
    assert scheduler.submissions[1][1] == records[0].slurm_job_id
    with pytest.raises(DuplicateSubmissionError):
        submit_soft_grid(plans, registry, scheduler, repository=repo_root)

    writer = MetricWriter(prepared.variants[0].run_dir / "metrics.jsonl")
    writer.write("epoch", epoch=1, train_loss=1.1, validation_loss=1.2)
    writer.write("epoch", epoch=2, train_loss=0.9, validation_loss=1.0)
    with (prepared.variants[0].run_dir / "metrics.jsonl").open("ab") as stream:
        stream.write(b'{"event": "epoch"')
    summary = summarize_soft_grid(prepared)
    baseline = summary.loc[summary["variant"] == "baseline"].iloc[0]
    assert baseline["epochs_completed"] == 2
    assert baseline["best_validation_loss"] == pytest.approx(1.0)

    classifier_ids = np.arange(1, 13, dtype=np.int64)
    classifier_labels = np.repeat(["first", "second"], 6)
    classifier_features = np.column_stack(
        (np.repeat([-2.0, 2.0], 6), np.linspace(-0.1, 0.1, 12))
    )
    export_embeddings(
        prepared.variants[0].run_dir / "embeddings" / "nucleus_texture_mae.npy",
        classifier_ids,
        classifier_features,
    )
    classifier_labels_path = tmp_path / "classifier_labels.tsv"
    pd.DataFrame(
        {"label_id": classifier_ids, "cell_type": classifier_labels}
    ).to_csv(classifier_labels_path, sep="\t", index=False)
    classifier_summary = compare_soft_grid_classifiers(
        prepared,
        classifier_labels_path,
        output=tmp_path / "classifier_comparison.tsv",
        folds=3,
    )
    assert classifier_summary.iloc[0]["mean_accuracy"] == pytest.approx(1.0)

    # A standalone real-data config may point at a notebook run directory.
    # Its immutable SLURM snapshot must instead agree with registry artifacts.
    from morphofeatures.experiments import plan_job, submit_plan
    from morphofeatures.slurm import DryRunScheduler
    from morphofeatures.workflows import WorkflowRequest

    misplaced_values = yaml.safe_load(
        prepared.variants[0].config_path.read_text(encoding="utf-8")
    )
    misplaced_values["paths"]["run_dir"] = str(tmp_path / "notebook-output")
    misplaced = tmp_path / "misplaced-real.yaml"
    misplaced.write_text(yaml.safe_dump(misplaced_values, sort_keys=False), encoding="utf-8")
    normalized_plan = plan_job(
        WorkflowRequest("mae_train", misplaced),
        ClusterProfile("test", partition="gpu", gpus=1),
        run_id="normalized-real-run",
        output_root=output_root,
    )
    normalized_record = submit_plan(
        normalized_plan, registry, DryRunScheduler(), repository=repo_root
    )
    normalized = yaml.safe_load(
        Path(normalized_record.config_snapshot).read_text(encoding="utf-8")
    )
    assert normalized["paths"]["run_dir"] == str(normalized_plan.working_directory)
    assert normalized["paths"]["metrics"] == str(normalized_plan.metrics_path)
    assert normalized["paths"]["checkpoint"] == str(normalized_plan.checkpoint_path)


def test_soft_grid_rejects_arbitrary_or_excessive_dimensions():
    base = {"training": {"learning_rate": 0.001}}
    with pytest.raises(ValueError, match="Unsupported"):
        expand_soft_grid(base, {"parameters": {"runtime.command": ["whoami"]}})
    with pytest.raises(ValueError, match="exceeding"):
        expand_soft_grid(
            base,
            {
                "mode": "cartesian",
                "include_base": False,
                "max_runs": 2,
                "parameters": {
                    "training.learning_rate": [0.1, 0.01],
                    "training.scheduler": ["constant", "cosine"],
                },
            },
        )


def test_shallow_classifier_joins_partial_labels_and_exports_interpretation(tmp_path):
    rng = np.random.default_rng(4)
    label_ids = np.arange(100, 160, dtype=np.int64)
    labels = np.repeat(np.arange(3), 20)
    features = rng.normal(scale=0.25, size=(60, 8))
    features[:, :3] += np.eye(3)[labels] * 3
    embedding_path = export_embeddings(tmp_path / "embedding.npy", label_ids, features)
    label_frame = pd.DataFrame(
        {
            "label_id": np.concatenate((label_ids, [999, 1000])),
            "cell_type": [f"type-{value}" for value in labels] + ["rare", "rare"],
        }
    )
    labels_path = tmp_path / "labels.tsv"
    label_frame.to_csv(labels_path, sep="\t", index=False)
    output = tmp_path / "classifier"
    result = evaluate_embedding_classifier(
        embedding_path,
        labels_path,
        output_dir=output,
        folds=4,
        seed=9,
    )
    assert result.mean_accuracy > 0.9
    assert result.confusion.sum() == 60
    assert np.array_equal(np.sort(result.label_ids), label_ids)
    assert (output / "classifier_summary.json").is_file()
    assert (output / "confusion_matrix.tsv").is_file()
    assert (output / "cross_validated_predictions.tsv").is_file()
