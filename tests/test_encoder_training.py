"""Tests for high-level encoder training orchestration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from _path import add_src_to_path

add_src_to_path()

from morphofeatures.training.encoders import train_shape_encoder, train_texture_encoder


class EncoderTrainingTests(unittest.TestCase):
    """Validate dry-run training orchestration without optional GPU dependencies."""

    def test_shape_training_dry_run_describes_checkpoint_dir(self) -> None:
        """Shape dry-run validates config and returns expected artifacts."""

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            npz_path = tmp_path / "shape.npz"
            np.savez(npz_path, points=np.zeros((2, 4, 3), dtype=np.float32), ids=np.array([1, 2]))
            config_path = tmp_path / "shape.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "experiment_dir": str(tmp_path / "run"),
                        "device": "cpu",
                        "data": {"npz_path": str(npz_path)},
                        "loader": {"batch_size": 1},
                        "model": {"name": "DeepGCN", "kwargs": {}},
                        "optimizer": {"name": "Adam", "kwargs": {"lr": 0.001}},
                        "criterion": {"name": "NTXentLoss", "kwargs": {}},
                        "training": {"epochs": 1},
                    }
                ),
                encoding="utf-8",
            )
            run = train_shape_encoder(config_path, dry_run=True, strict_paths=True)

        self.assertEqual(run.kind, "shape")
        self.assertEqual(run.checkpoint_dir.name, "checkpoints")
        self.assertIn("morphofeatures-shape embed", run.embedding_command)

    def test_texture_training_dry_run_describes_weights_dir(self) -> None:
        """Texture dry-run validates legacy project config files."""

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "train_config.yml").write_text(
                yaml.safe_dump(
                    {
                        "model_name": "ExampleModel",
                        "model_kwargs": {},
                        "loss": "MSELoss",
                        "loss_kwargs": {},
                        "training_optimizer_kwargs": {"optimizer": "Adam", "lr": 0.001},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "data_config.yml").write_text(
                yaml.safe_dump(
                    {
                        "contrastive": True,
                        "data": {"version": "1.0.1"},
                        "loader_config": {"batch_size": 1},
                        "val_loader_config": {"batch_size": 1},
                    }
                ),
                encoding="utf-8",
            )
            run = train_texture_encoder(run_dir, dry_run=True)

        self.assertEqual(run.kind, "texture")
        self.assertEqual(run.checkpoint_dir.name, "Weights")
        self.assertIn("morphofeatures-texture predict", run.embedding_command)


if __name__ == "__main__":
    unittest.main()
