"""Training orchestration for shape MorphoFeatures encoders."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from morphofeatures.config.loading import load_yaml_config


class ShapeTrainer:
    """Train a DeepGCN shape encoder with contrastive metric-learning losses.

    Args:
        config: Training configuration dictionary.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize trainer state from a configuration dictionary."""

        try:
            import torch
        except ImportError as exc:
            raise ImportError("Shape training requires torch.") from exc

        self.torch = torch
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        self.checkpoint_dir = Path(config["experiment_dir"]) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.build_loaders()
        self.reset()

    def reset(self) -> None:
        """Reset model, optimizer, and counters."""

        self.build_model()
        self.best_val_loss: float | None = None
        self.epoch = 0
        self.step = 0

    def build_loaders(self) -> None:
        """Build training and validation dataloaders from configuration."""

        from morphofeatures.shape.loaders import get_train_val_loaders

        loaders = get_train_val_loaders(self.config["data"], self.config.get("loader", {}))
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]

    def build_model(self) -> None:
        """Build model, optimizer, optional scheduler, and loss criterion."""

        import torch
        from pytorch_metric_learning import losses

        from morphofeatures.shape.network import DeepGCN

        self.model = DeepGCN(**self.config["model"].get("kwargs", {}))
        if torch.cuda.device_count() > 1 and self.device.type == "cuda":
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))
            self.model.cuda()
        else:
            self.model = self.model.to(self.device)

        self.optimizer = getattr(torch.optim, self.config["optimizer"]["name"])(
            self.model.parameters(),
            **self.config["optimizer"].get("kwargs", {}),
        )
        criterion_config = self.config["criterion"]
        self.criterion = getattr(losses, criterion_config["name"])(**criterion_config.get("kwargs", {}))

        scheduler_config = self.config.get("scheduler")
        if scheduler_config:
            self.scheduler = getattr(torch.optim.lr_scheduler, scheduler_config["name"])(
                self.optimizer,
                **scheduler_config.get("kwargs", {}),
            )
        else:
            self.scheduler = None

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move tensor batch values onto the configured device."""

        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }

    def checkpoint(self, force: bool = True) -> Path | None:
        """Save a checkpoint when requested by force or epoch cadence."""

        save = force or (self.epoch % self.config["training"].get("checkpoint_every", 1) == 0)
        if not save:
            return None

        info = {
            "epoch": self.epoch,
            "iteration": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config/model/name": self.config["model"].get("name", "DeepGCN"),
            "config/model/kwargs": self.config["model"].get("kwargs", {}),
        }
        if self.scheduler is not None:
            info["scheduler"] = self.scheduler.state_dict()
        checkpoint_name = f"best_ckpt_iter_{self.step}.pt" if force else f"ckpt_iter_{self.step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        self.torch.save(info, checkpoint_path)
        return checkpoint_path

    def train_epoch(self) -> None:
        """Run one training epoch over the training loader."""

        import torch
        from tqdm import tqdm

        self.model.train()
        for batch in tqdm(self.train_loader, desc="Iteration"):
            data = self._move_batch_to_device(batch)
            out, _ = self.model(data["points"], data["features"])
            labels = torch.arange(out.size(0) // 2).repeat_interleave(2).to(self.device)
            loss = self.criterion(out, labels)
            if torch.isnan(loss).item():
                print(f"Skipping NaN loss at step {self.step}.")
                continue

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self._log({"training/loss": loss.item()}, step=self.step)
            self.step += 1

    def validate_epoch(self) -> float | None:
        """Run validation when scheduled and save a best checkpoint."""

        import torch
        from tqdm import tqdm

        validate_every = self.config["training"].get("validate_every", 1)
        if self.epoch % validate_every != 0:
            return None

        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                data = self._move_batch_to_device(batch)
                out, _ = self.model(data["points"], data["features"])
                labels = torch.arange(out.size(0) // 2).repeat_interleave(2).to(self.device)
                total_loss += self.criterion(out, labels).item()

        average_loss = total_loss / max(1, len(self.val_loader))
        if self.best_val_loss is None or average_loss < self.best_val_loss:
            self.best_val_loss = average_loss
            self.checkpoint(True)
        self._log({"validation/average_loss": average_loss}, step=self.step)
        return average_loss

    def train(self) -> None:
        """Run the configured training loop."""

        from tqdm import tqdm

        for epoch_num in tqdm(range(self.config["training"]["epochs"]), desc="Epochs"):
            self.train_epoch()
            self.validate_epoch()
            if self.scheduler is not None:
                self.scheduler.step()
            self.checkpoint(False)
            self.epoch = epoch_num + 1

    def run(self) -> None:
        """Run training inside a Weights & Biases context when available."""

        try:
            import wandb
        except ImportError:
            self.validate_epoch()
            self.train()
            return

        with wandb.init(project=self.config.get("wandb_project", "MorphoFeatures")):
            self.validate_epoch()
            self.train()

    def _log(self, values: dict[str, float], step: int) -> None:
        """Log metrics to Weights & Biases when it is active."""

        try:
            import wandb

            if wandb.run is not None:
                wandb.log(values, step=step)
        except ImportError:
            return


def main(argv: list[str] | None = None) -> None:
    """Run the shape training CLI."""

    parser = argparse.ArgumentParser(description="Train a shape MorphoFeatures model.")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    trainer = ShapeTrainer(load_yaml_config(args.config))
    trainer.run()


if __name__ == "__main__":
    main()
