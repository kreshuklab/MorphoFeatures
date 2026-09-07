"""Train the legacy DeepGCN shape encoder with portable runtime behavior."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from morphofeatures.losses import nt_xent_loss
from morphofeatures.metrics import MetricWriter, configured_metrics_path
from morphofeatures.shape.loader import get_train_val_loaders
from morphofeatures.shape.network import DeepGCN
from morphofeatures.training_runtime import ExperimentLogger, resolve_device, save_checkpoint


class ShapeTrainer:
    def __init__(self, config):
        self.config = config
        self.device = resolve_device(str(config.get("device", "auto")))
        self.checkpoint_dir = Path(config["experiment_dir"]) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        loaders = get_train_val_loaders(config["data"], config["loader"])
        self.train_loader, self.val_loader = loaders["train"], loaders["val"]
        self.epoch, self.step, self.best_val_loss = 0, 0, None
        self.metric_writer = MetricWriter(
            configured_metrics_path(config, Path(config["experiment_dir"]) / "metrics.jsonl")
        )
        self._build_model()

    def _build_model(self):
        self.model = DeepGCN(**self.config["model"].get("kwargs", {})).to(self.device)
        parallel = bool(self.config.get("training", {}).get("data_parallel", True))
        if self.device.type == "cuda" and parallel and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        optimizer = self.config.get("optimizer", {"name": "AdamW", "kwargs": {"lr": 1e-3}})
        self.optimizer = getattr(torch.optim, optimizer["name"])(
            self.model.parameters(), **optimizer.get("kwargs", {})
        )
        scheduler = self.config.get("scheduler")
        self.scheduler = None
        if scheduler:
            self.scheduler = getattr(torch.optim.lr_scheduler, scheduler["name"])(
                self.optimizer, **scheduler.get("kwargs", {})
            )

    def _loss(self, projection):
        criterion = self.config.get("criterion", {})
        if criterion.get("name") and criterion["name"] != "NTXentLoss":
            try:
                from pytorch_metric_learning import losses
            except ImportError as error:
                raise RuntimeError("This criterion requires pytorch-metric-learning") from error
            labels = torch.arange(projection.size(0) // 2, device=self.device).repeat_interleave(2)
            return getattr(losses, criterion["name"])(**criterion.get("kwargs", {}))(
                projection, labels
            )
        temperature = float(criterion.get("kwargs", {}).get("temperature", 0.1))
        return nt_xent_loss(projection, temperature=temperature)

    def _forward(self, batch):
        projection, embedding = self.model(
            batch["points"].to(self.device, non_blocking=True),
            batch["features"].to(self.device, non_blocking=True),
        )
        return self._loss(projection), embedding

    def train_epoch(self, logger):
        self.model.train()
        losses = []
        for batch in tqdm(self.train_loader, desc="train", leave=False):
            loss, _ = self._forward(batch)
            if not torch.isfinite(loss):
                continue
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach()))
            logger.log({"training/loss": losses[-1]}, step=self.step)
            self.step += 1
        return float(sum(losses) / max(1, len(losses)))

    def validate_epoch(self):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="validate", leave=False):
                loss, _ = self._forward(batch)
                if torch.isfinite(loss):
                    losses.append(float(loss))
        return float(sum(losses) / max(1, len(losses)))

    def checkpoint(self, name, metrics):
        path = save_checkpoint(
            self.checkpoint_dir / name, self.model, optimizer=self.optimizer,
            scheduler=self.scheduler, epoch=self.epoch, step=self.step,
            config=self.config, metrics=metrics,
        )
        self.metric_writer.write(
            "checkpoint", path=str(path), epoch=self.epoch + 1, step=self.step, metrics=metrics
        )
        return path

    def run(self):
        training = self.config.get("training", {})
        wandb_config = self.config.get("wandb", {})
        self.metric_writer.write(
            "started", workflow="shape_train", device=str(self.device),
            epochs=int(training.get("epochs", 10)),
        )
        try:
            with ExperimentLogger(enabled=bool(wandb_config.get("enabled", False)),
                                  project=wandb_config.get("project", "MorphoFeatures"),
                                  config=self.config) as logger:
                for self.epoch in tqdm(range(int(training.get("epochs", 10))), desc="epochs"):
                    train_loss = self.train_epoch(logger)
                    metrics = {"train_loss": train_loss}
                    if self.epoch % int(training.get("validate_every", 1)) == 0:
                        validation_loss = self.validate_epoch()
                        metrics["validation_loss"] = validation_loss
                        logger.log(metrics, step=self.step)
                        if self.best_val_loss is None or validation_loss < self.best_val_loss:
                            self.best_val_loss = validation_loss
                            self.checkpoint("best.pt", metrics)
                    self.metric_writer.write(
                        "epoch", epoch=self.epoch + 1, step=self.step,
                        train_loss=train_loss,
                        validation_loss=metrics.get("validation_loss"),
                        learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                    )
                    if self.scheduler is not None:
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(metrics.get("validation_loss", train_loss))
                        else:
                            self.scheduler.step()
                    if (self.epoch + 1) % int(training.get("checkpoint_every", 1)) == 0:
                        self.checkpoint("epoch_{:04d}.pt".format(self.epoch + 1), metrics)
            self.metric_writer.write("completed", workflow="shape_train", epoch=self.epoch + 1)
        except Exception as error:
            self.metric_writer.write("failed", workflow="shape_train", error=str(error))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    base = args.config.resolve().parent
    for key in ("manifest", "root"):
        if config.get("data", {}).get(key):
            value = Path(config["data"][key])
            if not value.is_absolute():
                config["data"][key] = str(base / value)
    experiment = Path(config["experiment_dir"])
    if not experiment.is_absolute():
        config["experiment_dir"] = str(base / experiment)
    ShapeTrainer(config).run()


if __name__ == "__main__":
    main()
