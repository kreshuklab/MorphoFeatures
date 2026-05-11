"""Training orchestration for texture MorphoFeatures encoders."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any

from morphofeatures.config.loading import load_yaml_config
from morphofeatures.texture.loaders import CellLoaders
from morphofeatures.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def compile_criterion(criterion: str | dict[str, Any], **criterion_kwargs: Any) -> Any:
    """Create an Inferno/PyTorch criterion from config.

    Args:
        criterion: Criterion name or mapping of criterion names to kwargs.
        **criterion_kwargs: Extra kwargs passed to composite criteria.

    Returns:
        Instantiated criterion.
    """

    try:
        import inferno.extensions.criteria as inferno_criteria
        import torch.nn as nn
        from inferno.extensions.criteria import Criteria
    except ImportError as exc:
        raise ImportError("Texture training requires torch and inferno.") from exc

    if isinstance(criterion, str):
        criterion_class = getattr(nn, criterion, getattr(inferno_criteria, criterion, None))
        if criterion_class is None:
            raise ValueError(f"Unknown criterion: {criterion}")
        return criterion_class(**criterion_kwargs)
    if isinstance(criterion, dict):
        criteria = []
        for name, kwargs in criterion.items():
            criterion_class = getattr(nn, name, getattr(inferno_criteria, name, None))
            if criterion_class is None:
                raise ValueError(f"Unknown criterion: {name}")
            criteria.append(criterion_class(**kwargs))
        return Criteria(criteria, **criterion_kwargs)
    raise TypeError("criterion must be a string or dictionary.")


def set_up_training(project_directory: str | Path, config: dict[str, Any]) -> Any:
    """Build an Inferno trainer for texture model training."""

    try:
        import neurofire.models as models
        from inferno.trainers.basic import Trainer
        from inferno.trainers.callbacks.essentials import GarbageCollection, SaveAtBestValidationScore
        from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
        from inferno.trainers.callbacks.scheduling import AutoLR
    except ImportError as exc:
        raise ImportError("Texture training requires neurofire and inferno.") from exc

    model_name = config.get("model_name")
    model = getattr(models, model_name)(**config.get("model_kwargs", {}))
    criterion = compile_criterion(config.get("loss"), **config.get("loss_kwargs", {}))
    project_path = Path(project_directory)
    smoothness = config.get("smoothness", 0.95)

    logger.info("Building trainer.")
    trainer = (
        Trainer(model)
        .set_backprop_every(config.get("backprop_every", 1))
        .save_every((1000, "iterations"), to_directory=str(project_path / "Weights"))
        .build_criterion(criterion)
        .build_validation_criterion(criterion)
        .build_optimizer(**config.get("training_optimizer_kwargs", {}))
        .validate_every((100, "iterations"), for_num_iterations=20)
        .register_callback(SaveAtBestValidationScore(smoothness=smoothness, verbose=True))
        .register_callback(
            AutoLR(
                factor=0.98,
                patience="100 iterations",
                monitor="validation_loss_averaged",
                monitor_while="validating",
                monitor_momentum=smoothness,
                consider_improvement_with_respect_to="previous",
            )
        )
        .register_callback(GarbageCollection())
    )

    logger.info("Building tensorboard logger.")
    tensorboard = TensorboardLogger(
        log_scalars_every=(1, "iteration"),
        log_images_every=(100, "iterations"),
        send_image_at_channel_indices="mid",
    ).observe_state("validation", observe_while="validating")
    trainer.build_logger(tensorboard, log_directory=str(project_path / "Logs"))
    return trainer


def train_texture_model(
    project_directory: str | Path,
    train_configuration_file: str | Path,
    data_configuration_file: str | Path,
    devices: str = "0",
    from_checkpoint: bool = False,
) -> None:
    """Train a texture encoder from configuration files."""

    try:
        import torch
        import torch.nn as nn
        from inferno.trainers.basic import Trainer
    except ImportError as exc:
        raise ImportError("Texture training requires torch and inferno.") from exc

    project_path = Path(project_directory)
    logger.info("Loading config from %s.", train_configuration_file)
    config = load_yaml_config(train_configuration_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = devices

    if from_checkpoint:
        trainer = Trainer().load(from_directory=str(project_path), filename="Weights/checkpoint.pytorch")
    else:
        trainer = set_up_training(project_path, config)

    logger.info("Loading training and validation dataloaders from %s.", data_configuration_file)
    loader_factory = CellLoaders(data_configuration_file)
    train_loader, validation_loader = loader_factory.get_train_loaders()
    trainer.set_max_num_epochs(config.get("num_epochs", 10))
    trainer.bind_loader("train", train_loader).bind_loader("validate", validation_loader)

    if isinstance(trainer.model, torch.nn.DataParallel):
        trainer.model = trainer.model.module
    trainer.cuda([0])
    trainer.apex_opt_level = config.get("opt_level", "O1")
    trainer.mixed_precision = config.get("mixed_precision", False)

    if len(devices.split(",")) > 1:
        trainer.model = nn.DataParallel(trainer.model)

    trainer.pickle_module = "dill"
    logger.info("Starting texture training.")
    start = time.time()
    trainer.fit()
    elapsed = time.time() - start
    logger.info("Texture training took %.2f hours.", elapsed / 3600)


def main(argv: list[str] | None = None) -> None:
    """Run the legacy-compatible texture training CLI."""

    configure_logging()
    parser = argparse.ArgumentParser(description="Train a texture MorphoFeatures model.")
    parser.add_argument("project_directory", type=Path)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--from-checkpoint", action="store_true")
    args = parser.parse_args(argv)

    if not args.project_directory.exists():
        raise FileNotFoundError("Create a project directory with config files first.")

    train_texture_model(
        args.project_directory,
        args.project_directory / "train_config.yml",
        args.project_directory / "data_config.yml",
        devices=args.devices,
        from_checkpoint=args.from_checkpoint,
    )


if __name__ == "__main__":
    main()
