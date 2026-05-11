"""Unified encoder-training command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from morphofeatures.training.encoders import train_shape_encoder, train_texture_encoder


def _print_run(run: object) -> None:
    """Print an encoder training run description."""

    print(run.describe())  # type: ignore[attr-defined]


def main(argv: list[str] | None = None) -> None:
    """Run unified encoder training commands."""

    parser = argparse.ArgumentParser(description="Train new MorphoFeatures encoders.")
    subparsers = parser.add_subparsers(dest="encoder", required=True)

    shape = subparsers.add_parser("shape", help="Train a DeepGCN shape encoder.")
    shape.add_argument("--config", type=Path, required=True)
    shape.add_argument("--dry-run", action="store_true")
    shape.add_argument("--strict-paths", action="store_true")

    texture = subparsers.add_parser("texture", help="Train a texture encoder.")
    texture.add_argument("project_directory", type=Path)
    texture.add_argument("--train-config", type=Path, default=None)
    texture.add_argument("--data-config", type=Path, default=None)
    texture.add_argument("--devices", type=str, default="0")
    texture.add_argument("--from-checkpoint", action="store_true")
    texture.add_argument("--dry-run", action="store_true")
    texture.add_argument("--strict-paths", action="store_true")

    args = parser.parse_args(argv)
    if args.encoder == "shape":
        run = train_shape_encoder(args.config, dry_run=args.dry_run, strict_paths=args.strict_paths)
    elif args.encoder == "texture":
        run = train_texture_encoder(
            args.project_directory,
            train_config=args.train_config,
            data_config=args.data_config,
            devices=args.devices,
            from_checkpoint=args.from_checkpoint,
            dry_run=args.dry_run,
            strict_paths=args.strict_paths,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unknown encoder type: {args.encoder}")
    _print_run(run)


if __name__ == "__main__":
    main()
