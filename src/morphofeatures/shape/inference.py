"""Inference utilities for shape MorphoFeatures encoders."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from morphofeatures.config.loading import load_yaml_config


def load_model(config: dict[str, Any], device: Any) -> Any:
    """Load a DeepGCN model from a checkpoint config."""

    try:
        import torch
    except ImportError as exc:
        raise ImportError("Shape inference requires torch.") from exc

    from morphofeatures.shape.network import DeepGCN

    model = DeepGCN(**config.get("kwargs", {}))
    checkpoint = torch.load(config["checkpoint"], map_location=device)
    state_dict = checkpoint["model"]
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    if torch.cuda.device_count() > 1 and getattr(device, "type", str(device)) == "cuda":
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        model.cuda()
    else:
        model = model.to(device)
    return model


def generate_embeddings(model: Any, loader: Any, device: Any | None = None) -> np.ndarray:
    """Generate sorted ``label_id + embedding`` rows for all batches."""

    try:
        import torch
    except ImportError as exc:
        raise ImportError("Shape inference requires torch.") from exc

    model.eval()
    embeddings = []
    with torch.no_grad():
        for data in loader:
            if device is not None:
                data = {key: value.to(device) if hasattr(value, "to") else value for key, value in data.items()}
            ids = data["id"]
            _, hidden = model(data["points"], data["features"])
            embeddings.append(torch.cat((ids.unsqueeze(1).to(hidden.device), hidden), dim=1).detach().cpu())

    if not embeddings:
        raise ValueError("Cannot generate embeddings from an empty loader.")
    matrix = torch.cat(embeddings, dim=0).numpy()
    return matrix[matrix[:, 0].argsort()]


def run_inference(config_path: str | Path, save_to: str | Path | None = None) -> np.ndarray:
    """Run shape inference from a YAML config file."""

    try:
        import torch
    except ImportError as exc:
        raise ImportError("Shape inference requires torch.") from exc

    from morphofeatures.shape.loaders import get_simple_loader

    config = load_yaml_config(config_path)
    device = torch.device(config.get("device", "cpu"))
    loader = get_simple_loader(config["data"], config.get("loader", {}))
    model = load_model(config["model"], device)
    embeddings = generate_embeddings(model, loader, device)
    if save_to:
        np.save(save_to, embeddings)
    return embeddings


def main(argv: list[str] | None = None) -> None:
    """Run the shape embedding CLI."""

    parser = argparse.ArgumentParser(description="Generate shape embeddings.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--save-to", type=Path)
    args = parser.parse_args(argv)
    run_inference(args.config, args.save_to)


if __name__ == "__main__":
    main()
