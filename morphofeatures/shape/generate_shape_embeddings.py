"""Generate label_id-first shape embeddings from every inference batch."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from morphofeatures.data.io import export_embeddings
from morphofeatures.shape.loader import get_simple_loader
from morphofeatures.shape.network import DeepGCN
from morphofeatures.training_runtime import load_checkpoint, resolve_device


def load_model(config, device):
    model = DeepGCN(**config.get("kwargs", {})).to(device)
    load_checkpoint(Path(config["checkpoint"]), model, device=device)
    if device.type == "cuda" and bool(config.get("data_parallel", True)) and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


def generate_embeddings(model, loader, device):
    model.eval()
    ids, embeddings = [], []
    with torch.no_grad():
        for data in loader:
            _, hidden = model(data["points"].to(device), data["features"].to(device))
            ids.append(data["id"].detach().cpu().numpy())
            embeddings.append(hidden.detach().cpu().numpy())
    if not ids:
        raise ValueError("Inference loader produced no batches")
    all_ids, all_embeddings = np.concatenate(ids), np.concatenate(embeddings)
    order = np.argsort(all_ids, kind="stable")
    return all_ids[order], all_embeddings[order]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--save-to", type=Path, required=True)
    args = parser.parse_args(argv)
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    base = args.config.resolve().parent
    for key in ("manifest", "root"):
        if config.get("data", {}).get(key):
            value = Path(config["data"][key])
            if not value.is_absolute():
                config["data"][key] = str(base / value)
    checkpoint = Path(config["model"]["checkpoint"])
    if not checkpoint.is_absolute():
        config["model"]["checkpoint"] = str(base / checkpoint)
    device = resolve_device(str(config.get("device", "auto")))
    ids, embeddings = generate_embeddings(
        load_model(config["model"], device),
        get_simple_loader(config["data"], config["loader"]),
        device,
    )
    export_embeddings(args.save_to, ids, embeddings)


if __name__ == "__main__":
    main()
