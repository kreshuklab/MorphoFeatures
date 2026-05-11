"""Inference utilities for texture MorphoFeatures encoders."""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from morphofeatures.texture.loaders import CellLoaders


def predict(model: Any, loader: CellLoaders, path_to_save: str | Path) -> None:
    """Generate one embedding per cell and save it as ``label_id + features``."""

    try:
        import torch
    except ImportError as exc:
        raise ImportError("Texture prediction requires torch.") from exc

    pred_loader = loader.get_predict_loaders()
    labels = pred_loader.dataset.indices
    encoded = []
    with torch.no_grad():
        for batch_index, samples in enumerate(pred_loader):
            print(f"Batch {batch_index}")
            if loader.config.get("texture_contrastive", False):
                prediction = model(samples[0].cuda(), just_encode=True).cpu().numpy()
                if np.any(np.isnan(prediction)):
                    warnings.warn("NaN spotted in predictions", stacklevel=2)
                encoded.append(np.nanmean(prediction, axis=0))
            else:
                prediction = model(samples.cuda(), just_encode=True).cpu().numpy()
                encoded.extend(list(prediction))
    np.savetxt(path_to_save, np.c_[labels, np.asarray(encoded)])


def predict_patches(model: Any, loader: CellLoaders, path_to_save: str | Path, feature_size: int = 80) -> None:
    """Generate one embedding per texture patch and save it in a z5 container."""

    try:
        import torch
        import z5py
    except ImportError as exc:
        raise ImportError("Patch prediction requires torch and z5py.") from exc

    if not loader.config.get("texture_contrastive", False):
        raise ValueError("Patch prediction requires texture_contrastive=True.")

    pred_loader = loader.get_predict_loaders()
    batch_size = pred_loader.batch_size
    positions = pred_loader.dataset.positions
    output = z5py.File(str(path_to_save))
    predictions_ds = output.create_dataset(
        "preds",
        shape=(positions.shape[0], feature_size),
        dtype="float64",
        compression="gzip",
    )
    with torch.no_grad():
        for batch_index, samples in enumerate(pred_loader):
            print(f"Batch {batch_index}")
            prediction = model(samples.cuda(), just_encode=True).cpu().numpy()
            if np.any(np.isnan(prediction)):
                warnings.warn("NaN spotted in predictions", stacklevel=2)
            predictions_ds[batch_index * batch_size : batch_index * batch_size + prediction.shape[0]] = prediction
    ids = pred_loader.dataset.positions[:, 0].astype("int64")
    output.create_dataset("ids", data=ids, dtype="int64", compression="gzip")


def aggregate_patches(z5_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Average patch embeddings into one embedding per cell."""

    try:
        import z5py
    except ImportError as exc:
        raise ImportError("Aggregating patch predictions requires z5py.") from exc

    input_path = Path(z5_path)
    destination = Path(output_path) if output_path is not None else input_path.with_name("avg_encoded_patches_aggr.np")
    handle = z5py.File(str(input_path))
    ids = handle["ids"][:]
    labels = np.unique(ids)
    aggregated_features = np.zeros((labels.shape[0], handle["preds"].shape[1]))
    for index, label_id in enumerate(labels):
        patch_ids = np.where(ids == label_id)[0]
        expected_ids = np.arange(patch_ids[0], patch_ids[-1] + 1)
        if not np.all(patch_ids == expected_ids):
            raise ValueError(f"Patch IDs for label {label_id} are not contiguous.")
        aggregated_features[index] = np.mean(handle["preds"][slice(patch_ids[0], patch_ids[-1] + 1)], axis=0)
    np.savetxt(destination, np.c_[labels, aggregated_features])
    return destination


def run_prediction(path: str | Path, devices: str = "0", save_patches: bool = False, aggregate: bool = False) -> Path:
    """Run texture inference from an experiment directory."""

    try:
        import torch
        from inferno.trainers.basic import Trainer
    except ImportError as exc:
        raise ImportError("Texture prediction requires torch and inferno.") from exc

    experiment_path = Path(path)
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    encoded_path = experiment_path / "avg_encoded.np"
    if save_patches:
        encoded_path = experiment_path / "avg_encoded_patches.z5"

    if not encoded_path.exists():
        model_path = experiment_path / "Weights"
        best_model = Trainer().load(from_directory=str(model_path), best=True).model
        if len(devices.split(",")) == 1 and isinstance(best_model, torch.nn.DataParallel):
            best_model = best_model.module
        elif len(devices.split(",")) > 1 and not isinstance(best_model, torch.nn.DataParallel):
            best_model = torch.nn.DataParallel(best_model)

        if save_patches:
            cell_loader = CellLoaders(experiment_path / "test_config_patches.yml")
            predict_patches(best_model, cell_loader, encoded_path)
        else:
            cell_loader = CellLoaders(experiment_path / "test_config.yml")
            predict(best_model, cell_loader, encoded_path)

    if save_patches and aggregate:
        aggregate_patches(encoded_path)
    return encoded_path


def main(argv: list[str] | None = None) -> None:
    """Run the texture prediction CLI."""

    parser = argparse.ArgumentParser(description="Generate texture embeddings.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--save-patches", action="store_true")
    parser.add_argument("--aggregate-patches", action="store_true")
    args = parser.parse_args(argv)
    output = run_prediction(args.path, args.devices, args.save_patches, args.aggregate_patches)
    print(f"Saved embeddings at {output}")


if __name__ == "__main__":
    main()
