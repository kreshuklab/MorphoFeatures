"""Encode texture crops and aggregate patch-level features."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

from morphofeatures.data.io import export_embeddings
from morphofeatures.embedding_base import aggregate_patch_embeddings
from morphofeatures.texture.cell_loader import CellLoaders
from morphofeatures.texture.models import LegacyTextureAutoencoder3D
from morphofeatures.training_runtime import load_checkpoint, resolve_device


def predict(model, loader, path_to_save, device='cpu'):
    pred_loader = loader.get_predict_loaders()
    labels = np.asarray(pred_loader.dataset.indices, dtype=np.int64)
    encoded = []
    with torch.no_grad():
        for samples in pred_loader:
            prediction = model(samples.to(device), just_encode=True).cpu().numpy()
            if np.any(np.isnan(prediction)):
                warnings.warn("NaN spotted in predictions", stacklevel=2)
            encoded.append(prediction)
    features = np.concatenate(encoded)
    positions = getattr(pred_loader.dataset, 'positions', None)
    if positions is not None and len(positions) == len(features):
        labels, features = aggregate_patch_embeddings(np.asarray(positions[:, 0]), features)
    else:
        labels = labels[:len(features)]
    export_embeddings(Path(path_to_save), labels, features)
    return labels, features


def predict_patches(model, loader, path_to_save, device='cpu'):
    assert loader.config.get('texture_contrastive')
    pred_loader = loader.get_predict_loaders()
    predictions = []
    with torch.no_grad():
        for samples in pred_loader:
            prediction = model(samples.to(device), just_encode=True).cpu().numpy()
            if np.any(np.isnan(prediction)):
                warnings.warn("NaN spotted in predictions", stacklevel=2)
            predictions.append(prediction)
    ids = pred_loader.dataset.positions[:, 0].astype('int64')
    predictions = np.concatenate(predictions)
    destination = Path(path_to_save)
    if destination.suffix == '.z5':
        try:
            import z5py
        except ImportError as error:
            raise RuntimeError('Writing historical .z5 patch files requires z5py') from error
        container = z5py.File(str(destination))
        container.create_dataset('preds', data=predictions, compression='gzip')
        container.create_dataset('ids', data=ids, compression='gzip')
    else:
        np.savez_compressed(destination, preds=predictions, ids=ids)
    return ids, predictions


def aggregate_patches(patch_path, output=None):
    source = Path(patch_path)
    if source.suffix == '.z5':
        try:
            import z5py
        except ImportError as error:
            raise RuntimeError('Reading historical .z5 patch files requires z5py') from error
        container = z5py.File(str(source), 'r')
        ids, predictions = container['ids'][:], container['preds'][:]
    else:
        archive = np.load(source)
        ids, predictions = archive['ids'], archive['preds']
    labels, features = aggregate_patch_embeddings(ids, predictions)
    destination = Path(output) if output else source.with_name('avg_encoded_patches_aggr.npy')
    export_embeddings(destination, labels, features)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path, help='experiment directory')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--save-patches', action='store_true')
    parser.add_argument('--aggregate-patches', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    with (args.path / 'train_config.yml').open('r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream) or {}
    model = LegacyTextureAutoencoder3D(**config.get('model_kwargs', {})).to(device)
    checkpoint = args.checkpoint or args.path / 'checkpoints' / 'best.pt'
    load_checkpoint(checkpoint, model, device=device)
    model.eval()
    if args.save_patches:
        if args.output and not args.aggregate_patches:
            destination = args.output
        elif args.output:
            destination = args.output.with_suffix('.patches.npz')
        else:
            destination = args.path / 'encoded_patches.npz'
        predict_patches(model, CellLoaders(args.path / 'test_config_patches.yml'), destination, device)
        if args.aggregate_patches:
            aggregate_patches(destination, output=args.output)
    else:
        predict(model, CellLoaders(args.path / 'test_config.yml'),
                args.output or args.path / 'avg_encoded.npy', device)


if __name__ == '__main__':
    main()
