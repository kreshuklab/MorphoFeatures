"""Train the maintained legacy-style 3D texture autoencoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from morphofeatures.losses import texture_objective
from morphofeatures.metrics import MetricWriter, configured_metrics_path
from morphofeatures.texture.cell_loader import CellLoaders
from morphofeatures.texture.models import LegacyTextureAutoencoder3D
from morphofeatures.training_runtime import (
    ExperimentLogger,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
)


def _batch_loss(model, batch, device, loss_config):
    inputs, targets = batch
    if isinstance(targets, (tuple, list)):
        targets = targets[0]
    inputs, targets = inputs.to(device), targets.to(device)
    reconstruction, embedding, projection = model(inputs)
    return texture_objective(reconstruction, targets, projection, embedding, **loss_config)


def training(project_directory, train_configuration_file, data_configuration_file,
             device='auto', from_checkpoint=False, resume_checkpoint=None):
    project = Path(project_directory)
    with Path(train_configuration_file).open('r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream) or {}
    runtime_device = resolve_device(device)
    loaders = CellLoaders(data_configuration_file)
    train_loader, validation_loader = loaders.get_train_loaders()
    model = LegacyTextureAutoencoder3D(**config.get('model_kwargs', {})).to(runtime_device)
    optimizer_config = config.get('optimizer', {'name': 'AdamW', 'kwargs': {'lr': 1e-3}})
    optimizer = getattr(torch.optim, optimizer_config['name'])(
        model.parameters(), **optimizer_config.get('kwargs', {}))
    start_epoch = 0
    checkpoint_path = project / 'checkpoints' / 'last.pt'
    source_checkpoint = Path(resume_checkpoint) if resume_checkpoint else checkpoint_path
    if from_checkpoint and not source_checkpoint.exists():
        raise FileNotFoundError('Resume checkpoint does not exist: {}'.format(source_checkpoint))
    if from_checkpoint:
        payload = load_checkpoint(source_checkpoint, model, device=runtime_device, optimizer=optimizer)
        start_epoch = int(payload.get('epoch', 0)) + 1
    best_loss = float('inf')
    loss_config = config.get('loss_kwargs', {})
    wandb_config = config.get('wandb', {})
    metric_writer = MetricWriter(configured_metrics_path(config, project / 'metrics.jsonl'))
    metric_writer.write('started', workflow='texture_train', device=str(runtime_device),
                        epochs=int(config.get('num_epochs', 10)))
    try:
        with ExperimentLogger(enabled=bool(wandb_config.get('enabled', False)),
                              project=wandb_config.get('project', 'MorphoFeatures'),
                              config=config) as logger:
            for epoch in range(start_epoch, int(config.get('num_epochs', 10))):
                model.train()
                train_metrics = []
                for batch in train_loader:
                    loss, metrics = _batch_loss(model, batch, runtime_device, loss_config)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    train_metrics.append(metrics['loss'])
                model.eval()
                validation_metrics = []
                with torch.no_grad():
                    for batch in validation_loader:
                        _, metrics = _batch_loss(model, batch, runtime_device, loss_config)
                        validation_metrics.append(metrics['loss'])
                values = {'training/loss': sum(train_metrics) / max(1, len(train_metrics)),
                          'validation/loss': sum(validation_metrics) / max(1, len(validation_metrics))}
                logger.log(values, step=epoch)
                save_checkpoint(checkpoint_path, model, optimizer=optimizer, epoch=epoch,
                                config=config, metrics=values)
                metric_writer.write('checkpoint', path=str(checkpoint_path), epoch=epoch + 1,
                                    metrics=values)
                if values['validation/loss'] < best_loss:
                    best_loss = values['validation/loss']
                    best_path = project / 'checkpoints' / 'best.pt'
                    save_checkpoint(best_path, model, optimizer=optimizer,
                                    epoch=epoch, config=config, metrics=values)
                    metric_writer.write('checkpoint', path=str(best_path), epoch=epoch + 1,
                                        metrics=values)
                metric_writer.write(
                    'epoch', epoch=epoch + 1,
                    train_loss=values['training/loss'],
                    validation_loss=values['validation/loss'],
                    learning_rate=float(optimizer.param_groups[0]['lr']),
                )
                print('epoch={} training_loss={:.6f} validation_loss={:.6f}'.format(
                    epoch + 1, values['training/loss'], values['validation/loss']))
        metric_writer.write('completed', workflow='texture_train', checkpoint=str(project / 'checkpoints' / 'best.pt'))
    except Exception as error:
        metric_writer.write('failed', workflow='texture_train', error=str(error))
        raise
    return project / 'checkpoints' / 'best.pt'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project_directory', type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--from-checkpoint', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args(argv)
    training(args.project_directory, args.project_directory / 'train_config.yml',
             args.project_directory / 'data_config.yml', args.device, args.from_checkpoint,
             args.checkpoint)


if __name__ == '__main__':
    main()
