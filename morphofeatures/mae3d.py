"""Pragmatic masked-autoencoder pathway for small 3D crops and patches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as error:  # pragma: no cover
    raise RuntimeError("The MAE pathway requires morphofeatures[modern-training]") from error

from morphofeatures.config import load_config
from morphofeatures.data.crop_storage import load_crop_array
from morphofeatures.data.io import export_embeddings
from morphofeatures.embedding_base import EmbeddingMethod
from morphofeatures.mae_contract import (
    GROUPED_PATCH_MAE_ARCHITECTURE_VERSION,
    LEGACY_MAE_ARCHITECTURE_VERSION,
    MAE_ARCHITECTURE_VERSION,
    configured_mae_architecture,
)
from morphofeatures.metrics import MetricWriter, configured_metrics_path
from morphofeatures.training_runtime import load_checkpoint, resolve_device, save_checkpoint


def _triple(value: Sequence[int]) -> tuple[int, int, int]:
    if len(value) != 3 or any(int(item) <= 0 for item in value):
        raise ValueError("Expected three positive z, y, x values")
    return int(value[0]), int(value[1]), int(value[2])


def sinusoidal_position_encoding(length: int, dimension: int, device, dtype):
    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    even_indices = torch.arange(0, dimension, 2, device=device, dtype=dtype)
    frequencies = torch.exp(-np.log(10000.0) * even_indices / max(1, dimension))
    encoding = torch.zeros((length, dimension), device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    if dimension > 1:
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding


def sinusoidal_position_encoding_3d(grid_shape, dimension: int, device, dtype):
    """Return flattened separable sinusoidal positions in explicit z, y, x order."""

    depth, height, width = _triple(grid_shape)
    if int(dimension) < 3:
        raise ValueError("3D positional encoding requires at least three dimensions")
    base, remainder = divmod(int(dimension), 3)
    axis_dimensions = [base + (axis < remainder) for axis in range(3)]
    z_position = sinusoidal_position_encoding(depth, axis_dimensions[0], device, dtype)
    y_position = sinusoidal_position_encoding(height, axis_dimensions[1], device, dtype)
    x_position = sinusoidal_position_encoding(width, axis_dimensions[2], device, dtype)
    z_grid = z_position[:, None, None, :].expand(depth, height, width, -1)
    y_grid = y_position[None, :, None, :].expand(depth, height, width, -1)
    x_grid = x_position[None, None, :, :].expand(depth, height, width, -1)
    return torch.cat((z_grid, y_grid, x_grid), dim=-1).reshape(-1, int(dimension))


@dataclass
class MAEOutput:
    reconstruction: torch.Tensor
    mask: torch.Tensor
    embedding: torch.Tensor
    loss: torch.Tensor
    visible_mean_baseline_loss: torch.Tensor | None = None


@dataclass
class GroupedPatchMAEOutput:
    """Output for an MAE whose tokens are complete texture patches."""

    reconstruction: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor
    embedding: torch.Tensor
    loss: torch.Tensor
    visible_mean_baseline_loss: torch.Tensor


class MaskedAutoencoder3D(nn.Module):
    def __init__(self, input_channels=1, patch_size=(4, 4, 4), embedding_dim=128,
                 encoder_depth=2, encoder_heads=4, decoder_dim=None):
        super().__init__()
        self.input_channels = int(input_channels)
        self.patch_size = _triple(patch_size)
        self.embedding_dim = int(embedding_dim)
        patch_values = self.input_channels * int(np.prod(self.patch_size))
        self.patch_embedding = nn.Conv3d(
            self.input_channels, self.embedding_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim, nhead=int(encoder_heads),
            dim_feedforward=self.embedding_dim * 4, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(encoder_depth))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        hidden = int(decoder_dim or self.embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden), nn.GELU(), nn.Linear(hidden, patch_values)
        )
        nn.init.normal_(self.mask_token, std=0.02)
        self.architecture_version = MAE_ARCHITECTURE_VERSION

    def patchify(self, inputs):
        batch, channels, depth, height, width = inputs.shape
        pz, py, px = self.patch_size
        if depth % pz or height % py or width % px:
            raise ValueError("Crop dimensions must be divisible by patch_size")
        patches = inputs.reshape(batch, channels, depth // pz, pz, height // py, py, width // px, px)
        return patches.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(batch, -1, channels * pz * py * px)

    def unpatchify(self, patches, output_shape):
        batch, channels, depth, height, width = output_shape
        pz, py, px = self.patch_size
        values = patches.reshape(batch, depth // pz, height // py, width // px, channels, pz, py, px)
        return values.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(output_shape)

    def _token_components(self, inputs):
        content = self.patch_embedding(inputs).flatten(2).transpose(1, 2)
        pz, py, px = self.patch_size
        grid_shape = (
            inputs.shape[-3] // pz,
            inputs.shape[-2] // py,
            inputs.shape[-1] // px,
        )
        position = sinusoidal_position_encoding_3d(
            grid_shape, content.shape[-1], content.device, content.dtype
        )
        if position.shape[0] != content.shape[1]:
            raise ValueError("Input shape must be divisible by patch_size")
        return content, position.unsqueeze(0)

    def _tokens(self, inputs):
        content, position = self._token_components(inputs)
        return content + position

    def encode(self, inputs):
        return self.encoder(self._tokens(inputs)).mean(dim=1)

    def mask_volume(self, mask, output_shape):
        """Expand a token mask into a boolean voxel mask for quality-control plots."""
        batch, _, depth, height, width = output_shape
        pz, py, px = self.patch_size
        expected = (depth // pz) * (height // py) * (width // px)
        if mask.shape != (batch, expected):
            raise ValueError("mask shape is incompatible with output_shape and patch_size")
        grid = mask.reshape(batch, depth // pz, height // py, width // px)
        return grid.repeat_interleave(pz, 1).repeat_interleave(py, 2).repeat_interleave(px, 3)

    def composite_reconstruction(self, inputs, output: MAEOutput):
        """Keep visible input patches and insert predictions only at masked patches."""
        target = self.patchify(inputs)
        predicted = self.patchify(output.reconstruction)
        combined = torch.where(output.mask.unsqueeze(-1), predicted, target)
        return self.unpatchify(combined, tuple(inputs.shape))

    def forward(self, inputs, mask_ratio: float = 0.75, loss_mask=None) -> MAEOutput:
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between zero and one")
        content, position = self._token_components(inputs)
        batch, count, _ = content.shape
        loss_weights = None
        if loss_mask is not None:
            if loss_mask.ndim == 4:
                loss_mask = loss_mask.unsqueeze(1)
            if loss_mask.shape != (batch, 1) + tuple(inputs.shape[-3:]):
                raise ValueError("loss_mask must have shape (batch, 1, z, y, x)")
            loss_weights = self.patchify(loss_mask.to(dtype=inputs.dtype))
            foreground_tokens = loss_weights.sum(dim=-1) > 0
            if not torch.all(foreground_tokens.any(dim=1)):
                raise ValueError("Each loss_mask must contain foreground voxels")
        masked_count = max(1, min(count - 1, int(round(count * mask_ratio))))
        mask_indices = torch.rand((batch, count), device=inputs.device).argsort(dim=1)[:, :masked_count]
        mask = torch.zeros((batch, count), dtype=torch.bool, device=inputs.device)
        mask.scatter_(1, mask_indices, True)
        if loss_weights is not None:
            for sample_index in range(batch):
                if not torch.any(mask[sample_index] & foreground_tokens[sample_index]):
                    add_index = torch.nonzero(foreground_tokens[sample_index], as_tuple=False)[0, 0]
                    remove_candidates = torch.nonzero(
                        mask[sample_index] & ~foreground_tokens[sample_index], as_tuple=False
                    )
                    if len(remove_candidates):
                        mask[sample_index, remove_candidates[0, 0]] = False
                    mask[sample_index, add_index] = True
        # Replace patch content, not the complete token. Positional information
        # must remain after masking or all hidden locations become indistinguishable.
        masked_content = torch.where(
            mask.unsqueeze(-1), self.mask_token.expand_as(content), content
        )
        encoded = self.encoder(masked_content + position)
        predicted_patches = self.decoder(encoded)
        target_patches = self.patchify(inputs)
        if loss_weights is None:
            objective_weights = torch.ones_like(target_patches)
        else:
            objective_weights = loss_weights
            if self.input_channels > 1:
                objective_weights = objective_weights.repeat(1, 1, self.input_channels)
        active_weights = objective_weights * mask.unsqueeze(-1)
        denominator = active_weights.sum()
        loss = ((predicted_patches - target_patches).square() * active_weights).sum() / denominator

        # A normalized MSE can look deceptively small. Report the loss of a
        # no-spatial-information predictor using each sample's visible mean.
        visible_weights = objective_weights * (~mask).unsqueeze(-1)
        visible_denominator = visible_weights.sum(dim=(1, 2), keepdim=True)
        visible_sum = (target_patches * visible_weights).sum(dim=(1, 2), keepdim=True)
        fallback_denominator = objective_weights.sum(dim=(1, 2), keepdim=True).clamp_min(1)
        fallback_mean = (
            target_patches * objective_weights
        ).sum(dim=(1, 2), keepdim=True) / fallback_denominator
        visible_mean = torch.where(
            visible_denominator > 0,
            visible_sum / visible_denominator.clamp_min(1),
            fallback_mean,
        )
        baseline_loss = (
            (visible_mean - target_patches).square() * active_weights
        ).sum() / denominator
        visible = encoded[~mask].reshape(batch, count - masked_count, self.embedding_dim)
        reconstruction = self.unpatchify(predicted_patches, tuple(inputs.shape))
        return MAEOutput(
            reconstruction,
            mask,
            visible.mean(dim=1),
            loss,
            baseline_loss,
        )


def coordinate_position_encoding_3d(coordinates, dimension: int):
    """Encode arbitrary relative z, y, x patch coordinates with sinusoids."""

    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape (batch, patches, 3)")
    if int(dimension) < 3:
        raise ValueError("3D positional encoding requires at least three dimensions")
    base, remainder = divmod(int(dimension), 3)
    axis_dimensions = [base + (axis < remainder) for axis in range(3)]
    encoded = []
    for axis, axis_dimension in enumerate(axis_dimensions):
        values = coordinates[..., axis : axis + 1]
        even_indices = torch.arange(
            0, axis_dimension, 2, device=coordinates.device, dtype=coordinates.dtype
        )
        frequencies = torch.exp(-np.log(10000.0) * even_indices / max(1, axis_dimension))
        axis_encoding = torch.zeros(
            coordinates.shape[:2] + (axis_dimension,),
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        axis_encoding[..., 0::2] = torch.sin(values * frequencies)
        if axis_dimension > 1:
            axis_encoding[..., 1::2] = torch.cos(
                values * frequencies[: axis_encoding[..., 1::2].shape[-1]]
            )
        encoded.append(axis_encoding)
    return torch.cat(encoded, dim=-1)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock3D(nn.Module):
    """A compact residual block for local 3D EM texture."""

    def __init__(self, channels: int):
        super().__init__()
        groups = _group_count(int(channels))
        self.layers = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.GELU()

    def forward(self, values):
        return self.activation(values + self.layers(values))


class ResNetPatchEncoder3D(nn.Module):
    """Hierarchical 3D convolutional encoder producing one token per patch.

    GroupNorm keeps the result well-defined for the small and variable number
    of visible patches encountered in grouped masked-autoencoder training.
    """

    def __init__(self, embedding_dim: int, channels=(16, 32, 64), blocks=(1, 1, 1)):
        super().__init__()
        channels = tuple(int(value) for value in channels)
        blocks = tuple(int(value) for value in blocks)
        if not channels or len(channels) != len(blocks):
            raise ValueError("resnet_channels and resnet_blocks must have equal nonzero length")
        if any(value <= 0 for value in channels + blocks):
            raise ValueError("ResNet channels and block counts must be positive")
        layers = [
            nn.Conv3d(1, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels[0]), channels[0]),
            nn.GELU(),
        ]
        for stage, (width, depth) in enumerate(zip(channels, blocks)):
            if stage:
                layers.extend(
                    (
                        nn.Conv3d(
                            channels[stage - 1],
                            width,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                        nn.GroupNorm(_group_count(width), width),
                        nn.GELU(),
                    )
                )
            layers.extend(ResidualBlock3D(width) for _ in range(depth))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Linear(channels[-1], int(embedding_dim))

    def forward(self, patches):
        batch, count = patches.shape[:2]
        values = patches.reshape(batch * count, *patches.shape[2:])
        values = self.pool(self.features(values)).flatten(start_dim=1)
        return self.projection(values).reshape(batch, count, -1)


class GroupedPatchMaskedAutoencoder3D(nn.Module):
    """MAE for a spatial group of complete 3D texture patches.

    Each token is one stored ``(z, y, x)`` EM patch. The encoder receives only
    visible patches from one biological parent and the decoder reconstructs a
    lower-resolution version of each hidden patch. This follows the verified
    sampling unit of the legacy nucleus MAE without depending on legacy code.
    """

    def __init__(
        self,
        *,
        input_shape=(32, 32, 32),
        reconstruction_shape=(8, 8, 8),
        embedding_dim=80,
        encoder_depth=4,
        encoder_heads=4,
        decoder_dim=80,
        decoder_depth=4,
        decoder_heads=4,
        norm_pix_loss=True,
        patch_encoder="linear",
        resnet_channels=(16, 32, 64),
        resnet_blocks=(1, 1, 1),
    ):
        super().__init__()
        self.input_shape = _triple(input_shape)
        self.reconstruction_shape = _triple(reconstruction_shape)
        self.embedding_dim = int(embedding_dim)
        self.norm_pix_loss = bool(norm_pix_loss)
        output_values = int(np.prod(self.reconstruction_shape))
        self.patch_encoder_name = str(patch_encoder)
        if self.patch_encoder_name == "linear":
            # Keep the historical Sequential layout so v3 linear checkpoints
            # retain their ``patch_encoder.1.*`` state-dict keys.
            self.patch_encoder = nn.Sequential(
                nn.Flatten(start_dim=2),
                nn.Linear(int(np.prod(self.input_shape)), self.embedding_dim),
            )
        elif self.patch_encoder_name == "resnet3d":
            self.patch_encoder = ResNetPatchEncoder3D(
                self.embedding_dim,
                channels=resnet_channels,
                blocks=resnet_blocks,
            )
        else:
            raise ValueError("patch_encoder must be 'linear' or 'resnet3d'")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=int(encoder_heads),
            dim_feedforward=self.embedding_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(encoder_depth))
        self.encoder_norm = nn.LayerNorm(self.embedding_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        self.decoder_dim = int(decoder_dim)
        self.decoder_projection = nn.Linear(self.embedding_dim, self.decoder_dim)
        self.decoder_cls_projection = nn.Linear(self.embedding_dim, self.decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.decoder_dim,
            nhead=int(decoder_heads),
            dim_feedforward=self.decoder_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=int(decoder_depth))
        self.decoder_norm = nn.LayerNorm(self.decoder_dim)
        self.decoder_prediction = nn.Linear(self.decoder_dim, output_values)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.architecture_version = GROUPED_PATCH_MAE_ARCHITECTURE_VERSION

    def _validate_inputs(self, patches, positions, valid):
        if patches.ndim == 5:
            patches = patches.unsqueeze(2)
        expected = (patches.shape[0], patches.shape[1], 1) + self.input_shape
        if tuple(patches.shape) != expected:
            raise ValueError(
                "grouped patches must have shape (batch, patches, 1, z, y, x)"
            )
        if positions.shape != patches.shape[:2] + (3,):
            raise ValueError("positions must have shape (batch, patches, 3)")
        if valid.shape != patches.shape[:2]:
            raise ValueError("valid must have shape (batch, patches)")
        valid = valid.to(dtype=torch.bool)
        if not torch.all(valid.sum(dim=1) >= 2):
            raise ValueError("Every grouped sample must contain at least two real patches")
        return patches, positions.to(dtype=patches.dtype), valid

    def _encode_visible(self, patches, positions, visible):
        batch = patches.shape[0]
        keep_counts = visible.sum(dim=1)
        max_keep = int(keep_counts.max().item())
        visible_patches = patches.new_zeros((batch, max_keep) + tuple(patches.shape[2:]))
        visible_positions = positions.new_zeros((batch, max_keep, 3))
        padding = torch.ones((batch, max_keep), dtype=torch.bool, device=patches.device)
        visible_indices = []
        for sample in range(batch):
            indices = torch.nonzero(visible[sample], as_tuple=False).flatten()
            visible_indices.append(indices)
            count = len(indices)
            visible_patches[sample, :count] = patches[sample, indices]
            visible_positions[sample, :count] = positions[sample, indices]
            padding[sample, :count] = False
        visible_tokens = self.patch_encoder(visible_patches)
        visible_tokens = visible_tokens + coordinate_position_encoding_3d(
            visible_positions, self.embedding_dim
        ).to(dtype=visible_tokens.dtype)
        cls = self.cls_token.to(dtype=visible_tokens.dtype).expand(batch, -1, -1)
        encoded = self.encoder_norm(
            self.encoder(
                torch.cat((cls, visible_tokens), dim=1),
                src_key_padding_mask=torch.cat(
                    (
                        torch.zeros((batch, 1), dtype=torch.bool, device=patches.device),
                        padding,
                    ),
                    dim=1,
                ),
            )
        )
        return encoded, visible_indices, padding

    def encode(self, patches, positions, valid):
        patches, positions, valid = self._validate_inputs(patches, positions, valid)
        encoded, _, _ = self._encode_visible(patches, positions, valid)
        return encoded[:, 0]

    def _target(self, patches):
        batch, count = patches.shape[:2]
        target = torch.nn.functional.adaptive_avg_pool3d(
            patches.reshape(batch * count, 1, *self.input_shape),
            self.reconstruction_shape,
        ).reshape(batch, count, -1)
        if not self.norm_pix_loss:
            ones = torch.ones_like(target[..., :1])
            zeros = torch.zeros_like(ones)
            return target, target, zeros, ones
        mean = target.mean(dim=-1, keepdim=True)
        std = target.var(dim=-1, keepdim=True, unbiased=False).add(1e-6).sqrt()
        return (target - mean) / std, target, mean, std

    def forward(self, patches, positions, valid, mask_ratio: float = 0.75):
        if not 0.0 < float(mask_ratio) < 1.0:
            raise ValueError("mask_ratio must be between zero and one")
        patches, positions, valid = self._validate_inputs(patches, positions, valid)
        batch, count = patches.shape[:2]
        mask = torch.zeros_like(valid)
        visible = torch.zeros_like(valid)
        for sample in range(batch):
            real_indices = torch.nonzero(valid[sample], as_tuple=False).flatten()
            masked_count = max(
                1,
                min(len(real_indices) - 1, int(round(len(real_indices) * float(mask_ratio)))),
            )
            shuffled = real_indices[torch.randperm(len(real_indices), device=patches.device)]
            mask[sample, shuffled[:masked_count]] = True
            visible[sample, shuffled[masked_count:]] = True
        encoded, visible_indices, padding = self._encode_visible(patches, positions, visible)

        projected_visible = self.decoder_projection(encoded[:, 1:])
        decoder_tokens = (
            self.mask_token.to(dtype=projected_visible.dtype).expand(batch, count, -1).clone()
        )
        for sample, indices in enumerate(visible_indices):
            decoder_tokens[sample, indices] = projected_visible[sample, : len(indices)]
        decoder_position = coordinate_position_encoding_3d(
            positions, self.decoder_dim
        ).to(dtype=decoder_tokens.dtype)
        decoder_tokens = decoder_tokens + decoder_position
        decoder_cls = self.decoder_cls_projection(encoded[:, :1])
        decoded = self.decoder_norm(
            self.decoder(
                torch.cat((decoder_cls, decoder_tokens), dim=1),
                src_key_padding_mask=torch.cat(
                    (torch.zeros_like(padding[:, :1]), ~valid), dim=1
                ),
            )
        )[:, 1:]
        prediction = self.decoder_prediction(decoded)
        objective_target, target, target_mean, target_std = self._target(patches)
        active = mask.unsqueeze(-1)
        loss = ((prediction - objective_target).square() * active).sum() / (
            active.sum() * prediction.shape[-1]
        )

        visible_weights = visible.unsqueeze(-1)
        visible_mean = (objective_target * visible_weights).sum(dim=1, keepdim=True) / (
            visible_weights.sum(dim=1, keepdim=True).clamp_min(1)
        )
        baseline_loss = ((visible_mean - objective_target).square() * active).sum() / (
            active.sum() * prediction.shape[-1]
        )
        display_prediction = prediction * target_std + target_mean
        return GroupedPatchMAEOutput(
            reconstruction=display_prediction.reshape(
                batch, count, 1, *self.reconstruction_shape
            ),
            target=target.reshape(batch, count, 1, *self.reconstruction_shape),
            mask=mask,
            embedding=encoded[:, 0],
            loss=loss,
            visible_mean_baseline_loss=baseline_loss,
        )


class MAEEmbeddingMethod(EmbeddingMethod):
    def __init__(self, model, optimizer, device, epochs=1, mask_ratio=0.75):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.epochs = int(epochs)
        self.mask_ratio = float(mask_ratio)

    def train(self, train_loader, validation_loader=None, metric_writer=None) -> dict[str, float]:
        metrics = {"loss": float("nan"), "train_loss": float("nan")}
        for epoch in range(self.epochs):
            self.model.train()
            losses = []
            baseline_losses = []
            for batch in train_loader:
                crops, loss_mask = self._training_batch(batch)
                output = self.model(
                    crops.to(self.device),
                    mask_ratio=self.mask_ratio,
                    loss_mask=None if loss_mask is None else loss_mask.to(self.device),
                )
                self.optimizer.zero_grad()
                output.loss.backward()
                self.optimizer.step()
                losses.append(float(output.loss.detach()))
                baseline_losses.append(float(output.visible_mean_baseline_loss.detach()))
            train_loss = float(np.mean(losses))
            train_baseline = float(np.mean(baseline_losses))
            metrics = {
                "loss": train_loss,
                "train_loss": train_loss,
                "train_visible_mean_baseline_loss": train_baseline,
                "train_improvement_over_visible_mean": 1.0 - train_loss / max(train_baseline, 1e-12),
            }
            if validation_loader is not None:
                self.model.eval()
                validation_losses = []
                validation_baselines = []
                with torch.no_grad():
                    for batch in validation_loader:
                        crops, loss_mask = self._training_batch(batch)
                        output = self.model(
                            crops.to(self.device),
                            mask_ratio=self.mask_ratio,
                            loss_mask=None if loss_mask is None else loss_mask.to(self.device),
                        )
                        validation_losses.append(float(output.loss.detach()))
                        validation_baselines.append(
                            float(output.visible_mean_baseline_loss.detach())
                        )
                metrics["validation_loss"] = float(np.mean(validation_losses))
                validation_baseline = float(np.mean(validation_baselines))
                metrics["validation_visible_mean_baseline_loss"] = validation_baseline
                metrics["validation_improvement_over_visible_mean"] = (
                    1.0
                    - metrics["validation_loss"] / max(validation_baseline, 1e-12)
                )
            if metric_writer is not None:
                metric_writer.write(
                    "epoch",
                    epoch=epoch + 1,
                    step=(epoch + 1) * len(train_loader),
                    train_loss=metrics["train_loss"],
                    validation_loss=metrics.get("validation_loss"),
                    train_visible_mean_baseline_loss=metrics[
                        "train_visible_mean_baseline_loss"
                    ],
                    validation_visible_mean_baseline_loss=metrics.get(
                        "validation_visible_mean_baseline_loss"
                    ),
                    train_improvement_over_visible_mean=metrics[
                        "train_improvement_over_visible_mean"
                    ],
                    validation_improvement_over_visible_mean=metrics.get(
                        "validation_improvement_over_visible_mean"
                    ),
                    learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                )
        return metrics

    @staticmethod
    def _training_batch(batch):
        if isinstance(batch, (tuple, list)):
            if len(batch) == 1:
                return batch[0], None
            if len(batch) == 2:
                return batch[0], batch[1]
            raise ValueError("MAE training batches contain crops and an optional loss mask")
        return batch, None

    def encode_cells(self, loader, metric_writer=None):
        self.model.eval()
        ids, embeddings, next_id = [], [], 0
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    batch_ids, crops = batch
                else:
                    crops = batch[0] if isinstance(batch, (tuple, list)) else batch
                    batch_ids = torch.arange(next_id, next_id + len(crops))
                next_id += len(crops)
                ids.append(batch_ids.detach().cpu().numpy())
                embeddings.append(self.model.encode(crops.to(self.device)).cpu().numpy())
                if metric_writer is not None:
                    metric_writer.write("encoding_progress", processed=next_id, total=len(loader.dataset))
        return np.concatenate(ids).astype(np.int64), np.concatenate(embeddings)


def _load_crops(config: dict, seed: int):
    mae_config, data_config = config.get("mae", {}), config.get("data", {})
    if data_config.get("crops"):
        crops = load_crop_array(data_config)
    else:
        shape = _triple(mae_config.get("input_shape", (16, 16, 16)))
        crops = np.random.default_rng(seed).normal(size=(8,) + shape).astype(np.float32)
    if crops.ndim == 4:
        crops = crops[:, None]
    if crops.ndim != 5:
        raise ValueError("MAE crops must have shape (n, z, y, x) or (n, c, z, y, x)")
    if np.issubdtype(crops.dtype, np.integer):
        limits = np.iinfo(crops.dtype)
        crops = (crops.astype(np.float32) - limits.min) / float(limits.max - limits.min)
    else:
        crops = crops.astype(np.float32, copy=False)
    if not np.all(np.isfinite(crops)):
        raise ValueError("MAE crops contain NaN or infinite values")
    return crops


def _load_label_ids(config: dict, count: int) -> np.ndarray:
    value = config.get("data", {}).get("label_ids")
    if value is None:
        return np.arange(1, count + 1, dtype=np.int64)
    if isinstance(value, (str, Path)):
        ids = load_crop_array(config["data"], "label_ids")
    else:
        ids = np.asarray(value)
    ids = np.asarray(ids)
    if ids.ndim != 1 or len(ids) != count or not np.all(np.isfinite(ids)):
        raise ValueError("data.label_ids must contain one finite ID per crop")
    if np.issubdtype(ids.dtype, np.integer):
        if len(ids) and (int(ids.min()) < 0 or int(ids.max()) > np.iinfo(np.int64).max):
            raise ValueError("data.label_ids must fit nonnegative int64")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("data.label_ids must be unique integer-valued labels")
        return ids.astype(np.int64)
    if not np.equal(ids, np.rint(ids)).all() or len(np.unique(ids)) != len(ids):
        raise ValueError("data.label_ids must be unique integer-valued labels")
    if np.any(np.abs(ids) > 2**53):
        raise ValueError("Large label IDs must use an integer array to avoid precision loss")
    return np.rint(ids).astype(np.int64)


def _load_loss_masks(config: dict, count: int, crop_shape: Sequence[int]):
    value = config.get("data", {}).get("loss_masks")
    if value is None:
        return None
    masks = (
        load_crop_array(config["data"], "loss_masks")
        if isinstance(value, (str, Path))
        else np.asarray(value)
    )
    if masks.ndim == 4:
        masks = masks[:, None]
    expected = (count, 1) + tuple(int(item) for item in crop_shape)
    if masks.shape != expected:
        raise ValueError("data.loss_masks must have shape (n, z, y, x) or (n, 1, z, y, x)")
    if not np.all(np.isfinite(masks)) or np.any(masks < 0):
        raise ValueError("data.loss_masks must contain finite non-negative values")
    if not np.all(np.any(masks > 0, axis=(1, 2, 3, 4))):
        raise ValueError("Each MAE crop must have foreground in data.loss_masks")
    return masks.astype(np.float32, copy=False)


def build_mae_model(config: dict) -> MaskedAutoencoder3D | GroupedPatchMaskedAutoencoder3D:
    """Build the configured MAE for training, encoding, or notebook inspection."""
    mae = config.get("mae", {})
    version = str(mae.get("architecture_version", MAE_ARCHITECTURE_VERSION))
    if version == GROUPED_PATCH_MAE_ARCHITECTURE_VERSION:
        return GroupedPatchMaskedAutoencoder3D(
            input_shape=mae.get("input_shape", (32, 32, 32)),
            reconstruction_shape=mae.get("reconstruction_shape", (8, 8, 8)),
            embedding_dim=int(mae.get("embedding_dim", 80)),
            encoder_depth=int(mae.get("encoder_depth", 4)),
            encoder_heads=int(mae.get("encoder_heads", 4)),
            decoder_dim=int(mae.get("decoder_dim", mae.get("embedding_dim", 80))),
            decoder_depth=int(mae.get("decoder_depth", 4)),
            decoder_heads=int(mae.get("decoder_heads", mae.get("encoder_heads", 4))),
            norm_pix_loss=bool(mae.get("norm_pix_loss", True)),
            patch_encoder=str(mae.get("patch_encoder", "linear")),
            resnet_channels=mae.get("resnet_channels", (16, 32, 64)),
            resnet_blocks=mae.get("resnet_blocks", (1, 1, 1)),
        )
    if version != MAE_ARCHITECTURE_VERSION:
        raise ValueError(
            f"Unsupported MAE architecture_version {version!r}; "
            f"expected {MAE_ARCHITECTURE_VERSION!r} or "
            f"{GROUPED_PATCH_MAE_ARCHITECTURE_VERSION!r}"
        )
    return MaskedAutoencoder3D(
        input_channels=int(mae.get("input_channels", 1)), patch_size=mae.get("patch_size", (4, 4, 4)),
        embedding_dim=int(mae.get("embedding_dim", 128)), encoder_depth=int(mae.get("encoder_depth", 2)),
        encoder_heads=int(mae.get("encoder_heads", 4)), decoder_dim=mae.get("decoder_dim"),
    )


def load_mae_checkpoint(
    path: Path,
    model,
    device="cpu",
    optimizer=None,
    scheduler=None,
    scaler=None,
    strict=True,
):
    """Load only checkpoints matching the model's explicit architecture contract."""

    def validate(payload):
        checkpoint_config = payload.get("config")
        version = configured_mae_architecture(checkpoint_config)
        expected = str(getattr(model, "architecture_version", MAE_ARCHITECTURE_VERSION))
        if version != expected:
            detail = (
                "does not declare an architecture version"
                if version == LEGACY_MAE_ARCHITECTURE_VERSION
                else f"declares {version!r}"
            )
            raise ValueError(
                f"MAE checkpoint {Path(path)} {detail}, but this model expects {expected!r}. "
                "The checkpoint cannot be resumed or encoded because its sampling unit or "
                "masking objective is incompatible. Retrain from scratch with the expected "
                "architecture."
            )
        if expected == GROUPED_PATCH_MAE_ARCHITECTURE_VERSION:
            configured = checkpoint_config if isinstance(checkpoint_config, dict) else {}
            observed_encoder = str(configured.get("mae", {}).get("patch_encoder", "linear"))
            expected_encoder = str(getattr(model, "patch_encoder_name", "linear"))
            if observed_encoder != expected_encoder:
                raise ValueError(
                    f"MAE checkpoint {Path(path)} uses patch_encoder={observed_encoder!r}, "
                    f"but this model expects {expected_encoder!r}. Patch encoders have different "
                    "state and inductive biases; train a new run or select the matching config."
                )

    return load_checkpoint(
        Path(path),
        model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        strict=strict,
        validate_payload=validate,
    )


def train_from_config(config_path: Path, output: Path | None = None) -> Path:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    config.setdefault("mae", {}).setdefault(
        "architecture_version", MAE_ARCHITECTURE_VERSION
    )
    if config.get("data", {}).get("source") == "n5_masked_patches":
        from morphofeatures.real_mae import resolve_real_mae_config, train_real_mae

        resolved = resolve_real_mae_config(
            config_path, profile=config.get("active_profile") or config.get("resolved_profile")
        )
        return train_real_mae(resolved, checkpoint=output)
    if config.get("data", {}).get("crops"):
        crops_path = Path(config["data"]["crops"])
        if not crops_path.is_absolute():
            config["data"]["crops"] = str(config_path.parent / crops_path)
    if isinstance(config.get("data", {}).get("label_ids"), str):
        labels_path = Path(config["data"]["label_ids"])
        if not labels_path.is_absolute():
            config["data"]["label_ids"] = str(config_path.parent / labels_path)
    if isinstance(config.get("data", {}).get("loss_masks"), str):
        masks_path = Path(config["data"]["loss_masks"])
        if not masks_path.is_absolute():
            config["data"]["loss_masks"] = str(config_path.parent / masks_path)
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(str(config.get("device", "auto")))
    crops = _load_crops(config, seed)
    loss_masks = _load_loss_masks(config, len(crops), crops.shape[-3:])
    training = config.get("training", {})
    validation_fraction = float(training.get("validation_fraction", 0.25))
    order = np.random.default_rng(seed).permutation(len(crops))
    validation_size = 0
    if len(crops) > 1 and validation_fraction > 0:
        validation_size = max(1, min(len(crops) - 1, int(round(len(crops) * validation_fraction))))
    validation_indices = order[:validation_size]
    training_indices = order[validation_size:]
    generator = torch.Generator().manual_seed(seed)
    training_tensors = [torch.from_numpy(crops[training_indices])]
    if loss_masks is not None:
        training_tensors.append(torch.from_numpy(loss_masks[training_indices]))
    loader = DataLoader(
        TensorDataset(*training_tensors),
        batch_size=int(training.get("batch_size", 2)),
        shuffle=True,
        generator=generator,
    )
    validation_loader = None
    if validation_size:
        validation_tensors = [torch.from_numpy(crops[validation_indices])]
        if loss_masks is not None:
            validation_tensors.append(torch.from_numpy(loss_masks[validation_indices]))
        validation_loader = DataLoader(
            TensorDataset(*validation_tensors),
            batch_size=int(training.get("batch_size", 2)),
            shuffle=False,
        )
    model = build_mae_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training.get("learning_rate", 1e-3)),
                                  weight_decay=float(training.get("weight_decay", 0.01)))
    method = MAEEmbeddingMethod(model, optimizer, device,
                                epochs=int(training.get("epochs", 2)),
                                mask_ratio=float(config.get("mae", {}).get("mask_ratio", 0.75)))
    destination = output or load_config(config_path).paths.output_root / "mae" / "checkpoint.pt"
    writer = MetricWriter(configured_metrics_path(config, Path(destination).parent / "metrics.jsonl"))
    writer.write(
        "started",
        workflow="mae_train",
        device=str(device),
        epochs=method.epochs,
        mae_architecture_version=MAE_ARCHITECTURE_VERSION,
    )
    try:
        metrics = method.train(loader, validation_loader, metric_writer=writer)
    except Exception as error:
        writer.write("failed", workflow="mae_train", error=str(error))
        raise
    save_checkpoint(
        destination,
        model,
        optimizer=optimizer,
        epoch=method.epochs,
        step=method.epochs * len(loader),
        config=config,
        metrics=metrics,
    )
    writer.write("checkpoint", path=str(destination), epoch=method.epochs, metrics=metrics)
    writer.write("completed", workflow="mae_train", checkpoint=str(destination), metrics=metrics)
    print("MAE checkpoint saved to {} with loss {:.6f}".format(destination, metrics["train_loss"]))
    return destination


def encode_from_config(config_path: Path, checkpoint: Path, output: Path) -> Path:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    config.setdefault("mae", {}).setdefault(
        "architecture_version", MAE_ARCHITECTURE_VERSION
    )
    if config.get("data", {}).get("source") == "n5_masked_patches":
        from morphofeatures.real_mae import encode_real_mae, resolve_real_mae_config

        resolved = resolve_real_mae_config(
            config_path, profile=config.get("active_profile") or config.get("resolved_profile")
        )
        return encode_real_mae(resolved, checkpoint, output=output)
    if config.get("data", {}).get("crops"):
        crops_path = Path(config["data"]["crops"])
        if not crops_path.is_absolute():
            config["data"]["crops"] = str(config_path.parent / crops_path)
    if isinstance(config.get("data", {}).get("label_ids"), str):
        labels_path = Path(config["data"]["label_ids"])
        if not labels_path.is_absolute():
            config["data"]["label_ids"] = str(config_path.parent / labels_path)
    device = resolve_device(str(config.get("device", "auto")))
    model = build_mae_model(config).to(device)
    load_mae_checkpoint(checkpoint, model, device=device)
    crops = _load_crops(config, int(config.get("seed", 42)))
    label_ids = _load_label_ids(config, len(crops))
    loader = DataLoader(TensorDataset(torch.from_numpy(label_ids), torch.from_numpy(crops)),
                        batch_size=int(config.get("inference", {}).get("batch_size", 4)))
    writer = MetricWriter(configured_metrics_path(config, Path(output).parent / "metrics.jsonl"))
    ids, features = MAEEmbeddingMethod(model, optimizer=None, device=device).encode_cells(loader, metric_writer=writer)
    return export_embeddings(output, ids, features)
