"""Pragmatic masked-autoencoder pathway for small 3D crops and patches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import yaml

try:
    import torch
    from torch import nn
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as error:  # pragma: no cover
    raise RuntimeError("The MAE pathway requires morphofeatures[modern-training]") from error

from morphofeatures.config import load_config
from morphofeatures.data.io import export_embeddings
from morphofeatures.embedding_base import EmbeddingMethod
from morphofeatures.training_runtime import load_checkpoint, resolve_device, save_checkpoint


def _triple(value: Sequence[int]) -> Tuple[int, int, int]:
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


@dataclass
class MAEOutput:
    reconstruction: torch.Tensor
    mask: torch.Tensor
    embedding: torch.Tensor
    loss: torch.Tensor


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

    def _tokens(self, inputs):
        tokens = self.patch_embedding(inputs).flatten(2).transpose(1, 2)
        position = sinusoidal_position_encoding(tokens.shape[1], tokens.shape[2], tokens.device, tokens.dtype)
        return tokens + position.unsqueeze(0)

    def encode(self, inputs):
        return self.encoder(self._tokens(inputs)).mean(dim=1)

    def forward(self, inputs, mask_ratio: float = 0.75) -> MAEOutput:
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between zero and one")
        tokens = self._tokens(inputs)
        batch, count, _ = tokens.shape
        masked_count = max(1, min(count - 1, int(round(count * mask_ratio))))
        mask_indices = torch.rand((batch, count), device=inputs.device).argsort(dim=1)[:, :masked_count]
        mask = torch.zeros((batch, count), dtype=torch.bool, device=inputs.device)
        mask.scatter_(1, mask_indices, True)
        encoded = self.encoder(torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(tokens), tokens))
        predicted_patches = self.decoder(encoded)
        target_patches = self.patchify(inputs)
        loss = functional.mse_loss(predicted_patches[mask], target_patches[mask])
        visible = encoded[~mask].reshape(batch, count - masked_count, self.embedding_dim)
        reconstruction = self.unpatchify(predicted_patches, tuple(inputs.shape))
        return MAEOutput(reconstruction, mask, visible.mean(dim=1), loss)


class MAEEmbeddingMethod(EmbeddingMethod):
    def __init__(self, model, optimizer, device, epochs=1, mask_ratio=0.75):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.epochs = int(epochs)
        self.mask_ratio = float(mask_ratio)

    def train(self, train_loader, validation_loader=None) -> Dict[str, float]:
        self.model.train()
        last_loss = float("nan")
        for _ in range(self.epochs):
            for batch in train_loader:
                crops = batch[-1] if isinstance(batch, (tuple, list)) else batch
                output = self.model(crops.to(self.device), mask_ratio=self.mask_ratio)
                self.optimizer.zero_grad()
                output.loss.backward()
                self.optimizer.step()
                last_loss = float(output.loss.detach())
        return {"loss": last_loss}

    def encode_cells(self, loader):
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
        return np.concatenate(ids).astype(np.int64), np.concatenate(embeddings)


def _load_crops(config: Dict, seed: int):
    mae_config, data_config = config.get("mae", {}), config.get("data", {})
    if data_config.get("crops"):
        crops = np.load(Path(data_config["crops"]))
    else:
        shape = _triple(mae_config.get("input_shape", (16, 16, 16)))
        crops = np.random.default_rng(seed).normal(size=(8,) + shape).astype(np.float32)
    if crops.ndim == 4:
        crops = crops[:, None]
    if crops.ndim != 5:
        raise ValueError("MAE crops must have shape (n, z, y, x) or (n, c, z, y, x)")
    return crops.astype(np.float32, copy=False)


def _build_model(config: Dict) -> MaskedAutoencoder3D:
    mae = config.get("mae", {})
    return MaskedAutoencoder3D(
        input_channels=int(mae.get("input_channels", 1)), patch_size=mae.get("patch_size", (4, 4, 4)),
        embedding_dim=int(mae.get("embedding_dim", 128)), encoder_depth=int(mae.get("encoder_depth", 2)),
        encoder_heads=int(mae.get("encoder_heads", 4)), decoder_dim=mae.get("decoder_dim"),
    )


def train_from_config(config_path: Path, output: Optional[Path] = None) -> Path:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if config.get("data", {}).get("crops"):
        crops_path = Path(config["data"]["crops"])
        if not crops_path.is_absolute():
            config["data"]["crops"] = str(config_path.parent / crops_path)
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(str(config.get("device", "auto")))
    crops = _load_crops(config, seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(crops)),
                        batch_size=int(config.get("training", {}).get("batch_size", 2)), shuffle=True)
    model = _build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("training", {}).get("learning_rate", 1e-3)))
    method = MAEEmbeddingMethod(model, optimizer, device,
                                epochs=int(config.get("training", {}).get("epochs", 2)),
                                mask_ratio=float(config.get("mae", {}).get("mask_ratio", 0.75)))
    metrics = method.train(loader)
    destination = output or load_config(config_path).paths.output_root / "mae" / "checkpoint.pt"
    save_checkpoint(destination, model, optimizer=optimizer, config=config, metrics=metrics)
    print("MAE checkpoint saved to {} with loss {:.6f}".format(destination, metrics["loss"]))
    return destination


def encode_from_config(config_path: Path, checkpoint: Path, output: Path) -> Path:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if config.get("data", {}).get("crops"):
        crops_path = Path(config["data"]["crops"])
        if not crops_path.is_absolute():
            config["data"]["crops"] = str(config_path.parent / crops_path)
    device = resolve_device(str(config.get("device", "auto")))
    model = _build_model(config).to(device)
    load_checkpoint(checkpoint, model, device=device)
    crops = _load_crops(config, int(config.get("seed", 42)))
    loader = DataLoader(TensorDataset(torch.from_numpy(crops)),
                        batch_size=int(config.get("inference", {}).get("batch_size", 4)))
    ids, features = MAEEmbeddingMethod(model, optimizer=None, device=device).encode_cells(loader)
    return export_embeddings(output, ids, features)
