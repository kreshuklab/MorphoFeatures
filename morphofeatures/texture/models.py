"""Maintained 3D autoencoder preserving the legacy 80-dimensional texture contract."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as functional


class ConvBlock3D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.GELU(),
            nn.Conv3d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.GELU(),
        )

    def forward(self, inputs):
        return self.block(inputs)


class LegacyTextureAutoencoder3D(nn.Module):
    def __init__(self, input_channels=1, base_channels=16, embedding_dim=80, projection_dim=80):
        super().__init__()
        self.encoder1 = ConvBlock3D(input_channels, base_channels)
        self.encoder2 = ConvBlock3D(base_channels, base_channels * 2, stride=2)
        self.encoder3 = ConvBlock3D(base_channels * 2, base_channels * 4, stride=2)
        self.to_embedding = nn.Linear(base_channels * 4, embedding_dim)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.GELU(), nn.Linear(embedding_dim, projection_dim)
        )
        self.decoder2 = ConvBlock3D(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.decoder1 = ConvBlock3D(base_channels * 2 + base_channels, base_channels)
        self.reconstruction = nn.Conv3d(base_channels, input_channels, 1)

    def encode(self, inputs):
        level1 = self.encoder1(inputs)
        level2 = self.encoder2(level1)
        level3 = self.encoder3(level2)
        embedding = self.to_embedding(functional.adaptive_avg_pool3d(level3, 1).flatten(1))
        return embedding, (level1, level2, level3)

    def forward(self, inputs, just_encode: bool = False):
        embedding, (level1, level2, level3) = self.encode(inputs)
        if just_encode:
            return embedding
        decoded2 = functional.interpolate(level3, size=level2.shape[-3:], mode="trilinear", align_corners=False)
        decoded2 = self.decoder2(torch.cat((decoded2, level2), dim=1))
        decoded1 = functional.interpolate(decoded2, size=level1.shape[-3:], mode="trilinear", align_corners=False)
        decoded1 = self.decoder1(torch.cat((decoded1, level1), dim=1))
        return self.reconstruction(decoded1), embedding, self.projection(embedding)
