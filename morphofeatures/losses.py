"""Objectives used by restored contrastive workflows."""

from __future__ import annotations


def nt_xent_loss(projections, temperature: float = 0.1):
    """NT-Xent for adjacent augmented pairs: [a0, a1, b0, b1, ...]."""
    import torch
    import torch.nn.functional as functional

    if projections.ndim != 2 or projections.shape[0] % 2:
        raise ValueError("projections must have shape (2 * batch, embedding_dim)")
    normalized = functional.normalize(projections, dim=1)
    similarity = normalized @ normalized.T / temperature
    diagonal = torch.eye(len(projections), dtype=torch.bool, device=projections.device)
    similarity = similarity.masked_fill(diagonal, float("-inf"))
    positive_indices = torch.arange(len(projections), device=projections.device) ^ 1
    positive = similarity[torch.arange(len(projections), device=projections.device), positive_indices]
    return (-positive + torch.logsumexp(similarity, dim=1)).mean()


def texture_objective(
    reconstruction,
    target,
    projection,
    embedding,
    contrastive_weight: float = 1.0,
    reconstruction_weight: float = 1.0,
    bottleneck_weight: float = 1e-4,
    temperature: float = 0.1,
):
    import torch.nn.functional as functional

    reconstruction_loss = functional.mse_loss(reconstruction, target)
    contrastive_loss = nt_xent_loss(projection, temperature=temperature)
    bottleneck_loss = embedding.square().mean()
    total = (
        contrastive_weight * contrastive_loss
        + reconstruction_weight * reconstruction_loss
        + bottleneck_weight * bottleneck_loss
    )
    return total, {
        "loss": float(total.detach()),
        "contrastive_loss": float(contrastive_loss.detach()),
        "reconstruction_loss": float(reconstruction_loss.detach()),
        "bottleneck_loss": float(bottleneck_loss.detach()),
    }
