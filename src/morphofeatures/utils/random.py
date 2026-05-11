"""Randomness helpers."""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int | None) -> None:
    """Seed Python, NumPy, and PyTorch if available.

    Args:
        seed: Seed value. ``None`` leaves global RNG state unchanged.
    """

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return
