"""Backward-compatible shape data-loading package."""

from __future__ import annotations

from morphofeatures.shape.loaders import get_simple_loader, get_train_val_loaders

__all__ = ["get_simple_loader", "get_train_val_loaders"]
