"""Lightweight version contract for portable MAE configurations and checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MAE_ARCHITECTURE_VERSION = "position-aware-3d-v2"
GROUPED_PATCH_MAE_ARCHITECTURE_VERSION = "grouped-nucleus-patches-v3"
LEGACY_MAE_ARCHITECTURE_VERSION = "position-blind-v1"


def configured_mae_architecture(
    config: Mapping[str, Any] | None,
    *,
    missing: str = LEGACY_MAE_ARCHITECTURE_VERSION,
) -> str:
    """Return an explicit version, treating historical missing values as legacy."""

    if not isinstance(config, Mapping):
        return missing
    mae = config.get("mae", {})
    if not isinstance(mae, Mapping):
        return missing
    return str(mae.get("architecture_version", missing))
