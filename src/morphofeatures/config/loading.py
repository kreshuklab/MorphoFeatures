"""Helpers for loading YAML configuration files."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


ConfigDict = dict[str, Any]


def load_yaml_config(path: str | Path) -> ConfigDict:
    """Load a YAML file into a dictionary.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        A dictionary containing the parsed configuration. Empty YAML files
        return an empty dictionary.

    Raises:
        FileNotFoundError: If the path does not exist.
        TypeError: If the YAML root is not a mapping.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise TypeError(f"Configuration root must be a mapping: {config_path}")
    return loaded


def merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> ConfigDict:
    """Recursively merge two configuration dictionaries.

    Args:
        base: Default values.
        override: Values that should replace or extend ``base``.

    Returns:
        A new merged dictionary. Neither input mapping is mutated.
    """

    merged: ConfigDict = deepcopy(dict(base))
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = merge_dicts(existing, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path, defaults_path: str | Path | None = None) -> ConfigDict:
    """Load a YAML config with optional recursive defaults.

    Args:
        path: Path to the run-specific YAML file.
        defaults_path: Optional YAML file containing default values.

    Returns:
        The merged configuration dictionary.
    """

    config = load_yaml_config(path)
    if defaults_path is None:
        return config
    defaults = load_yaml_config(defaults_path)
    return merge_dicts(defaults, config)
