"""Configuration loading with repository-relative paths and explicit overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DATA_ROOT_ENV = "MORPHOFEATURES_DATA_ROOT"
OUTPUT_ROOT_ENV = "MORPHOFEATURES_OUTPUT_ROOT"


@dataclass(frozen=True)
class PathsConfig:
    repo_root: Path
    data_root: Path
    analysis_data: Path
    mobie_data: Path
    output_root: Path


@dataclass(frozen=True)
class PipelineConfig:
    paths: PathsConfig
    seed: int = 42
    device: str = "auto"
    raw: Optional[Dict[str, Any]] = None


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(path: str, base: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def load_config(path: Optional[Path] = None) -> PipelineConfig:
    """Load YAML config without depending on the current working directory."""
    repo_root = repository_root()
    config_path = Path(path).resolve() if path else repo_root / "configs" / "default.yaml"
    raw: Dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}

    path_config = raw.get("paths", {})
    configured_repo = _resolve(path_config.get("repo_root", str(repo_root)), config_path.parent)
    data_root_value = os.environ.get(
        DATA_ROOT_ENV, path_config.get("data_root", str(configured_repo))
    )
    output_root_value = os.environ.get(
        OUTPUT_ROOT_ENV, path_config.get("output_root", "outputs")
    )
    data_root = _resolve(data_root_value, configured_repo)
    paths = PathsConfig(
        repo_root=configured_repo,
        data_root=data_root,
        analysis_data=_resolve(path_config.get("analysis_data", "analysis/data"), configured_repo),
        mobie_data=_resolve(path_config.get("mobie_data", "data_mobie"), configured_repo),
        output_root=_resolve(output_root_value, configured_repo),
    )
    return PipelineConfig(
        paths=paths,
        seed=int(raw.get("seed", 42)),
        device=str(raw.get("device", "auto")),
        raw=raw,
    )
