"""Path utilities for command-line workflows."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a ``Path``."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a path relative to an optional base directory."""

    candidate = Path(path)
    if candidate.is_absolute() or base_dir is None:
        return candidate.expanduser().resolve()
    return (Path(base_dir) / candidate).expanduser().resolve()
