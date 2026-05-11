"""Helpers for importing the local src package during unittest discovery."""

from __future__ import annotations

import sys
from pathlib import Path


def add_src_to_path() -> None:
    """Add the repository's src directory to ``sys.path``."""

    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
