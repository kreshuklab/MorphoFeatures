"""Logging setup helpers."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a concise stream logger for CLI commands."""

    logging.basicConfig(
        format="[+][%(asctime)-15s][%(name)s %(levelname)s] %(message)s",
        stream=sys.stdout,
        level=level,
    )
