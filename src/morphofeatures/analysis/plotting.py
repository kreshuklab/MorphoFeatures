"""Shared plotting helpers."""

from __future__ import annotations


def require_matplotlib() -> object:
    """Import and return ``matplotlib.pyplot`` with a clear error message."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires matplotlib.") from exc
    return plt
