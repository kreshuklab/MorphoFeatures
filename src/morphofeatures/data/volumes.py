"""Volume-opening utilities for z5/n5 and BigDataViewer-backed data."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_bdv_data_path(xml_path: str | Path) -> str:
    """Resolve a BigDataViewer XML file to its backing dataset path.

    Args:
        xml_path: BigDataViewer XML path.

    Returns:
        Path to the backing data container.

    Raises:
        ImportError: If ``pybdv`` is not installed.
    """

    try:
        from pybdv.metadata import get_data_path
    except ImportError as exc:
        raise ImportError("Resolving BDV XML paths requires pybdv.") from exc
    return get_data_path(str(xml_path), True)


def open_z5_dataset(container_path: str | Path, dataset_path: str, mode: str = "r") -> Any:
    """Open a dataset from a z5/n5 container.

    Args:
        container_path: z5/n5 container path.
        dataset_path: Internal dataset path.
        mode: File opening mode.

    Returns:
        z5py dataset object.

    Raises:
        ImportError: If ``z5py`` is not installed.
    """

    try:
        import z5py
    except ImportError as exc:
        raise ImportError("Opening z5/n5 data requires z5py.") from exc

    handle = z5py.File(str(container_path), mode)
    return handle[dataset_path]
