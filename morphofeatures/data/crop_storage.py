"""Shared, explicitly keyed access to prepared NumPy, HDF5 and N5 crops."""

from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

import numpy as np


def crop_storage_backend(output_format):
    """Validate the format and optional dependency before starting a volume scan."""
    if output_format == "npy":
        return None
    if output_format not in {"h5", "n5"}:
        raise ValueError("output_format must be npy, h5, or n5")
    module = "h5py" if output_format == "h5" else "z5py"
    try:
        return import_module(module)
    except ImportError as error:
        raise ValueError(
            f"{output_format} crop storage requires the optional {module} package"
        ) from error


@contextmanager
def open_crop_array(data, field="crops"):
    """Yield an array/dataset without loading all pixels; close it on exit.

    Container paths use a companion ``<field>_key``. An explicit key prevents
    confusing whole-object N5 crops with the separate grouped-patch layout.
    """
    value = data.get(field)
    if value is None:
        raise ValueError(f"Select data.{field} before reading prepared crops")
    if not isinstance(value, (str, Path)):
        yield np.asarray(value)
        return
    path = Path(value).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            yield array
        finally:
            array._mmap.close()
        return
    if suffix not in {".h5", ".hdf5", ".hdf", ".n5"}:
        raise ValueError(f"data.{field} must be a NumPy .npy array or an HDF5/N5 container")
    key = data.get(field + "_key")
    if not isinstance(key, str) or not key.strip():
        raise ValueError(
            f"Specify data.{field}_key for the dataset inside {path.name}, "
            "or load the generated mae_config.yaml to fill in the paths and keys"
        )
    backend = crop_storage_backend("n5" if suffix == ".n5" else "h5")
    with backend.File(str(path), "r") as store:
        if key not in store:
            raise ValueError(f"Dataset key {key!r} for data.{field} is missing from {path}")
        dataset = store[key]
        if not hasattr(dataset, "shape"):
            raise ValueError(f"data.{field}_key must point to an array, not a group")
        yield dataset


def load_crop_array(data, field="crops"):
    """Return an owned array for consumers such as the in-memory crop MAE."""
    with open_crop_array(data, field) as array:
        # Container slicing already returns owned memory; memmaps need a copy
        # before their backing file is closed.
        return np.array(array) if isinstance(array, np.ndarray) else np.asarray(array[:])
