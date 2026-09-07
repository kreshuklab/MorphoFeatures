"""Bounded reads from HDF5, N5, Zarr and NumPy in explicit ZYX coordinates."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np


class SpatialVolume:
    def __init__(self, dataset, axes, channel=0):
        self.dataset = dataset
        self.axes = str(axes).lower()
        self.channel = int(channel)
        if (
            set(self.axes) not in ({"z", "y", "x"}, {"z", "y", "x", "c"})
            or len(set(self.axes)) != len(self.axes)
            or len(self.axes) != len(dataset.shape)
        ):
            raise ValueError("axes must describe exactly z,y,x and optionally one channel axis")
        if "c" in self.axes and not 0 <= channel < dataset.shape[self.axes.index("c")]:
            raise ValueError("Selected channel is outside the dataset")
        self.shape = tuple(dataset.shape[self.axes.index(axis)] for axis in "zyx")
        self.dtype = np.dtype(dataset.dtype)
        attrs = getattr(dataset, "attrs", {})
        known = attrs.get("DIMENSION_LABELS")
        if known is not None:
            known = "".join(v.decode() if isinstance(v, bytes) else str(v) for v in known)
            if known and known != self.axes:
                raise ValueError(
                    f"Declared axes {self.axes!r} conflict with dataset DIMENSION_LABELS {known!r}"
                )

    def read(self, slices):
        indexing = tuple(
            self.channel if axis == "c" else slices["zyx".index(axis)] for axis in self.axes
        )
        block = np.asarray(self.dataset[indexing])
        spatial = self.axes.replace("c", "")
        return np.transpose(block, tuple(spatial.index(axis) for axis in "zyx"))


@contextmanager
def open_volume(path, key, axes="zyx", channel=0):
    path = Path(path).expanduser()
    close = None
    if path.suffix in {".h5", ".hdf5", ".hdf"}:
        import h5py

        store = h5py.File(path, "r")
        close = store.close
    elif path.suffix == ".npy":
        store = None
        dataset = np.load(path, mmap_mode="r", allow_pickle=False)
    elif path.suffix == ".n5":
        import z5py

        store = z5py.File(str(path), "r")
    else:
        import zarr

        store = zarr.open(str(path), mode="r")
    try:
        if store is not None:
            if not key or key not in store:
                raise ValueError(f"Dataset key {key!r} is missing from {path}")
            dataset = store[key]
        yield SpatialVolume(dataset, axes, channel)
    finally:
        if close:
            close()
