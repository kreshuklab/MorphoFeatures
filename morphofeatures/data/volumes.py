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
            raise ValueError(
                f"Stored array shape is {tuple(dataset.shape)}; axes {self.axes!r} must describe "
                "every dimension as z,y,x and optionally one channel axis"
            )
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
def open_volume(path, key, axes="zyx", channel=0, *, remote_options=None):
    from morphofeatures.data.remote_n5 import HttpN5Array, is_remote_url

    if is_remote_url(path):
        if axes != "zyx" or channel != 0:
            raise ValueError("Remote N5 volumes use ZYX axes without a channel dimension")
        yield SpatialVolume(HttpN5Array(path, key, remote_options), axes)
        return
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

        if (path / "zarr.json").is_file() and int(zarr.__version__.split(".")[0]) < 3:
            import z5py

            store = z5py.File(str(path), "r")
        else:
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


def inspect_volume(path, key=None, axes="zyx", channel=0):
    """Read array metadata without loading volume pixels."""
    with open_volume(path, key, axes, channel) as volume:
        return {
            "stored_shape": tuple(volume.dataset.shape),
            "spatial_shape": volume.shape,
            "dtype": str(volume.dtype),
            "axes": volume.axes,
            "channel": volume.channel,
            "chunks": getattr(volume.dataset, "chunks", None),
        }


def preview_volume_pair(settings, *, axis=0, index=None, max_side=512):
    """Read one bounded, aligned ROI plane without scanning object IDs or chunks.

    Large planes are center-cropped, not downsampled: instance IDs remain exact
    and every backend needs only basic contiguous slicing.
    """
    if axis not in (0, 1, 2) or not 1 <= max_side <= 1024:
        raise ValueError("Select a spatial axis and a preview side between 1 and 1024")
    with (
        open_volume(
            settings["raw"],
            settings.get("raw_key"),
            settings.get("raw_axes", "zyx"),
            settings.get("raw_channel", 0),
        ) as raw,
        open_volume(
            settings["segmentation"],
            settings.get("segmentation_key"),
            settings.get("segmentation_axes", "zyx"),
            settings.get("segmentation_channel", 0),
        ) as labels,
    ):
        if raw.shape != labels.shape:
            raise ValueError(
                f"Spatial dimensions differ: raw {raw.shape}, segmentation {labels.shape}"
            )
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Instance segmentation must contain integer IDs")
        bounds = np.asarray(settings.get("roi") or [[0, 0, 0], list(raw.shape)])
        if (
            bounds.shape != (2, 3)
            or not np.isfinite(bounds).all()
            or not np.equal(bounds, np.floor(bounds)).all()
        ):
            raise ValueError("ROI requires integer start and stop coordinates in Z, Y, X")
        start, stop = bounds.astype(np.int64)
        if np.any(start < 0) or np.any(stop > raw.shape) or np.any(stop <= start):
            raise ValueError(f"ROI must be nonempty and inside Z, Y, X dimensions {raw.shape}")
        plane = (int(start[axis]) + int(stop[axis]) - 1) // 2 if index is None else int(index)
        if not start[axis] <= plane < stop[axis]:
            raise ValueError("Preview slice is outside the selected ROI")
        lower, upper = start.copy(), stop.copy()
        for dim in range(3):
            if dim != axis and stop[dim] - start[dim] > max_side:
                lower[dim] = (start[dim] + stop[dim] - max_side) // 2
                upper[dim] = lower[dim] + max_side
        lower[axis], upper[axis] = plane, plane + 1
        slices = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
        return {
            "raw": np.take(raw.read(slices), 0, axis=axis),
            "segmentation": np.take(labels.read(slices), 0, axis=axis),
            "start": lower.tolist(),
            "stop": upper.tolist(),
            "roi_shape": (stop - start).tolist(),
            "cropped": any(stop[d] - start[d] > max_side for d in range(3) if d != axis),
        }
