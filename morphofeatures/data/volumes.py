"""Bounded reads from HDF5, N5, Zarr and NumPy in explicit ZYX coordinates."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np


class SpatialVolume:
    def __init__(self, dataset, axes, channel=0, timepoint=0):
        self.dataset = dataset
        self.axes = str(axes).lower()
        self.channel = int(channel)
        self.timepoint = int(timepoint)
        if (
            not {"z", "y", "x"} <= set(self.axes) <= {"z", "y", "x", "c", "t"}
            or len(set(self.axes)) != len(self.axes)
            or len(self.axes) != len(dataset.shape)
        ):
            raise ValueError(
                f"Stored array shape is {tuple(dataset.shape)}; axes {self.axes!r} must describe "
                "every dimension as z,y,x and optionally channel (c) and time (t) axes"
            )
        for name, value in (("c", channel), ("t", timepoint)):
            limit = dataset.shape[self.axes.index(name)] if name in self.axes else 1
            if int(value) != value or not 0 <= value < limit:
                raise ValueError("Selected channel/timepoint is outside the dataset")
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
            self.channel
            if axis == "c"
            else self.timepoint
            if axis == "t"
            else slices["zyx".index(axis)]
            for axis in self.axes
        )
        block = np.asarray(self.dataset[indexing])
        spatial = self.axes.replace("c", "").replace("t", "")
        return np.transpose(block, tuple(spatial.index(axis) for axis in "zyx"))


@contextmanager
def open_volume(path, key, axes="zyx", channel=0, *, remote_options=None, timepoint=0):
    from morphofeatures.data.remote_n5 import HttpN5Array, is_remote_url

    if is_remote_url(path):
        if (remote_options or {}).get("format") == "ome_zarr":
            from morphofeatures.data.ome_zarr import HttpZarrArray

            yield SpatialVolume(HttpZarrArray(path, key, remote_options), axes, channel, timepoint)
            return
        if axes != "zyx" or channel != 0 or timepoint != 0:
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
        yield SpatialVolume(dataset, axes, channel, timepoint)
    finally:
        if close:
            close()


def inspect_volume(path, key=None, axes="zyx", channel=0, *, remote_options=None, timepoint=0):
    """Read array metadata without loading volume pixels."""
    with open_volume(
        path, key, axes, channel, remote_options=remote_options, timepoint=timepoint
    ) as volume:
        return {
            "stored_shape": tuple(volume.dataset.shape),
            "spatial_shape": volume.shape,
            "dtype": str(volume.dtype),
            "axes": volume.axes,
            "channel": volume.channel,
            "timepoint": volume.timepoint,
            "chunks": getattr(volume.dataset, "chunks", None),
        }


def preview_volume_region(settings, kind="raw", *, axis=0, index=None, max_side=512):
    """Read a central, bounded ROI plane from one raw or segmentation volume."""
    if kind not in {"raw", "segmentation"} or axis not in (0, 1, 2) or not 1 <= max_side <= 1024:
        raise ValueError(
            "Choose raw/segmentation, a spatial axis, and a preview side between 1 and 1024"
        )
    with open_volume(
        settings[kind],
        settings.get(kind + "_key"),
        settings.get(kind + "_axes", "zyx"),
        settings.get(kind + "_channel", 0),
        remote_options=settings.get("remote_options"),
        timepoint=settings.get(kind + "_timepoint", 0),
    ) as volume:
        if kind == "segmentation" and not np.issubdtype(volume.dtype, np.integer):
            raise ValueError("Instance segmentation must contain integer IDs")
        bounds = np.asarray(settings.get("roi") or [[0, 0, 0], list(volume.shape)])
        if (
            bounds.shape != (2, 3)
            or not np.isfinite(bounds).all()
            or not np.equal(bounds, np.floor(bounds)).all()
        ):
            raise ValueError("ROI requires integer start and stop coordinates in Z, Y, X")
        start, stop = bounds.astype(np.int64)
        if np.any(start < 0) or np.any(stop > volume.shape) or np.any(stop <= start):
            raise ValueError(f"ROI must be nonempty and inside Z, Y, X dimensions {volume.shape}")
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
            kind: np.take(volume.read(slices), 0, axis=axis),
            "start": lower.tolist(),
            "stop": upper.tolist(),
            "roi_shape": (stop - start).tolist(),
            "cropped": any(stop[d] - start[d] > max_side for d in range(3) if d != axis),
        }


def preview_volume_pair(settings, *, axis=0, index=None, max_side=512):
    """Read matching bounded ROI planes; preserve exact integer instance IDs."""
    metadata = {
        kind: inspect_volume(
            settings[kind],
            settings.get(kind + "_key"),
            settings.get(kind + "_axes", "zyx"),
            settings.get(kind + "_channel", 0),
            remote_options=settings.get("remote_options"),
            timepoint=settings.get(kind + "_timepoint", 0),
        )
        for kind in ("raw", "segmentation")
    }
    if metadata["raw"]["spatial_shape"] != metadata["segmentation"]["spatial_shape"]:
        raise ValueError("Spatial dimensions differ between raw and segmentation")
    if settings.get("raw_unit") != settings.get("segmentation_unit"):
        raise ValueError("Raw and segmentation coordinate units differ")
    for name in ("spacing_zyx", "origin_zyx"):
        raw, labels = settings.get("raw_" + name), settings.get("segmentation_" + name)
        if raw is not None and labels is not None and not np.allclose(raw, labels):
            raise ValueError(
                "Raw and segmentation grids differ; choose aligned resolutions for an overlay"
            )
    raw = preview_volume_region(settings, "raw", axis=axis, index=index, max_side=max_side)
    labels = preview_volume_region(
        settings, "segmentation", axis=axis, index=index, max_side=max_side
    )
    return {**raw, "segmentation": labels["segmentation"]}
