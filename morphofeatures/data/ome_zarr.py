"""Read-only, cached OME-Zarr HTTP sources with explicit NGFF grid selection.

Zarr-Python decodes chunks; the HTTP transport shares N5's persistent bounded
cache. No directory listing or whole-volume read is needed for discovery.
"""

import json
from collections.abc import MutableMapping
from copy import deepcopy

import numpy as np

from morphofeatures.data.remote_n5 import MAX_FILE_BYTES, HttpCache, remote_url


def _key(value):
    value = str(value)
    if (
        not value
        or any(p in {"", ".", ".."} for p in value.split("/"))
        or any(c in value for c in "?#\\")
    ):
        raise ValueError("OME-Zarr dataset paths must be relative paths within the supplied group")
    return value


class CachedHttpMapping(MutableMapping):
    """Zarr v2 mapping. Metadata supplies array paths; HTTP listing is unnecessary."""

    def __init__(self, url, options=None):
        self.url, self.cache = remote_url(url), HttpCache(options)

    def __getitem__(self, key):
        try:
            value = self.cache.get(self.url + "/" + _key(key), missing_ok=True)
            if value is None:
                raise KeyError(key)
            return value
        except FileNotFoundError as error:
            raise KeyError(key) from error

    def __setitem__(self, key, value):
        raise PermissionError("Remote image stores are read-only")

    def __delitem__(self, key):
        raise PermissionError("Remote image stores are read-only")

    def __iter__(self):
        raise NotImplementedError("Use NGFF metadata to select arrays; HTTP listing is unavailable")

    def __len__(self):
        raise NotImplementedError("HTTP listing is unavailable")


def _zarr_store(mapping):
    import zarr

    if int(zarr.__version__.split(".")[0]) < 3:
        return mapping
    from zarr.storage import MemoryStore

    class CachedHttpStore(MemoryStore):
        # Use MemoryStore's read-only lifecycle and implement its byte interface.
        # Pixel bytes live in HttpCache, not in MemoryStore's in-memory dictionary.
        supports_listing = False

        def __init__(self):
            super().__init__(read_only=True)

        async def get(self, key, prototype, byte_range=None):
            import asyncio

            try:
                content = await asyncio.to_thread(mapping.__getitem__, key)
            except KeyError:
                return None
            if byte_range is not None:
                if hasattr(byte_range, "start"):
                    content = content[byte_range.start : byte_range.end]
                elif hasattr(byte_range, "offset"):
                    content = content[byte_range.offset :]
                elif hasattr(byte_range, "suffix"):
                    content = content[-byte_range.suffix :] if byte_range.suffix else b""
                else:
                    raise ValueError("Unsupported Zarr byte-range request")
            return prototype.buffer.from_bytes(content)

        async def exists(self, key):
            import asyncio

            return await asyncio.to_thread(mapping.__contains__, key)

        async def getsize(self, key):
            import asyncio

            try:
                return len(await asyncio.to_thread(mapping.__getitem__, key))
            except KeyError as error:
                raise FileNotFoundError(key) from error

        async def get_partial_values(self, prototype, key_ranges):
            return [await self.get(key, prototype, byte_range) for key, byte_range in key_ranges]

    return CachedHttpStore()


class HttpZarrArray:
    """Delegate codecs/order/fill values to Zarr while bounding each array read."""

    def __init__(self, url, key, options=None):
        try:
            import zarr
        except ImportError as error:
            raise ValueError(
                "OME-Zarr streaming requires Zarr-Python; install the MorphoFeatures workspace extra in the UI and worker environment"
            ) from error

        mapping = CachedHttpMapping(url, options)
        key = _key(key)
        # Metadata is small and lets v2 installations reject v3 with a useful error.
        metadata = mapping.cache.get(mapping.url + "/" + key + "/zarr.json", missing_ok=True)
        if metadata is not None and int(zarr.__version__.split(".")[0]) < 3:
            raise ValueError(
                "This source uses Zarr v3; use Zarr-Python >=3 in the UI and worker environment"
            )
        kwargs = (
            {"zarr_format": 3 if metadata is not None else 2}
            if int(zarr.__version__.split(".")[0]) >= 3
            else {}
        )
        self.array = zarr.open_array(store=_zarr_store(mapping), path=key, mode="r", **kwargs)
        self.shape, self.chunks = tuple(self.array.shape), tuple(self.array.chunks)
        self.dtype, self.ndim = np.dtype(self.array.dtype), self.array.ndim
        self.attrs, self.cache = self.array.attrs, mapping.cache
        if self.ndim not in {3, 4, 5} or self.dtype.kind not in "buif":
            raise ValueError(
                "Streaming requires numeric 3D volumes with optional channel/time axes"
            )
        # Shards are downloaded as complete cache entries, subject to the same limit.
        storage_chunks = getattr(self.array, "shards", None) or self.chunks
        if np.prod(storage_chunks, dtype=object) * self.dtype.itemsize > MAX_FILE_BYTES:
            raise ValueError(
                "OME-Zarr chunks/shards exceed the 64 MiB decoded read limit; choose another level/source"
            )

    def __getitem__(self, indexing):
        if len(indexing) != self.ndim:
            raise ValueError("Remote Zarr reads must specify every axis")
        voxels, chunks = 1, 1
        for value, size, chunk in zip(indexing, self.shape, self.chunks):
            if isinstance(value, (int, np.integer)):
                if not 0 <= value < size:
                    raise ValueError("Channel/time index is outside the volume")
                continue
            if not isinstance(value, slice) or value.step not in (None, 1):
                raise ValueError("Remote Zarr reads require contiguous slices")
            start, stop, _ = value.indices(size)
            count = max(stop - start, 0)
            voxels *= count
            chunks *= (stop - 1) // chunk - start // chunk + 1 if count else 0
        if voxels > 512**3 or voxels * self.dtype.itemsize > 512 * 1024**2 or chunks > 512:
            raise ValueError(
                "Remote region exceeds the 512-chunk/512-MiB read limit; reduce the ROI or choose a coarser level"
            )
        return self.array[indexing]


def discover_ome_zarr(url, options=None, *, image_index=0):
    """Read multiscale metadata only; resolution selection need not load chunks."""
    url, cache = remote_url(url), HttpCache(options)
    contents = cache.get(url + "/.zattrs", missing_ok=True)
    if contents is not None:
        attrs = json.loads(contents)
    else:
        contents = cache.get(url + "/zarr.json")
        attrs = json.loads(contents).get("attributes", {})
    multiscales = attrs.get("multiscales", attrs.get("ome", {}).get("multiscales", []))
    if not isinstance(image_index, int) or not 0 <= image_index < len(multiscales):
        raise ValueError(
            "Select an OME-Zarr image group with multiscales metadata and a valid image_index"
        )
    image = multiscales[image_index]
    axes = image.get("axes", [])
    names = [a if isinstance(a, str) else a.get("name") for a in axes]
    if (
        len(names) not in {3, 4, 5}
        or len(set(names)) != len(names)
        or not {"z", "y", "x"} <= set(names) <= {"t", "c", "z", "y", "x"}
    ):
        raise ValueError("NGFF axes must name X, Y, Z and optionally channel (c) and time (t)")
    units = [
        a.get("unit") if isinstance(a, dict) else None
        for a in axes
        if (a if isinstance(a, str) else a.get("name")) in "zyx"
    ]
    factors = {
        "nanometer": 0.001,
        "micrometer": 1.0,
        "millimeter": 1000.0,
        "meter": 1e6,
        "nm": 0.001,
        "um": 1.0,
        "µm": 1.0,
    }
    if all(u is None for u in units):
        unit, spatial_factors = "voxel", [1.0, 1.0, 1.0]
    elif any(u not in factors for u in units):
        raise ValueError(
            "NGFF spatial units must be declared consistently (nm/µm/mm/m), or all omitted"
        )
    else:
        unit = "um"
        spatial_factors = [factors[axes[names.index(a)].get("unit")] for a in "zyx"]
    levels = []
    for index, dataset in enumerate(image.get("datasets", [])):
        scale, origin = np.ones(len(names)), np.zeros(len(names))
        transforms = dataset.get("coordinateTransformations", []) + image.get(
            "coordinateTransformations", []
        )
        if not transforms:
            raise ValueError(
                "Each NGFF resolution needs scale/translation coordinate transformations"
            )
        for transform in transforms:
            kind = transform.get("type")
            if kind not in {"scale", "translation"}:
                raise ValueError(
                    "Only axis-aligned NGFF scale/translation transforms are supported"
                )
            value = np.asarray(transform[kind], dtype=float)
            if value.shape != scale.shape or not np.isfinite(value).all():
                raise ValueError("NGFF transform dimensions must match the declared axes")
            if kind == "scale":
                if np.any(value <= 0):
                    raise ValueError("NGFF scales must be positive")
                scale *= value
                origin *= value
            else:
                origin += value
        levels.append(
            {
                "level": index,
                "key": _key(dataset["path"]),
                "spacing_zyx": [
                    float(scale[names.index(a)] * f) for a, f in zip("zyx", spatial_factors)
                ],
                "origin_zyx": [
                    float(origin[names.index(a)] * f) for a, f in zip("zyx", spatial_factors)
                ],
            }
        )
    if not levels:
        raise ValueError("OME-Zarr metadata contains no resolution datasets")
    return {
        "url": url,
        "axes": "".join(names),
        "unit": unit,
        "name": image.get("name", ""),
        "levels": levels,
        "image_count": len(multiscales),
    }


def resolve_ome_zarr_source(source, options=None):
    metadata = discover_ome_zarr(source["url"], options, image_index=source.get("image_index", 0))
    level = source.get("level", 0)
    if not isinstance(level, int) or not 0 <= level < len(metadata["levels"]):
        raise ValueError("Choose a resolution level listed in the source metadata")
    result = {
        **metadata,
        **metadata["levels"][level],
        "channel": source.get("channel", 0),
        "timepoint": source.get("timepoint", 0),
    }
    from morphofeatures.data.volumes import inspect_volume

    result.update(
        inspect_volume(
            result["url"],
            result["key"],
            result["axes"],
            result["channel"],
            remote_options={**(options or {}), "format": "ome_zarr"},
            timepoint=result["timepoint"],
        )
    )
    return result


def resolve_ome_zarr_preprocessing(settings, *, require_pair=True):
    """Resolve selected grids; raw/segmentation previews can connect independently."""
    result = deepcopy(settings)
    for key in ("object_index", "index_reference", "platybrowser", "inspection_config"):
        result.pop(key, None)
    options = {**settings.get("remote_options", {}), "format": "ome_zarr"}
    options["cache_directory"] = str(HttpCache(options).root)
    sources = settings.get("ome_zarr", {})
    resolved = {}
    for kind in ("raw", "segmentation"):
        if not sources.get(kind, {}).get("url"):
            for name in (
                kind,
                kind + "_key",
                kind + "_axes",
                kind + "_channel",
                kind + "_timepoint",
                kind + "_unit",
                kind + "_spacing_zyx",
                kind + "_origin_zyx",
            ):
                result.pop(name, None)
            if require_pair:
                raise ValueError(
                    "OME-Zarr preprocessing needs both raw and instance-segmentation URLs; either can be previewed alone"
                )
            continue
        value = resolve_ome_zarr_source(sources[kind], options)
        resolved[kind] = value
        result[kind] = value["url"]
        for name in ("key", "axes", "channel", "timepoint", "spacing_zyx", "origin_zyx", "unit"):
            result[kind + "_" + name] = value[name]
    if not resolved:
        raise ValueError("Enter a raw or segmentation OME-Zarr URL")
    reference = resolved.get("raw", resolved.get("segmentation"))
    result.update(
        source="ome_zarr",
        remote_options=options,
        remote_provenance=resolved,
        spacing_zyx=reference["spacing_zyx"],
        origin_zyx=reference["origin_zyx"],
        unit=reference["unit"],
    )
    if require_pair:
        raw, labels = resolved["raw"], resolved["segmentation"]
        if (
            raw["spatial_shape"] != labels["spatial_shape"]
            or raw["unit"] != labels["unit"]
            or any(
                not np.allclose(raw[k], labels[k], rtol=1e-6, atol=1e-8)
                for k in ("spacing_zyx", "origin_zyx")
            )
        ):
            raise ValueError(
                "Raw and segmentation resolutions must share dimensions, spacing and origin before preprocessing; choose aligned levels or resample first"
            )
    return result
