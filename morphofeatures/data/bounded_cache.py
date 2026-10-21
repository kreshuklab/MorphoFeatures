"""Small bounded caches and process-safe locks for read-only image work."""

from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def file_lock(path, *, blocking=True):
    """An OS-owned lock is released even if its worker is killed."""
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another worker owns {path.parent}; retry after it finishes."
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class ByteLRU:
    """Decoded arrays are bounded by their actual bytes, rather than object count."""

    def __init__(self, limit):
        self.limit = int(limit)
        self.values = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = 0

    def get(self, key, loader):
        if key in self.values:
            self.hits += 1
            self.values.move_to_end(key)
            return self.values[key]
        self.misses += 1
        value = loader()
        size = value.nbytes
        if self.limit > 0 and size <= self.limit:
            while self.values and (self.bytes + size > self.limit or len(self.values) >= 4096):
                _, removed = self.values.popitem(last=False)
                self.bytes -= removed.nbytes
                self.evictions += 1
            self.values[key] = value
            self.bytes += size
        return value

    def stats(self):
        return {key: getattr(self, key) for key in ("bytes", "hits", "misses", "evictions")}


class ReadVolume:
    """Time bounded reads and reuse decoded local/Zarr chunks in spatial order."""

    def __init__(self, volume, cache_mb=128):
        import numpy as np

        self.volume = volume
        self.shape, self.dtype = volume.shape, volume.dtype
        self.dataset = getattr(volume, "dataset", None)
        chunks = getattr(self.dataset, "chunks", None)
        self.chunks = tuple(chunks[volume.axes.index(a)] for a in "zyx") if chunks else (64,) * 3
        # Remote N5 already has a decoder-level cache; avoid keeping a second copy.
        self.cached = ByteLRU(int(cache_mb * 1024**2))
        self.use_cache = (
            cache_mb > 0
            and chunks is not None
            and np.prod(self.chunks, dtype=object) * self.dtype.itemsize <= self.cached.limit
            and not hasattr(self.dataset, "decoded")
        )
        self.seconds = 0.0
        self.reads = self.bytes = 0
        self._np = np

    def read(self, slices):
        import itertools
        import time

        np = self._np
        started = time.perf_counter()
        try:
            if not self.use_cache:
                result = self.volume.read(slices)
            else:
                lower = np.array([s.indices(n)[0] for s, n in zip(slices, self.shape)])
                upper = np.array([s.indices(n)[1] for s, n in zip(slices, self.shape)])
                result = np.empty(tuple(upper - lower), dtype=self.dtype)
                grid = [
                    range(int(a // c), int((b + c - 1) // c))
                    for a, b, c in zip(lower, upper, self.chunks)
                ]
                for pos in itertools.product(*grid):
                    start = np.asarray(pos) * self.chunks
                    stop = np.minimum(self.shape, start + self.chunks)
                    block = self.cached.get(
                        pos,
                        lambda start=start, stop=stop: self.volume.read(
                            tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
                        ),
                    )
                    lo, hi = np.maximum(start, lower), np.minimum(stop, upper)
                    result[tuple(slice(int(a), int(b)) for a, b in zip(lo - lower, hi - lower))] = (
                        block[tuple(slice(int(a), int(b)) for a, b in zip(lo - start, hi - start))]
                    )
            self.reads += 1
            self.bytes += result.nbytes
            return result
        finally:
            self.seconds += time.perf_counter() - started

    def stats(self):
        result = {"read_seconds": self.seconds, "reads": self.reads, "requested_bytes": self.bytes}
        result["decoded"] = getattr(self.dataset, "decoded", self.cached).stats()
        if hasattr(self.dataset, "cache"):
            result["http"] = self.dataset.cache.stats()
        result["decode_seconds"] = getattr(self.dataset, "decode_seconds", 0.0)
        return result


class ObjectMaskVolume:
    """A previously read label mask, retaining the source's global voxel indices."""

    def __init__(self, mask, lower, shape):
        self.mask, self.lower, self.shape = mask, lower, shape

    def read(self, slices):
        import numpy as np

        lo = np.array([s.start for s in slices])
        hi = np.array([s.stop for s in slices])
        result = np.zeros(tuple(hi - lo), dtype=bool)
        start, stop = np.maximum(lo, self.lower), np.minimum(hi, self.lower + self.mask.shape)
        if np.all(stop > start):
            result[tuple(slice(int(a), int(b)) for a, b in zip(start - lo, stop - lo))] = self.mask[
                tuple(slice(int(a), int(b)) for a, b in zip(start - self.lower, stop - self.lower))
            ]
        return result
