"""Read-only HTTP N5 regions with a bounded, persistent download cache.

Implements scalar 3D N5 blocks (mode 0, gzip/raw), independently of Zarr's
deprecated N5 adapters. See https://github.com/saalfeldlab/n5#file-system-specification.
"""

import gzip
import hashlib
import io
import itertools
import json
import os
import struct
import sqlite3
import time
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np

from morphofeatures.data.bounded_cache import ByteLRU, file_lock

MAX_FILE_BYTES = 64 * 1024**2


def is_remote_url(value):
    return isinstance(value, str) and urlsplit(value).scheme in {"http", "https"}


def remote_url(value):
    parts = urlsplit(str(value))
    if (
        parts.scheme not in {"http", "https"}
        or not parts.netloc
        or parts.username
        or parts.password
    ):
        raise ValueError("Use an HTTP(S) URL without embedded credentials")
    if parts.query or parts.fragment:
        raise ValueError("Remote dataset URLs must not contain a query or fragment")
    return str(value).rstrip("/")


@lru_cache(maxsize=32)
def _cache_lock(root):
    return threading.RLock()


class HttpCache:
    """Atomic downloads with indexed eviction and independent entry locks.

    Only HTTP 404 denotes an absent N5 block. Transport/authentication/server errors
    are surfaced rather than silently interpreted as empty segmentation voxels.
    """

    def __init__(self, options=None):
        options = options or {}
        self.root = (
            Path(options.get("cache_directory", "~/.cache/morphofeatures/remote"))
            .expanduser()
            .resolve()
        )
        self.limit = int(float(options.get("cache_size_mb", 1024)) * 1024**2)
        self.timeout = float(options.get("timeout_seconds", 20))
        self.version = str(options.get("cache_version", "1"))
        if not 1 <= self.limit <= 65536 * 1024**2 or not 1 <= self.timeout <= 120:
            raise ValueError(
                "Remote cache must be positive and at most 64 GiB; timeout must be 1–120 seconds"
            )
        self.downloads = self.downloaded_bytes = self.hits = 0
        self.download_seconds = self.evictions = 0
        self._touches = {}
        self._last_touch = time.monotonic()
        self.root.mkdir(parents=True, exist_ok=True)
        with _cache_lock(str(self.root)), file_lock(self.root / "index.lock"):
            with sqlite3.connect(self.root / "index.sqlite") as db:
                exists = db.execute(
                    "SELECT name FROM sqlite_master WHERE name='entries'"
                ).fetchone()
                db.execute(
                    "CREATE TABLE IF NOT EXISTS entries (name TEXT PRIMARY KEY, size INTEGER, used REAL)"
                )
                db.execute("CREATE INDEX IF NOT EXISTS entries_used ON entries(used)")
                if not exists:
                    # One migration scan for older caches, never one scan per download.
                    for entry in (self.root / "downloads").glob("*.cache"):
                        try:
                            stat = entry.stat()
                            db.execute(
                                "INSERT OR REPLACE INTO entries VALUES (?, ?, ?)",
                                (entry.name, stat.st_size, stat.st_mtime),
                            )
                        except FileNotFoundError:
                            pass

    def _index(self, name=None, size=0):
        with _cache_lock(str(self.root)), file_lock(self.root / "index.lock"):
            with sqlite3.connect(self.root / "index.sqlite") as db:
                db.executemany(
                    "UPDATE entries SET used=? WHERE name=?",
                    [(used, key) for key, used in self._touches.items()],
                )
                self._touches.clear()
                self._last_touch = time.monotonic()
                if name is not None:
                    db.execute(
                        "INSERT OR REPLACE INTO entries VALUES (?, ?, ?)", (name, size, time.time())
                    )
                total, count = db.execute(
                    "SELECT COALESCE(SUM(size), 0), COUNT(*) FROM entries"
                ).fetchone()
                if total > self.limit or count > 10000:
                    for key, length in db.execute(
                        "SELECT name, size FROM entries ORDER BY used"
                    ).fetchall():
                        if total <= self.limit and count <= 10000:
                            break
                        (self.root / "downloads" / key).unlink(missing_ok=True)
                        db.execute("DELETE FROM entries WHERE name=?", (key,))
                        self.evictions += 1
                        total -= length
                        count -= 1

    def stats(self):
        return {
            key: getattr(self, key)
            for key in ("downloads", "downloaded_bytes", "hits", "evictions", "download_seconds")
        }

    def _download(self, url):
        try:
            with urlopen(
                Request(url, headers={"User-Agent": "MorphoFeatures/0.2"}), timeout=self.timeout
            ) as response:
                content = response.read(MAX_FILE_BYTES + 1)
        except HTTPError as error:
            if error.code == 404:
                raise FileNotFoundError(f"Remote object not found: {url}") from error
            raise OSError(f"Remote read failed (HTTP {error.code}): {url}") from error
        except (URLError, TimeoutError) as error:
            raise OSError(f"Cannot read {url}: {error}") from error
        if len(content) > MAX_FILE_BYTES:
            raise ValueError("Remote object exceeds the 64 MiB download limit")
        self.downloads += 1
        self.downloaded_bytes += len(content)
        return content

    def get(self, url, *, missing_ok=False):
        url = remote_url(url)
        name = hashlib.sha256((self.version + "\n" + url).encode()).hexdigest()
        # Prefix byte distinguishes a real empty object from a cached HTTP 404.
        path = self.root / "downloads" / (name + ".cache")
        # Striped entry locks include only this download, allowing other URLs in parallel.
        with file_lock(self.root / "locks" / (name[:2] + ".lock")):
            try:
                cached = path.read_bytes()
                if not cached or cached[:1] not in (b"0", b"1"):
                    raise ValueError("Invalid cache entry; change cache_version to rebuild it")
                with _cache_lock(str(self.root)):
                    self._touches[path.name] = time.time()
                    if len(self._touches) >= 128 or time.monotonic() - self._last_touch > 5:
                        self._index()
                self.hits += 1
                content = None if cached[:1] == b"0" else cached[1:]
            except FileNotFoundError:
                started = time.perf_counter()
                try:
                    content = self._download(url)
                except FileNotFoundError:
                    if not missing_ok:
                        raise
                    content = None
                finally:
                    self.download_seconds += time.perf_counter() - started
                encoded = b"0" if content is None else b"1" + content
                if len(encoded) <= self.limit:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                        temporary = Path(stream.name)
                        stream.write(encoded)
                    try:
                        os.replace(temporary, path)
                    finally:
                        temporary.unlink(missing_ok=True)
                    self._index(path.name, len(encoded))
            if content is None and not missing_ok:
                raise FileNotFoundError(f"Remote object not found: {url}")
            return content

    def json(self, url):
        return json.loads(self.get(url))


class HttpN5Array:
    """Array interface restricted to bounded unit-stride ZYX slices."""

    def __init__(self, root, key, options=None):
        if not key or any(part in {"", ".", ".."} for part in str(key).split("/")):
            raise ValueError(
                "Remote N5 requires an explicit dataset key, e.g. setup0/timepoint0/s0"
            )
        self.url = remote_url(root) + "/" + str(key)
        self.cache = HttpCache(options)
        self.decoded = ByteLRU(int(float((options or {}).get("decoded_cache_mb", 128)) * 1024**2))
        self.decode_seconds = 0.0
        self.attrs = self.cache.json(self.url + "/attributes.json")
        self.shape = tuple(reversed(self.attrs["dimensions"]))
        self.chunks = tuple(reversed(self.attrs["blockSize"]))
        self.dtype = np.dtype(self.attrs["dataType"])
        self.ndim = 3
        self.compression = self.attrs["compression"]["type"]
        if (
            len(self.shape) != 3
            or len(self.chunks) != 3
            or any(int(v) != v or v <= 0 for v in (*self.shape, *self.chunks))
            or self.dtype.kind not in "uif"
            or self.compression not in {"gzip", "raw"}
            or self.attrs.get("isLabelMultiset")
        ):
            raise ValueError("Streaming supports 3D scalar N5 arrays with gzip or raw compression")
        if np.prod(self.chunks, dtype=object) * self.dtype.itemsize > MAX_FILE_BYTES:
            raise ValueError("N5 chunks exceed the 64 MiB decompressed block limit")

    def _decode(self, content):
        if len(content) < 16 or struct.unpack(">HH", content[:4]) != (0, 3):
            raise ValueError("Unsupported/corrupt N5 block: expected mode 0 with three dimensions")
        shape = tuple(reversed(struct.unpack(">III", content[4:16])))
        if any(n < 1 or n > maximum for n, maximum in zip(shape, self.chunks)):
            raise ValueError("N5 block dimensions disagree with the dataset metadata")
        expected = int(np.prod(shape)) * self.dtype.itemsize
        if self.compression == "gzip":
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(content[16:])) as stream:
                    decoded = stream.read(expected + 1)
            except (OSError, EOFError) as error:
                raise ValueError(
                    "Corrupt compressed N5 block; change cache_version to re-fetch"
                ) from error
        else:
            decoded = content[16:]
        if len(decoded) != expected:
            raise ValueError("N5 block payload size disagrees with its header")
        return np.frombuffer(decoded, dtype=self.dtype.newbyteorder(">")).reshape(shape)

    def __getitem__(self, slices):
        if len(slices) != 3 or any(
            not isinstance(s, slice) or s.step not in (None, 1) for s in slices
        ):
            raise ValueError("Remote reads require three contiguous ZYX slices")
        lower = np.array([s.indices(n)[0] for s, n in zip(slices, self.shape)])
        upper = np.array([s.indices(n)[1] for s, n in zip(slices, self.shape)])
        size = upper - lower
        if np.any(size < 0) or np.prod(size, dtype=object) > 512**3:
            raise ValueError("Remote region exceeds the 512-cubed voxel read limit")
        grid = [
            range(int(a // c), int((b + c - 1) // c)) for a, b, c in zip(lower, upper, self.chunks)
        ]
        if np.prod([len(v) for v in grid], dtype=object) > 512:
            raise ValueError(
                "Region requires more than 512 chunks; reduce the view or choose a coarser level"
            )
        result = np.zeros(tuple(size), dtype=self.dtype)
        if not result.size:
            return result
        for position in itertools.product(*grid):

            def load(position=position):
                content = self.cache.get(
                    self.url + "/" + "/".join(map(str, reversed(position))), missing_ok=True
                )
                if content is None:
                    return np.empty((0, 0, 0), dtype=self.dtype)
                started = time.perf_counter()
                block = self._decode(content)
                self.decode_seconds += time.perf_counter() - started
                return block

            block = self.decoded.get(position, load)
            if not block.size:
                continue  # Sparse N5 blocks are defined to be zero.
            start = np.asarray(position) * self.chunks
            lo, hi = np.maximum(lower, start), np.minimum(upper, start + block.shape)
            if np.any(hi <= lo):
                continue
            result[tuple(slice(int(a), int(b)) for a, b in zip(lo - lower, hi - lower))] = block[
                tuple(slice(int(a), int(b)) for a, b in zip(lo - start, hi - start))
            ]
        return result
