"""Optional byte-bounded disk reuse of actual DINO images across checkpoints.

Entries own complete objects. An OS lock pins an entry during reading or
writing; unfinished entries are never used as prepared data.
"""

import hashlib
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path

import numpy as np

from morphofeatures.data.bounded_cache import file_lock
from morphofeatures.data.dino_views import VIEW_VERSION, source_identity


class PreparedViewCache:
    def __init__(self, root, limit):
        self.root, self.limit = Path(root), int(limit)
        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.root / "index.lock"), sqlite3.connect(self.root / "index.sqlite") as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS entries (name TEXT PRIMARY KEY, size INTEGER, used REAL)"
            )

    def reserve(self, key, size):
        """Reserve bytes before a write. Pinned entries are never evicted."""
        with file_lock(self.root / "index.lock"), sqlite3.connect(self.root / "index.sqlite") as db:
            own = db.execute("SELECT size FROM entries WHERE name=?", (key,)).fetchone()
            if (own[0] if own else 0) + size > self.limit:
                return False
            used = db.execute("SELECT COALESCE(SUM(size), 0) FROM entries").fetchone()[0]
            for name, length in db.execute(
                "SELECT name, size FROM entries ORDER BY used"
            ).fetchall():
                if used + size <= self.limit:
                    break
                if name == key:
                    continue
                try:
                    with file_lock(self.root / "locks" / (name + ".lock"), blocking=False):
                        shutil.rmtree(self.root / name, ignore_errors=True)
                        db.execute("DELETE FROM entries WHERE name=?", (name,))
                        used -= length
                except RuntimeError:
                    continue
            if used + size > self.limit:
                return False
            db.execute(
                "INSERT INTO entries VALUES (?, ?, ?) ON CONFLICT(name) DO UPDATE SET size=size+excluded.size, used=excluded.used",
                (key, size, time.time()),
            )
            return True

    def remove(self, key):
        with file_lock(self.root / "index.lock"), sqlite3.connect(self.root / "index.sqlite") as db:
            shutil.rmtree(self.root / key, ignore_errors=True)
            db.execute("DELETE FROM entries WHERE name=?", (key,))

    def events(self, key, factory):
        folder = self.root / key
        with file_lock(self.root / "locks" / (key + ".lock")):
            marker = folder / "complete.json"
            if marker.exists():
                index = json.loads(marker.read_text())
                self.reserve(key, 0)  # Touch the indexed LRU without a directory scan.
                for item in index:
                    path = folder / item["file"]
                    if item["kind"] == "images":
                        with np.load(path, allow_pickle=False) as arrays:
                            yield (
                                "images",
                                item["label_id"],
                                (
                                    arrays["images"],
                                    arrays["masks"],
                                    json.loads(str(arrays["infos"])),
                                ),
                            )
                    else:
                        payload = json.loads(path.read_text())
                        if item["kind"] == "done":
                            payload["prepared_cache_hit"] = True
                            payload["timings"] = {}
                            payload["io"] = {}
                        yield item["kind"], item["label_id"], payload
                return
            self.remove(key)  # Interrupted preparation is cheap to regenerate.
            folder.mkdir()
            index = []
            caching = True
            for kind, label_id, payload in factory():
                if caching:
                    name = str(len(index)) + (".npz" if kind == "images" else ".json")
                    # Serialize a bounded image message, never a complete object in RAM.
                    if kind == "images":
                        import io

                        buffer = io.BytesIO()
                        images, masks, infos = payload
                        np.savez(buffer, images=images, masks=masks, infos=json.dumps(infos))
                        data = buffer.getvalue()
                    else:
                        data = json.dumps(payload).encode()
                    if self.reserve(key, len(data)):
                        (folder / name).write_bytes(data)
                        index.append({"kind": kind, "label_id": label_id, "file": name})
                    else:
                        caching = False
                        self.remove(key)
                yield kind, label_id, payload
            if caching:
                data = json.dumps(index).encode()
                if self.reserve(key, len(data)):
                    temporary = folder / "complete.tmp"
                    temporary.write_bytes(data)
                    os.replace(temporary, marker)
                else:
                    self.remove(key)


def cached_object_events(reader, row, modes, chunk_size, runtime, factory):
    if not runtime.get("prepared_cache_mb"):
        yield from factory()
        return
    source = reader.source
    views = {
        k: v
        for k, v in reader.views.items()
        if k not in {"batch_size", "features", "feature", "aggregation", "_runtime"}
    }
    scientific_source = {
        k: v
        for k, v in source.items()
        if k not in {"remote_options", "inspection_config", "platybrowser"}
    }
    key = hashlib.sha256(
        json.dumps(
            {
                "algorithm": VIEW_VERSION,
                "source": scientific_source,
                "dataset": source_identity(source),
                "row": row,
                "views": views,
                "modes": modes,
                "chunk_size": chunk_size,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    root = runtime.get("scratch_directory") or runtime.get("cache_directory")
    cache = PreparedViewCache(
        Path(root).expanduser() / "prepared-dino-views", runtime["prepared_cache_mb"] * 1024**2
    )
    yield from cache.events(key, factory)
