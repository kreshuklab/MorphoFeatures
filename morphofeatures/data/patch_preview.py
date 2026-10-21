"""Cancellable object lookup and patch previews; no Streamlit calls in workers.

An unindexed label ID has no spatial address. Scan the selected segmentation ROI
once in bounded chunks, cache complete bounds, then read one object's raw image.
Never treat a cancelled/partial scan as a complete index or read the whole raw
volume. Source contents are versioned, not exhaustively hashed.
"""

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from morphofeatures.artifacts import write_json_atomic
from morphofeatures.data.dino_views import index_source
from morphofeatures.representations import fingerprint_file


class PreviewCancelled(Exception):
    pass


def cached_bounds(source, root, *, progress=None, cancel=None):
    """Reuse all IDs in one grid/ROI, independent of patch settings and object choice."""
    source = deepcopy(source)
    identity = {key: value for key, value in source.items() if key.startswith("segmentation")}
    identity.update(
        {
            key: source.get(key)
            for key in ("roi", "id_mapping", "remote_provenance", "data_version", "remote_options")
        }
    )
    for key in ("segmentation", "id_mapping"):
        path = source.get(key)
        if path and not path.startswith(("https://", "http://")):
            identity[key + "_file"] = fingerprint_file(path)
    path = source["segmentation"]
    if not path.startswith(("https://", "http://")):
        dataset = Path(path) / (source.get("segmentation_key") or "")
        for name in (".zarray", "zarr.json", "attributes.json"):
            if (dataset / name).is_file():
                identity[name] = fingerprint_file(dataset / name)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = Path(root) / key
    marker = directory / "index.json"
    if marker.is_file() and (directory / "objects.tsv").is_file():
        return directory / "objects.tsv"

    def tick(**values):
        if cancel is not None and cancel.is_set():
            raise PreviewCancelled
        if progress:
            progress(**values)

    tick(phase="Locating objects in segmentation", blocks=0, total_blocks=1)
    source.update(max_objects=0, object_ids=None, object_index=None)
    objects = index_source(source, directory, progress=tick)
    tick(phase="Object lookup complete", blocks=1, total_blocks=1)
    write_json_atomic(marker, {"source": identity, "objects": len(objects), "complete": True})
    return directory / "objects.tsv"


class PatchPreview:
    def __init__(self):
        self.cancel = threading.Event()
        self.lock = threading.Lock()
        self.status = {"phase": "Opening image metadata", "blocks": 0, "total_blocks": 1}
        self.future = None

    def update(self, **values):
        if self.cancel.is_set():
            raise PreviewCancelled
        with self.lock:
            self.status.update(values)

    def snapshot(self):
        with self.lock:
            return dict(self.status)


@lru_cache(maxsize=1)
def _executor():
    # Only one mask/preview is assembled at once to bound memory use in the UI process.
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="patch-preview")


def start_patch_preview(settings, root, *, object_id=None):
    """Return immediately. Cache bounds on disk; retain at most six patches in memory."""
    from morphofeatures.data.distributed_patches import preview_patches
    from morphofeatures.workflow_validation import FieldError

    task = PatchPreview()
    settings = deepcopy(settings)

    def run():
        try:
            images, summary = preview_patches(
                settings,
                object_id=object_id,
                index_directory=root,
                progress=task.update,
                cancel=task.cancel,
            )
            task.update(phase="Preview ready", blocks=1, total_blocks=1)
            return {"state": "complete", "images": images, "summary": summary}
        except PreviewCancelled:
            return {"state": "cancelled"}
        except Exception as error:
            return {
                "state": "error",
                "error": str(error),
                "fields": error.fields
                if isinstance(error, FieldError)
                else ("segmentation", "object_index"),
            }

    task.future = _executor().submit(run)
    return task
