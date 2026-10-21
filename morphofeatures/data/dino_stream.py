"""Spawned CPU preparation with independent readers and bounded image queues."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import tempfile
import time
import traceback
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import numpy as np

from morphofeatures.data.dino_views import PhysicalObjects
from morphofeatures.data.object_eligibility import ObjectEligibilityError
from morphofeatures.data.prepared_view_cache import cached_object_events
from morphofeatures.dino_memory import MIB, object_estimates
from morphofeatures.dino_runtime import MemorySampler
from morphofeatures.workflow_validation import FieldError


class SpilledRecords:
    """Keep arbitrarily many excluded-view records off the preparation heap."""

    def __init__(self):
        self.stream = tempfile.TemporaryFile(mode="w+")

    def append(self, value):
        self.stream.write(json.dumps(value) + "\n")

    def __iter__(self):
        self.stream.flush()
        self.stream.seek(0)
        try:
            for line in self.stream:
                yield json.loads(line)
        finally:
            self.stream.seek(0, 2)

    def close(self):
        self.stream.close()


def _difference(after, before):
    return {
        key: _difference(value, before.get(key, {}))
        if isinstance(value, dict)
        else value - before.get(key, 0)
        for key, value in after.items()
    }


def _object_events(reader, row, modes, chunk_size):
    reader.exclusion_factory = SpilledRecords
    label_id = int(row["label_id"])
    yield ("start", label_id, row)
    started = time.perf_counter()
    before = {"raw": reader.raw.stats(), "segmentation": reader.seg.stats()}
    coverage, excluded, plane_seconds = [], [], 0.0
    stop = None
    try:
        obj = reader.prepare(row)
        for mode in modes:
            images, masks, infos = [], [], []
            count = 0
            last = {}
            low, high = np.full(3, np.inf), np.full(3, -np.inf)
            planes = iter(reader.planes(obj, mode))
            while True:
                start = time.perf_counter()
                try:
                    image, mask, info = next(planes)
                except StopIteration:
                    plane_seconds += time.perf_counter() - start
                    break
                plane_seconds += time.perf_counter() - start
                images.append(image)
                masks.append(mask)
                infos.append(info)
                count += 1
                last = info
                low, high = np.minimum(low, info["lower_um"]), np.maximum(high, info["upper_um"])
                if len(images) == chunk_size:
                    yield ("images", label_id, (np.stack(images), np.stack(masks), infos))
                    images, masks, infos = [], [], []
            if images:
                yield ("images", label_id, (np.stack(images), np.stack(masks), infos))
            if not count:
                reasons = {}
                for item in reader.excluded_views:
                    if item.get("mode") == mode:
                        reasons[item["reason"]] = None
                    if len(reasons) >= 8:
                        break
                reason = (
                    "; ".join(reasons)
                    or "No eligible foreground views; check fine tile/image cutoffs and selected slice positions."
                )
                if reader.source.get("on_object_error", "skip") == "error":
                    raise ValueError(reason)
                excluded.append({"label_id": label_id, "mode": mode, "reason": reason})
                coverage.append(
                    {
                        "label_id": label_id,
                        "mode": mode,
                        "views": 0,
                        **reader.sampling_summary,
                        "coverage": "excluded: " + reason,
                    }
                )
            else:
                coverage.append(
                    {
                        "label_id": label_id,
                        "mode": mode,
                        "views": count,
                        "coverage_scope": obj.get("coverage_scope", "complete_object"),
                        "volume_faces": obj.get("volume_faces", ""),
                        "roi_faces": obj.get("roi_faces", ""),
                        "eligible_tiles": last["eligible_tiles"],
                        "selected_tiles": last["selected_tiles"],
                        "tile_anchor": reader.views.get("tile_anchor", "bbox_min"),
                        "min_foreground_fraction": reader.views.get(
                            "min_foreground_fraction", 0.01
                        ),
                        "min_slice_foreground_fraction": reader.views.get(
                            "min_slice_foreground_fraction", 0
                        ),
                        "represented_min_zyx_um": json.dumps(low.tolist()),
                        "represented_max_zyx_um": json.dumps(high.tolist()),
                        "coverage": "sampled cross-sections, not exhaustive voxel coverage",
                    }
                )
    except (ValueError, OSError) as error:
        record = (
            error.record
            if isinstance(error, ObjectEligibilityError)
            else {"reason_code": "object_error"}
        )
        excluded.append({**record, "label_id": label_id, "reason": str(error)})
        if record.get("decision") == "error" or (
            not isinstance(error, ObjectEligibilityError)
            and reader.source.get("on_object_error", "skip") == "error"
        ):
            stop = str(error)
        # Pending images from an incomplete mode are not encoded or counted.
        # Already emitted partial-mode images may exist; the consumer uses the
        # explicitly completed coverage modes when deciding which vectors to save.
    after = {"raw": reader.raw.stats(), "segmentation": reader.seg.stats()}
    for skipped in reader.excluded_views:
        yield ("excluded_view", label_id, {**skipped, "status": "excluded"})
    yield (
        "done",
        label_id,
        {
            "coverage": coverage,
            "excluded": excluded,
            "stop": stop,
            "timings": {
                **getattr(reader, "timings", {}),
                "planes_seconds": plane_seconds,
                "object_wall_seconds": time.perf_counter() - started,
            },
            "io": _difference(after, before),
            "worker_peak_rss_mb": reader.memory.peak if hasattr(reader, "memory") else 0,
        },
    )


def _reader_source(source, runtime):
    source = deepcopy(source)
    options = {
        **(source.get("remote_options") or {}),
        "decoded_cache_mb": runtime["decoded_cache_mb"],
    }
    if runtime.get("scratch_directory"):
        options["cache_directory"] = str(Path(runtime["scratch_directory"]).expanduser() / "remote")
    source["remote_options"] = options
    return source


def _worker(source, views, runtime, modes, chunk_size, tasks, results):
    # Spawn starts without inherited HDF5/N5 handles or a CUDA context.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(runtime["worker_threads"])
    try:
        from threadpoolctl import threadpool_limits

        threads = threadpool_limits(runtime["worker_threads"])
    except ImportError:
        threads = nullcontext()
    try:
        with (
            threads,
            MemorySampler() as memory,
            PhysicalObjects(
                _reader_source(source, runtime), None, {**views, "_runtime": runtime}
            ) as reader,
        ):
            reader.memory = memory
            while True:
                row = tasks.get()
                if row is None:
                    break
                for event in cached_object_events(
                    reader,
                    row,
                    modes,
                    chunk_size,
                    runtime,
                    lambda row=row: _object_events(reader, row, modes, chunk_size),
                ):
                    results.put(event)
                # Release the potentially large object mask before admitting another object.
                reader.__dict__.pop("_object_mask", None)
    except BaseException:
        results.put(("fatal", None, traceback.format_exc()))


def prepared_events(source, objects, views, runtime, modes):
    """Yield CPU-prepared events. No process ever owns a model or CUDA tensor.

    Admission is based on a conservative 128 bytes per object-box voxel for
    masks, coordinates, tile intermediates and reader scratch. Queue capacity
    is independently bounded in bytes, including serialization copies.
    """
    rows = objects.to_dict("records")
    if runtime["spatial_order"]:
        rows.sort(
            key=lambda row: (
                tuple(row.get(f"bbox_min_{axis}", 0) for axis in "zyx") + (int(row["label_id"]),)
            )
        )
    size = int(views.get("size", 224))
    # Image float32 + mask bool, plus manifests and one serialization copy.
    image_bytes = 2 * (size * size * 5 + 4096)
    queue_bytes = runtime["prefetch_mb"] * 1024**2
    if image_bytes > queue_bytes:
        raise FieldError(
            "runtime.prefetch_mb cannot fit one DINO image; increase the queue budget",
            "runtime.prefetch_mb",
        )
    chunk_size = min(8, max(1, queue_bytes // image_bytes), int(views.get("batch_size", 16)))
    # Enforce the same admission estimate in serial and parallel mode. Serial
    # execution must not bypass a job limit merely because it has no queue workers.
    workers = min(runtime["prepare_workers"], len(rows))
    memory = runtime["prepare_memory_mb"] * MIB
    cache_reservation = max(1, workers) * runtime["decoded_cache_mb"] * MIB * 2
    cache_reservation += workers * runtime.get("worker_overhead_mb", 0) * MIB
    estimates = {key: value * MIB for key, value in object_estimates(source, objects).items()}
    if cache_reservation >= memory:
        raise FieldError("Preparation RAM budget cannot fit worker overhead and raw/segmentation caches. Use automatic DINO memory budgets or reduce workers/caches.", "runtime.prepare_memory_mb")
    for label_id, size in estimates.items():
        if size + cache_reservation > memory:
            raise FieldError(f"Object {label_id} needs approximately {size / MIB:.0f} MiB preparation scratch plus {cache_reservation / MIB:.0f} MiB caches/worker overhead. Increase the job RAM budget or enable automatic memory correction. No object has been silently skipped.", "runtime.prepare_memory_mb", "runtime.job_memory_mb")
    if not workers:
        with PhysicalObjects(
            _reader_source(source, runtime), objects, {**views, "_runtime": runtime}
        ) as reader:
            for row in rows:
                yield from cached_object_events(
                    reader,
                    row,
                    modes,
                    chunk_size,
                    runtime,
                    lambda row=row: _object_events(reader, row, modes, chunk_size),
                )
        return
    context = mp.get_context("spawn")
    tasks = context.Queue(maxsize=workers)
    results = context.Queue(maxsize=max(1, queue_bytes // (image_bytes * chunk_size)))
    processes = [
        context.Process(
            target=_worker,
            args=(source, views, runtime, modes, chunk_size, tasks, results),
            daemon=True,
        )
        for _ in range(workers)
    ]
    for process in processes:
        process.start()
    active, index = {}, 0
    try:
        while active or index < len(rows):
            while (
                index < len(rows)
                and len(active) < workers
                and cache_reservation
                + sum(active.values())
                + estimates[int(rows[index]["label_id"])]
                <= memory
            ):
                row = rows[index]
                active[int(row["label_id"])] = estimates[int(row["label_id"])]
                tasks.put(row)
                index += 1
            try:
                event = results.get(timeout=1)
            except queue.Empty:
                if any(not process.is_alive() for process in processes):
                    raise RuntimeError(
                        "DINO preparation worker exited unexpectedly; completed object checkpoints are retained. Check worker memory limits and retry."
                    ) from None
                continue
            if event[0] == "fatal":
                raise RuntimeError("DINO preparation worker failed:\n" + event[2])
            if event[0] == "done":
                active.pop(event[1])
            yield event
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
        tasks.close()
        results.close()
        tasks.cancel_join_thread()
        results.cancel_join_thread()
