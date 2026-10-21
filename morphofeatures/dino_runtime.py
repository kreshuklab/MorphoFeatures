"""Bounded DINO execution, numerical provenance and durable object checkpoints.

Runtime tuning never changes object selection or view geometry. Precision is the
exception: its effective value participates in the extraction cache identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import threading
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import numpy as np

from morphofeatures.artifacts import write_json_atomic
from morphofeatures.data.bounded_cache import file_lock
from morphofeatures.workflow_validation import FieldError

RUNTIME_VERSION = "dino-runtime-v1"
DEFAULTS = {
    "memory_mode": "auto",
    "job_memory_mb": 0,
    "auto_adjust_memory": True,
    "resume": True,
    "prepare_workers": 0,
    "worker_threads": 1,
    "prefetch_mb": 256,
    "prepare_memory_mb": 4096,
    "decoded_cache_mb": 128,
    "scratch_directory": None,
    "spatial_order": False,
    "precision": "float32",
    "pin_memory": True,
    "async_transfer": True,
    "batch_across_objects": True,
    "prepared_cache_mb": 0,
    "group_normalization_reads": True,
    "reuse_object_mask": True,
}


class MemorySampler:
    """Sample actual process RSS; ru_maxrss can inherit a parent's pre-exec peak."""

    def __init__(self):
        self.peak = 0.0
        self.method = "RSS sampled every 50 ms from /proc; shared pages included"
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self):
        try:
            rss = (
                int(Path("/proc/self/statm").read_text().split()[1])
                * os.sysconf("SC_PAGE_SIZE")
                / 1024**2
            )
        except (OSError, ValueError, IndexError):
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            self.method = "process-lifetime ru_maxrss (live RSS unavailable)"
        self.peak = max(self.peak, rss)

    def _run(self):
        while not self.stop.wait(0.05):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.sample()
        self.stop.set()
        self.thread.join(timeout=1)


def runtime_options(settings):
    raw = settings.get("runtime", {})
    if not isinstance(raw, dict):
        raise FieldError("DINO runtime settings must be a mapping", "runtime")
    result = {**DEFAULTS, **raw}
    for key, minimum in (
        ("job_memory_mb", 0),
        ("prepare_workers", 0),
        ("worker_threads", 1),
        ("prefetch_mb", 1),
        ("prepare_memory_mb", 1),
        ("decoded_cache_mb", 0),
        ("prepared_cache_mb", 0),
    ):
        try:
            value = result[key]
            if isinstance(value, bool) or int(value) != float(value) or int(value) < minimum:
                raise ValueError()
            result[key] = int(value)
        except (ValueError, TypeError, OverflowError) as error:
            raise FieldError(
                f"{key} must be a whole number of at least {minimum}", "runtime." + key
            ) from error
    if result["prepare_workers"] > 32:
        raise FieldError(
            "Use at most 32 preparation workers; start with 2–4 and measure source load.",
            "runtime.prepare_workers",
        )
    for key in (
        "auto_adjust_memory",
        "resume",
        "spatial_order",
        "pin_memory",
        "async_transfer",
        "batch_across_objects",
        "group_normalization_reads",
        "reuse_object_mask",
    ):
        if not isinstance(result[key], bool):
            raise FieldError(f"{key} must be true or false", "runtime." + key)
    if result["memory_mode"] not in {"auto", "manual"}:
        raise FieldError("Choose automatic or manual DINO memory budgets", "runtime.memory_mode")
    for key in ("allocation_memory_mb", "allocation_cpus"):
        if key in result and (
            isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 1
        ):
            raise FieldError(
                "The submitted allocation must be a positive whole number", "runtime." + key
            )
    if result["precision"] not in ("float32", "bfloat16", "float16"):
        raise FieldError("Choose float32, bfloat16, or float16 inference", "runtime.precision")
    scratch = result["scratch_directory"]
    if scratch is not None and (not isinstance(scratch, str) or not scratch.strip()):
        raise FieldError(
            "Enter a scratch directory or null to use the normal cache location",
            "runtime.scratch_directory",
        )
    return result


def numerical_settings(settings, device):
    from importlib.metadata import PackageNotFoundError, version

    import torch

    options = runtime_options(settings)
    requested = options["precision"]
    effective = requested
    fallback = None
    with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
        supported = device.type == "cuda" and (
            requested != "bfloat16" or torch.cuda.is_bf16_supported()
        )
    if requested != "float32" and not supported:
        effective = "float32"
        fallback = f"{requested} is unavailable on {device}; using float32"
    try:
        xformers = version("xformers")
    except PackageNotFoundError:
        xformers = None
    return {
        "runtime_version": RUNTIME_VERSION,
        "requested_precision": requested,
        "precision": effective,
        "fallback": fallback,
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "xformers": xformers,
        "attention": "upstream backbone (unchanged)",
    }


def identity_settings(settings, numerical):
    """Execution limits can change on resume; the numerical contract cannot."""
    result = deepcopy(settings)
    result.pop("runtime", None)
    result.setdefault("views", {}).pop("batch_size", None)
    result["inference_contract"] = {
        key: numerical[key]
        for key in (
            "runtime_version",
            "precision",
            "torch",
            "device",
            "matmul_precision",
            "xformers",
        )
    }
    # Hot-cache paths/budgets are not a different scientific source.
    volume = result.get("config", {}).get("data", {}).get("volume", {})
    if "remote_options" in volume:
        volume["remote_options"] = {
            k: v
            for k, v in (volume["remote_options"] or {}).items()
            if k not in {"cache_directory", "cache_size_mb", "decoded_cache_mb", "timeout_seconds"}
        }
    return result


class ObjectJournal:
    """Commit vectors and matching manifest byte offsets once per complete object.

    The caller holds the extraction lock. A committed object is an atomically
    renamed directory; an interrupted manifest append is truncated on resume.
    """

    def __init__(self, root, identity, *, resume=True):
        self.root, self.identity = Path(root), identity
        self.objects = self.root / "object-checkpoints"
        self.pending = self.root / "pending-objects"
        self.manifest_path = self.root / "views.jsonl"
        self.records = {}
        self.root.mkdir(parents=True, exist_ok=True)
        signature = self.root / "resume.json"
        if signature.exists() and json.loads(signature.read_text()).get("identity") != identity:
            raise ValueError(
                "Cannot resume: extraction signature differs. Use a separate cache directory."
            )
        if not resume:
            shutil.rmtree(self.objects, ignore_errors=True)
            self.manifest_path.unlink(missing_ok=True)
        self.objects.mkdir(exist_ok=True)
        shutil.rmtree(self.pending, ignore_errors=True)
        self.pending.mkdir()
        for path in self.objects.iterdir():
            if not path.is_dir():
                continue
            record = json.loads((path / "object.json").read_text())
            if record["identity"] != identity or str(record["label_id"]) != path.name:
                raise ValueError(f"Incompatible object checkpoint: {path}")
            # NumPy validates ZIP CRCs while reading; a damaged commit is an error,
            # never silently accepted as already completed work.
            with np.load(path / "vectors.npz", allow_pickle=False) as values:
                for key in values:
                    if not np.isfinite(values[key]).all():
                        raise ValueError(f"Nonfinite checkpoint features: {path}")
            self.records[record["label_id"]] = record
        end = 0
        # Excluded objects may have empty manifests. Put these before a
        # nonempty commit at the same byte offset, regardless of directory order.
        ordered = sorted(
            self.records.values(), key=lambda row: (row["manifest_start"], row["manifest_end"])
        )
        for record in ordered:
            if record["manifest_start"] != end:
                raise ValueError(
                    "Incomplete/corrupt object manifest offsets; use a fresh extraction cache"
                )
            end = record["manifest_end"]
        if end and (not self.manifest_path.exists() or self.manifest_path.stat().st_size < end):
            raise ValueError("Object checkpoints refer to missing view manifest bytes")
        if end:
            with self.manifest_path.open("rb") as stream:
                for record in ordered:
                    digest = hashlib.sha256()
                    stream.seek(record["manifest_start"])
                    remaining = record["manifest_end"] - record["manifest_start"]
                    while remaining:
                        chunk = stream.read(min(1024**2, remaining))
                        if not chunk:
                            raise ValueError("Committed view manifest is truncated")
                        digest.update(chunk)
                        remaining -= len(chunk)
                    if digest.hexdigest() != record["manifest_sha256"]:
                        raise ValueError(
                            f"Corrupt committed view manifest for object {record['label_id']}; use a fresh extraction cache"
                        )
        self.manifest = self.manifest_path.open("r+b" if self.manifest_path.exists() else "w+b")
        self.manifest.truncate(end)
        self.manifest.seek(end)
        self.resumed = len(self.records)
        write_json_atomic(signature, {"schema": RUNTIME_VERSION, "identity": identity})

    def start(self, label_id):
        path = self.pending / str(label_id)
        path.mkdir()
        return path

    def commit(self, label_id, record, vectors):
        path = self.pending / str(label_id)
        with (path / "vectors.npz").open("wb") as stream:
            np.savez(stream, **vectors)
            stream.flush()
            os.fsync(stream.fileno())
        start = self.manifest.tell()
        digest = hashlib.sha256()
        with (path / "views.jsonl").open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024**2), b""):
                self.manifest.write(chunk)
                digest.update(chunk)
        self.manifest.flush()
        os.fsync(self.manifest.fileno())
        record = {
            **record,
            "label_id": label_id,
            "identity": self.identity,
            "manifest_start": start,
            "manifest_end": self.manifest.tell(),
            "manifest_sha256": digest.hexdigest(),
        }
        write_json_atomic(path / "object.json", record)
        with (path / "object.json").open("rb") as stream:
            os.fsync(stream.fileno())
        (path / "views.jsonl").unlink()
        os.replace(path, self.objects / str(label_id))
        directory = os.open(self.objects, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self.records[label_id] = record
        return record

    def close(self):
        self.manifest.close()


class BatchEncoder:
    """One GPU owner; float32 pooling, float64 object sums, optional CUDA autocast."""

    def __init__(self, model, device, settings, writer=None):
        self.model, self.device = model.to(device).eval(), device
        self.options = runtime_options(settings)
        self.writer = writer
        self.batch_limit = int(settings.get("views", {}).get("batch_size", 16))
        self.memory_adjustments = []
        self.numerical = numerical_settings(settings, device)
        self.times = {
            k: 0.0
            for k in (
                "host_batch_seconds",
                "transfer_seconds",
                "inference_seconds",
                "pooling_seconds",
                "readback_seconds",
            )
        }
        self.batches = self.images = 0
        if device.type == "cuda":
            import torch

            torch.cuda.reset_peak_memory_stats(device)

    def encode(self, images, masks, features, stride):
        import torch

        if len(images) > self.batch_limit:
            limit = self.batch_limit
            parts = [
                self.encode(images[i : i + limit], masks[i : i + limit], features, stride)
                for i in range(0, len(images), limit)
            ]
            return {key: np.concatenate([part[key] for part in parts]) for key in features}
        try:
            return self._encode(images, masks, features, stride)
        except torch.cuda.OutOfMemoryError:
            # Leave the except scope before retrying: its traceback holds failed tensors.
            if len(images) == 1 or not self.options["auto_adjust_memory"]:
                raise FieldError(
                    "DINO inference ran out of GPU memory. A single image cannot fit, or automatic batch recovery is disabled. Use a GPU with more memory or explicitly reduce Images per model batch. Completed object checkpoints are retained; precision, input size and views have not been changed.",
                    "views.batch_size",
                    "runtime.auto_adjust_memory",
                ) from None
        old = self.batch_limit
        self.batch_limit = max(1, len(images) // 2)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        adjustment = {
            "requested": old,
            "effective": self.batch_limit,
            "reason": "GPU out of memory; retry the same images in smaller batches.",
        }
        self.memory_adjustments.append(adjustment)
        message = f"DINO GPU memory adjustment: image batch {old} → {self.batch_limit}; retrying without changing images or precision."
        print(message, flush=True)
        if self.writer:
            self.writer.write("dino_gpu_memory_adjustment", message=message, **adjustment)
        return self.encode(images, masks, features, stride)

    def _encode(self, images, masks, features, stride):
        import torch

        from morphofeatures.dino import pool_tokens

        started = time.perf_counter()
        inputs = torch.as_tensor(np.stack(images), dtype=torch.float32)[:, None].expand(
            -1, 3, -1, -1
        )
        inputs = (inputs - torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]) / torch.tensor(
            [0.229, 0.224, 0.225]
        )[None, :, None, None]
        masks = torch.as_tensor(np.stack(masks), dtype=torch.float32)
        cuda = self.device.type == "cuda"
        if cuda and self.options["pin_memory"]:
            inputs, masks = inputs.pin_memory(), masks.pin_memory()
        self.times["host_batch_seconds"] += time.perf_counter() - started
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)] if cuda else None
        if events:
            events[0].record(torch.cuda.current_stream(self.device))
        start = time.perf_counter()
        nonblocking = cuda and self.options["pin_memory"] and self.options["async_transfer"]
        inputs, masks = (
            inputs.to(self.device, non_blocking=nonblocking),
            masks.to(self.device, non_blocking=nonblocking),
        )
        self.times["transfer_seconds"] += time.perf_counter() - start if not cuda else 0
        if events:
            events[1].record(torch.cuda.current_stream(self.device))
        precision = self.numerical["precision"]
        context = (
            torch.autocast("cuda", dtype=getattr(torch, precision))
            if precision != "float32"
            else nullcontext()
        )
        start = time.perf_counter()
        with torch.inference_mode(), context:
            result = self.model.forward_features(inputs)
        self.times["inference_seconds"] += time.perf_counter() - start if not cuda else 0
        if events:
            events[2].record(torch.cuda.current_stream(self.device))
        start = time.perf_counter()
        # Never sum half-precision tokens, even inside a mixed-precision experiment.
        result = {
            key: value.float() if torch.is_tensor(value) else value for key, value in result.items()
        }
        values = {f: pool_tokens(result, masks, f, stride) for f in features}
        self.times["pooling_seconds"] += time.perf_counter() - start if not cuda else 0
        if events:
            events[3].record(torch.cuda.current_stream(self.device))
        start = time.perf_counter()
        values = {key: value.cpu().numpy() for key, value in values.items()}
        self.times["readback_seconds"] += time.perf_counter() - start
        if events:
            for index, name in enumerate(
                ("transfer_seconds", "inference_seconds", "pooling_seconds")
            ):
                self.times[name] += events[index].elapsed_time(events[index + 1]) / 1000
        self.images += len(images)
        self.batches += 1
        return values

    def report(self):
        import sys

        import torch

        attention = {}
        for name, module in self.model.named_modules():
            if name.endswith("attn"):
                kind = type(module).__module__ + "." + type(module).__name__
                owner = sys.modules.get(type(module).__module__)
                attention[kind] = {"xformers_available": getattr(owner, "XFORMERS_AVAILABLE", None)}
        return {
            **self.times,
            "images": self.images,
            "batches": self.batches,
            "effective_batch_size": self.batch_limit,
            "gpu_memory_adjustments": self.memory_adjustments,
            "attention_modules": attention,
            "numerical": self.numerical,
            "peak_process_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device)
            if self.device.type == "cuda"
            else 0,
        }


def extraction_lock(root):
    return file_lock(Path(root) / "extraction.lock", blocking=False)
