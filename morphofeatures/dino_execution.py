"""Physical-view scheduling and per-object aggregation around one frozen backbone."""

import json
import shutil
import time
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd

from morphofeatures.artifacts import write_json_atomic
from morphofeatures.data.dino_stream import prepared_events
from morphofeatures.dino_runtime import BatchEncoder, MemorySampler, ObjectJournal, runtime_options
from morphofeatures.workflow_validation import FieldError


def encode_physical_objects(
    settings, root, identity, source, objects, model, device, modes, writer=None
):
    from morphofeatures.dino_memory import announce_plan, plan_memory

    memory_plan = plan_memory(settings, source=source, objects=objects, detect=True)
    announce_plan(memory_plan, root, writer, phase="encoding after model load")
    settings = deepcopy(settings)
    settings["runtime"] = memory_plan["runtime"]
    settings["views"]["batch_size"] = memory_plan["batch_size"]
    runtime = runtime_options(settings)
    runtime["cache_directory"] = str(root.parent)
    views = settings["views"]
    features = views.get("features", [views.get("feature", "cls")])
    stride = 14 if settings["model"] == "dinov2" else 16
    encoder = BatchEncoder(model, device, settings, writer=writer)
    journal = ObjectJournal(root, identity, resume=runtime["resume"])
    remaining = objects[~objects.label_id.isin(journal.records)]
    states, images, masks, owners = {}, [], [], []
    started = time.perf_counter()
    committed_before = len(journal.records)
    allocated_previews = defaultdict(int)
    for record in journal.records.values():
        for mode, count in record.get("preview_counts", {}).items():
            allocated_previews[mode] += count
    preparation_wait = export_seconds = 0.0
    stopped = None

    def commit_ready(label_id):
        nonlocal export_seconds, stopped
        state = states[label_id]
        if state["done"] is None or state["received"] != state["encoded"]:
            return
        record = state["done"]
        if state.get("inference_error"):
            record["excluded"].append(
                {
                    "label_id": label_id,
                    "reason": state["inference_error"],
                    "reason_code": "nonfinite_features",
                }
            )
            for item in record["coverage"]:
                item.update(views=0, coverage="excluded: " + state["inference_error"])
            if source.get("on_object_error", "skip") == "error":
                record["stop"] = state["inference_error"]
        if record.get("stop"):
            stopped = record["stop"]
            return
        vectors = {}
        for item in record["coverage"]:
            mode, count = item["mode"], item["views"]
            if not count:
                continue
            for feature in features:
                value = state["totals"][mode + "." + feature]
                vectors[mode + "." + feature] = (
                    value if views.get("aggregation", "mean") == "max" else value / count
                ).astype(np.float32)
        state["manifest"].close()
        export_started = time.perf_counter()
        journal.commit(
            label_id,
            {
                **record,
                "dataset_identity": settings["dataset_identity"],
                "object": state["row"],
                "preview_counts": dict(state["previews"]),
            },
            vectors,
        )
        export_seconds += time.perf_counter() - export_started
        states.pop(label_id)
        elapsed = time.perf_counter() - started
        fresh = len(journal.records) - committed_before
        progress = {
            "completed_objects": len(journal.records),
            "total_objects": len(objects),
            "resumed_objects": committed_before,
            "encoded_images_this_session": encoder.images,
            "objects_per_hour": fresh * 3600 / elapsed,
            "images_per_second": encoder.images / elapsed,
            "estimated_remaining_seconds": (len(objects) - len(journal.records)) * elapsed / fresh,
            "last_label_id": label_id,
        }
        write_json_atomic(root / "progress.json", progress)
        if writer:
            writer.write(
                "object_encoded",
                label_id=label_id,
                processed=len(journal.records),
                **{k: v for k, v in progress.items() if k != "last_label_id"},
            )

    def flush():
        if not images:
            return
        values = encoder.encode(images, masks, features, stride)
        touched = set()
        for index, (label_id, mode) in enumerate(owners):
            state = states[label_id]
            if any(not np.isfinite(values[feature][index]).all() for feature in features):
                state["inference_error"] = (
                    "Backbone returned nonfinite features; try runtime.precision=float32 if mixed precision is enabled."
                )
            for feature in features:
                name = mode + "." + feature
                value = values[feature][index].astype(np.float64)
                old = state["totals"].get(name)
                state["totals"][name] = (
                    value
                    if old is None
                    else np.maximum(old, value)
                    if views.get("aggregation", "mean") == "max"
                    else old + value
                )
            state["encoded"] += 1
            touched.add(label_id)
        images.clear()
        masks.clear()
        owners.clear()
        for label_id in sorted(touched):
            commit_ready(label_id)

    events = prepared_events(source, remaining, views, runtime, modes)
    memory = MemorySampler()
    memory.__enter__()
    try:
        while True:
            waiting = time.perf_counter()
            try:
                kind, label_id, payload = next(events)
            except StopIteration:
                preparation_wait += time.perf_counter() - waiting
                break
            preparation_wait += time.perf_counter() - waiting
            if kind == "start":
                path = journal.start(label_id)
                states[label_id] = {
                    "path": path,
                    "row": payload,
                    "manifest": (path / "views.jsonl").open("w"),
                    "totals": {},
                    "received": 0,
                    "encoded": 0,
                    "done": None,
                    "previews": defaultdict(int),
                }
            elif kind == "images":
                state = states[label_id]
                planes, foregrounds, infos = payload
                for image, mask, info in zip(planes, foregrounds, infos):
                    state["manifest"].write(json.dumps(info) + "\n")
                    mode = info["mode"]
                    preview_index = state["previews"][mode]
                    if allocated_previews[mode] < 6:
                        np.savez_compressed(
                            state["path"] / f"preview-{mode}-{preview_index}.npz",
                            image=image,
                            mask=mask,
                            metadata=json.dumps(info),
                        )
                        state["previews"][mode] += 1
                        allocated_previews[mode] += 1
                    images.append(image)
                    masks.append(mask)
                    owners.append((label_id, mode))
                    state["received"] += 1
                    if len(images) >= encoder.batch_limit:
                        flush()
            elif kind == "excluded_view":
                states[label_id]["manifest"].write(json.dumps(payload) + "\n")
            elif kind == "done":
                states[label_id]["done"] = payload
                if not runtime["batch_across_objects"]:
                    flush()
                    if label_id not in states:
                        continue
                commit_ready(label_id)
            if stopped:
                break
        if not stopped:
            flush()
    finally:
        memory.__exit__()
        events.close()
        for state in states.values():
            state["manifest"].close()
        journal.close()
        report = {
            **encoder.report(),
            "peak_process_rss_mb": memory.peak,
            "memory_measurement": memory.method,
            "runtime": runtime,
            "memory_plan": memory_plan,
            "resumed_objects": journal.resumed,
            "completed_objects": len(journal.records),
            "total_objects": len(objects),
            "preparation_wait_seconds": preparation_wait,
            "checkpoint_seconds": export_seconds,
            "wall_seconds": time.perf_counter() - started,
            "worker_peak_rss_mb": max(
                (r.get("worker_peak_rss_mb", 0) for r in journal.records.values()), default=0
            ),
        }
        records = list(journal.records.values())
        report["preparation_timings"] = {
            key: sum(r.get("timings", {}).get(key, 0) for r in records)
            for key in {k for r in records for k in r.get("timings", {})}
        }
        report["io"] = {
            kind: {
                key: sum(r.get("io", {}).get(kind, {}).get(key, 0) for r in records)
                for key in ("read_seconds", "reads", "requested_bytes", "decode_seconds")
            }
            for kind in ("raw", "segmentation")
        }
        for kind in report["io"]:
            for cache in ("decoded", "http"):
                report["io"][kind][cache] = {
                    key: sum(
                        r.get("io", {}).get(kind, {}).get(cache, {}).get(key, 0) for r in records
                    )
                    for key in {
                        k for r in records for k in r.get("io", {}).get(kind, {}).get(cache, {})
                    }
                }
        write_json_atomic(root / "performance.json", report)
        coverage = [item for r in records for item in r["coverage"]]
        excluded = [item for r in records for item in r["excluded"]]
        # Retain the offending object's diagnostic even when policy stops before commit.
        for state in states.values():
            if state["done"]:
                excluded.extend(state["done"]["excluded"])
        pd.DataFrame(coverage).to_csv(root / "coverage.tsv", sep="\t", index=False)
        pd.DataFrame(
            excluded,
            columns=list(
                dict.fromkeys(
                    ["label_id", "mode", "reason", "reason_code", *(k for r in excluded for k in r)]
                )
            ),
        ).to_csv(root / "excluded.tsv", sep="\t", index=False)
    if stopped:
        raise FieldError(
            f"Stopped at object {label_id} because its error policy is Stop run. {stopped} See {root / 'excluded.tsv'}.",
            "on_object_error",
        )
    outputs = {mode + "." + feature: ([], []) for mode in modes for feature in features}
    previews = defaultdict(int)
    for label_id in map(int, objects.label_id):
        record = journal.records[label_id]
        folder = journal.objects / str(label_id)
        with np.load(folder / "vectors.npz", allow_pickle=False) as vectors:
            for name in vectors:
                outputs[name][0].append(label_id)
                outputs[name][1].append(vectors[name])
        for mode in modes:
            for index in range(record.get("preview_counts", {}).get(mode, 0)):
                path = folder / f"preview-{mode}-{index}.npz"
                if previews[mode] < 6:
                    shutil.copyfile(path, root / f"preview-{mode}-{previews[mode]}.npz")
                    previews[mode] += 1
    return outputs, excluded, coverage, report
