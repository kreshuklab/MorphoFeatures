"""Repeatable physical-DINO timing and feature-drift experiments on a fixed cohort."""

import itertools
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from morphofeatures.artifacts import write_json_atomic
from morphofeatures.data.dino_views import index_source, validate_source
from morphofeatures.data.io import load_embeddings
from morphofeatures.dino_physical import extract_physical


def feature_drift(reference, candidate, k=5):
    from sklearn.neighbors import NearestNeighbors

    left, right = load_embeddings(reference), load_embeddings(candidate)
    if set(map(int, left.label_ids)) != set(map(int, right.label_ids)):
        raise ValueError(
            "Benchmark variants encoded different objects; inspect their exclusion reports"
        )
    lookup = {int(label): i for i, label in enumerate(right.label_ids)}
    x, y = (
        left.features.astype(np.float64),
        right.features[[lookup[int(label)] for label in left.label_ids]].astype(np.float64),
    )
    norms = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    cosine = np.divide(np.einsum("ij,ij->i", x, y), norms, out=np.zeros(len(x)), where=norms > 0)
    count = min(k, len(x) - 1)
    overlap = None
    nearest_agreement = None
    if count:
        neighbors = []
        nearest = []
        for values in (x, y):
            indices = (
                NearestNeighbors(n_neighbors=count + 1)
                .fit(values)
                .kneighbors(values, return_distance=False)
            )
            # With duplicate vectors, ties can keep the query itself out of the
            # returned candidates. Always retain exactly k other objects.
            others = [[int(j) for j in row if j != i][:count] for i, row in enumerate(indices)]
            neighbors.append([set(row) for row in others])
            nearest.append([row[0] for row in others])
        overlap = float(np.mean([len(a & b) / count for a, b in zip(*neighbors)]))
        nearest_agreement = float(np.mean(np.asarray(nearest[0]) == nearest[1]))
    return {
        "objects": len(x),
        "max_absolute_difference": float(np.max(np.abs(x - y))),
        "mean_cosine_distance": float(np.maximum(0, 1 - cosine).mean()),
        "neighbor_k": count,
        "neighbor_overlap": overlap,
        "nearest_neighbor_agreement": nearest_agreement,
    }


def run_benchmark(
    settings,
    destination,
    *,
    workers=(0, 2),
    batch_sizes=(16, 32),
    precisions=("float32",),
    max_objects=3,
    annotations=None,
    backbone=None,
):
    """Application-cold/warm caches; OS/page/server caches are not purged.

    All variants consume identical IDs and scientific views. Fitted fold-based
    classifier checks are optional; very small cohorts report why evaluation
    cannot be estimated instead of fabricating an accuracy.
    """
    from morphofeatures.dino import load_backbone

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    settings = deepcopy(settings)
    source = validate_source(settings["config"]["data"]["volume"])
    source["max_objects"] = 0
    objects = index_source(source, destination / "cohort")
    if max_objects < 1:
        raise ValueError("Benchmark max_objects must be positive")
    if len(objects) > max_objects:
        sizes = np.prod(
            np.column_stack([objects["bbox_max_" + a] - objects["bbox_min_" + a] for a in "zyx"]),
            axis=1,
        )
        ordered = np.argsort(sizes, kind="stable")
        selected = ordered[np.linspace(0, len(ordered) - 1, max_objects).round().astype(int)]
        objects = objects.iloc[np.sort(selected)].copy()
    objects.to_csv(destination / "cohort.tsv", sep="\t", index=False)
    model = load_backbone(settings) if backbone is None else backbone
    baseline_runtime = {
        **settings.get("runtime", {}),
        "memory_mode": "manual",  # Preserve this deliberately uncached baseline.
        "prepare_workers": 0,
        "precision": "float32",
        "decoded_cache_mb": 0,
        "prepared_cache_mb": 0,
        "group_normalization_reads": False,
        "reuse_object_mask": False,
        "batch_across_objects": False,
    }
    cases = [("serial-baseline", settings.get("views", {}).get("batch_size", 16), baseline_runtime)]
    for worker, batch, precision in itertools.product(workers, batch_sizes, precisions):
        cases.append(
            (
                f"workers-{worker}-batch-{batch}-{precision}",
                batch,
                {**settings.get("runtime", {}), "prepare_workers": worker, "precision": precision},
            )
        )
    report = {
        "cohort": str(destination / "cohort.tsv"),
        "object_ids": [int(i) for i in objects.label_id],
        "settings": settings,
        "cache_note": "Cold/warm refers to the application caches only. OS, storage-server and network caches are not purged. Baseline uses serial preparation, uncached decoding and original mask/normalization reads, with one object per model-batch sequence; it still saves recovery checkpoints.",
        "memory_note": "Requested workers and batches identify each trial. Read performance.runtime and effective_batch_size for allocation-aware adjustments. Automatic host planning and GPU retries may reduce the requested concurrency/batch without changing sampled views.",
        "trials": [],
    }
    reference = None
    for name, batch, runtime in cases:
        scratch_base = Path(
            runtime.get("scratch_directory") or destination / "hot-caches"
        ).expanduser()
        scratch = scratch_base / (destination.resolve().name + "-" + name)
        if scratch.exists():
            raise ValueError(
                f"Use a new benchmark output/scratch directory for a cold trial: {scratch}"
            )
        for temperature in ("cold", "warm"):
            trial = destination / name / temperature
            configured = deepcopy(settings)
            configured["runtime"] = {**runtime, "resume": True, "scratch_directory": str(scratch)}
            configured["views"]["batch_size"] = int(batch)
            configured["cache"] = str(trial / "recovery")  # Always infer; never hit vector cache.
            started = time.perf_counter()
            row = {
                "case": name,
                "cache": temperature,
                "workers": runtime["prepare_workers"],
                "batch_size": batch,
                "precision": runtime["precision"],
            }
            try:
                paths = extract_physical(
                    configured, trial, source=source, objects=objects, backbone=model
                )
                performance_path = next((trial / "recovery").glob("views-*/performance.json"))
                row["performance"] = json.loads(performance_path.read_text())
                row["embeddings"] = {k: str(v) for k, v in paths.items()}
                if reference is None:
                    reference = paths
                row["feature_drift"] = {
                    key: feature_drift(reference[key], path) for key, path in paths.items()
                }
                if annotations:
                    from morphofeatures.analysis.classification import evaluate_embedding_classifier

                    row["classification"] = {}
                    for key, path in paths.items():
                        try:
                            result = evaluate_embedding_classifier(
                                path,
                                annotations,
                                folds=3,
                                seed=42,
                                model="logistic",
                                output_dir=trial / (key + "-classification"),
                            )
                            row["classification"][key] = {
                                "mean_accuracy": result.mean_accuracy,
                                "fold_accuracies": result.scores.tolist(),
                            }
                        except ValueError as error:
                            row["classification"][key] = {"unavailable": str(error)}
            except (ValueError, RuntimeError, OSError) as error:
                row["error"] = str(error)
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
            row["elapsed_seconds"] = time.perf_counter() - started
            report["trials"].append(row)
            write_json_atomic(destination / "benchmark.json", report)
    summary = []
    for row in report["trials"]:
        p = row.get("performance", {})
        summary.append(
            {
                **{
                    key: row.get(key)
                    for key in (
                        "case",
                        "cache",
                        "workers",
                        "batch_size",
                        "precision",
                        "elapsed_seconds",
                        "error",
                    )
                },
                "images": p.get("images"),
                "images_per_second": p.get("images", 0)
                / max(p.get("wall_seconds", row["elapsed_seconds"]), 1e-9),
                "peak_process_rss_mb": p.get("peak_process_rss_mb"),
                "worker_peak_rss_mb": p.get("worker_peak_rss_mb"),
                "gpu_peak_bytes": p.get("cuda_peak_allocated_bytes"),
                "effective_precision": p.get("numerical", {}).get("precision"),
            }
        )
    pd.DataFrame(summary).to_csv(destination / "benchmark.tsv", sep="\t", index=False)
    return destination / "benchmark.json"


def benchmark_command(args):
    import yaml

    from morphofeatures.dino_experiments import resolve_paths

    document = resolve_paths(yaml.safe_load(args.config.read_text()), args.config.resolve().parent)
    stage = document["stages"][args.stage] if "stages" in document else document
    annotations = None
    if stage.get("action") == "dino_experiment":
        dataset = stage["datasets"][args.dataset]
        annotations = dataset.get("annotations")
        settings = {
            **stage["backbones"][args.backbone],
            "views": deepcopy(stage["views"]),
            "runtime": stage.get("runtime", {}),
            "config": {
                "device": args.device or stage.get("device", "auto"),
                "data": {"volume": dataset["source"]},
            },
        }
        settings["views"]["tile_size_um"] = stage.get(
            "tile_sizes_um", [stage["views"].get("tile_size_um", 1.0)]
        )[args.scale]
    else:
        settings = stage
        if args.device:
            settings.setdefault("config", {})["device"] = args.device
    if settings.get("views", {}).get("mode", "stored") == "stored":
        raise ValueError(
            "This benchmark takes a physical whole_label/fixed_tiles/both extraction or dino_experiment stage"
        )
    path = run_benchmark(
        settings,
        args.output,
        workers=args.workers,
        batch_sizes=args.batch_sizes,
        precisions=args.precisions,
        max_objects=args.objects,
        annotations=annotations,
    )
    print(path)
    return 0
