"""Frozen DINO extraction from physical views, with reusable multi-feature outputs."""

from __future__ import annotations

import hashlib
import json
import resource
import shutil
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np

from morphofeatures.artifacts import write_json_atomic
from morphofeatures.data.dino_views import (
    VIEW_VERSION,
    index_source,
    inspection_config,
    source_identity,
    validate_source,
)
from morphofeatures.data.io import export_embeddings
from morphofeatures.data.object_eligibility import write_eligibility
from morphofeatures.workflow_validation import FieldError, at_fields


def validate_physical_settings(settings, *, check_source=True):
    v = settings.get("views", {})
    if v.get("mode") not in {"whole_label", "fixed_tiles", "both"}:
        raise FieldError("Choose stored, whole_label, fixed_tiles, or both views", "views.mode")
    for name, default in (
        ("coarse_fractions", [0.1, 0.3, 0.5, 0.7, 0.9]),
        ("fine_fractions", [0.5]),
    ):
        with at_fields("views." + name):
            fractions = v.get(name, default)
            if (
                not isinstance(fractions, (list, tuple))
                or not fractions
                or any(not 0 <= float(f) <= 1 for f in fractions)
            ):
                raise ValueError(
                    "Enter slice positions as a list between 0 and 1, for example [0.5] for one middle slice or [0.25, 0.5, 0.75] for three slices."
                )
    with at_fields("views.tile_size_um"):
        width = np.broadcast_to(np.array(v.get("tile_size_um", 1.0), dtype=float), (3,))
        if not np.isfinite(width).all() or np.any(width <= 0):
            raise ValueError("Tile size must be positive micrometers")
    for name, default in (("max_candidate_tiles", 250000), ("max_tiles", 64)):
        with at_fields("views." + name):
            value = v.get(name, default)
            if (
                isinstance(value, bool)
                or not np.isfinite(float(value))
                or int(value) != float(value)
                or int(value) < 1
            ):
                raise ValueError(
                    "Enter a positive whole-number tile limit. To use every eligible tile, enable Process all eligible tiles instead of setting the cap to zero."
                )
    for name, default in (
        ("min_foreground_fraction", 0.01),
        ("min_slice_foreground_fraction", 0.0),
    ):
        with at_fields("views." + name):
            if not 0 <= float(v.get(name, default)) <= 1:
                raise ValueError(
                    "Nucleus-content cutoff must be between 0 and 1: 0.25 = 25%, 0.50 = 50%. "
                    "The tile cutoff measures a 3D volume; the image cutoff measures each 2D DINO input."
                )
    if v.get("tile_anchor", "bbox_min") not in {"bbox_min", "bbox_center", "centroid"}:
        raise FieldError(
            "Choose bounding-box corner (bbox_min), bounding-box center (bbox_center), or foreground centroid (centroid).",
            "views.tile_anchor",
        )
    if v.get("normalization", "foreground_percentile") != "foreground_percentile":
        raise FieldError(
            "Physical views share foreground_percentile scaling per object", "views.normalization"
        )
    if v.get("resize", "letterbox") != "letterbox":
        raise FieldError(
            "Physical views preserve physical aspect ratio; use letterbox", "views.resize"
        )
    features = v.get("features", [v.get("feature", "cls")])
    if not features or any(f not in {"cls", "patch_mean", "foreground_mean"} for f in features):
        raise FieldError("Select cls, patch_mean and/or foreground_mean", "views.features")
    if check_source:
        from morphofeatures.data.object_selection import selection_active

        source = deepcopy(settings.get("config", {}).get("data", {}).get("volume", {}))
        stage_selection = selection_active(settings.get("object_selection"))
        if stage_selection:
            if selection_active(source.get("object_selection")):
                raise FieldError("Configure Target objects either on the extraction stage or in its volume settings, not both.", "object_selection", "config.data.volume.object_selection")
            source["object_selection"] = settings["object_selection"]
        try:
            return validate_source(source)
        except FieldError as error:
            raise FieldError(
                str(error), *(path if stage_selection and path[:1] == ("object_selection",)
                              else ("config", "data", "volume") + path for path in error.fields)
            ) from error


def image_tensor(image):
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(image)).float()[None].expand(3, -1, -1)
    return (tensor - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor(
        [0.229, 0.224, 0.225]
    )[:, None, None]


def extract_physical(
    settings, destination, *, writer=None, backbone=None, source=None, objects=None
):
    from morphofeatures.dino import validate_dino_settings
    from morphofeatures.dino_runtime import extraction_lock, identity_settings, numerical_settings
    from morphofeatures.representations import extraction_identity, fingerprint_file
    from morphofeatures.training_runtime import resolve_device

    started = time.perf_counter()
    settings = deepcopy(settings)
    validate_dino_settings(settings, require_data=False)
    validate_physical_settings(settings, check_source=False)
    source = source or validate_physical_settings(settings)
    settings.setdefault("config", {}).setdefault("data", {}).update(
        source="physical_volume", volume=source
    )
    settings["view_algorithm"] = VIEW_VERSION
    settings["dataset_identity"] = source_identity(source)
    settings["source_files"] = {
        k: fingerprint_file(source[k])
        for k in ("raw", "segmentation", "object_index", "id_mapping")
        if source.get(k) and Path(source[k]).exists()
    }
    if objects is not None:
        settings["cohort_hash"] = hashlib.sha256(objects.to_csv(index=False).encode()).hexdigest()
    device = resolve_device(settings.get("config", {}).get("device", "auto"))
    numerical = numerical_settings(settings, device)
    if numerical["fallback"] and writer:
        writer.write("precision_fallback", message=numerical["fallback"])
    identity, provenance = extraction_identity(identity_settings(settings, numerical))
    root = Path(settings.get("cache", destination)) / ("views-" + identity)
    with extraction_lock(root):
        return _extract_locked(
            settings,
            destination,
            root,
            identity,
            provenance,
            source,
            objects,
            backbone,
            device,
            writer,
            started,
        )


def _extract_locked(
    settings,
    destination,
    root,
    identity,
    provenance,
    source,
    objects,
    backbone,
    device,
    writer,
    started,
):
    from morphofeatures.analysis.annotations import read_id_table
    from morphofeatures.dino import load_backbone
    from morphofeatures.dino_execution import encode_physical_objects
    from morphofeatures.dino_runtime import runtime_options

    report_path = root / "extraction.json"
    Path(destination).mkdir(parents=True, exist_ok=True)
    if report_path.exists() and runtime_options(settings)["resume"]:
        report = json.loads(report_path.read_text())
        required = [Path(p) for p in report["embeddings"].values()]
        required += [p.with_suffix(".metadata.json") for p in required]
        required += [
            root / name
            for name in (
                "objects.tsv",
                "eligibility.tsv",
                "eligibility.json",
                "views.jsonl",
                "coverage.tsv",
                "excluded.tsv",
                "input_preview.png",
                "input_preview.svg",
            )
        ]
        if all(p.is_file() for p in required):
            return materialize_extraction(root, destination, cache_hit=True)
    root.mkdir(parents=True, exist_ok=True)
    report_path.unlink(missing_ok=True)
    indexing_started = time.perf_counter()
    cohort_path = root / "objects.tsv"
    cohort_marker = root / "cohort.json"
    reused_index = False
    if (
        objects is None
        and runtime_options(settings)["resume"]
        and cohort_path.exists()
        and cohort_marker.exists()
    ):
        marker = json.loads(cohort_marker.read_text())
        if (
            marker.get("identity") != identity
            or marker.get("sha256") != hashlib.sha256(cohort_path.read_bytes()).hexdigest()
        ):
            raise ValueError(
                "The saved DINO object index is incompatible or corrupt; use a fresh extraction cache"
            )
        objects = read_id_table(cohort_path)
        objects["segmentation_id"] = read_id_table(
            cohort_path, id_column="segmentation_id"
        ).segmentation_id
        reused_index = True
    if objects is None:
        objects = index_source(source, root)
    temporary = root / "objects.tsv.tmp"
    objects.to_csv(temporary, sep="\t", index=False)
    temporary.replace(cohort_path)
    write_json_atomic(
        cohort_marker,
        {"identity": identity, "sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest()},
    )
    _, eligibility_summary = write_eligibility(source, objects, root)
    from morphofeatures.dino_memory import announce_plan, plan_memory

    # Check indexed boxes before loading a new backbone. Rechecked in the encoder
    # after model loading against the worker's current resident set and limits.
    announce_plan(plan_memory(settings, source=source, objects=objects, detect=True), root, writer)
    indexing_seconds = time.perf_counter() - indexing_started
    modes = (
        ["fixed_tiles", "whole_label"]
        if settings["views"]["mode"] == "both"
        else [settings["views"]["mode"]]
    )
    model = backbone if backbone is not None else load_backbone(settings)
    stride = 14 if settings["model"] == "dinov2" else 16
    input_config = inspection_config(source, root / "objects.tsv")
    outputs, excluded, coverage, performance = encode_physical_objects(
        settings,
        root,
        identity,
        source,
        objects,
        model,
        device,
        modes,
        writer,
    )
    performance["indexing_seconds"] = indexing_seconds
    performance["reused_object_index"] = reused_index
    export_started = time.perf_counter()
    paths = {}
    for name, (ids, vectors) in outputs.items():
        if not ids:
            relevant = [
                r for r in excluded if not r.get("mode") or r.get("mode") == name.split(".")[0]
            ]
            counts = Counter(r.get("reason_code", "no_eligible_views") for r in relevant)
            details = "; ".join(dict.fromkeys(r["reason"] for r in relevant))
            raise FieldError(
                f"No objects encoded for {name}: {len(relevant)} object exclusions ({dict(counts)}). {details[:1800]} Reports: {root / 'excluded.tsv'} and {root / 'coverage.tsv'}.",
                "views.min_foreground_fraction",
                "views.min_slice_foreground_fraction",
                "views.tile_size_um",
                "on_object_error",
            )
        path = export_embeddings(root / (name + ".npz"), ids, np.stack(vectors))
        mode, feature = name.split(".")
        config = deepcopy(settings)
        config["views"].update(mode=mode, feature=feature)
        config["views"].pop("features", None)
        config["config"] = {**settings["config"], **input_config}
        metadata = {
            **provenance,
            "settings": config,
            "model": settings["model"],
            "identity": identity + ":" + name,
            "dataset_identity": settings["dataset_identity"],
            "rows": len(ids),
            "dimensions": len(vectors[0]),
            "view_manifest": str(root / "views.jsonl"),
            "coverage": str(root / "coverage.tsv"),
            "excluded_objects": excluded,
            "eligibility": eligibility_summary,
            "extraction_seconds": time.perf_counter() - started,
        }
        write_json_atomic(path.with_suffix(".metadata.json"), metadata)
        paths[name] = path.resolve()
    save_input_preview(root, stride)
    performance["export_seconds"] = time.perf_counter() - export_started
    write_json_atomic(root / "performance.json", performance)
    write_json_atomic(
        report_path,
        {
            "peak_process_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "schema": "morphofeatures.dino_views.v1",
            "performance": performance,
            "embeddings": {k: str(p) for k, p in paths.items()},
            "source": source,
            "settings": settings,
            "eligibility": eligibility_summary,
            "seconds": time.perf_counter() - started,
        },
    )
    return materialize_extraction(root, destination, cache_hit=False)


def materialize_extraction(root, destination, *, cache_hit):
    """Keep worker exports self-contained even when inference uses a shared cache."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    local = root if root.is_relative_to(destination) else destination / "views"
    if local != root:
        shutil.copytree(
            root,
            local,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                "object-checkpoints", "pending-objects", "extraction.lock"
            ),
        )
        for path in local.glob("*.json"):
            path.write_text(path.read_text().replace(str(root), str(local)))
    report = json.loads((local / "extraction.json").read_text())
    write_json_atomic(
        destination / "result.json",
        {"report": str(local / "extraction.json"), "cache_hit": cache_hit},
    )
    return {k: Path(p) for k, p in report["embeddings"].items()}


def save_input_preview(root, stride):
    """Actual model inputs with physical captions; masks and token cells stay separate."""
    import matplotlib.pyplot as plt

    paths = sorted(Path(root).glob("preview-*.npz"))
    if not paths:
        return
    figure, axes = plt.subplots(
        len(paths), 2, figsize=(6.8, 3.3 * len(paths)), squeeze=False, layout="constrained"
    )
    for row, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as data:
            image, mask = data["image"], data["mask"]
            info = json.loads(str(data["metadata"]))
        for axis in axes[row]:
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
            axis.set_axis_off()
        axes[row, 0].set_title(
            f"ID {info['label_id']} · {info['mode']} · {'ZYX'[info['axis']]} slice\n{info['pixel_size_um']:.4g} µm/pixel · {image.shape[0] * info['pixel_size_um']:.3g} µm canvas",
            fontsize=9,
        )
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = [0.1, 0.8, 0.3, 0.3]
        axes[row, 1].imshow(overlay)
        for edge in range(stride, image.shape[0], stride):
            axes[row, 1].axhline(edge - 0.5, color="cyan", alpha=0.4, lw=0.3)
            axes[row, 1].axvline(edge - 0.5, color="cyan", alpha=0.4, lw=0.3)
        axes[row, 1].set_title(
            f"Foreground and spatial token grid\n{100 * info['foreground_fraction']:.1f}% foreground",
            fontsize=9,
        )
    for suffix in ("png", "svg"):
        figure.savefig(Path(root) / ("input_preview." + suffix), dpi=120, facecolor="white")
    plt.close(figure)
