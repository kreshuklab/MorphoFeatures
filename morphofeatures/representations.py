"""ID-preserving, cached extraction shared by MAE and pretrained backbones."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path

import yaml

from morphofeatures.artifacts import write_json_atomic
from morphofeatures.data.io import export_embeddings, load_embeddings
from morphofeatures.metrics import utc_now


def fingerprint_file(path):
    path = Path(path).resolve()
    stat = path.stat()
    value = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if path.is_file() and (stat.st_size < 1024 * 1024 or path.suffix in {".pt", ".pth"}):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        value["sha256"] = digest.hexdigest()
    if path.is_dir() and (path / "attributes.json").exists():
        value["attributes"] = json.loads((path / "attributes.json").read_text())
    return value


def extraction_identity(settings):
    paths = {"checkpoint": fingerprint_file(settings["checkpoint"])}
    for key in ("crops", "label_ids", "loss_masks", "patches_container", "positions_container"):
        value = settings.get("config", {}).get("data", {}).get(key)
        if isinstance(value, str) and value:
            paths[key] = fingerprint_file(value)
    if settings.get("model_repository"):
        from morphofeatures.experiments import collect_provenance

        paths["model_repository"] = collect_provenance(Path(settings["model_repository"]))
        paths["model_repository"].pop("timestamp", None)
    scientific = {k: v for k, v in settings.items() if k not in {"cache", "action"}}
    payload = {"settings": scientific, "sources": paths, "schema": "morphofeatures.embedding.v1"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def extract_representation(settings, destination, writer=None):
    data = settings.get("config", {}).get("data", {})
    if (
        data.get("crops")
        and data.get("label_ids") is None
        and not settings.get("sequential_ids", False)
    ):
        raise ValueError(
            "Provide data.label_ids for segmentation correspondence, or explicitly enable sequential_ids for IDs 1..N"
        )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    identity, provenance = extraction_identity(settings)
    cache = Path(settings.get("cache", destination))
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / f"{settings.get('model', 'mae')}-{identity}.npz"
    metadata_path = output.with_suffix(".metadata.json")
    if output.exists() and metadata_path.exists():
        load_embeddings(output)
        write_json_atomic(
            destination / "result.json", {"embedding": str(output.resolve()), "cache_hit": True}
        )
        if writer:
            writer.write("cache_hit", embedding=str(output))
        return output
    started = time.perf_counter()
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    excluded = []
    if settings.get("model", "mae") == "mae":
        from morphofeatures.mae3d import encode_from_config
        from morphofeatures.workspace_jobs import scope_training_outputs

        config_path = destination / "extraction_config.yaml"
        config_path.write_text(
            yaml.safe_dump(scope_training_outputs(settings["config"], destination), sort_keys=False)
        )
        encode_from_config(config_path, Path(settings["checkpoint"]), output)
    else:
        from morphofeatures.dino import extract_dino

        ids, features, excluded = extract_dino(settings, progress=writer)
        export_embeddings(output, ids, features)
    table = load_embeddings(output)
    metadata = {
        **provenance,
        "identity": identity,
        "created": utc_now(),
        "model": settings.get("model", "mae"),
        "rows": len(table.label_ids),
        "dimensions": table.features.shape[1],
        "dtype": str(table.features.dtype),
        "feature_bytes": table.features.nbytes,
        "artifact_bytes": output.stat().st_size,
        "extraction_seconds": time.perf_counter() - started,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "training_cost": settings.get("training_cost", "not measured here"),
        "excluded_objects": excluded,
    }
    if torch.cuda.is_available():
        metadata["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    from importlib.metadata import PackageNotFoundError, version

    metadata["packages"] = {}
    for package in ("numpy", "torch", "morphofeatures"):
        try:
            metadata["packages"][package] = version(package)
        except PackageNotFoundError:
            pass
    preprocessing = settings.get("config", {}).get("data", {}).get("preprocessing")
    if preprocessing and Path(preprocessing).is_file():
        metadata["preprocessing"] = json.loads(Path(preprocessing).read_text())
    if output.with_suffix(output.suffix + ".metadata.json").exists():
        metadata["mae_details"] = json.loads(
            output.with_suffix(output.suffix + ".metadata.json").read_text()
        )
    write_json_atomic(metadata_path, metadata)
    write_json_atomic(
        destination / "result.json", {"embedding": str(output.resolve()), "cache_hit": False}
    )
    return output
