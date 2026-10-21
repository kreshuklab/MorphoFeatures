"""A single exploratory coordinate system for independent datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from morphofeatures.analysis.annotations import join_labels, read_annotations
from morphofeatures.analysis.projection import project_embeddings
from morphofeatures.analysis.visualization import (
    category_colors,
    label_palette,
    save_projection_figure,
)
from morphofeatures.artifacts import write_json_atomic
from morphofeatures.data.io import export_embeddings, load_embeddings
from morphofeatures.representations import fingerprint_file
from morphofeatures.workflow_validation import FieldError, at_fields

SCHEMA = "morphofeatures.joint_analysis.v1"


def extraction_metadata(path, _seen=None):
    """Accept extraction artifacts and the copies kept in saved analysis reports."""
    path = Path(path)
    seen = set() if _seen is None else _seen
    identity = str(path.absolute())
    if identity in seen or len(seen) >= 8:
        return {}
    seen.add(identity)
    sidecar = path.with_suffix(".metadata.json")
    if sidecar.is_file():
        return json.loads(sidecar.read_text())
    report = path.parent / "analysis.json"
    if report.is_file():
        value = json.loads(report.read_text())
        if value.get("embedding") == path.name:
            if value.get("extraction"):
                return value["extraction"]
            # Older re-analysis reports dropped inherited metadata. Recover it
            # only while the recorded input still matches its saved fingerprint.
            source = value.get("source", {})
            if source.get("path"):
                original = Path(source["path"])
                if not original.is_absolute():
                    original = report.parent / original
                if original.is_file() and "size" in source and "mtime_ns" in source:
                    stat = original.stat()
                    if stat.st_size == source["size"] and stat.st_mtime_ns == source["mtime_ns"]:
                        return extraction_metadata(original, seen)
    return {}


def _signature(metadata):
    settings = metadata.get("settings", {})
    model = metadata.get("model", settings.get("model"))
    if model == "embedding_distillation":
        recipe = metadata.get("representation_recipe", {})
        return {"model": model, "checkpoint": recipe.get("parameters", {}).get("sha256"),
                "method": recipe.get("method"), "algorithm": recipe.get("algorithm")}
    if model == "spatial_context":
        return {"model": model, "checkpoint": "input context recipe", "context": metadata.get("representation_recipe", {})}
    signature = {
        "model": model,
        "checkpoint": metadata.get("sources", {}).get("checkpoint", {}).get("sha256"),
    }
    if model or metadata.get("adaptation"):
        signature["adaptation"] = metadata.get("adaptation", {"condition": "none"})
    if model and model not in {"dinov2", "dinov3", "fusion"}:
        data = settings.get("config", {}).get("data", {})
        signature["input layout"] = (
            "grouped patches" if data.get("source") == "n5_masked_patches" else "object crops"
        )
        if data.get("source") == "n5_masked_patches":
            signature["patch sampling"] = {
                key: data.get(key, default)
                for key, default in {
                    "group_size": 200,
                    "normalization": "dtype",
                    "mask_mode": "nonzero",
                    "min_foreground_fraction": 0.01,
                    "position_stride_zyx": [8, 8, 8],
                }.items()
            }
    if model in {"dinov2", "dinov3"}:
        signature["variant"] = settings.get("variant")
        signature["views"] = {
            **dict(
                axes=[0, 1, 2],
                fractions=[0.25, 0.5, 0.75],
                size=224,
                feature="cls",
                aggregation="mean",
                normalization="foreground_percentile",
                percentiles=[1, 99],
                mask=True,
                resize="stretch",
            ),
            **{k: v for k, v in settings.get("views", {}).items() if k != "batch_size"},
        }
        views = signature["views"]
        data = settings.get("config", {}).get("data", {})
        if data.get("source") == "grouped_crops":
            views["normalization"] = settings.get("views", {}).get("normalization", "unit")
            views["resize"] = settings.get("views", {}).get("resize", "letterbox")
            recipe = dict(data.get("patch_recipe", {}))
            if recipe.get("sampling", "physical_resampled") == "physical_resampled":
                # Default physical preparation is unchanged; older artifacts omit these fields.
                recipe.pop("sampling", None)
                recipe.pop("saved_voxel_size_um", None)
            recipe.pop("max_candidate_tiles", None)
            if recipe.get("exhaustive"):
                recipe.pop("max_tiles", None)
            signature["distributed patch recipe"] = recipe
        mode = views.get("mode", "stored")
        if mode == "stored":
            views.pop("mode", None)  # Omitted and explicit legacy defaults agree.
        else:
            # Compare settings actually used by this output, not other outputs
            # produced in the same multi-view extraction or resource safeguards.
            views.pop("fractions", None)
            views.pop("max_candidate_tiles", None)
            views.setdefault("resize", "letterbox")
            if "resize" not in settings.get("views", {}):
                views["resize"] = "letterbox"
            if mode == "whole_label":
                views.setdefault("coarse_fractions", [0.1, 0.3, 0.5, 0.7, 0.9])
                for key in (
                    "tile_size_um",
                    "tile_anchor",
                    "max_tiles",
                    "exhaustive",
                    "fine_fractions",
                    "min_foreground_fraction",
                    "min_slice_foreground_fraction",
                ):
                    views.pop(key, None)
            else:
                views.pop("coarse_fractions", None)
                for key, default in dict(
                    tile_size_um=1.0,
                    tile_anchor="bbox_min",
                    max_tiles=64,
                    exhaustive=False,
                    fine_fractions=[0.5],
                    min_foreground_fraction=0.01,
                    min_slice_foreground_fraction=0.0,
                ).items():
                    views.setdefault(key, default)
                if views["exhaustive"]:
                    views.pop("max_tiles", None)
        signature["view_algorithm"] = settings.get("view_algorithm", "stored-v1")
    if model == "fusion":
        from morphofeatures.analysis.recipe_compatibility import canonical_recipe

        signature["fusion"] = canonical_recipe(metadata.get("fusion_recipe"))
        # This is a derived recipe digest, not a model checkpoint. Older digests
        # included dataset preparation; compare the actual component recipes.
        if signature["fusion"]:
            signature["checkpoint"] = "component recipes"
    return signature


def validate_joint_datasets(settings, *, available=(), require_files=True):
    """Check dataset identities and provenance without fitting or allocating a run."""
    from morphofeatures.analysis.projection import validate_projection_settings
    from morphofeatures.analysis.variants import validate_variants

    validate_projection_settings(settings)
    validate_variants(settings)
    datasets = settings.get("datasets")
    if not isinstance(datasets, list) or len(datasets) < 2:
        raise FieldError("Shared projection requires at least two datasets.", "datasets")
    names = set()
    signatures = []
    metadata_by_name = {}
    unverified = []
    for i, dataset in enumerate(datasets):
        prefix = ("datasets", i)
        if not isinstance(dataset, dict):
            raise FieldError("Each dataset must be a mapping.", prefix)
        name = dataset.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise FieldError("Give every dataset a unique, nonempty name.", prefix + ("name",))
        names.add(name)
        link = dataset.get("from_embedding")
        if link and link not in available:
            raise FieldError(
                "Choose an earlier named extraction stage.", prefix + ("from_embedding",)
            )
        path = dataset.get("embedding")
        if not path and not link:
            raise FieldError(
                "Select saved embeddings or an earlier extraction for this dataset.",
                prefix + ("embedding",),
            )
        if path and link:
            raise FieldError(
                "Choose either saved embeddings or an earlier extraction.",
                prefix + ("embedding",),
                prefix + ("from_embedding",),
            )
        for key in ("embedding", "annotations", "input_config"):
            if dataset.get(key) and require_files and not Path(dataset[key]).is_file():
                raise FieldError(
                    f"{name}: {key.replace('_', ' ')} file does not exist.", prefix + (key,)
                )
        if not require_files:
            continue
        if dataset.get("annotations"):
            with at_fields(prefix + ("annotations",), prefix + ("label_column",)):
                read_annotations(dataset["annotations"], dataset)
        if path:
            with at_fields(prefix + ("embedding",)):
                metadata = extraction_metadata(path)
                signature = _signature(metadata)
                metadata_by_name[name] = metadata
            if not signature["model"] or not signature["checkpoint"]:
                unverified.append(name)
            signatures.append((i, name, signature))
    from morphofeatures.analysis.recipe_compatibility import (
        _verified,
        compare_recipes,
        compatibility_message,
    )

    comparisons = []
    unverified = [name for _, name, signature in signatures if not _verified(signature)]
    for offset, (i, name, _) in enumerate(signatures):
        for j, other_name, _ in signatures[offset + 1 :]:
            comparison = compare_recipes(
                metadata_by_name[name], metadata_by_name[other_name], name, other_name
            )
            if comparison["differences"]:
                raise FieldError(
                    compatibility_message(comparison),
                    ("datasets", i, "embedding"),
                    ("datasets", j, "embedding"),
                )
            if comparison["preparation_differences"]:
                comparisons.append(comparison)
    if unverified and not settings.get("assume_same_model", False):
        raise FieldError(
            "Model provenance is unavailable for "
            + ", ".join(unverified)
            + ". Supply extraction metadata, or confirm that these files used the same model and feature settings.",
            "assume_same_model",
        )
    with at_fields("max_objects_per_dataset"):
        limit = int(settings.get("max_objects_per_dataset", 0))
        if limit < 0:
            raise ValueError("Maximum objects per dataset must be zero (all) or positive.")
    if settings.get("subset", 0):
        raise FieldError(
            "Use max_objects_per_dataset to sample a joint projection; subset must be zero.",
            "subset",
        )
    if settings.get("classify"):
        raise FieldError(
            "Joint projection does not fit classifiers. Use a separate analysis with specimen-aware evaluation.",
            "classify",
        )
    return {
        "verified": not unverified,
        "unverified_datasets": unverified,
        **({"input_compatibility": comparisons} if comparisons else {}),
    }


def joint_analyze(settings, destination):
    """Pool rows, fit once, and retain native-ID reports for each spatial source."""
    from morphofeatures.representation_analysis import package_results

    verification = validate_joint_datasets(settings)
    tables = [load_embeddings(d["embedding"]).sorted() for d in settings["datasets"]]
    if any(not len(t.label_ids) for t in tables):
        raise FieldError("Each dataset must contain at least one object.", "datasets")
    if len({t.features.shape[1] for t in tables}) != 1:
        raise FieldError("All datasets must have the same embedding feature dimension.", "datasets")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(settings.get("seed", 42)))
    limit = int(settings.get("max_objects_per_dataset", 0))
    index_parts, features, selected_ids = [], [], []
    offset = 1
    for dataset, table in zip(settings["datasets"], tables):
        ids = np.arange(offset, offset + len(table.label_ids), dtype=np.int64)
        offset += len(ids)
        labels = join_labels(table.label_ids, dataset).rename(
            columns={"label_id": "source_label_id"}
        )
        labels.insert(0, "label_id", ids)
        labels.insert(1, "dataset", dataset["name"])
        rows = np.arange(len(ids))
        if limit and len(rows) > limit:
            rows = np.sort(rng.choice(rows, limit, replace=False))
        labels["projected"] = np.isin(ids, ids[rows])
        index_parts.append(labels)
        features.append(table.features[rows])
        selected_ids.append(ids[rows])
    index = pd.concat(index_parts, ignore_index=True)
    pooled = np.concatenate(features)
    ids = np.concatenate(selected_ids)
    frame, diagnostics = project_embeddings(ids, pooled, settings)
    frame = frame.merge(index, on="label_id", validate="one_to_one")
    palette = label_palette(index)
    cluster_colors = category_colors(frame.cluster)
    frame.attrs.update(class_colors=palette, cluster_colors=cluster_colors)
    result = {
        "schema": SCHEMA,
        "settings": settings,
        "coordinates": "coordinates.tsv",
        "object_index": "object_index.tsv",
        "embedding": "embeddings.npz",
        "class_colors": palette,
        "cluster_colors": cluster_colors,
        "diagnostics": diagnostics,
        "model_verification": verification,
        "datasets": {},
        "interpretation": (
            "Normalization, PCA, UMAP (if enabled), and clustering are fitted once to pooled objects. "
            "Dataset separation may reflect biology, preparation or acquisition effects. Larger datasets "
            "have greater influence unless capped. These coordinates are exploratory, not a classification "
            "test or a fixed reference transform. Joint label_id values are internal: use dataset and "
            "source_label_id, or a per-dataset report, for spatial correspondence."
        ),
    }
    export_embeddings(destination / "embeddings.npz", ids, pooled)
    index.to_csv(destination / "object_index.tsv", sep="\t", index=False)
    frame.to_csv(destination / "coordinates.tsv", sep="\t", index=False)
    save_projection_figure(
        frame, destination / "projection.svg", "Shared space across datasets", settings
    )
    for i, (dataset, table) in enumerate(zip(settings["datasets"], tables)):
        folder = destination / f"dataset-{i}"
        folder.mkdir(exist_ok=True)
        native = (
            frame[frame.dataset == dataset["name"]]
            .drop(columns=["label_id", "dataset", "projected"])
            .rename(columns={"source_label_id": "label_id"})
        )
        native.attrs.update(class_colors=palette, cluster_colors=cluster_colors)
        export_embeddings(folder / "embeddings.npz", table.label_ids, table.features)
        native.to_csv(folder / "coordinates.tsv", sep="\t", index=False)
        labels = join_labels(table.label_ids, dataset).merge(
            native[["label_id", "cluster"]], on="label_id", how="left", validate="one_to_one"
        )
        labels.to_csv(folder / "object_labels.tsv", sep="\t", index=False)
        extraction = extraction_metadata(dataset["embedding"])
        if extraction:
            write_json_atomic(folder / "embeddings.metadata.json", extraction)
        report = {
            "schema": "morphofeatures.analysis.v1",
            "settings": {**settings, **dataset, "classify": False},
            "source": fingerprint_file(dataset["embedding"]),
            "extraction": extraction,
            "embedding": "embeddings.npz",
            "coordinates": "coordinates.tsv",
            "object_labels": "object_labels.tsv",
            "class_colors": palette,
            "cluster_colors": cluster_colors,
            "shared_projection": {"report": "../joint_analysis.json", "dataset": dataset["name"]},
            "diagnostics": diagnostics,
            "interpretation": "Coordinates and cluster IDs are taken from the pooled projection. Object IDs belong to this dataset. No projection was refitted for this report.",
        }
        write_json_atomic(folder / "analysis.json", report)
        save_projection_figure(
            native, folder / "projection.svg", dataset["name"] + " · shared coordinates", settings
        )
        result["datasets"][dataset["name"]] = {
            "directory": folder.name,
            "report": folder.name + "/analysis.json",
            "objects": len(table.label_ids),
            "projected_objects": len(native),
            "source": report["source"],
        }
    path = write_json_atomic(destination / "joint_analysis.json", result)
    from morphofeatures.analysis.variants import run_configured_variants

    run_configured_variants(path, settings)
    package_results(destination)
    return path
