"""Reproducible DINO fine/coarse experiments composed from normal analysis artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from morphofeatures.analysis.annotations import read_annotations
from morphofeatures.analysis.fusion import fuse_embeddings
from morphofeatures.analysis.joint import joint_analyze
from morphofeatures.analysis.transfer import transfer_labels
from morphofeatures.artifacts import write_json_atomic
from morphofeatures.data.dino_views import index_source, validate_source
from morphofeatures.data.object_eligibility import write_eligibility
from morphofeatures.dino import VARIANTS, validate_dino_settings
from morphofeatures.dino_physical import extract_physical
from morphofeatures.representation_analysis import compare, package_results
from morphofeatures.workflow_validation import FieldError, at_fields

FUSION_METHODS = {
    "balanced": "Balanced block L2",
    "raw": "Raw concatenation",
}
FEATURE_BLOCKS = {
    "fine_cls": ("Fine CLS", "fixed_tiles.cls"),
    "fine_foreground": ("Fine foreground", "fixed_tiles.foreground_mean"),
    "coarse_cls": ("Coarse CLS", "whole_label.cls"),
    "coarse_foreground": ("Coarse foreground", "whole_label.foreground_mean"),
}
FUSION_COMBINATIONS = {
    "cls_cls": ("Fine CLS + coarse CLS", ["fine_cls", "coarse_cls"]),
    "cls_foreground": ("Fine CLS + coarse foreground", ["fine_cls", "coarse_foreground"]),
    "foreground_foreground": ("Fine foreground + coarse foreground", ["fine_foreground", "coarse_foreground"]),
    "fine_both": ("Both fine features", ["fine_cls", "fine_foreground"]),
    "coarse_both": ("Both coarse features", ["coarse_cls", "coarse_foreground"]),
    "all_four": ("All four features", list(FEATURE_BLOCKS)),
}


def fusion_recipes(settings):
    """Expand readable presets and legacy pairings without changing old recipes."""
    from morphofeatures.analysis.fusion import validate_fusion_settings

    modes = settings.get("combination_methods", ["balanced"])
    selected = settings.get("additional_combinations", [])
    for values, allowed, key in (
        (modes, FUSION_METHODS, "combination_methods"),
        (selected, FUSION_COMBINATIONS, "additional_combinations"),
    ):
        if (not isinstance(values, list) or any(not isinstance(v, str) or v not in allowed for v in values)
                or len(set(values)) != len(values) or (key == "combination_methods" and not values)):
            raise FieldError(f"Choose distinct options for {key}: {', '.join(allowed)}.", key)
    weights = settings.get("fusion_weights", [0.5, 0.5])
    with at_fields("fusion_weights"):
        validate_fusion_settings({"embeddings": {"fine": "", "coarse": ""}, "weights": weights})
    specs = [("Fine + coarse", ["fine_foreground", "coarse_cls"], weights, modes)]
    for code in selected:
        name, blocks = FUSION_COMBINATIONS[code]
        block_weights = [weights[0 if block.startswith("fine_") else 1] /
                         sum(other.split("_")[0] == block.split("_")[0] for other in blocks)
                         for block in blocks]
        # A within-scale pairing is meaningful even when that scale has zero
        # weight in a separate fine/coarse experiment.
        if all(b.startswith("fine_") for b in blocks) or all(b.startswith("coarse_") for b in blocks):
            block_weights = [1 / len(blocks)] * len(blocks)
        specs.append((name, blocks, block_weights, modes))
    extra = settings.get("extra_pairings", [])
    if not isinstance(extra, list):
        raise FieldError("Additional pairings must be a list of named fine/coarse recipes.", "extra_pairings")
    for i, pairing in enumerate(extra):
        if not isinstance(pairing, dict) or any(pairing.get(k) not in {"cls", "foreground_mean"} for k in ("fine", "coarse")):
            raise FieldError("Choose cls or foreground_mean for fine and coarse.", ("extra_pairings", i))
        blocks = [f"{prefix}_" + ("cls" if pairing[prefix] == "cls" else "foreground") for prefix in ("fine", "coarse")]
        specs.append((pairing.get("name"), blocks, pairing.get("weights", weights),
                      [pairing["combination"]] if "combination" in pairing else modes))
    result, names = [], {v[0] for v in FEATURE_BLOCKS.values()}
    for name, blocks, block_weights, methods in specs:
        if not isinstance(name, str) or not name.strip():
            raise FieldError("Each combination needs a nonempty name.", "extra_pairings")
        for method in methods:
            if method not in FUSION_METHODS:
                raise FieldError("Combination must be balanced or raw.", "extra_pairings")
            title = f"{name} · {FUSION_METHODS[method]}" if len(methods) > 1 else name
            if title in names:
                raise FieldError(f"Combination name is repeated: {title}", "extra_pairings", "additional_combinations")
            names.add(title)
            recipe = {"name": title, "blocks": blocks, "weights": block_weights, "combination": method}
            validate_fusion_settings({**recipe, "embeddings": {b: "" for b in blocks}})
            result.append(recipe)
    return result


def resolve_paths(value, base):
    """Resolve nested scientific inputs while preserving URLs and dataset names."""
    paths = {
        "raw",
        "segmentation",
        "object_index",
        "ids_file",
        "mapping_file",
        "id_mapping",
        "annotations",
        "embedding",
        "checkpoint",
        "model_repository",
        "cache",
        "cache_directory",
        "scratch_directory",
        "reference_model",
        "input_config",
    }
    if isinstance(value, list):
        return [resolve_paths(v, base) for v in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for k, v in value.items():
        if k in paths and isinstance(v, str) and v and not v.startswith(("https://", "http://")):
            result[k] = str((Path(base) / Path(v).expanduser()).resolve())
        elif k == "embeddings" and isinstance(v, dict):
            result[k] = {
                name: str((Path(base) / Path(p).expanduser()).resolve()) for name, p in v.items()
            }
        else:
            result[k] = resolve_paths(v, base)
    return result


def experiment_defaults():
    return {
        "action": "dino_experiment",
        "datasets": [
            {
                "name": "Platynereis",
                "source": {
                    "source": "local",
                    "raw": "",
                    "segmentation": "",
                    "unit": "um",
                    "raw_spacing_zyx": [0.025, 0.020, 0.020],
                    "segmentation_spacing_zyx": [0.1, 0.08, 0.08],
                },
                "annotations": None,
                "label_column": "auto",
                "group_column": None,
            }
        ],
        "backbones": [
            {
                "model": "dinov2",
                "variant": "dinov2_vits14",
                "checkpoint": "",
                "model_repository": "",
            }
        ],
        "views": {
            "mode": "both",
            "size": 224,
            "axes": [0, 1, 2],
            "fine_fractions": [0.5],
            "coarse_fractions": [0.1, 0.3, 0.5, 0.7, 0.9],
            "tile_size_um": 1.0,
            "max_tiles": 64,
            "min_foreground_fraction": 0.01,
            "min_slice_foreground_fraction": 0.0,
            "tile_anchor": "bbox_min",
            "exhaustive": False,
            "features": ["cls", "foreground_mean"],
            "normalization": "foreground_percentile",
            "percentiles": [1, 99],
            "resize": "letterbox",
            "mask": True,
            "aggregation": "mean",
            "batch_size": 16,
        },
        "tile_sizes_um": [1.0],
        "fusion_weights": [0.5, 0.5],
        "max_objects": 100,
        "seed": 42,
        "device": "auto",
        "umap": True,
        "clusters": 8,
        "classify": True,
        "transfer": {
            "enabled": False,
            "reference": "Platynereis",
            "conditions": ["uncorrected", "coral"],
            "classifier": "logistic",
            "min_confidence": 0.6,
            "distance_quantile": 0.99,
        },
    }


def validate_experiment(settings, *, sources_only=False):
    from morphofeatures.analysis.projection import validate_projection_settings
    from morphofeatures.analysis.variants import validate_variants

    if not sources_only:
        validate_projection_settings(settings)
        validate_variants(settings)
    datasets = settings.get("datasets", [])
    if not datasets:
        raise FieldError("Add at least one dataset", "datasets")
    names = [d.get("name") for d in datasets]
    if any(not isinstance(n, str) or not n.strip() for n in names) or len(set(names)) != len(names):
        raise FieldError("Dataset names must be nonempty and unique", "datasets")
    for i, dataset in enumerate(datasets):
        try:
            with at_fields(("datasets", i, "source")):
                validate_source(dataset.get("source", {}))
        except FieldError as error:
            if error.fields and error.fields[0][:1] != ("datasets",):
                raise FieldError(
                    str(error), *(("datasets", i, "source") + path for path in error.fields)
                ) from error
            raise
        if not sources_only and dataset.get("annotations"):
            with at_fields(("datasets", i, "annotations")):
                read_annotations(dataset["annotations"], dataset)
    from morphofeatures.dino_physical import validate_physical_settings

    validate_physical_settings({"views": settings.get("views", {})}, check_source=False)
    # Prepare data can be checked before weights or execution options are ready.
    # Reuse the same image-contract validation as extraction for each chosen family.
    families = {b.get("model") for b in settings.get("backbones", [])} & set(VARIANTS)
    image_views = {k: v for k, v in settings.get("views", {}).items() if k in {"size", "axes", "percentiles"}}
    for family in sorted(families or {"dinov2"}):
        validate_dino_settings({"model": family, "views": image_views}, require_data=False)
    with at_fields("max_objects"):
        if int(settings.get("max_objects", 100)) < 0:
            raise ValueError("Object limit must be zero (all) or positive")
    scales = settings.get("tile_sizes_um", [1.0])
    if not scales:
        raise FieldError("Select at least one physical tile width", "tile_sizes_um")
    for scale in scales:
        with at_fields("tile_sizes_um"):
            values = np.broadcast_to(np.asarray(scale, dtype=float), (3,))
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError("Every tile width must be positive micrometers")
    if sources_only:
        return
    if not settings.get("backbones"):
        raise FieldError("Configure a local DINO checkpoint", "backbones")
    for i, backbone in enumerate(settings["backbones"]):
        with at_fields(("backbones", i)):
            family = backbone.get("model")
            if backbone.get("variant") not in VARIANTS.get(family, set()):
                raise ValueError("Choose a supported DINOv2/v3 variant")
            if not Path(backbone.get("checkpoint", "")).is_file():
                raise ValueError("Select existing local backbone weights")
            if not (Path(backbone.get("model_repository", "")) / "hubconf.py").is_file():
                raise ValueError("Select an official local DINO checkout")
            validate_dino_settings(
                {**backbone, "views": settings.get("views", {}), "runtime": settings.get("runtime", {})}, require_data=False
            )
    names_used = {v[0] for v in FEATURE_BLOCKS.values()} | {r["name"] for r in fusion_recipes(settings)}
    for i, dataset in enumerate(datasets):
        baseline_names = set(names_used)
        for baseline in dataset.get("baselines", []):
            with at_fields(("datasets", i, "baselines")):
                if not baseline.get("name") or baseline["name"] in baseline_names:
                    raise ValueError(
                        "Baseline names must be unique and differ from generated names"
                    )
                baseline_names.add(baseline["name"])
                if not Path(baseline.get("embedding", "")).is_file():
                    raise ValueError("Choose an existing saved baseline embedding")
    transfer = settings.get("transfer", {})
    if transfer.get("enabled"):
        from morphofeatures.analysis.transfer import validate_transfer_settings

        try:
            validate_transfer_settings(transfer)
        except FieldError as error:
            raise FieldError(
                str(error), *(("transfer",) + path for path in error.fields)
            ) from error
        if len(datasets) < 2 or transfer.get("reference") not in names:
            raise FieldError(
                "Label transfer needs a reference and at least one other dataset",
                "transfer.reference",
            )
        if not datasets[names.index(transfer["reference"])].get("annotations"):
            raise FieldError(
                "Reference annotations are required for transfer",
                ("datasets", names.index(transfer["reference"]), "annotations"),
            )


def run_experiment(settings, destination, *, writer=None, backbones=None):
    """Extract first, then compare every method on one matched cohort per dataset."""
    import json

    import numpy as np

    from morphofeatures.analysis.fusion import dataset_namespace
    from morphofeatures.analysis.joint import extraction_metadata
    from morphofeatures.analysis.projection import analysis_normalizations
    from morphofeatures.data.io import export_embeddings, load_embeddings
    from morphofeatures.dino import load_backbone

    pairings = fusion_recipes(settings)
    normalizations = analysis_normalizations(settings, default="l2")

    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    datasets = settings["datasets"]
    sources, cohorts, runs, eligibility = [], [], [], []
    for index, dataset in enumerate(datasets):
        try:
            source = validate_source(dataset["source"])
            source.update(max_objects=settings.get("max_objects", 100), seed=settings.get("seed", 42))
            sources.append(source)
            cohorts.append(index_source(source, destination / f"dataset-{index}" / "input"))
            input_root = destination / f"dataset-{index}" / "input"
            _, summary = write_eligibility(source, cohorts[-1], input_root)
            eligibility.append(
                {
                    "dataset": dataset["name"],
                    "summary": summary,
                    "table": str(input_root / "eligibility.tsv"),
                }
            )
        except FieldError as error:
            raise FieldError(
                f"Dataset {dataset['name']}: {error}",
                *(("datasets", index, "source") + path for path in error.fields),
            ) from error
    report = {
        "schema": "morphofeatures.dino_experiment.v1",
        "settings": settings,
        "fusion_recipes": pairings,
        "analysis_normalizations": normalizations,
        "comparisons": [],
        "joint_analyses": [],
        "transfers": [],
        "embeddings": [],
        "original_embeddings": [],
        "execution_state": "partial",
        "cohorts": [],
        "object_eligibility": eligibility,
        "object_selections": [
            {"dataset": dataset["name"], "report": str(destination / f"dataset-{i}" / "input/selection.json"),
             "table": str(destination / f"dataset-{i}" / "input/selection.tsv")}
            for i, dataset in enumerate(datasets)
            if (destination / f"dataset-{i}" / "input/selection.json").exists()
        ],
        "interpretation": "Every representation uses the same matched cohort and evaluation folds within each dataset. Each representation has its own fitted PCA/UMAP. Fusion concatenates full feature vectors before projection.",
    }
    def publish():
        write_json_atomic(destination / "experiment.json", report)

    publish()
    from morphofeatures.dino_memory import announce_plan, plan_memory

    report["memory_plans"] = []
    for index, source in enumerate(sources):
        plan = plan_memory(settings, source=source, objects=cohorts[index], detect=True)
        announce_plan(plan, destination / f"dataset-{index}" / "input", writer, phase=f"experiment preflight: {datasets[index]['name']}")
        report["memory_plans"].append({"dataset": datasets[index]["name"], **plan})
    names = {
        "Fine CLS": "fixed_tiles.cls",
        "Fine foreground": "fixed_tiles.foreground_mean",
        "Coarse CLS": "whole_label.cls",
        "Coarse foreground": "whole_label.foreground_mean",
    }
    for b, backbone in enumerate(settings["backbones"]):
        family = f"{backbone['model']}-{b}"
        model = load_backbone(backbone) if backbones is None else backbones[b]
        coarse_outputs = {}
        for scale_index, scale in enumerate(
            settings.get("tile_sizes_um", [settings["views"].get("tile_size_um", 1.0)])
        ):
            maps = []
            for index, dataset in enumerate(datasets):
                root = destination / family / f"scale-{scale_index}" / f"dataset-{index}"
                views = {
                    **settings["views"],
                    "mode": "both" if scale_index == 0 else "fixed_tiles",
                    "tile_size_um": scale,
                    "features": ["cls", "foreground_mean"],
                }
                extraction = {
                    **backbone,
                    "views": views,
                    "runtime": settings.get("runtime", {}),
                    "config": {
                        "device": settings.get("device", "auto"),
                        "data": {"source": "physical_volume", "volume": sources[index]},
                    },
                }
                if settings.get("cache"):
                    extraction["cache"] = settings["cache"]
                try:
                    outputs = extract_physical(
                        extraction,
                        root / "extract",
                        source=sources[index],
                        objects=cohorts[index],
                        writer=writer,
                        backbone=model,
                    )
                except (ValueError, OSError) as error:
                    fields = []
                    if isinstance(error, FieldError):
                        for path in error.fields:
                            if path[:3] == ("config", "data", "volume"):
                                fields.append(("datasets", index, "source") + path[3:])
                            elif path[:1] in {("views",), ("runtime",)}:
                                fields.append(path)
                            else:
                                fields.append(("datasets", index, "source") + path)
                    raise FieldError(
                        f"Dataset {dataset['name']} · {backbone.get('variant', family)} · fine width {scale} µm: {error}",
                        *(
                            fields
                            or [("datasets", index, "source"), ("views",), ("tile_sizes_um",)]
                        ),
                    ) from error
                if scale_index == 0:
                    coarse_outputs[index] = {
                        k: v for k, v in outputs.items() if k.startswith("whole_label.")
                    }
                else:
                    outputs.update(coarse_outputs[index])
                representations = {name: outputs[key] for name, key in names.items()}
                report["original_embeddings"].extend(
                    {"dataset": dataset["name"], "backbone": family, "scale_um": scale,
                     "representation": name, "path": str(path), "original_path": str(path)}
                    for name, path in representations.items()
                )
                publish()
                for j, pairing in enumerate(pairings):
                    representations[pairing["name"]] = fuse_embeddings(
                        {
                            "embeddings": {
                                block: str(outputs[FEATURE_BLOCKS[block][1]]) for block in pairing["blocks"]
                            },
                            "weights": pairing["weights"],
                            "combination": pairing.get("combination", "balanced"),
                        },
                        root / f"fusion-{j}",
                    )
                    report["original_embeddings"].append(
                        {"dataset": dataset["name"], "backbone": family, "scale_um": scale,
                         "representation": pairing["name"], "path": str(representations[pairing["name"]]),
                         "original_path": str(representations[pairing["name"]])}
                    )
                    publish()
                for baseline_index, baseline in enumerate(dataset.get("baselines", [])):
                    if baseline["name"] in representations:
                        raise ValueError(
                            "Saved baseline names must differ from generated representation names"
                        )
                    meta = extraction_metadata(baseline["embedding"])
                    namespace = meta.get("dataset_identity")
                    expected = dataset_namespace(extraction_metadata(next(iter(outputs.values()))))
                    if namespace and namespace != expected:
                        raise ValueError(
                            "Saved baseline belongs to a different dataset/object namespace"
                        )
                    if not namespace and not baseline.get("assume_same_dataset", False):
                        raise ValueError(
                            "Saved baseline has no dataset provenance; check IDs and explicitly set assume_same_dataset"
                        )
                    table = load_embeddings(baseline["embedding"])
                    saved = export_embeddings(
                        root / f"baseline-{baseline_index}" / "embeddings.npz",
                        table.label_ids,
                        table.features,
                    )
                    write_json_atomic(
                        saved.with_suffix(".metadata.json"),
                        {
                            **meta,
                            "dataset_identity": expected,
                            "dataset_identity_verified": "recorded provenance"
                            if namespace
                            else "user confirmation",
                            "baseline_source": str(baseline["embedding"]),
                        },
                    )
                    representations[baseline["name"]] = saved
                maps.append(representations)
            runs.append(
                {"backbone": family, "scale_um": scale, "scale_index": scale_index, "maps": maps}
            )
        del model
    # Intersect across checkpoints, scales, pairings and saved baselines, once.
    for index, dataset in enumerate(datasets):
        selected = set(cohorts[index].label_id)
        matched = selected.copy()
        available = []
        for run in runs:
            for name, path in run["maps"][index].items():
                ids = set(load_embeddings(path).label_ids)
                matched &= ids
                available.append((run, name, path, ids))
        if len(matched) < 3:
            raise ValueError(
                f"{dataset['name']}: fewer than three matched objects across all variants; inspect extraction exclusions or remove incomplete baselines"
            )
        ids = np.array(sorted(matched), dtype=np.int64)
        exclusions = []
        for run, name, path, present in available:
            exclusions.extend(
                {
                    "label_id": int(i),
                    "backbone": run["backbone"],
                    "scale_um": str(run["scale_um"]),
                    "representation": name,
                    "reason": "Missing from representation",
                }
                for i in sorted(selected - present)
            )
            table = load_embeddings(path).sorted()
            root = (
                destination / run["backbone"] / f"scale-{run['scale_index']}" / f"dataset-{index}"
            )
            n = list(run["maps"][index]).index(name)
            artifact = export_embeddings(
                root / f"matched-{n}" / "embeddings.npz",
                ids,
                table.features[np.searchsorted(table.label_ids, ids)],
            )
            if settings.get("export_tsv", False):
                export_embeddings(
                    artifact.with_suffix(".tsv"),
                    ids,
                    table.features[np.searchsorted(table.label_ids, ids)],
                )
            metadata = extraction_metadata(path)
            write_json_atomic(
                artifact.with_suffix(".metadata.json"),
                {
                    **metadata,
                    "cohort_selection": {
                        "original_artifact": str(path),
                        "matched_objects": len(ids),
                    },
                },
            )
            run["maps"][index][name] = artifact
            report["embeddings"].append(
                {
                    "dataset": dataset["name"],
                    "backbone": run["backbone"],
                    "scale_um": run["scale_um"],
                    "representation": name,
                    "path": str(artifact),
                    "original_path": str(path),
                }
            )
            publish()
        cohort_root = destination / f"dataset-{index}"
        pd.DataFrame({"label_id": ids}).to_csv(
            cohort_root / "matched_ids.tsv", sep="\t", index=False
        )
        pd.DataFrame(
            exclusions, columns=["label_id", "backbone", "scale_um", "representation", "reason"]
        ).to_csv(cohort_root / "excluded_from_comparison.tsv", sep="\t", index=False)
        report["cohorts"].append(
            {
                "dataset": dataset["name"],
                "selected": len(selected),
                "matched": len(ids),
                "matched_ids": f"dataset-{index}/matched_ids.tsv",
            }
        )
    projection_options = {
        k: settings[k]
        for k in (
            "umap",
            "clusters",
            "cluster_method",
            "neighbors",
            "min_dist",
            "umap_epochs",
            "umap_pca_components",
            "cluster_pca_components",
            "analysis_variants",
            "resolution",
            "seed",
            "unlabeled_opacity",
        )
        if k in settings
    }
    metrics = []
    for run in runs:
        family, scale, scale_index, maps = (
            run[k] for k in ("backbone", "scale_um", "scale_index", "maps")
        )
        for index, dataset in enumerate(datasets):
            root = destination / family / f"scale-{scale_index}" / f"dataset-{index}"
            options = {
                k: v
                for k, v in settings.items()
                if k
                in {
                    "umap",
                    "clusters",
                    "neighbors",
                    "min_dist",
                    "seed",
                    "folds",
                    "classifier_models",
                    "classifier_space",
                    "classifier_dimensions",
                    "classifier_standardize",
                    "classifier_umap_neighbors",
                    "classifier_umap_min_dist",
                    "classifier_umap_epochs",
                    "evaluation_pca",
                    "hidden_dimensions",
                    "max_iter",
                    "linear_c",
                    "knn_k",
                }
            }
            options.update(projection_options)
            options.update(
                normalization=normalizations[0],
                analysis_normalizations=normalizations,
                annotations=dataset.get("annotations"),
                label_column=dataset.get("label_column", "auto"),
                group_column=dataset.get("group_column"),
                classify=settings.get("classify", True),
                input_config=None,
            )
            path = compare(
                {n: str(p) for n, p in maps[index].items()}, root / "comparison", options
            )
            report["comparisons"].append(
                {
                    "dataset": dataset["name"],
                    "backbone": family,
                    "scale_um": scale,
                    "path": str(Path(path).relative_to(destination)),
                }
            )
            publish()
            for name, entry in json.loads(Path(path).read_text())["representations"].items():
                for row in entry.get("evaluation", []):
                    metrics.append(
                        {
                            "dataset": dataset["name"],
                            "backbone": family,
                            "scale_um": scale,
                            "representation": name,
                            "normalization": entry.get("normalization", normalizations[0]),
                            **row,
                        }
                    )
        if len(datasets) > 1:
            common_names = [n for n in maps[0] if all(n in m for m in maps)]
            for n, name in enumerate(common_names):
                inputs = [
                    {
                        "name": d["name"],
                        "embedding": str(maps[i][name]),
                        "annotations": d.get("annotations"),
                        "label_column": d.get("label_column", "auto"),
                    }
                    for i, d in enumerate(datasets)
                ]
                root = destination / family / f"scale-{scale_index}" / f"representation-{n}"
                for normalization in normalizations:
                    path = joint_analyze(
                        {
                            **projection_options,
                            "datasets": inputs,
                            "normalization": normalization,
                            "umap": settings.get("umap", True),
                            "clusters": settings.get("clusters", 8),
                            "seed": settings.get("seed", 42),
                        },
                        root / ("joint" if len(normalizations) == 1 else f"joint-{normalization}"),
                    )
                    report["joint_analyses"].append(
                        {
                            "backbone": family,
                            "scale_um": scale,
                            "representation": name,
                            "normalization": normalization,
                            "path": str(Path(path).relative_to(destination)),
                        }
                    )
                    publish()
                transfer = settings.get("transfer", {})
                if transfer.get("enabled"):
                    reference = next(
                        i for i, d in enumerate(datasets) if d["name"] == transfer["reference"]
                    )
                    reference_spec = {
                        **inputs[reference],
                        "group_column": datasets[reference].get("group_column"),
                    }
                    path = transfer_labels(
                        {
                            **projection_options,
                            "export_tsv": settings.get("export_tsv", False),
                            "folds": settings.get("folds", 5),
                            **transfer,
                            "reference": reference_spec,
                            "targets": [d for i, d in enumerate(inputs) if i != reference],
                            "umap": settings.get("umap", True),
                            "clusters": settings.get("clusters", 8),
                            "seed": settings.get("seed", 42),
                        },
                        root / "transfer",
                    )
                    report["transfers"].append(
                        {
                            "backbone": family,
                            "scale_um": scale,
                            "representation": name,
                            "path": str(Path(path).relative_to(destination)),
                        }
                    )
                    publish()
        write_json_atomic(destination / "experiment.json", report)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(destination / "metrics.tsv", sep="\t", index=False)
    if not metric_frame.empty:
        groups = [
            c
            for c in ("dataset", "backbone", "scale_um", "representation", "normalization", "model")
            if c in metric_frame
        ]
        scores = [
            c
            for c in metric_frame.select_dtypes(include="number")
            if c not in groups + ["fold", "train_objects", "test_objects", "k"]
        ]
        if scores:
            # Scalar scale labels keep anisotropic tile presets groupable too.
            metric_frame["scale_um"] = metric_frame["scale_um"].map(str)
            summary = (
                metric_frame.groupby(groups, dropna=False)[scores]
                .agg(["mean", "std"])
                .reset_index()
            )
            summary.columns = ["_".join(str(v) for v in c if v) for c in summary.columns]
            summary.to_csv(destination / "metrics_summary.tsv", sep="\t", index=False)
    report["execution_state"] = "completed"
    publish()
    package_results(destination)
    return destination / "experiment.json"
