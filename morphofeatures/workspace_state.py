"""Editable workflow drafts and lossless imports, independent of Streamlit."""

from __future__ import annotations

import fcntl
import json
import uuid
from copy import deepcopy
from pathlib import Path

from morphofeatures.artifacts import write_json_atomic
from morphofeatures.config import repository_root
from morphofeatures.configuration_editor import load_document, parse_document
from morphofeatures.workspace_jobs import resolve_job

STARTING_POINTS = {
    "distill": "Distill embeddings with cell types (adapter / PCA)",
    "published": "Explore published MorphoFeatures / MorphoContextFeatures",
    "bilateral": "Compare bilateral-partner retrieval",
    "context": "Build spatial context from embeddings",
    "dino_experiment": "Compare DINO fine/coarse representations",
    "fusion": "Combine saved feature blocks",
    "transfer": "Transfer labels from saved embeddings",
    "raw": "Raw image and instance segmentation",
    "platybrowser": "Stream PlatyBrowser raw and segmentation",
    "ome_zarr": "Stream OME-Zarr raw and segmentation",
    "preprocess": "Just Run Preprocessing",
    "crops": "Prepared crops or grouped N5 patches",
    "dino": "DINO features from prepared objects",
    "checkpoint": "An existing model checkpoint",
    "embeddings": "Saved embeddings",
    "joint_embeddings": "Shared space across independent datasets",
    "demo": "Synthetic demonstration (eight crops)",
}


def with_fusion_comparison(document, index):
    """Keep the configured fusion and add its raw/balanced counterpart plus analysis."""
    result = deepcopy(document)
    stages = result["stages"]
    original = stages[index]
    if original.get("action") != "fuse":
        raise ValueError("Select a Combine embeddings stage first.")
    if not original.get("name"):
        original["name"] = "Combined"
    alternative = deepcopy(original)
    alternative["combination"] = "raw" if original.get("combination", "balanced") == "balanced" else "balanced"
    base = original["name"] + " · " + alternative["combination"]
    names = {stage.get("name") for stage in stages}
    name, number = base, 2
    while name in names:
        name, number = f"{base} {number}", number + 1
    alternative["name"] = name
    following = stages[index + 1] if index + 1 < len(stages) else {}
    reuse = following.get("action") == "analyze" and following.get("from_embedding") == original["name"]
    analysis = deepcopy(following) if reuse else {"umap": True, "clusters": 8, "normalization": "none"}
    analysis.pop("from_embedding", None)
    analysis.pop("embedding", None)
    analysis.update(action="compare", from_embeddings=[original["name"], name])
    analysis.setdefault("analysis_normalizations", ["none", "l2"])
    stages[index + 1:index + (2 if reuse else 1)] = [alternative, analysis]
    return result
STAGE_LABELS = {
    "distill": "Distill embeddings",
    "export_bilateral": "Export bilateral pairs in 3D",
    "bilateral": "Bilateral-partner retrieval",
    "context": "Build spatial context features",
    "dino_experiment": "DINO fine/coarse experiments",
    "fuse": "Combine embeddings",
    "transfer": "Transfer labels",
    "preprocess": "Prepare data",
    "train": "Train",
    "extract": "Extract embeddings",
    "analyze": "Analyze",
    "compare": "Compare representations",
    "joint_analyze": "Shared space across datasets",
    "export_labels": "Export labels to segmentation",
    "export_mobie": "Export to MoBIE",
}


def training_defaults(*, demo=False):
    values = load_document(repository_root() / "configs/smoke.yaml")
    if not demo:
        values["device"] = "auto"
        values["data"] = {"crops": None, "label_ids": None, "loss_masks": None}
        values["mae"].update(
            input_shape=[32, 32, 32], embedding_dim=192, encoder_depth=4, encoder_heads=6
        )
        values["training"].update(epochs=100, batch_size=4)
    return values


def preprocessing_defaults():
    return {
        "action": "preprocess",
        "layout": "single_crop",
        "raw": "",
        "segmentation": "",
        "raw_key": "exported_data",
        "segmentation_key": "exported_data",
        "raw_axes": "zyxc",
        "segmentation_axes": "zyxc",
        "raw_channel": 0,
        "segmentation_channel": 0,
        "spacing_zyx": [1.0, 1.0, 1.0],
        "origin_zyx": [0.0, 0.0, 0.0],
        "unit": "voxel",
        "roi": [[0, 0, 0], [64, 64, 64]],
        "crop_shape": [32, 32, 32],
        "output_format": "npy",
        "max_objects": 8,
        "block_shape": [32, 32, 32],
        "min_voxels": 10,
        "center": "bbox",
        "oversized": "skip",
        "boundary": "pad",
        "roi_boundary": "skip",
        "normalization": "dtype",
        "background": 0.0,
    }


def analysis_defaults(*, linked=True):
    return {
        "action": "analyze",
        **({"from_extraction": True} if linked else {"embedding": ""}),
        "normalization": "standardize",
        "umap": False,
        "clusters": 8,
        "min_dist": 0.0,
        "umap_epochs": 50,
        "seed": 42,
    }


def add_dataset_extraction(document, stage_index, dataset_index):
    """Insert an extraction, copying model settings but requiring a new data source."""
    document = deepcopy(document)
    earlier = document["stages"][:stage_index]
    template = next((s for s in earlier if s["action"] == "extract"), None)
    stage = deepcopy(template) if template else new_draft("checkpoint")["document"]["stages"][0]
    if "config" not in stage and stage.get("from_training"):
        stage["config"] = deepcopy(
            next(s["config"] for s in reversed(earlier) if s["action"] == "train")
        )
    config = stage.setdefault("config", training_defaults())
    data = config.setdefault("data", {})
    for key in ("crops", "label_ids", "loss_masks", "patches_container", "positions_container"):
        if key in data:
            data[key] = None
    for key in (
        "preprocessing",
        "object_table",
        "qc_raw_container",
        "foreground_mask_container",
        "masks_container",
        "manifest",
        "root",
        "raw",
        "id_mapping",
    ):
        data.pop(key, None)
    config.pop("inspection", None)
    stage.pop("from_preprocessing", None)
    stage["name"] = "dataset-" + uuid.uuid4().hex[:8]
    dataset = document["stages"][stage_index]["datasets"][dataset_index]
    dataset.pop("embedding", None)
    dataset["from_embedding"] = stage["name"]
    document["stages"].insert(stage_index, stage)
    return document


def stages_after_preparation(preparation, *, mae="none", dino=False, compare=False, model_defaults=None):
    """Build the same crop handoff for local and streamed sources."""
    if mae not in {"none", "train", "checkpoint"}:
        raise ValueError("Choose none, train, or checkpoint for MAE")
    if preparation.get("layout") == "distributed_patches" and mae != "none":
        raise ValueError(
            "Distributed patches currently feed DINO; select No MAE or single_crop preparation"
        )
    stages = [deepcopy(preparation)]
    names = []
    if mae != "none":
        config = training_defaults()
        config["mae"]["input_shape"] = preparation.get("crop_shape", [32, 32, 32])
        if preparation.get("source") in {"platybrowser", "ome_zarr"}:
            config["mae"]["patch_size"] = [
                8 if side % 8 == 0 else 4 if side % 4 == 0 else 1
                for side in config["mae"]["input_shape"]
            ]
        if mae == "train":
            stages.append({"action": "train", "config": config, "from_preprocessing": True})
        stages.append(
            {
                "action": "extract",
                "name": "MAE",
                "model": "mae",
                **(
                    {"from_training": True}
                    if mae == "train"
                    else {
                        "from_preprocessing": True,
                        "config": config,
                        "checkpoint": "",
                    }
                ),
            }
        )
        stages.append(analysis_defaults())
        names.append("MAE")
    if dino:
        stage = new_draft("dino", model_defaults=model_defaults)["document"]["stages"][0]
        stage.update(name="DINO", from_preprocessing=True)
        if preparation.get("layout") == "distributed_patches":
            stage.setdefault("views", {}).update(
                normalization="unit", resize="letterbox", fractions=[0.5]
            )
        stages.extend([stage, analysis_defaults()])
        names.append("DINO")
    if compare and len(names) > 1:
        analysis = analysis_defaults(linked=False)
        analysis.pop("embedding", None)
        stages.append({**analysis, "action": "compare", "from_embeddings": names})
    return stages


def new_draft(start="crops", *, document=None, source=None, model_defaults=None):
    from morphofeatures.dino_defaults import backbone_defaults

    if document is None:
        stages = []
        if start in {"raw", "preprocess", "platybrowser", "ome_zarr"}:
            stages.append(preprocessing_defaults())
            if start == "platybrowser":
                from morphofeatures.analysis.platybrowser import DEFAULTS

                stages[0].update(
                    source="platybrowser",
                    platybrowser={**DEFAULTS, "object_ids": "segmentation"},
                    roi=None,
                    crop_shape=[64, 64, 64],
                    output_format="h5",
                    oversized="clip",
                    read_budget_side=256,
                )
        if start == "ome_zarr":
            stages[0].update(
                source="ome_zarr",
                ome_zarr={},
                roi=None,
                output_format="h5",
                normalization="percentile",
                read_budget_side=256,
                max_scan_voxels=128**3,
            )
        if start in {"raw", "crops", "demo"}:
            stages.append(
                {
                    "action": "train",
                    "config": training_defaults(demo=start == "demo"),
                    **({"from_preprocessing": True} if start == "raw" else {}),
                }
            )
            stages.append({"action": "extract", "model": "mae", "from_training": True})
        if start == "checkpoint":
            stages.append(
                {
                    "action": "extract",
                    "model": "mae",
                    "checkpoint": "",
                    "config": training_defaults(),
                }
            )
        if start == "dino":
            stages.append(
                {
                    "action": "extract",
                    **backbone_defaults(model_defaults),
                    "config": {
                        "device": "auto",
                        "seed": 42,
                        "data": {"crops": None, "label_ids": None, "loss_masks": None},
                    },
                }
            )
        if start == "joint_embeddings":
            stages.append(
                {
                    **analysis_defaults(linked=False),
                    "action": "joint_analyze",
                    "datasets": [
                        {"name": "Dataset A", "embedding": ""},
                        {"name": "Dataset B", "embedding": ""},
                    ],
                    "assume_same_model": False,
                    "max_objects_per_dataset": 0,
                }
            )
            stages[-1].pop("embedding")
        if start == "dino_experiment":
            from morphofeatures.dino_experiments import experiment_defaults

            stages.append(experiment_defaults())
            stages[-1]["backbones"] = [backbone_defaults(model_defaults)]
        if start == "fusion":
            stages.extend([{"action": "fuse", "name": "Combined", "embeddings": {}, "weights": [.5, .5]}, {"action": "analyze", "from_embedding": "Combined", "normalization": "none", "umap": True, "clusters": 8}])
        if start == "transfer":
            stages.append({"action": "transfer", "reference": {"name": "Reference", "embedding": "", "annotations": ""}, "targets": [{"name": "Target", "embedding": "", "annotations": None}]})
        if start == "distill":
            from morphofeatures.analysis.distillation import defaults

            stages.append(defaults())
        if start == "published":
            from morphofeatures.analysis.published import references

            name = "MorphoContextFeatures (200 dimensions)"
            stages.append({**analysis_defaults(linked=False), "embedding": references()[name]["embedding"], "published_reference": name, "use_published_layout": True, "umap": True})
        if start == "bilateral":
            from morphofeatures.analysis.bilateral import defaults
            from morphofeatures.analysis.published import references

            stages.append({"action": "bilateral", **defaults(), "embeddings": {name: value["embedding"] for name, value in references().items()}, "same_specimen": True})
        if start == "context":
            stages.extend([{"action": "context", "name": "Spatial context", "embedding": "", "graph_mode": "adjacency_table", "adjacency_table": "", "include_self": True, "combination": "raw", "reduction": "none"}, {**analysis_defaults(linked=False), "from_embedding": "Spatial context"}])
            stages[-1].pop("embedding")
        if start not in {"preprocess", "platybrowser", "ome_zarr", "joint_embeddings", "dino_experiment", "fusion", "transfer", "published", "bilateral", "context", "distill"}:
            stages.append(analysis_defaults(linked=start != "embeddings"))
        document = {
            "stages": stages,
            "slurm": {
                "partition": "compute",
                "cpus": 4,
                "gpus": 0,
                "memory": "8G",
                "time": "01:00:00",
            },
        }
    document = deepcopy(document)
    document.pop("worker", None)
    identifier = uuid.uuid4().hex
    return {
        "schema": "morphofeatures.draft.v1",
        "id": identifier,
        "name": STARTING_POINTS.get(start, "Imported workflow"),
        "run_id": "run-" + identifier[:8],
        "document": document,
        "source": source or STARTING_POINTS.get(start, "Imported workflow"),
        "source_document": deepcopy(document),
        "allow_synthetic": start == "demo",
        "revision": 0,
        "saved_revision": -1,
        "storage_revision": 0,
        "execution": "local",
        "dependency": "",
    }


def import_workflow(path, profile=None):
    path = Path(path).expanduser().resolve()
    values = parse_document(path.read_text())
    if "stages" in values:
        return resolve_job(values, path.parent, validate=False)
    config = load_document(path, profile=profile)
    return {
        "stages": [
            {"action": "train", "config": config},
            {"action": "extract", "model": "mae", "from_training": True},
            analysis_defaults(),
        ],
        "slurm": deepcopy(config.get("slurm", {})),
    }


def validate_draft_inputs(document, *, allow_synthetic=False):
    """UI real-data routes must not fall through to the worker's synthetic fixture."""
    for index, stage in enumerate(document["stages"]):
        if stage.get("action") not in {"train", "extract"}:
            continue
        if stage.get("from_preprocessing") or stage.get("from_training"):
            continue
        data = stage.get("config", {}).get("data", {})
        if (
            not allow_synthetic
            and not data.get("crops")
            and data.get("source") != "n5_masked_patches"
        ):
            raise ValueError(
                f"Stage {index + 1} ({STAGE_LABELS.get(stage['action'])}): select prepared crops "
                "or load a grouped N5 data configuration. Synthetic data requires the explicit demonstration option."
            )


def workflow_document(draft):
    """Use the same derived settings for export, review, and execution."""
    document = deepcopy(draft["document"])
    prepared = None
    for stage in document["stages"]:
        if stage.get("action") == "fuse":
            from morphofeatures.analysis.fusion import freeze_component_order

            freeze_component_order(stage)
        if stage.get("action") == "preprocess":
            prepared = stage
        elif (
            stage.get("from_preprocessing")
            and prepared
            and prepared.get("layout") != "distributed_patches"
        ):
            stage.setdefault("config", {}).setdefault("mae", {})["input_shape"] = prepared.get(
                "crop_shape", [32, 32, 32]
            )
    return document


def draft_path(root, identifier):
    if len(identifier) != 32 or any(c not in "0123456789abcdef" for c in identifier):
        raise ValueError("Invalid draft identifier")
    return Path(root) / ".morphofeatures" / "drafts" / (identifier + ".json")


def save_draft(root, draft):
    path = draft_path(root, draft["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stored = json.loads(path.read_text()) if path.exists() else None
        if stored and stored["storage_revision"] != draft["storage_revision"]:
            raise ValueError(
                "This draft was saved in another session. Download your changes, then reopen the saved draft before saving again."
            )
        saved = deepcopy(draft)
        for stage in saved.get("document", {}).get("stages", []):
            if stage.get("action") == "fuse":
                from morphofeatures.analysis.fusion import freeze_component_order

                freeze_component_order(stage)
        saved["saved_revision"] = saved["revision"]
        saved["storage_revision"] += 1
        write_json_atomic(path, saved)
    return saved


def load_draft(root, identifier):
    draft = json.loads(draft_path(root, identifier).read_text())
    if draft.get("schema") != "morphofeatures.draft.v1" or draft.get("id") != identifier:
        raise ValueError("Unsupported or invalid draft file")
    if not isinstance(draft.get("document", {}).get("stages"), list):
        raise ValueError("Draft has no stage list")
    return draft


def change_summary(before, after, prefix=""):
    """Leaf differences for review; unknown settings participate as well."""
    if isinstance(before, dict) and isinstance(after, dict):
        return [
            row
            for key in sorted(set(before) | set(after))
            for row in change_summary(
                before.get(key), after.get(key), f"{prefix}.{key}".lstrip(".")
            )
        ]
    if (
        isinstance(before, list)
        and isinstance(after, list)
        and all(isinstance(value, dict) for value in before + after)
    ):
        return [
            row
            for index in range(max(len(before), len(after)))
            for row in change_summary(
                before[index] if index < len(before) else None,
                after[index] if index < len(after) else None,
                f"{prefix}[{index}]",
            )
        ]
    return (
        [] if before == after else [{"Setting": prefix, "Loaded": str(before), "Draft": str(after)}]
    )
