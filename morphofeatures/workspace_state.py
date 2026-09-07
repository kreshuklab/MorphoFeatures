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
    "raw": "Raw image and instance segmentation",
    "crops": "Prepared crops or grouped N5 patches",
    "checkpoint": "An existing model checkpoint",
    "embeddings": "Saved embeddings",
    "demo": "Synthetic demonstration (eight crops)",
}
STAGE_LABELS = {
    "preprocess": "Prepare data",
    "train": "Train",
    "extract": "Extract embeddings",
    "analyze": "Analyze",
    "compare": "Compare representations",
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
        "clusters": 2,
        "seed": 42,
    }


def new_draft(start="crops", *, document=None, source=None):
    if document is None:
        stages = []
        if start == "raw":
            stages.append(preprocessing_defaults())
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
        if stage.get("action") == "preprocess":
            prepared = stage
        elif stage.get("from_preprocessing") and prepared:
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
