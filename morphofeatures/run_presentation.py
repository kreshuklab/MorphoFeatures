"""Readable run identities derived from the configuration actually submitted."""

from functools import lru_cache
from pathlib import Path

import yaml

from morphofeatures.workflows import WORKFLOWS


def stage_description(stage):
    action = stage.get("action", "unknown")
    model = str(stage.get("model", "mae")).replace("dinov", "DINOv")
    if model == "mae":
        model = "MAE"
    if action == "preprocess":
        return (
            "Prepare distributed fine patches"
            if stage.get("layout") == "distributed_patches"
            else "Prepare object crops"
        )
    if action == "train":
        return "MAE training"
    if action == "extract":
        return f"{model} embeddings"
    if action == "dino_experiment":
        families = list(
            dict.fromkeys(
                str(b.get("model", "dino")).replace("dinov", "DINOv")
                for b in stage.get("backbones", [])
            )
        )
        return f"{' / '.join(families) or 'DINO'} fine/coarse comparison"
    if action in {"analyze", "compare", "joint_analyze"}:
        if stage.get("use_published_layout"):
            return "Published map: " + str(stage.get("published_reference", "features"))
        projection = "PCA + UMAP" if stage.get("umap", True) else "PCA"
        prefix = {
            "analyze": "",
            "compare": "Compare representations: ",
            "joint_analyze": "Shared datasets: ",
        }[action]
        return prefix + projection + " & clustering"
    return {
        "distill": "Neural adapter / PCA distillation",
        "export_bilateral": "Bilateral 3D export",
        "bilateral": "Bilateral-partner retrieval",
        "context": "Spatial context features",
        "transfer": "Label transfer",
        "fuse": "Combine embeddings",
        "export_mobie": "MoBIE export",
        "export_labels": "Export labels to segmentation",
    }.get(action, action.replace("_", " ").capitalize())


@lru_cache(maxsize=1024)
def _snapshot_label(path, modified, size):
    del modified, size  # These invalidate the cache when a snapshot is replaced.
    try:
        document = yaml.safe_load(Path(path).read_text())
        stages = document.get("stages", []) if isinstance(document, dict) else []
        if stages and all(isinstance(stage, dict) for stage in stages):
            return " → ".join(stage_description(stage) for stage in stages)
    except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError):
        pass
    return "Workflow (configuration unavailable)"


def workflow_label(record):
    """Keep internal dispatch keys unchanged, including for historical runs."""
    if record.workflow != "workspace_run":
        definition = WORKFLOWS.get(record.workflow)
        return definition.label if definition else record.workflow.replace("_", " ").capitalize()
    try:
        path = Path(record.config_snapshot)
        stat = path.stat()
        return _snapshot_label(str(path), stat.st_mtime_ns, stat.st_size)
    except (OSError, TypeError):
        return "Workflow (configuration unavailable)"


def select_table_run(state, table_key, record_ids):
    """Translate a table event against exactly the rows rendered for that event."""
    rows = state.get(table_key, {}).get("selection", {}).get("rows", [])
    selected = list(dict.fromkeys(record_ids[row] for row in rows if 0 <= row < len(record_ids)))
    previous = state.get("runs:selected_ids", [])
    state["runs:selected_ids"] = selected
    added = [job_id for job_id in selected if job_id not in previous]
    # Keep one inspector while the table also holds the batch deletion selection.
    if added:
        state["runs:selected"] = added[-1]
    elif selected and state.get("runs:selected") not in selected:
        state["runs:selected"] = selected[0]
