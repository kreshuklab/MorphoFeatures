"""Experiment and artifact descriptors from declared outputs, never folder scans."""

from pathlib import Path

from morphofeatures.artifact_availability import probe, read_report
from morphofeatures.artifact_locations import artifact_path
from morphofeatures.workflow_outputs import OUTPUTS, ordered_artifacts, report_action


def vector_artifacts(metadata, path):
    path = Path(path)
    result = {}

    def add(name, value, parent=path.parent):
        if value:
            result[str(name)] = str(artifact_path(parent / str(value)).resolve())

    schema = metadata.get("schema")
    if path.suffix.lower() != ".json":
        if probe(path).kind == "vectors":
            add(path.stem, path)
    elif schema == "morphofeatures.distillation.v1":
        for entry in metadata.get("representations", {}).values():
            add(entry.get("name", "Representation") + " · all objects", entry.get("embedding"))
            add(entry.get("name", "Representation") + " · held-out population", entry.get("held_out"))
    elif schema == "morphofeatures.dino_experiment.v1":
        for i, entry in enumerate(metadata.get("embeddings") or metadata.get("original_embeddings", [])):
            add(" · ".join(str(entry.get(k, "")) for k in ("dataset", "backbone", "scale_um", "representation")) + f" [{i + 1}]", entry.get("path"))
    elif schema == "morphofeatures.transfer.v1":
        add("Reference", metadata.get("reference_embedding", "reference/embeddings.npz"))
        for i, entry in enumerate(metadata.get("targets", [])):
            folder = path.parent / entry["directory"]
            title = f"{entry['name']} · {entry['condition']} [{i + 1}]"
            add(title, "embeddings.npz", folder)
            add(title + " · reference in this space", "reference_embeddings.npz", folder)
    elif schema in {"morphofeatures.comparison.v1", "morphofeatures.bilateral.v1"}:
        for name, entry in metadata.get("representations", {}).items():
            if entry.get("directory"):
                add(name, str(Path(entry["directory"]) / "embeddings.npz"))
    elif schema in {"morphofeatures.analysis.v1", "morphofeatures.fusion.v1", "morphofeatures.context.v1"}:
        add(metadata.get("display_name", path.parent.name), metadata.get("embedding", "embeddings.npz"))
    elif schema == "morphofeatures.joint_analysis.v1":
        for name, entry in metadata.get("datasets", {}).items():
            if entry.get("directory"):
                add(name, str(Path(entry["directory"]) / "embeddings.npz"))
    elif schema == "morphofeatures.dino_views.v1":
        for name, embedding in metadata.get("embeddings", {}).items():
            add(name, embedding)
    elif schema == "morphofeatures.embedding.v1":
        add(metadata.get("display_name", "Embedding"), metadata.get("embedding"))
    return result


def run_artifacts(record):
    paths = [*record.artifacts, record.checkpoint_path, record.embedding_path]
    # status.json is a known worker contract, not a guessed recursive search.
    try:
        status = read_report(Path(record.working_directory) / "status.json")
        if not isinstance(status.get("stages", []), list) or any(not isinstance(stage, dict) for stage in status.get("stages", [])):
            raise ValueError("Invalid worker stage index")
        paths.extend(status.get("artifacts", []))
        paths.extend(stage.get("result") for stage in status.get("stages", []) if stage.get("state") == "completed")
        paths.extend(path for stage in status.get("stages", []) for path in stage.get("reports", []))
    except (OSError, ValueError, TypeError):
        pass
    return ordered_artifacts(paths)


def run_results(record):
    results, unavailable = [], []
    for path in run_artifacts(record):
        if Path(path).suffix.lower() not in {".json", ".npz", ".npy", ".tsv", ".csv", ".yaml", ".yml", ".pt", ".pth"}:
            continue
        status = probe(path)
        if status.state in {"missing", "invalid", "unreadable"}:
            unavailable.append(status)
        elif status.kind in {"report", "vectors", "coordinates", "prepared", "checkpoint"}:
            metadata = status.metadata
            action = report_action(metadata)
            name = metadata.get("display_name") or OUTPUTS.get(action, ({"vectors": "Saved vectors", "coordinates": "Saved map", "prepared": "Prepared data", "checkpoint": "Trained checkpoint"}.get(status.kind, "Saved result"),))[0]
            results.append({"path": status.path, "name": name, "kind": status.kind,
                            "action": action, "metadata": metadata})
    # The parent compound report already indexes its children. Keep independent
    # stages visible, without repeating every vector beside a valid parent.
    parents = [r for r in results if r["action"] in {"dino_experiment", "distill", "transfer", "compare", "joint_analyze"}]
    if parents:
        results = [r for r in results if r in parents or not any(
            Path(p["path"]).parent in Path(r["path"]).parents for p in parents
        )]
    return results, unavailable


def report_choices(entries, parent, key="path"):
    """Only readable supported reports can become active child selectors."""
    return [entry for entry in entries if (
        (status := probe(Path(parent) / entry.get(key, ""))).ready and status.kind == "report"
    )]


def search_names(entry):
    metadata = entry["metadata"]
    names = [entry["name"], metadata.get("display_name", "")]
    for field in ("datasets", "representations", "embeddings", "original_embeddings", "targets"):
        values = metadata.get(field, {})
        if isinstance(values, dict):
            names.extend(values)
            values = values.values()
        for value in values:
            if isinstance(value, dict):
                names.extend(str(value[k]) for k in ("name", "dataset", "representation", "condition") if k in value)
    return " ".join(names)
