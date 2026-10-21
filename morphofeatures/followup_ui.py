"""Artifact-based follow-up drafts; opening controls never starts computation."""

from pathlib import Path

import streamlit as st

CHOICES = {
    "analyze": "Projection, clustering and classification",
    "distill": "Distill embeddings: adapter + PCA + original",
    "bilateral": "Compare bilateral-partner retrieval",
    "context": "Build spatial context features",
    "fuse": "Combine feature blocks",
    "joint_analyze": "Shared map across independent datasets",
    "transfer": "Transfer labels to another dataset",
}


def embedding_artifacts(metadata, path):
    """Discover only declared vector artifacts, never scan folders or load arrays."""
    path = Path(path)
    result = {}

    def add(name, value, parent=path.parent):
        if value:
            file = (parent / str(value)).resolve()
            result[str(name)] = str(file)

    schema = metadata.get("schema")
    if path.suffix != ".json":
        add(path.stem, path)
    elif schema == "morphofeatures.distillation.v1":
        for entry in metadata["representations"].values():
            add(entry["name"] + " · all objects", entry["embedding"])
            add(entry["name"] + " · held-out population", entry["held_out"])
    elif schema == "morphofeatures.dino_experiment.v1":
        for i, entry in enumerate(metadata.get("embeddings", [])):
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
            add(name, str(Path(entry["directory"]) / "embeddings.npz"))
    elif schema in {"morphofeatures.analysis.v1", "morphofeatures.fusion.v1", "morphofeatures.context.v1"}:
        add(metadata.get("display_name", path.parent.name), metadata.get("embedding", "embeddings.npz"))
    elif schema == "morphofeatures.joint_analysis.v1":
        for name, entry in metadata.get("datasets", {}).items():
            add(name, str(Path(entry["directory"]) / "embeddings.npz"))
    return result


def followup_document(embeddings, actions, *, source, settings=None, distill_then_compare=False):
    """Create editable stages with explicit dependencies and retained source lineage."""
    from morphofeatures.analysis.bilateral import defaults as bilateral_defaults
    from morphofeatures.analysis.distillation import defaults as distill_defaults
    from morphofeatures.workspace_state import new_draft

    if not embeddings or not actions:
        raise ValueError("Choose at least one representation and one follow-up analysis.")
    if set(actions) - set(CHOICES):
        raise ValueError("Unknown follow-up analysis.")
    settings = settings or {}
    stages = []
    label_settings = {k: settings[k] for k in ("annotations", "label_column", "group_column") if settings.get(k)}
    # A stable order makes selection order immaterial. Runs always use the shared
    # worker contract, with review before submission.
    for i, embedding in enumerate(embeddings.values()):
        if "distill" in actions:
            stage = {**distill_defaults(), **label_settings, "name": f"Distilled {i + 1}", "embedding": embedding}
            stages.append(stage)
        if "analyze" in actions:
            stages.append({"action": "analyze", "embedding": embedding, **label_settings,
                           "umap": True, "clusters": 8, "normalization": "standardize"})
        if "context" in actions:
            stages.append({"action": "context", "name": f"Context {i + 1}", "embedding": embedding,
                           "graph_mode": "adjacency_table", "adjacency_table": ""})
    if "bilateral" in actions:
        # For distillation use its internal held-out comparison by default;
        # appending an ordinary probe would silently include training objects.
        if distill_then_compare and "distill" in actions:
            for stage in stages:
                if stage["action"] == "distill":
                    stage["bilateral"] = {**bilateral_defaults(), "enabled": True, "same_specimen": True}
        else:
            stages.append({"action": "bilateral", **bilateral_defaults(), "embeddings": dict(embeddings),
                           "same_specimen": len(embeddings) == 1})
    if "fuse" in actions:
        if len(embeddings) < 2:
            raise ValueError("Combine feature blocks needs at least two selected representations.")
        stages.extend([
            {"action": "fuse", "name": "Combined", "embeddings": dict(embeddings), "combination": "balanced",
             "weights": [1 / len(embeddings)] * len(embeddings)},
            {"action": "analyze", "from_embedding": "Combined", "normalization": "none", "umap": True, "clusters": 8, **label_settings},
        ])
    if "joint_analyze" in actions:
        if len(embeddings) < 2:
            raise ValueError("A shared map needs at least two independent datasets with a common feature recipe.")
        stages.append({"action": "joint_analyze", "datasets": [{"name": name, "embedding": file} for name, file in embeddings.items()],
                       "umap": True, "clusters": 8, "normalization": "standardize", "assume_same_model": False})
    if "transfer" in actions:
        items = list(embeddings.items())
        stages.append({"action": "transfer", "reference": {"name": items[0][0], "embedding": items[0][1], **label_settings},
                       "targets": [{"name": name, "embedding": file} for name, file in items[1:]] or [{"name": "Target", "embedding": ""}]})
    document = new_draft()["document"]
    document.update(stages=stages, followup_of={"analysis": str(Path(source).resolve()), "embeddings": dict(embeddings)})
    return document


def controls(config, metadata, path):
    from morphofeatures.workflow_ui import navigate, use_draft
    from morphofeatures.workspace_state import new_draft

    choices = embedding_artifacts(metadata, path)
    if not choices:
        st.info("Open an embedding or an analysis that contains saved vectors to create a follow-up workflow.")
        return
    st.caption("Reuse saved vectors without re-embedding images. Select inputs and one or more tasks; Create draft opens settings for review and local/SLURM submission. Existing results remain unchanged.")
    key = "followup:" + str(path)
    is_distillation = metadata.get("schema") == "morphofeatures.distillation.v1"
    default = [name for name in choices if name.endswith("held-out population")] if is_distillation else list(choices)[:1]
    selected = st.multiselect("Input representations for follow-up", list(choices), default=default, key=key + ":inputs",
                              help="Compare feature approaches only within one specimen and ID namespace. For joint maps or transfer, select independent datasets that use the same fitted representation.")
    actions = st.multiselect("Follow-up analyses", list(CHOICES), format_func=CHOICES.get, key=key + ":actions")
    linked = False
    if "distill" in actions and "bilateral" in actions:
        linked = st.checkbox("Probe original, PCA and adapter together after distillation", True, key=key + ":linked",
                             help="Uses each distillation's held-out search population by default. Uncheck to probe the selected input vectors instead.")
    if is_distillation:
        st.warning("Full vectors include adapter training objects. Held-out vectors are the safer starting point for a new probe; related cells can still share biological information.")
    if st.button("Create follow-up workflow draft", key=key + ":create", disabled=not selected or not actions):
        try:
            document = followup_document({name: choices[name] for name in selected}, actions, source=path,
                                         settings=metadata.get("settings"), distill_then_compare=linked)
            use_draft(config, new_draft(document=document, source=str(path)), "Analyze")
            navigate("Workflow")
        except ValueError as error:
            st.error(str(error))


def download_file(path, label):
    path = Path(path)
    if not path.is_file():
        return
    if path.stat().st_size > 200 * 1024**2:
        st.caption(f"{label}: above the 200 MiB browser limit; retrieve from {path}")
    else:
        st.download_button(label, path.read_bytes(), path.name, key="file-download:" + str(path) + ":" + label)
