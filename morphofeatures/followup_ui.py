"""Artifact-based follow-up drafts; opening controls never starts computation."""

from copy import deepcopy
from pathlib import Path

import streamlit as st

CHOICES = {
    "analyze": "Projection, clustering and classification",
    "compare": "Compare representations of the same objects",
    "distill": "Distill embeddings: adapter + PCA + original",
    "bilateral": "Compare bilateral-partner retrieval",
    "context": "Build spatial context features",
    "fuse": "Combine feature blocks",
    "joint_analyze": "Shared map across independent datasets",
    "transfer": "Transfer labels to another dataset",
}


def embedding_artifacts(metadata, path):
    from morphofeatures.result_catalog import vector_artifacts

    return vector_artifacts(metadata, path)


def followup_document(embeddings, actions, *, source, settings=None, distill_then_compare=False, context=None):
    """Create editable stages with explicit dependencies and retained source lineage."""
    from morphofeatures.analysis.bilateral import defaults as bilateral_defaults
    from morphofeatures.analysis.distillation import defaults as distill_defaults
    from morphofeatures.workspace_state import new_draft

    if not embeddings or not actions:
        raise ValueError("Choose at least one representation and one follow-up analysis.")
    embeddings = {str(name): str(path) for name, path in embeddings.items()}
    if set(actions) - set(CHOICES):
        raise ValueError("Unknown follow-up analysis.")
    settings = settings or {}
    from morphofeatures.analysis.projection import PROJECTION_DEFAULTS

    inherited_keys = set(PROJECTION_DEFAULTS) | {
        "umap", "annotations", "label_column", "group_column", "input_config", "classify",
        "folds", "fold_policy", "minimum_class_count", "classifier_models", "prediction_model",
        "classifier_space", "classifier_dimensions", "classifier_standardize", "classifier_umap_neighbors",
        "classifier_umap_min_dist", "classifier_umap_epochs", "evaluation_pca", "class_weight",
        "linear_c", "knn_k", "hidden_dimensions", "max_iter", "predict_unlabeled", "analysis_normalizations",
    }
    inherited = {key: deepcopy(value) for key, value in settings.items() if key in inherited_keys}
    stages = []
    label_settings = {k: settings[k] for k in ("annotations", "label_column", "group_column") if settings.get(k)}
    # A stable order makes selection order immaterial. Runs always use the shared
    # worker contract, with review before submission.
    for i, embedding in enumerate(embeddings.values()):
        if "distill" in actions:
            stage = {**distill_defaults(), **label_settings, "name": f"Distilled {i + 1}", "embedding": embedding}
            stages.append(stage)
        if "analyze" in actions:
            stages.append({"action": "analyze", "embedding": embedding,
                           "umap": True, "clusters": 8, "normalization": "standardize", **inherited})
        if "context" in actions:
            stages.append({"action": "context", "name": f"Context {i + 1}", "embedding": embedding,
                           "graph_mode": "adjacency_table", "adjacency_table": ""})
    if "compare" in actions:
        inputs = dict(embeddings)
        if len(inputs) == 1:
            inputs["Another representation (required)"] = ""
        stages.append({"action": "compare", "embeddings": inputs,
                       "umap": True, "clusters": 8, "normalization": "standardize", **inherited})
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
    document.update(stages=stages, followup_of={"analysis": str(Path(source).resolve()), "embeddings": dict(embeddings),
                                             "selection": deepcopy(context or {}), "copied_settings": sorted(inherited)})
    return document


def controls(config, metadata, path, *, inputs=None):
    from morphofeatures.artifact_availability import availability_details, available_vectors
    from morphofeatures.workflow_ui import navigate, use_draft
    from morphofeatures.workspace_state import new_draft

    declared = embedding_artifacts(metadata, path) if inputs is None else inputs
    availability_details(declared, key="followup:" + str(path))
    choices = available_vectors(declared)
    key = "followup:" + str(path)
    with st.expander("Add saved vectors from another result"):
        extra_path = st.text_input("Additional embedding file", key=key + ":extra-path",
                                   help="Use the original feature vectors. A saved PCA/UMAP coordinate table cannot supply them.")
        extra_name = st.text_input("Additional dataset / representation name", "Additional representation", key=key + ":extra-name")
        if extra_path.strip():
            extra = {extra_name.strip() or "Additional representation": extra_path.strip()}
            availability_details(extra, key=key + ":extra")
            if set(extra) & set(choices):
                st.warning("Choose a distinct name for the additional representation.")
            else:
                choices.update(available_vectors(extra))
    if not choices:
        st.info("No validated embedding vectors are available for this selection. Locate missing vectors or check an unchecked artifact above. A saved projection alone cannot supply its original features.")
        return
    st.caption("Reuse saved vectors without re-embedding images. Select inputs and one or more tasks; Create draft opens settings for review and local/SLURM submission. Existing results remain unchanged.")
    is_distillation = metadata.get("schema") == "morphofeatures.distillation.v1" or any(name.endswith("held-out population") for name in choices)
    default = [name for name in choices if name.endswith("held-out population")] if is_distillation else list(choices)[:1]
    selected = st.multiselect("Input representations for follow-up", list(choices), default=default, key=key + ":inputs",
                              help="Compare feature approaches only within one specimen and ID namespace. For joint maps or transfer, select independent datasets that use the same fitted representation.")
    actions = st.multiselect("Follow-up analyses", list(CHOICES), format_func=CHOICES.get, key=key + ":actions")
    requirements = {
        "analyze": "Known-label evaluation needs annotations. New analyses recompute enabled evaluations; use Change map or clustering below to retain saved scores.",
        "compare": "Requires another representation of the same objects. Missing input can be supplied in the draft; comparisons use a matched cohort.",
        "distill": "Requires known cell types, repeated classes and a reserved validation split. No target labels fit the reducer.",
        "context": "Requires physical adjacency, calibrated centroids or segmentation, plus explicit cell/nucleus mappings where needed.",
        "bilateral": "Requires independently defined partner candidates and a shared specimen/ID namespace. Physical adjacency is not a candidate table.",
        "fuse": "Requires at least two full feature blocks for the same objects. Block order and weights are reviewed in the draft.",
        "joint_analyze": "Requires independent datasets with the same fitted representation recipe; equal dimensions or filenames are insufficient.",
        "transfer": "Requires a labelled reference and compatible target vectors. Target labels are optional and reserved for evaluation.",
    }
    for action in actions:
        st.caption(requirements[action])
    if "transfer" in actions and selected:
        reference = st.selectbox("Reference representation for label transfer", selected, key=key + ":reference")
        selected = [reference, *(name for name in selected if name != reference)]
    if actions:
        st.markdown("**Draft inputs:** " + ", ".join(selected))
        with st.expander("Settings and provenance copied to the draft"):
            st.write("Selected vectors, parent result and applicable projection/classifier settings are retained. New tasks receive their own defaults for settings absent from this result.")
            st.json(metadata.get("settings", {}), expanded=False)
    linked = False
    if "distill" in actions and "bilateral" in actions:
        linked = st.checkbox("Probe original, PCA and adapter together after distillation", True, key=key + ":linked",
                             help="Uses each distillation's held-out search population by default. Uncheck to probe the selected input vectors instead.")
    if is_distillation:
        st.warning("Full vectors include adapter training objects. Held-out vectors are the safer starting point for a new probe; related cells can still share biological information.")
    if st.button("Create follow-up workflow draft", key=key + ":create", disabled=not selected or not actions):
        try:
            document = followup_document({name: choices[name] for name in selected}, actions, source=path,
                                         settings=metadata.get("settings"), distill_then_compare=linked,
                                         context={"representations": selected, "display_name": metadata.get("display_name")})
            draft = new_draft(document=document, source=str(path))
            draft["name"] = "Follow-up · " + metadata.get("display_name", Path(path).parent.name)
            use_draft(config, draft, "Analyze")
            navigate("Workflow")
        except ValueError as error:
            st.error(str(error))


def download_file(path, label):
    from morphofeatures.artifact_locations import artifact_path

    path = artifact_path(path)
    from morphofeatures.artifact_availability import selected_artifact

    status = selected_artifact(path)
    if not status.ready:
        return
    if path.stat().st_size > 200 * 1024**2:
        st.caption(f"{label}: above the 200 MiB browser limit; retrieve from {path}")
    else:
        st.download_button(label, path.read_bytes(), path.name, key="file-download:" + str(path) + ":" + label)
