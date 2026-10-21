"""Small intent/input decisions before the existing scientific stage forms."""

import streamlit as st

from morphofeatures.workflow_catalog import (
    FAMILIES,
    FAMILY_HELP,
    INPUTS,
    RECIPES,
    SOURCES,
    TASKS,
    guided_draft,
)
from morphofeatures.workspace_state import STARTING_POINTS


def selection(config):
    from morphofeatures.workflow_ui import navigate

    def guided():
        st.session_state["start:direct"] = False

    st.subheader("What would you like to achieve?")
    family = st.radio("Your goal", list(FAMILIES), format_func=FAMILIES.get,
                      horizontal=True, key="start:family", on_change=guided)
    st.caption(FAMILY_HELP[family])
    inputs = st.selectbox("What do you already have?", list(INPUTS), format_func=INPUTS.get,
                          key="start:inputs", on_change=guided)
    task, source, recipe = None, "raw", "dino"
    if inputs in {"vectors", "published"} or (inputs in {"images", "prepared", "checkpoint"} and family != "create"):
        task = st.selectbox("Scientific question", list(TASKS[family]), format_func=TASKS[family].get,
                            key="start:task:" + family, on_change=guided)
    if inputs == "result":
        st.info("Open the saved result to view its available outputs and create a follow-up using the selected dataset and representation.")
        if st.button("Open Results", key="start:results"):
            navigate("Results")
    elif inputs in {"images", "prepared", "checkpoint"}:
        if inputs == "images":
            source = st.selectbox("Image source", list(SOURCES), format_func=SOURCES.get,
                                  key="start:source", on_change=guided)
        if inputs == "checkpoint":
            recipe = "checkpoint"
            st.caption("Reuse training. Supply the matching model configuration and prepared data; images can be prepared in a separate workflow.")
        else:
            recipes = (list(RECIPES) if inputs == "images" and family == "create" else ["dino", "checkpoint", "crops"]) + ["specialist"]
            recipe = st.selectbox("What should this workflow produce?", recipes, format_func=lambda k: "Published shape / texture models (specialist)" if k == "specialist" else RECIPES[k],
                                  key="start:recipe:" + inputs, on_change=guided)
            if recipe == "specialist":
                st.caption("Published shape and texture models keep their model-specific configuration and command contracts.")
                if st.button("Configure published shape / texture workflow"):
                    st.session_state["tools:selection"] = "Published shape / texture workflows and MAE sweeps"
                    navigate("Tools")
        if family != "create":
            st.info("The plan extracts embeddings first, then connects them to your scientific question. Comparison needs another representation; transfer uses these objects as the labelled reference and requires target vectors.")
        if recipe == "dino":
            st.caption("Frozen DINO needs local pretrained weights and its matching code checkout. No model training is required.")
        if recipe == "dino_experiment":
            st.caption("A distinct physical sampling experiment: calibrated volumes are required for fine/coarse coverage, multiple scales and matched comparisons.")
    elif inputs == "vectors":
        if task == "joint_embeddings":
            st.caption("Independent datasets need the same fitted representation recipe. Their native object IDs remain separate.")
        elif task == "compare":
            st.caption("Different feature methods are compared on shared objects; each method has its own fitted map.")
        elif task == "context":
            st.caption("Use a physical neighborhood graph. Bilateral evaluation partners and feature-space neighbors are different inputs.")
    elif inputs == "published":
        st.caption("Preserve the published coordinates and clusters, or explicitly configure a new analysis. Viewing the references is also available in Results.")

    def direct():
        st.session_state["start:direct"] = True

    with st.expander("Direct presets, demonstration and specialist workflows"):
        preset = st.selectbox("Starting point", [None, *STARTING_POINTS], format_func=lambda p: STARTING_POINTS.get(p, "Choose a direct preset"),
                              key="start:preset", on_change=direct)
        st.caption("Compatibility shortcuts retain the existing preset defaults. Guided choices above build a plan from your inputs.")
        if st.button("Open specialist tools"):
            navigate("Tools")
        if st.button("Try the synthetic demonstration"):
            st.session_state["start:demo"] = True
            st.session_state["start:direct"] = True
    if st.session_state.pop("start:demo", False):
        from morphofeatures.workspace_state import new_draft
        return "demo", new_draft("demo")
    if st.session_state.get("start:direct") and preset:
        return preset, None
    if inputs == "result" or recipe == "specialist":
        return None, None
    proposed = guided_draft(family=family, inputs=inputs, task=task, source=source, recipe=recipe,
                            model_defaults=st.session_state.get("workspace_dino_defaults"))
    start = "published" if inputs == "published" else task or (source if inputs == "images" else recipe)
    if inputs == "images" and recipe == "preprocess" and source == "raw":
        start = "preprocess"
    return "embeddings" if start == "compare" else start, proposed
