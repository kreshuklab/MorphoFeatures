"""Visual lineage and actionable representation comparisons in existing pages."""

import json

import pandas as pd
import streamlit as st

from morphofeatures.provenance import build_provenance, provenance_dot, workflow_overview


def compatibility_details(comparisons):
    for item in comparisons:
        label = f"{item['reference']} → {item['target']}"
        if item.get("preparation_differences"):
            verification = (
                "the recorded model/feature recipes match"
                if item.get("verified")
                else "the available settings do not conflict, but model provenance is incomplete"
            )
            st.warning(
                f"{label}: {verification}; input preparation differs. "
                "Transfer can run; differences in image scale, crop coverage or masking can affect biological accuracy. "
                "Use matched physical views for a controlled comparison."
            )
            rows = [
                {
                    "Input setting": row["parameter"],
                    f"{item['reference']} (reference)": "not applicable / not recorded"
                    if row["reference"] is None
                    else str(row["reference"]),
                    f"{item['target']} (target)": "not applicable / not recorded"
                    if row["target"] is None
                    else str(row["target"]),
                }
                for row in item["preparation_differences"]
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        elif item.get("verified"):
            st.success(f"{label}: recorded model, view and fusion recipes match.")
        else:
            st.warning(
                f"{label}: compatibility is user-confirmed because model provenance is incomplete."
            )


def transfer_input_check(stage):
    from morphofeatures.analysis.transfer import validate_transfer_inputs

    try:
        compatibility_details(validate_transfer_inputs(stage, allow_pending=True))
    except (ValueError, OSError, TypeError, KeyError) as error:
        st.error("Analyze → Transfer labels → Reference / target embedding files: " + str(error))


def provenance_view(*, key, document=None, base=None, artifact=None):
    with st.expander("Data provenance · workflows, settings and failures", expanded=False):
        st.caption(
            "Follow solid arrows from inputs to their processing steps and outputs. Dashed arrows show execution order only. "
            "Choose a node below to inspect its settings and recorded errors. This reads saved metadata, not image volumes."
        )
        # A collapsed Streamlit expander still executes; keep filesystem tracing
        # opt-in so browsing results does not repeatedly walk upstream files.
        if not st.checkbox(
            "Show provenance graph",
            key=key + ":show",
            help="Read the saved workflow and extraction metadata to show where this result came from. This does not rerun analysis or access image pixels.",
        ):
            return
        graph = build_provenance(document=document, base=base, artifact=artifact)
        for notice in graph["notices"]:
            st.info(notice)
        if not graph["nodes"]:
            st.info("No saved workflow or extraction metadata was found for this input.")
            return
        dot = provenance_dot(graph)
        details = st.checkbox(
            "Include individual files and detailed processing records",
            key=key + ":detail",
            help="The overview shows workflow steps and their dependencies. Enable this to also show source images, checkpoints, annotation files and saved extraction settings. The downloads always include the full recorded graph.",
        )
        st.graphviz_chart(
            dot if details else provenance_dot(workflow_overview(graph)), use_container_width=True
        )
        lookup = {n["id"]: n for n in graph["nodes"]}
        selected = st.selectbox(
            "Inspect a workflow step or data file",
            list(lookup),
            key=key + ":node",
            format_func=lambda v: lookup[v]["label"] + " · " + lookup[v]["state"],
            help="Choose a step to see its settings and failure details, or a file to see its path and saved metadata. All recorded nodes are available here, including files hidden in the overview.",
        )
        node = lookup[selected]
        st.code(node.get("path", ""), language=None)
        if node.get("execution", {}).get("error"):
            st.error("Recorded failure: " + node["execution"]["error"])
            issue = node["execution"].get("issue", {})
            if issue:
                st.caption("Location: " + issue["page"] + " → " + issue["stage_name"])
                for field in issue.get("fields", []):
                    st.code(".".join(map(str, field)), language=None)
        st.json(node.get("settings", {}))
        if node.get("naming"):
            st.caption("Run display name and rename history")
            st.json(node["naming"])
        if node.get("settings", {}).get("action") == "transfer":
            st.caption("Current input check (the historical run status above is preserved):")
            transfer_input_check(node["settings"])
        if node.get("resources"):
            with st.expander("Execution resources"):
                st.json(node["resources"])
        left, right = st.columns(2)
        left.download_button(
            "Download provenance JSON",
            json.dumps(graph, indent=2),
            "provenance.json",
            "application/json",
            key=key + ":json",
        )
        right.download_button(
            "Download graph (DOT)", dot, "provenance.dot", "text/vnd.graphviz", key=key + ":dot"
        )
