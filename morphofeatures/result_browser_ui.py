"""Experiment-first browsing over the shared artifact availability contracts."""

from pathlib import Path

import pandas as pd
import streamlit as st

from morphofeatures.artifact_availability import (
    availability_details,
    probe,
    selected_artifact,
)
from morphofeatures.result_catalog import run_results, search_names, vector_artifacts
from morphofeatures.workspace_navigation import selector_key


def focus(path, metadata, *, vectors=None):
    vectors = vectors if vectors is not None else vector_artifacts(metadata, path)
    parent = st.session_state.get("results:parent", {})
    if parent.get("schema") == "morphofeatures.distillation.v1" and str(path) != st.session_state.get("results:root"):
        # Maps include training objects; default new experiments to the selected
        # method's held-out vectors, without losing the selected map's settings.
        from morphofeatures.artifact_locations import artifact_path

        matching = {entry["name"] + " · held-out population": str(artifact_path(Path(st.session_state["results:root"]).parent / entry["held_out"]))
                    for entry in parent.get("representations", {}).values()
                    if entry.get("held_out") and entry.get("name") in vectors}
        if matching:
            vectors = matching
    st.session_state["results:focus"] = {
        "root": st.session_state.get("results:root", str(path)),
        "path": str(path), "metadata": metadata,
        "vectors": vectors,
    }


def safe_choice(label, options, *, key, **kwargs):
    """Stale saved selections cannot reopen artifacts that are no longer available."""
    if not options:
        return None
    options = list(options)
    if st.session_state.get(key) not in options:
        st.session_state.pop(key, None)
    if len(options) == 1:
        st.session_state[key] = options[0]
        selector_key(key)
        st.caption(label + ": " + str(kwargs.get("format_func", str)(options[0])))
        return options[0]
    return st.selectbox(label, options, key=selector_key(key), **kwargs)


def published_results(config):
    from morphofeatures.analysis.published import references
    from morphofeatures.analysis_ui import projection_results
    from morphofeatures.followup_ui import download_file
    from morphofeatures.workflow_ui import navigate, use_draft
    from morphofeatures.workspace_state import new_draft

    catalog = references()
    name = st.selectbox("Published reference", list(catalog), key="results:published")
    reference = catalog[name]
    st.caption("Saved historical coordinates and clusters are displayed without fitting a new projection. The original context graph and fitted reducer are not bundled.")
    layout = reference.get("layout")
    if layout and selected_artifact(layout).ready:
        frame = pd.read_csv(layout, sep="\t", dtype={"label_id": "int64"})
        projection_results(frame, {"settings": {"use_published_layout": True}}, Path(layout))
        download_file(layout, "Download published coordinates and clusters")
    else:
        st.info("This reference has no usable saved layout here. Its vectors can still be used in a new analysis when available.")
    availability_details({"Published vectors": reference["embedding"], **({"Published layout": layout} if layout else {})}, key="published:" + name)
    st.subheader("Next steps")
    if st.button("Create an analysis draft for this reference"):
        draft = new_draft("published")
        draft["document"]["stages"][0].update(
            embedding=reference["embedding"], published_reference=name,
            use_published_layout=bool(layout), umap=bool(layout),
        )
        from copy import deepcopy
        draft["source_document"] = deepcopy(draft["document"])
        use_draft(config, draft, "Analyze")
        navigate("Workflow")


def render_result(config, metadata, path):
    schema = metadata.get("schema")
    if schema == "morphofeatures.distillation.v1":
        from morphofeatures.distillation_ui import results
        results(config, metadata, path)
    elif schema == "morphofeatures.bilateral_export.v1":
        from morphofeatures.spatial_probe_ui import bilateral_export_results
        bilateral_export_results(config, metadata, path)
    elif schema in {"morphofeatures.bilateral.v1", "morphofeatures.context.v1"}:
        from morphofeatures.spatial_probe_ui import probe_results
        probe_results(config, metadata, path)
    elif schema in {"morphofeatures.dino_experiment.v1", "morphofeatures.transfer.v1"}:
        from morphofeatures.dino_experiment_ui import experiment_results
        experiment_results(config, metadata, path)
    elif schema == "morphofeatures.joint_analysis.v1":
        from morphofeatures.analysis_ui import joint_projection_results
        joint_projection_results(config, metadata, path)
    elif schema == "morphofeatures.mobie_export.v1":
        from morphofeatures.analysis_ui import mobie_export_results
        mobie_export_results(metadata, path)
    elif schema in {"morphofeatures.analysis.v1", "morphofeatures.comparison.v1"}:
        from morphofeatures.result_panels import analysis_panels
        analysis_panels(config, metadata, path)
    else:
        from morphofeatures.analysis_ui import inspect_results, projection_results
        from morphofeatures.followup_ui import download_file

        status = selected_artifact(path)
        if status.kind in {"prepared", "checkpoint"} and status.ready:
            from morphofeatures.workflow_ui import continue_from_artifact
            st.write("Prepared data" if status.kind == "prepared" else "Trained checkpoint")
            st.caption(str(path))
            st.caption("Reuse this input in a new draft. Review checks the matching model, data and execution settings.")
            if st.button("Create workflow from this input", key="reuse:" + str(path)):
                continue_from_artifact(config, path, kind="checkpoint" if status.kind == "checkpoint" else None)
            if status.kind == "prepared" and st.button("Extract DINO from these prepared objects", key="reuse-dino:" + str(path)):
                continue_from_artifact(config, path, kind="dino")
        elif status.kind == "coordinates" and status.ready:
            frame = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",", dtype={"label_id": "int64"})
            st.info("Saved projection table. These coordinates are not the original embedding vectors.")
            projection_results(frame, {}, path)
            download_file(path, "Download saved coordinates")
        elif status.kind == "vectors" and status.ready:
            download_file(path, "Download embedding vectors")
            inspect_results(config, {path.stem: path}, {}, pd.DataFrame({"label_id": []}), path)
        elif metadata:
            from morphofeatures.workflow_outputs import show_result_outputs
            show_result_outputs(metadata, path)
            vectors = vector_artifacts(metadata, path)
            from morphofeatures.artifact_availability import available_vectors
            for name, vector in available_vectors(vectors).items():
                download_file(vector, "Download vectors · " + name)
            with st.expander("Saved report"):
                st.json(metadata)
        else:
            st.info("This file has no supported scientific view. Inspect its availability or open a report, embedding, or saved projection table.")
        availability_details({path.name: str(path)}, key="result:" + str(path))


def browser(config):
    from morphofeatures.artifact_locations import artifact_path, configure_locations
    try:
        configure_locations(config.paths.output_root)
    except (OSError, ValueError) as error:
        st.warning("Could not read recorded artifact locations: " + str(error))
    from morphofeatures.analysis.variants import REPORT_SCHEMAS
    from morphofeatures.analysis_variants_ui import result_controls, result_name_controls
    from morphofeatures.config import repository_root
    from morphofeatures.followup_ui import controls, download_file
    from morphofeatures.job_status import reconcile_job_states
    from morphofeatures.registry import JobRegistry
    from morphofeatures.run_deletion import DELETING
    from morphofeatures.workflow_outputs import show_result_outputs
    from morphofeatures.workspace_ui import retained_input

    registry = JobRegistry.under_output_root(config.paths.output_root)
    reconcile_job_states(registry)
    limit = st.session_state.get("results:limit", 100)
    records = [r for r in registry.list(limit=limit) if r.application_state not in DELETING]
    source = st.radio("Browse", ["Workspace experiments", "Open a file", "Published references"],
                      horizontal=True, key="results:source")
    if source == "Published references":
        published_results(config)
        return
    chosen = "Enter path"
    if source == "Workspace experiments":
        query = st.text_input("Find experiment, dataset or representation", key="results:search").casefold()
        catalog = [(r, *run_results(r)) for r in records]
        if query:
            catalog = [(r, entries, missing) for r, entries, missing in catalog if query in (
                r.display_name + " " + r.run_id + " " + " ".join(search_names(e) for e in entries)
            ).casefold()]
        usable = [(r, entries, missing) for r, entries, missing in catalog if entries]
        if usable:
            owners = {r.id: (r, entries, missing) for r, entries, missing in usable}
            identifier = safe_choice("Experiment", list(owners), key="results:experiment",
                                     format_func=lambda i: owners[i][0].display_name + " · " + owners[i][0].run_id)
            record, entries, missing = owners[identifier]
            st.caption(f"{record.application_state} · {len(entries)} saved result(s). Available stages can be reused even when a later stage failed.")
            options = {entry["path"]: entry for entry in entries}
            chosen = safe_choice("Result", list(options), key="results:registered",
                                 format_func=lambda p: options[p]["name"] + (" · " + Path(p).parent.name if p != "Enter path" and len(entries) > 1 else ""))
        else:
            st.info("No matching saved results yet. Open an existing file, browse published references, or start a workflow.")
        unavailable = [(r.display_name, item) for r, _, missing in catalog for item in missing]
        if unavailable:
            with st.expander(f"Older or unavailable artifacts · {len(unavailable)}"):
                st.dataframe([{"Experiment": name, "Status": item.state, "Reason": item.reason, "Path": item.path} for name, item in unavailable], hide_index=True)
                st.caption("Open a moved report at its new location. Original run provenance is retained.")
        if len(records) >= limit and st.button("Show older experiments"):
            st.session_state["results:limit"] = limit + 100
            st.rerun()
    with st.expander("Open an existing or moved result", expanded=source == "Open a file" or not records):
        manual = retained_input(
            "Saved analysis.json, comparison.json, joint_analysis.json, or embedding file", "", "results:path",
            help="Also accepts experiment, distillation, transfer, spatial-probe and export reports, or a saved PCA/UMAP table. Paths are on the app host. Opening does not rerun analysis.",
            on_change=lambda: st.session_state.update({"results:source": "Open a file", "results:registered": "Enter path"}),
        )
    if source == "Open a file":
        chosen = "Enter path"
    if chosen == "Enter path" and not manual.strip():
        return
    path = artifact_path(repository_root() / Path(manual if chosen == "Enter path" else chosen).expanduser()).resolve()
    st.session_state["results:root"] = str(path)
    status = probe(path)
    if not status.ready and status.state != "unchecked":
        st.warning(f"This result is {status.state}: {status.reason}")
        st.caption("Open the new location above if it moved. Other experiment results remain available.")
        return
    metadata = status.metadata if path.suffix == ".json" else {}
    st.session_state["results:parent"] = metadata
    section_key = "result-view:" + str(path)
    if st.session_state.get(section_key) == "Follow-up analysis":
        st.session_state[section_key] = "Next steps"
    section = st.radio("Result view", ["Explore results", "Next steps", "Files & provenance"],
                       horizontal=True, key=selector_key(section_key))
    if metadata.get("schema") in REPORT_SCHEMAS:
        path, metadata = result_controls(path, metadata, exploration=False)
    elif metadata:
        result_name_controls(path, metadata)
    top = (st.session_state["results:root"], str(path))
    if st.session_state.get("results:top") != top:
        focus(path, metadata)
        st.session_state["results:top"] = top
    if section == "Files & provenance":
        from morphofeatures.provenance_ui import provenance_view
        show_result_outputs(metadata or {"schema": "morphofeatures.embedding.v1"}, path)
        provenance_view(key="results:provenance:" + str(path), artifact=path)
        availability_details(vector_artifacts(metadata, path), key="files:" + str(path))
        with st.expander("All saved settings and metrics"):
            st.json(metadata)
        download_file(path.parent / "export.zip", "Download complete results archive")
        return
    if section == "Next steps":
        selected = st.session_state.get("results:focus", {})
        if selected.get("root") != st.session_state["results:root"]:
            focus(path, metadata)
            selected = st.session_state["results:focus"]
        st.caption("Using selected result: " + selected["metadata"].get("display_name", Path(selected["path"]).parent.name))
        selected_path, selected_metadata = Path(selected["path"]), selected["metadata"]
        if selected_path != path or selected["vectors"] != vector_artifacts(metadata, path):
            scope = st.radio("Follow-up scope", ["Selected view", "Whole result"], horizontal=True, key="followup-scope:" + str(path))
            if scope == "Whole result":
                selected_path, selected_metadata = path, metadata
                selected = {"vectors": vector_artifacts(metadata, path)}
        controls(config, selected_metadata, selected_path, inputs=selected["vectors"])
        if selected_metadata.get("schema") in REPORT_SCHEMAS:
            from morphofeatures.analysis_variants_ui import variant_draft_controls
            variant_draft_controls(config, selected_path, selected_metadata)
            from morphofeatures.analysis_ui import mobie_export_start
            mobie_export_start(config, selected_metadata, selected_path)
        if metadata.get("schema") == "morphofeatures.dino_experiment.v1":
            from morphofeatures.dino_experiment_ui import saved_combination_controls
            saved_combination_controls(config, metadata, path)
        return
    focus(path, metadata)
    try:
        render_result(config, metadata, path)
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        st.warning("Some saved outputs could not be displayed: " + str(error))
        st.caption("Other outputs and provenance remain accessible. Use Next steps to configure a new experiment from available vectors.")
    st.subheader("Next steps")
    selected = st.session_state.get("results:focus", {})
    names = list(selected.get("vectors", {}))
    if names:
        st.caption("Selected inputs: " + ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else ""))
    st.caption("Reuse selected outputs in a reviewable draft. Additional labels, representations or spatial sources may be required.")
    st.button("Choose a follow-up experiment", key="next:" + str(path),
              on_click=lambda: st.session_state.update({section_key: "Next steps"}))
