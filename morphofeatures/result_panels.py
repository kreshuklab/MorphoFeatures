"""Render one analysis section at a time without fitting or loading other views."""

from pathlib import Path

import pandas as pd
import streamlit as st


def analysis_panels(config, metadata, path):
    from morphofeatures.analysis_ui import (
        classification_results,
        inspect_results,
        mobie_export_start,
        projection_results,
    )
    from morphofeatures.artifact_availability import (
        availability_details,
        available_vectors,
        probe,
        selected_artifact,
    )
    from morphofeatures.artifact_locations import artifact_path
    from morphofeatures.followup_ui import download_file
    from morphofeatures.result_browser_ui import focus
    from morphofeatures.spatial_probe_ui import nested_bilateral_results
    from morphofeatures.workspace_navigation import selector_key

    entries = metadata.get("representations", {})
    declared = (
        {name: path.parent / value["directory"] / "embeddings.npz" for name, value in entries.items()}
        if entries else {"Embedding": path.parent / metadata.get("embedding", "embeddings.npz")}
    )
    embeddings = available_vectors(declared)
    coordinate_path = artifact_path(path.parent / metadata.get("coordinates", "coordinates.tsv"))
    coordinates = selected_artifact(coordinate_path)
    sections = []
    if coordinates.ready and coordinates.kind == "coordinates":
        sections.append("Projection")
    if any(v.get("classification", {}).get("classifiers") for v in (list(entries.values()) if entries else [metadata])):
        sections.append("Classification")
    if metadata.get("bilateral") and (status := probe(path.parent / metadata["bilateral"])).ready and status.kind == "report":
        sections.append("Bilateral probe")
    if embeddings:
        sections.append("Objects & meshes")
    if embeddings or coordinates.ready or probe(path.parent / "export.zip").ready:
        sections.append("Exports")
    availability_details({**declared, "Saved coordinates": coordinate_path}, key="analysis:" + str(path))
    if not sections:
        st.info("This report has no usable result views yet. Files & provenance retains its settings and recorded outputs.")
        return
    section_key = "analysis-section:" + str(path)
    if st.session_state.get(section_key) not in sections:
        st.session_state.pop(section_key, None)
    for entry in (list(entries.values()) if entries else [metadata]):
        supervision = entry.get("extraction", {}).get("supervision", {})
        if supervision.get("warning"):
            st.warning(supervision["warning"])
            break
    section = st.radio(
        "Analysis section", sections,
        horizontal=True, key=selector_key(section_key),
    )
    if section == "Classification":
        scores = [dict(representation=name, **row)
                  for name, value in entries.items() for row in value.get("evaluation", [])]
        if scores:
            with st.expander("Compare saved classifier scores across representations"):
                st.dataframe(pd.DataFrame(scores))
        classification_results(metadata, path)
        if not metadata.get("classification") and not any(v.get("classification") for v in entries.values()):
            st.info("This report has no classifier evaluation. Distillation scores are under its parent report's Scores section; for other embeddings, configure a classification follow-up with known labels.")
        return
    if section == "Bilateral probe":
        if metadata.get("bilateral"):
            nested_bilateral_results(config, metadata, path)
        else:
            st.info("No bilateral probe was included. Select Next steps to add one using these saved vectors.")
        return
    if section == "Exports":
        mobie_export_start(config, metadata, path)
        download_file(path.parent / "coordinates.tsv", "Export coordinates and clusters")
        download_file(path.parent / "export.zip", "Download report, embeddings, figures, and splits")
        for name, file in embeddings.items():
            download_file(file, "Download vectors · " + name)
            download_file(Path(file).with_suffix(".metadata.json"), "Download vector provenance · " + name)
        return
    frame = pd.read_csv(coordinate_path, sep="\t", dtype={"label_id": "int64"}) if coordinates.ready else pd.DataFrame({"label_id": []})
    if section == "Projection":
        if len(entries) > 1:
            selected = st.multiselect("Representations to display", list(entries), default=list(entries)[:1],
                                      key=selector_key("display-representations:" + str(path)),
                                      help="Show one representation at a time, or select several to compare maps. Each representation has a separately fitted coordinate space.")
            frame = frame.loc[frame.representation.isin(selected)].copy()
            focus(path, metadata, vectors={name: value for name, value in embeddings.items() if name in selected})
            if frame.empty:
                st.info("Select a representation to display its map.")
                return
        if metadata.get("interpretation"):
            st.caption(metadata["interpretation"])
        if metadata.get("settings", {}).get("use_published_layout"):
            st.info("Showing saved published coordinates and clusters, not a newly fitted projection.")
            missing = metadata.get("diagnostics", {}).get("objects_without_published_coordinates", [])
            if missing:
                st.warning(f"{len(missing):,} objects have no saved published coordinates; their vectors remain available. See Files & provenance for excluded IDs.")
        projection_results(frame, metadata, path)
    else:
        st.caption("Points selected on the Projection section remain available here for crop, mesh and volume inspection.")
        inspect_results(config, embeddings, metadata, frame, path)
