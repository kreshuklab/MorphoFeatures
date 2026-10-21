"""Saved-run comparisons, source connections and spatial-export workflow controls."""

import json
from pathlib import Path

import streamlit as st


def saved_run_controls(draft, path, settings):
    from morphofeatures.workflow_ui import field, rerun, touch

    field(
        draft,
        path + ["compare_saved_runs"],
        "Compare with previous bilateral runs",
        False,
        help="Reuse saved embedding vectors from any earlier bilateral run, including your own MAE/DINO experiments. All methods are recomputed on shared cells with this run's candidates and variants; the saved runs remain unchanged.",
    )
    if not settings.get("compare_saved_runs"):
        return
    from morphofeatures.registry import JobRegistry

    root = st.session_state.get("workspace_root")
    choices = {}
    if root:
        for record in JobRegistry.under_output_root(Path(root)).list(limit=500):
            for value in record.artifacts:
                artifact = Path(value)
                if artifact.name == "bilateral.json" and artifact.is_file():
                    name = f"{record.display_name} · {record.run_id} · {artifact.parent.name}"
                    choices[str(artifact)] = name
                elif artifact.name in {"analysis.json", "comparison.json"} and artifact.is_file():
                    # Also find probes included in Analyze/Compare results.
                    if artifact.stat().st_size < 8 * 1024**2:
                        try:
                            report = json.loads(artifact.read_text())
                        except (OSError, ValueError):
                            continue
                        if report.get("bilateral"):
                            nested = artifact.parent / report["bilateral"]
                            if nested.is_file():
                                name = f"{record.display_name} · {record.run_id} · {artifact.parent.name}"
                                choices[str(nested)] = name
    selected = st.multiselect(
        "Completed bilateral runs in this workspace",
        list(choices),
        format_func=lambda p: choices[p],
        key=str(path) + ":saved-runs",
        help="Choose runs, then add them to the comparison. Files outside this workspace can be entered in the named report box below.",
    )
    if st.button(
        "Add selected runs to this comparison", key=str(path) + ":add-runs", disabled=not selected
    ):
        settings.setdefault("comparison_reports", {}).update({choices[p]: p for p in selected})
        touch(draft)
        st.session_state["editor_epoch"] = st.session_state.get("editor_epoch", 0) + 1
        rerun()
    field(
        draft,
        path + ["comparison_reports"],
        "Named saved bilateral reports (YAML)",
        {},
        help="Map a unique run label to bilateral.json, for example {MAE earlier run: /path/run/workspace/02-bilateral/bilateral.json}. The full report folder, including saved embeddings.npz files, must be available. Remove an entry here to stop including it.",
    )
    st.info(
        "Comparison uses the intersection of cell IDs across new and saved vectors. A smaller shared population changes ranks. Results plots freshly computed curves; each saved run's original settings remain in provenance."
    )


def context_source_controls(draft, index):
    from morphofeatures.analysis.context_sources import context_catalog, resolve_context_source
    from morphofeatures.analysis.platybrowser import DEFAULTS
    from morphofeatures.workflow_ui import _local_stage_error, field, repository_root, rerun, touch

    path = ["document", "stages", index]
    stage = draft["document"]["stages"][index]
    prefix = str(draft.get("id", "")) + str(path) + ":context-source"
    field(
        draft,
        path + ["spatial_source"],
        "Spatial context segmentation source",
        "manual",
        options=["manual", "ome_zarr", "platybrowser"],
        format_func={
            "manual": "Enter a local container or remote array",
            "ome_zarr": "Stream an OME-Zarr segmentation (including Platy)",
            "platybrowser": "Choose a published PlatyBrowser segmentation",
        }.get,
        help="Choose the whole-animal instance segmentation used to define physical contacts. Only needed chunks are read. This step constructs context around existing embeddings; it does not re-embed raw images.",
    )
    source = stage.get("spatial_source", "manual")
    if source == "manual":
        return
    if stage.get("graph_mode") != "segmentation":
        st.info(
            "Streamed instance volumes define touching-label neighborhoods. Choose segmentation as the graph method, or select Manual above to use your physical position/edge table."
        )
        return
    cache_path = path + ["platybrowser" if source == "platybrowser" else "remote_options"]
    field(
        draft,
        cache_path + ["cache_directory"],
        "Context streaming cache directory",
        str(repository_root() / "outputs/remote_cache/context"),
        help="Writable worker-accessible folder for downloaded metadata and chunks. Repeated reads reuse cached data; this does not store a full-volume copy.",
    )
    field(
        draft,
        cache_path + ["cache_size_mb"],
        "Context streaming disk cache (MiB)",
        512,
        help="Bound the disk space used for remote chunks. Older cached chunks can be downloaded again when needed. This is independent of the graph scan and edge budgets.",
    )
    if source == "ome_zarr":
        for key, label, default, help_ in [
            (
                "url",
                "Context segmentation OME-Zarr URL",
                "",
                "Root of an integer instance-label OME-Zarr layer, not a raw image or binary foreground mask.",
            ),
            (
                "image_index",
                "OME-Zarr image in multiscales",
                0,
                "Zero-based image entry; most layers contain one image at index 0.",
            ),
            (
                "channel",
                "OME-Zarr segmentation channel",
                0,
                "Zero-based channel; 0 for a single-channel segmentation.",
            ),
            (
                "timepoint",
                "OME-Zarr segmentation time point",
                0,
                "Zero-based time index; 0 for a single time point.",
            ),
        ]:
            field(draft, path + ["ome_zarr", key], label, default, help=help_)
        spec = stage.get("ome_zarr", {})
        signature = (spec.get("url"), spec.get("image_index", 0))
        if st.button("Discover context segmentation resolutions", key=prefix + ":discover"):
            try:
                from morphofeatures.data.ome_zarr import discover_ome_zarr

                metadata = discover_ome_zarr(
                    spec["url"],
                    stage.get("remote_options", {}),
                    image_index=spec.get("image_index", 0),
                )
                st.session_state[prefix + ":levels"] = (signature, metadata)
            except Exception as error:
                _local_stage_error(
                    draft, index, error, "Context segmentation → Discover resolutions"
                )
        cached = st.session_state.get(prefix + ":levels")
        if cached and cached[0] == signature:
            levels = cached[1]["levels"]
            field(
                draft,
                path + ["ome_zarr", "level"],
                "Context segmentation resolution",
                0,
                options=list(range(len(levels))),
                format_func=lambda i: (
                    f"{i}: {levels[i]['key']} · spacing Z/Y/X {levels[i]['spacing_zyx']}"
                ),
                help="Coarser levels reduce streamed data and graph scan time, but can change contacts or remove small objects. Use the same level when comparing context recipes.",
            )
        else:
            field(
                draft,
                path + ["ome_zarr", "level"],
                "Context segmentation resolution index",
                0,
                help="Zero is usually finest. Discover resolutions to see physical voxel sizes before choosing.",
            )
    else:
        for key in ("project_url", "revision", "dataset"):
            field(
                draft,
                path + ["platybrowser", key],
                "Context PlatyBrowser " + key.replace("_", " "),
                DEFAULTS[key],
                help="Published project selection. Use a pinned revision for reproducibility; branches are resolved to a commit. The catalogue determines whether the data use N5 or OME-Zarr.",
            )
        signature = tuple(
            stage.get("platybrowser", {}).get(k, DEFAULTS[k])
            for k in ("project_url", "revision", "dataset")
        )
        if st.button("List published context segmentations", key=prefix + ":catalog"):
            try:
                catalog = context_catalog(stage.get("platybrowser", {}))
                st.session_state[prefix + ":catalog-result"] = (signature, catalog)
            except Exception as error:
                _local_stage_error(draft, index, error, "Context PlatyBrowser → List segmentations")
        cached = st.session_state.get(prefix + ":catalog-result")
        catalog = cached[1] if cached and cached[0] == signature else {}
        field(
            draft,
            path + ["platybrowser", "segmentation_source"],
            "Published context segmentation",
            "cells",
            options=catalog.get("segmentation") or None,
            help="Cells is appropriate for touching-cell context. Separated nuclei often have no contacts; use a physical position table and radius/KNN for that experiment.",
        )
        field(
            draft,
            path + ["platybrowser", "segmentation_level"],
            "Published context resolution level",
            0,
            help="Zero is finest. Coarser integer-label levels are faster but may alter adjacency; the selected format, spacing and dimensions are shown after connecting.",
        )
    if st.button("Connect context segmentation", key=prefix + ":connect"):
        try:
            stage.update(resolve_context_source(stage))
            st.session_state["editor_epoch"] = st.session_state.get("editor_epoch", 0) + 1
            touch(draft)
            rerun()
        except Exception as error:
            _local_stage_error(draft, index, error, "Context segmentation → Connect")
    metadata = stage.get("spatial_source_metadata")
    if metadata:
        st.caption(
            f"Connected {metadata['format']}: Z/Y/X {metadata['volume']['spatial_shape']}; voxel spacing {stage['segmentation_spacing_zyx']} {stage['segmentation_unit']}. Set a bounded ROI below; the voxel scan budget still applies."
        )
        st.caption(
            "The selected source and resolution determine the array URL, key, axes and channel below. To override those fields directly, switch the source to 'Enter a local container or remote array'."
        )
        plane = st.number_input(
            "Context segmentation preview Z slice",
            min_value=0,
            value=int(metadata["volume"]["spatial_shape"][0]) // 2,
            key=prefix + ":plane",
            help="Preview one bounded central XY window at this Z index. No whole-volume download occurs.",
        )
        if st.button("Preview context segmentation", key=prefix + ":preview"):
            try:
                import matplotlib.pyplot as plt

                from morphofeatures.data.volumes import preview_volume_region

                preview = preview_volume_region(
                    resolve_context_source(stage), "segmentation", axis=0, index=plane, max_side=256
                )
                figure, axis = plt.subplots(figsize=(4, 4))
                axis.imshow(
                    preview["segmentation"] % 256, cmap="nipy_spectral", interpolation="nearest"
                )
                axis.set_title("Instance labels · bounded XY preview")
                axis.axis("off")
                st.pyplot(figure)
                plt.close(figure)
                st.caption(
                    f"Voxel window: {preview['start']} to {preview['stop']}. Colors are for inspection only; graph construction uses the exact IDs."
                )
            except Exception as error:
                _local_stage_error(draft, index, error, "Context segmentation → Preview")


def export_controls(draft, index):
    from morphofeatures.analysis.bilateral_comparison import read_report
    from morphofeatures.workflow_ui import _export_mobie, field

    stage = draft["document"]["stages"][index]
    path = ["document", "stages", index]
    st.caption(
        "Export query→candidate links in physical 3D coordinates, with optional MoBIE pair views and separate query/partner mask layers. Bilateral candidates remain hypotheses; one cell can participate in several pairs."
    )
    if stage.get("from_bilateral"):
        st.info("Uses the bilateral result produced earlier in this workflow.")
    else:
        field(
            draft,
            path + ["analysis"],
            "Bilateral result to export (bilateral.json)",
            "",
            help="A completed bilateral report. Saved comparison representations can be exported in exactly the same way as new inputs.",
        )
    metadata = {}
    if stage.get("analysis") and Path(stage["analysis"]).is_file():
        try:
            _, metadata = read_report(stage["analysis"])
        except (ValueError, OSError) as error:
            st.error(str(error))
    names = list(metadata.get("representations", {}))
    field(
        draft,
        path + ["bilateral_representation"],
        "Representation to export (empty means all)",
        None,
        options=[None] + names if names else None,
        optional=True,
        help="Select one representation from the saved result, or leave empty to export every representation in separate groups.",
    )
    field(
        draft,
        path + ["bilateral_variant"],
        "Probe variant index to export (null means all)",
        None,
        help="Zero-based index in the report. Leave null for every variant; different distance/search settings remain separate groups.",
    )
    field(
        draft,
        path + ["query_ids"],
        "Specific query cell IDs (optional)",
        [],
        help="Cell IDs from the bilateral rank table, e.g. [300, 466]. Empty selects pairs automatically using the ordering below. Their best candidate IDs are looked up in the saved report.",
    )
    field(
        draft,
        path + ["pair_order"],
        "Automatic pair selection",
        "best",
        options=["best", "worst", "id"],
        help="Best selects lowest retrieval ranks; worst selects highest ranks; id selects ascending cell IDs. These choices only affect visualization, never the probe metrics.",
    )
    field(
        draft,
        path + ["max_pairs"],
        "Maximum pairs per representation / variant",
        20,
        help="Limit visualization/export size independently of the number of evaluated cells. Explicit ID selections exceeding this limit raise an error. Each MoBIE view renders just one pair.",
    )
    field(
        draft,
        path + ["make_mobie"],
        "Create a MoBIE project with 3D pair views",
        True,
        help="Export rank-colored objects and selectable query/candidate views, alongside the non-MoBIE geometry files. Choose published streaming or a portable spatial ROI below.",
    )
    field(
        draft,
        path + ["export_pair_masks"],
        "Also export query/partner masks as HDF5",
        False,
        help="One P×Z×Y×X stack with a separate layer per pair: 0 background, 1 query, 2 candidate. Uses a bounded portable ROI; shared cells are never overwritten by another pair.",
    )
    field(
        draft,
        path + ["max_mask_voxels"],
        "Maximum total voxels across pair-mask layers",
        256**3,
        help="Pairs × ROI volume must fit this total work/output budget. Each voxel uses one byte before compression; lower pair count or ROI size to reduce work.",
    )
    field(
        draft,
        path + ["positions"],
        "Physical position table for non-MoBIE geometry (optional)",
        None,
        optional=True,
        help="TSV/CSV with label_id and physical Z/Y/X coordinates. Platy default.tsv anchor columns work. If omitted, a MoBIE export supplies positions from its object catalogue or copied segmentation ROI.",
    )
    field(
        draft,
        path + ["position_columns"],
        "Physical position columns in Z,Y,X order",
        ["anchor_z", "anchor_y", "anchor_x"],
        help="Names of the three coordinate columns in your position table. These may be object anchors rather than exact centroids; record that distinction when interpreting connecting lines.",
    )
    field(
        draft,
        path + ["positions_ids"],
        "IDs in the physical position table",
        "cell",
        options=["cell", "segmentation"],
        help="Cell matches the probe. Segmentation uses the cell-to-segmentation mapping below, e.g. when using a nucleus catalogue.",
    )
    field(
        draft,
        path + ["positions_unit"],
        "Physical position table unit",
        "um",
        options=["um", "nm"],
        help="All exported link coordinates are converted to micrometers. Volume masks retain their configured voxel spacing and origin.",
    )
    if stage.get("make_mobie", True) or stage.get("export_pair_masks"):
        _export_mobie(draft, index)
    else:
        field(
            draft,
            path + ["id_mapping"],
            "Cell-to-segmentation ID mapping (optional)",
            None,
            optional=True,
            help="Required if the physical position table uses nucleus/segmentation IDs instead of cell IDs. Columns label_id and nucleus_id or segmentation_id.",
        )
