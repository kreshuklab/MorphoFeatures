"""Workflow controls and results for published references and spatial probes."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from morphofeatures.workspace_jobs import embedding_names


def identity_fields(draft, path):
    from morphofeatures.workflow_ui import field

    field(
        draft,
        path + ["id_namespace"],
        "IDs stored in these embeddings",
        "cell",
        options=["cell", "nucleus"],
        help="Published bilateral candidates and context graphs use cell IDs. Choose nucleus only if your embedding IDs are nucleus IDs; a mapping then joins them to their corresponding cells. Nuclear image content does not automatically mean nucleus IDs.",
    )
    field(
        draft,
        path + ["id_mapping"],
        "Cell-to-nucleus mapping (when needed)",
        None,
        optional=True,
        help="TSV/CSV with one cell and one nucleus ID per row. Positive IDs must be one-to-one; missing mappings raise a pointed error instead of guessing from numeric overlap.",
    )
    field(
        draft,
        path + ["cell_column"],
        "Cell ID column in mapping",
        "label_id",
        help="Column containing the cell IDs used in the published candidate table or spatial graph.",
    )
    field(
        draft,
        path + ["nucleus_column"],
        "Nucleus ID column in mapping",
        "nucleus_id",
        help="Column containing the nucleus IDs used in your embedding or segmentation.",
    )


def bilateral_fields(draft, path, settings, *, inline=False):
    from morphofeatures.analysis.bilateral import defaults
    from morphofeatures.workflow_ui import field, rerun, touch

    defaults_ = defaults()
    if inline:
        field(
            draft,
            path + ["enabled"],
            "Evaluate bilateral-partner retrieval",
            False,
            help="Ask how highly the closest listed bilateral candidate ranks among other cells in embedding space. This evaluates saved vectors; it does not use UMAP coordinates or create context features.",
        )
        if not settings.get("enabled"):
            return
    from morphofeatures.spatial_probe_controls import saved_run_controls

    saved_run_controls(draft, path, settings)
    st.caption(
        "Lower rank is better. Every compared representation uses the same shared cell search population. Candidate sets are anatomical hypotheses, not verified one-to-one pairs."
    )
    field(
        draft,
        path + ["candidates"],
        "Bilateral candidate partners",
        defaults_["candidates"],
        help="Published Platy candidate dictionary, or a TSV/CSV with source_id and target_id. These are cell IDs. Supply your own independently derived candidates for another specimen.",
    )
    field(
        draft,
        path + ["same_specimen"],
        "These representations describe the same specimen and cells",
        False,
        help="Required for multiple representations or comparisons with saved bilateral runs. Raw and balanced combinations from the same cells qualify; different feature values do not change cell identity. This is separate from the Combine embeddings identity setting. Matching numeric IDs from different animals does not establish correspondence. Verify that the candidate table belongs to this specimen too.",
    )
    stage = draft["document"]["stages"][path[2]]
    count = (
        len(set(stage.get("embeddings", {})) | set(stage.get("from_embeddings", [])))
        if stage.get("action") in {"bilateral", "compare"} else 1
    )
    if not settings.get("same_specimen") and (count > 1 or settings.get("compare_saved_runs")):
        st.warning(
            "Bilateral retrieval needs the confirmation above before this workflow can run. "
            "Check it after verifying that all representations describe the same specimen and "
            "corresponding cells. For fusion, projections and classification without this probe, "
            "turn off 'Evaluate bilateral-partner retrieval'."
            if inline else
            "Bilateral retrieval needs the confirmation above before this workflow can run. "
            "Verify that all representations and candidate partners belong to the same specimen "
            "and corresponding cells, then check the box."
        )
    field(
        draft,
        path + ["sides"],
        "Body-side table for opposite-side search",
        defaults_["sides"],
        help="TSV/CSV with label_id and side (True/False or 1/0). Opposite-side variants exclude ambiguous candidate sets that include same-side cells, following the historical protocol.",
    )
    identity_fields(draft, path)
    field(
        draft,
        path + ["top_k"],
        "Report retrieval within these ranks",
        defaults_["top_k"],
        help="For example [1, 5, 10, 50, 100]. Recall@10 is the fraction of evaluated cells with a listed candidate among the ten nearest other cells. A chance expectation accounts for candidate-set size.",
    )
    settings.setdefault("variants", defaults_["variants"])
    for i, variant in enumerate(settings["variants"]):
        with st.expander(
            f"Bilateral variant {i + 1}: {variant.get('name', 'Unnamed')}", expanded=i == 0
        ):
            p = path + ["variants", i]
            field(
                draft,
                p + ["name"],
                "Variant name",
                f"Variant {i + 1}",
                help="A unique label identifying these distance and search settings in result tables and figures.",
            )
            field(
                draft,
                p + ["normalization"],
                "Scale features before retrieval",
                "standardize",
                options=["standardize", "none", "l2"],
                help="Standardize each feature across the matched cohort for the published protocol. None preserves existing feature/block weights; L2 compares vector directions. This is exploratory retrieval, not held-out classification.",
            )
            field(
                draft,
                p + ["metric"],
                "Distance between feature vectors",
                "euclidean",
                options=["euclidean", "cosine"],
                help="Euclidean measures straight-line distance after scaling. Cosine compares directions. Neither option measures physical distance between cells.",
            )
            field(
                draft,
                p + ["search"],
                "Cells eligible as retrieved neighbors",
                "all",
                options=["all", "opposite"],
                help="All searches the entire matched population except the query. Opposite searches only the other body side and needs the side table. Both report one-based ranks; the old opposite-side script used zero-based indices.",
            )
            field(
                draft,
                p + ["pca"],
                "PCA dimensions before retrieval (optional)",
                None,
                help="Leave null to evaluate full embeddings as in the published default. A positive integer fits PCA on the shared population before distances; this changes the experiment and is reported separately.",
            )
            if len(settings["variants"]) > 1 and st.button(
                "Remove this bilateral variant", key=str(p) + ":remove"
            ):
                settings["variants"].pop(i)
                touch(draft)
                rerun()
    if st.button("Add bilateral settings variant", key=str(path) + ":add"):
        settings["variants"].append(
            {**defaults_["variants"][0], "name": f"Variant {len(settings['variants']) + 1}"}
        )
        touch(draft)
        rerun()
    with st.expander("Bilateral CPU resources and per-representation ID overrides"):
        field(
            draft,
            path + ["batch_size"],
            "Query cells per distance block",
            128,
            help="Only this many query-to-population distances are held at once. Lower values reduce memory; ranks remain exact. Full retrieval still requires quadratic distance work.",
        )
        field(
            draft,
            path + ["workers"],
            "CPU threads for distances",
            2,
            help="Maximum numerical-library threads used in distance calculations. Keep this within the CPUs requested on Review & run.",
        )
        field(
            draft,
            path + ["identities"],
            "ID settings by representation (optional YAML)",
            {},
            help="Override the common ID settings for named inputs, for example {DINO: {id_namespace: nucleus, id_mapping: /path/mapping.tsv}}. Published references normally use cell IDs.",
        )


def stage_form(draft, index):
    from morphofeatures.workflow_ui import field, validation_expander

    stage = draft["document"]["stages"][index]
    path = ["document", "stages", index]
    earlier = [name for s in draft["document"]["stages"][:index] for name in embedding_names(s)]
    if stage["action"] == "bilateral":
        field(
            draft,
            path + ["embeddings"],
            "Named saved embedding files (YAML)",
            {},
            help="Map a readable name to each NPZ/NPY/TSV/CSV path, for example {DINO: /path/embeddings.npz, MAE: /path/mae.npz}. The preset includes published baselines; add your own in this box.",
        )
        field(
            draft,
            path + ["from_embeddings"],
            "Earlier outputs to include",
            [],
            options=earlier,
            multiple=True,
            help="Include named extraction, combination or context outputs from earlier in this workflow. All inputs are matched by cell ID before distances are evaluated.",
        )
        bilateral_fields(draft, path, stage)
        return
    st.caption(
        "Paper recipe: keep each cell's own vector and concatenate the mean of itself plus its physical neighbors. No cell-type labels or bilateral candidates enter feature construction."
    )
    field(
        draft,
        path + ["name"],
        "Name for this context embedding",
        "Spatial context",
        help="Later Analyze, Compare or bilateral stages can select this name as an earlier output.",
    )
    field(
        draft,
        path + ["from_embedding"],
        "Earlier embedding output (optional)",
        None,
        options=[None] + earlier,
        help="Select a named output from an earlier stage, or leave empty to supply a saved file below.",
    )
    if not stage.get("from_embedding") and not stage.get("from_extraction"):
        field(
            draft,
            path + ["embedding"],
            "Cell or nucleus embedding file",
            "",
            help="Full object vectors from MAE, DINO or another encoder, with exact object IDs. Do not supply UMAP coordinates.",
        )
    identity_fields(draft, path)
    field(
        draft,
        path + ["graph_mode"],
        "How to define physical neighbors",
        "adjacency_table",
        options=["adjacency_table", "segmentation", "radius", "knn"],
        help="Use a saved spatial edge table, touching instance labels, a physical centroid radius, or the nearest physical centroids. Touching cell segments match the published idea. Nuclei rarely touch, so radius/KNN are useful but different experiments.",
    )
    mode = stage.get("graph_mode", "adjacency_table")
    from morphofeatures.spatial_probe_controls import context_source_controls

    context_source_controls(draft, index)
    field(
        draft,
        path + ["graph_id_namespace"],
        "IDs in spatial graph / segmentation / positions",
        "cell",
        options=["cell", "nucleus"],
        help="Graph IDs are converted through the mapping above if they are nucleus IDs. The resulting context vectors use cell IDs; native-ID mapping is exported for inspection.",
    )
    if mode == "adjacency_table":
        field(
            draft,
            path + ["adjacency_table"],
            "Physical adjacency TSV/CSV",
            "",
            help="Columns source_id and target_id, one touching/spatial pair per row. The graph is undirected. A sibling graph_nodes.tsv from an earlier context run preserves isolated and ROI-boundary cells. Never use bilateral candidate pairs here.",
        )
    elif mode == "segmentation":
        field(
            draft,
            path + ["segmentation"],
            "Instance segmentation container or URL",
            "",
            help="Local H5/N5/Zarr/NPY or a supported remote N5/OME-Zarr array. Positive integer IDs identify instances; zero is background. Graph construction scans bounded blocks in the worker.",
        )
        field(
            draft,
            path + ["segmentation_key"],
            "Segmentation dataset / resolution key",
            "",
            help="Dataset inside the container, such as volumes/labels or s0. Leave empty for a standalone NPY. Select the actual instance-label resolution you want to test; coarser levels can change contacts.",
        )
        field(
            draft,
            path + ["segmentation_axes"],
            "Segmentation dimension order",
            "zyx",
            help="Stored dimension order, e.g. zyx or tczyx. Spatial blocks are read in Z,Y,X order.",
        )
        field(
            draft,
            path + ["segmentation_channel"],
            "Segmentation channel",
            0,
            help="Zero-based channel to read if the array has a c axis.",
        )
        field(
            draft,
            path + ["segmentation_timepoint"],
            "Segmentation time point",
            0,
            help="Zero-based time index if the array has a t axis.",
        )
        field(
            draft,
            path + ["roi"],
            "Graph scan ROI (optional)",
            None,
            help="Voxel bounds [[z_start,y_start,x_start],[z_stop,y_stop,x_stop]], with exclusive stops. Null scans the full segmentation only when the explicit voxel budget permits it.",
        )
        field(
            draft,
            path + ["connectivity"],
            "Touching-label connectivity",
            1,
            options=[1, 2, 3],
            help="1 means shared faces (6-neighborhood), 2 also includes edges (18), and 3 includes corners (26). Faces are the conservative default; this setting can change the biological neighborhood.",
        )
        field(
            draft,
            path + ["remote_options"],
            "Remote reader settings (optional YAML)",
            {},
            help="For OME-Zarr use {format: ome_zarr}; for remote N5 leave empty. Existing reader cache and timeout settings can also be supplied here.",
        )
    else:
        field(
            draft,
            path + ["positions"],
            "Object centroid TSV/CSV",
            "",
            help="A row per graph object with label_id and three physical coordinates. Include neighbors even if they lack embeddings so coverage can be reported honestly.",
        )
        field(
            draft,
            path + ["position_columns"],
            "Coordinate columns in Z,Y,X order",
            ["z", "y", "x"],
            help="Names of the three measured coordinate columns. Convert voxel indices using the correct spacing before supplying this table.",
        )
        field(
            draft,
            path + ["unit"],
            "Centroid coordinate unit",
            "um",
            options=["um", "nm"],
            help="Unit of all three coordinate columns. Nanometers are converted to micrometers before computing physical distances.",
        )
        if mode == "radius":
            field(
                draft,
                path + ["radius_um"],
                "Neighborhood radius (µm)",
                10.0,
                help="Include every centroid within this physical radius. A biological length scale is needed; 10 µm is only a starting value, not a published optimum.",
            )
        else:
            field(
                draft,
                path + ["neighbors"],
                "Nearest physical neighbors per object",
                8,
                help="Connect each centroid to its k closest others, then make edges undirected. The final degree can exceed k when another cell selects this cell.",
            )
    with validation_expander(
        draft, "Context recipe, missing neighbors and scan resources", expanded=True
    ):
        field(
            draft,
            path + ["include_self"],
            "Include focal cell in the neighborhood mean",
            True,
            help="Enabled matches the paper's self-plus-neighbor mean. The focal vector is also retained as the first half of the final concatenation.",
        )
        field(
            draft,
            path + ["combination"],
            "Combine focal and neighborhood vectors",
            "raw",
            options=["raw", "balanced"],
            help="Raw concatenates both full blocks, following the paper's construction. Balanced first L2-normalizes each block and applies square-root weights; this is an alternative experiment.",
        )
        field(
            draft,
            path + ["self_weight"],
            "Focal block weight for balanced combination",
            0.5,
            help="Only used for balanced combination. 0.5 gives equal squared-distance contribution to focal and context blocks; context receives 1 minus this value.",
        )
        field(
            draft,
            path + ["reduction"],
            "Reduce context dimensions",
            "none",
            options=["none", "agglomeration"],
            help="None keeps all 2D dimensions. Agglomeration groups correlated feature columns using Ward clustering and averages each group. The published stored context baseline has 200 dimensions, but its fitted reducer is not available here.",
        )
        field(
            draft,
            path + ["n_features"],
            "Dimensions after feature agglomeration",
            200,
            help="Used only when agglomeration is enabled; cannot exceed twice the input dimension. This fits on the whole cohort for exploration, so use unreduced vectors for leakage-aware classifier evaluation.",
        )
        field(
            draft,
            path + ["missing_neighbors"],
            "When a graph neighbor lacks an embedding",
            "error",
            options=["error", "available"],
            help="Error stops and names the affected cell. Available averages only embedded neighbors, subject to the minimum fraction below, and reports the missing coverage.",
        )
        field(
            draft,
            path + ["minimum_coverage"],
            "Minimum fraction of neighbors with embeddings",
            1.0,
            help="1.0 requires all graph neighbors. Lower this explicitly when using available-neighbor averaging; e.g. 0.8 retains cells with at least 80% represented neighbors.",
        )
        field(
            draft,
            path + ["isolated"],
            "Cells with no available neighbors",
            "self",
            options=["self", "skip", "error"],
            help="Self keeps the cell with a self-only mean if Include focal cell is enabled. Skip records an exclusion. Error stops to flag a potentially unsuitable graph, such as separated nuclei in contact mode.",
        )
        field(
            draft,
            path + ["roi_boundary"],
            "Cells touching a segmentation / ROI edge",
            "skip",
            options=["skip", "allow"],
            help="Skip avoids silently using truncated physical neighborhoods. Allow keeps these cells with a recorded boundary flag. Imported graphs without boundary metadata cannot diagnose truncation.",
        )
        field(
            draft,
            path + ["block_shape"],
            "Segmentation scan block shape (Z,Y,X)",
            [64, 64, 64],
            help="Blocks have a one-voxel halo so contacts across block seams are retained. This limits memory, not the final graph's scientific definition.",
        )
        field(
            draft,
            path + ["max_scan_voxels"],
            "Maximum segmentation voxels to scan",
            100_000_000,
            help="Total work budget before scanning a volume/ROI. Increase deliberately for large jobs and request sufficient worker time; this is separate from per-block RAM.",
        )
        field(
            draft,
            path + ["max_edges"],
            "Maximum unique graph edges",
            5_000_000,
            help="Stops graph construction if too many contacts accumulate. Reduce the ROI/radius or raise this limit with adequate memory.",
        )
        field(
            draft,
            path + ["graph_cache"],
            "Reusable contact-graph cache directory (optional)",
            None,
            optional=True,
            help="Reuse completed segmentation graphs across MAE/DINO runs with identical source, resolution, ROI and connectivity. Partial scans are never cached.",
        )
        field(
            draft,
            path + ["source_version"],
            "Source revision for cache identity",
            "",
            help="Use a dataset revision or version tag. Change it after editing remote data or container chunks in place, because directory/URL metadata alone cannot detect every chunk edit.",
        )


def published_fields(draft, index):
    from morphofeatures.analysis.published import references
    from morphofeatures.workflow_ui import field, rerun, touch

    stage = draft["document"]["stages"][index]
    path = ["document", "stages", index]
    catalog = references()
    field(
        draft,
        path + ["published_reference"],
        "Published feature reference",
        "MorphoContextFeatures (200 dimensions)",
        options=list(catalog),
        help="Loads the bundled original feature vectors. MorphoFeatures includes cell/nucleus shape and texture; it is not a nuclear-texture-only control for MAE or DINO.",
    )
    if st.button("Use this published feature file", key=str(path) + ":published"):
        stage["embedding"] = catalog[stage["published_reference"]]["embedding"]
        stage["use_published_layout"] = "layout" in catalog[stage["published_reference"]]
        stage["umap"] = stage["use_published_layout"]
        touch(draft)
        rerun()
    field(
        draft,
        path + ["use_published_layout"],
        "Preserve published UMAP coordinates and clusters",
        True,
        help="Use the saved historical layout for the matching reference file. Turn off to run your current projection/clustering settings. Classification always uses the high-dimensional feature vectors.",
    )
    if stage.get("use_published_layout"):
        st.info(
            "The saved published UMAP and clusters will be preserved. Projection and clustering controls below apply only when this option is off. Classification and bilateral probes remain available."
        )


def _download(path, label):
    if path.is_file() and path.stat().st_size <= 200 * 1024**2:
        st.download_button(label, path.read_bytes(), path.name, key="spatial-download:" + str(path))
    elif path.is_file():
        st.caption(f"Large artifact: download from the worker at {path}")


def probe_results(config, metadata, path):
    from morphofeatures.analysis_ui import inspect_results

    path = Path(path)
    root = path.parent
    st.info(metadata["interpretation"])
    for warning in metadata.get("warnings", []):
        st.warning(warning)
    if metadata["schema"] == "morphofeatures.context.v1":
        st.write(
            f"{metadata['matched_objects']:,} context vectors · {metadata['input_dimensions']} input dimensions → {metadata['full_dimensions']} concatenated → {metadata['dimensions']} saved dimensions"
        )
        coverage = pd.read_csv(root / "coverage.tsv", sep="\t", keep_default_na=False)
        st.dataframe(coverage, hide_index=True)
        for name in (
            "embeddings.npz",
            "full_context.npz",
            "self.npz",
            "neighborhood_mean.npz",
            "coverage.tsv",
            "adjacency.tsv",
            "id_mapping.tsv",
        ):
            _download(root / name, "Download " + name)
        st.caption(
            "Context outputs use cell IDs. Use a cell-ID inspection configuration or the exported mapping; the original nucleus IDs remain in id_mapping.tsv."
        )
        inspect_results(
            config,
            {"Spatial context": root / "embeddings.npz"},
            metadata,
            pd.DataFrame({"label_id": pd.Series(dtype="int64")}),
            path,
        )
    else:
        st.write(
            f"Shared search population: {metadata['matched_objects']:,} cells. Rank convention: {metadata['rank_convention']}."
        )
        if metadata.get("comparison_protocol"):
            st.caption(metadata["comparison_protocol"])
        from morphofeatures.workflow_ui import navigate, use_draft
        from morphofeatures.workspace_state import new_draft

        if st.button("Export bilateral pairs in 3D", key=str(path) + ":export"):
            stage = {
                "action": "export_bilateral",
                "analysis": str(path),
                "mode": "platybrowser",
                "make_mobie": True,
                "platybrowser": {"segmentation_source": "cells", "object_ids": "segmentation"},
                "max_pairs": 20,
            }
            draft = new_draft(
                "bilateral",
                document={
                    "stages": [stage],
                    "slurm": {"cpus": 4, "gpus": 0, "memory": "8G", "time": "01:00:00"},
                },
                source="Bilateral result: " + str(path),
            )
            draft["name"] = "Export bilateral pairs in 3D"
            use_draft(config, draft, "Analyze")
            navigate("Workflow")
        st.dataframe(pd.read_csv(root / "summary.tsv", sep="\t"), hide_index=True)
        st.image(str(root / "bilateral.png"))
        for name in ("summary.tsv", "bilateral.svg", "bilateral.png"):
            _download(root / name, "Download " + name)
        name = st.selectbox(
            "Retrieval representation", list(metadata["representations"]), key=str(path) + ":rep"
        )
        representation = metadata["representations"][name]
        variant_index = st.selectbox(
            "Retrieval settings variant",
            range(len(representation["variants"])),
            format_func=lambda i: representation["variants"][i]["settings"]["name"],
            key=str(path) + ":variant",
        )
        variant = representation["variants"][variant_index]
        ranks_path = root / representation["directory"] / variant["ranks"]
        from morphofeatures.analysis.bilateral_comparison import read_ranks

        ranks = read_ranks(ranks_path)
        st.dataframe(ranks, hide_index=True)
        _download(ranks_path, "Download per-cell retrieval ranks")
        valid = ranks.dropna(subset=["best_partner_id"])
        if len(valid):
            selected = st.selectbox(
                "Cell / candidate pair to inspect", valid.label_id.tolist(), key=str(path) + ":pair"
            )
            row = valid.set_index("label_id").loc[selected]
            bridge = pd.read_csv(
                root / representation["directory"] / "id_mapping.tsv", sep="\t", dtype="int64"
            ).set_index("label_id")
            native_ids = [
                int(bridge.loc[i, "native_id"]) for i in (selected, int(row.best_partner_id))
            ]
            source = Path(representation["source"]["path"])
            if source.is_file():
                if st.button(
                    "Inspect this cell and its best candidate", key=str(path) + ":inspect"
                ):
                    st.session_state["inspection:" + str(path) + ":mode"] = "Chosen objects"
                    st.session_state["comparison:" + str(path) + ":" + name] = native_ids
                st.caption(
                    "Image / mesh inspection below uses the original embedding IDs, converted from the evaluated cell IDs through the exported mapping."
                )
                inspect_results(
                    config,
                    {name: source},
                    metadata,
                    pd.DataFrame({"label_id": pd.Series(dtype="int64")}),
                    path,
                )
            else:
                st.caption(
                    "Original embedding file is unavailable here. Use id_mapping.tsv to convert evaluated cell IDs before inspecting native objects."
                )
    with st.expander("Scientific settings, sources and exclusions"):
        st.json(metadata)
    _download(root / "export.zip", "Download complete spatial-probe results")


def nested_bilateral_results(config, metadata, path):
    if metadata.get("bilateral"):
        report = Path(path).parent / metadata["bilateral"]
        with st.expander("Bilateral-partner retrieval results"):
            probe_results(config, json.loads(report.read_text()), report)


def bilateral_export_results(config, metadata, path):
    path = Path(path)
    st.info(metadata["interpretation"])
    st.dataframe(pd.DataFrame(metadata["groups"]), hide_index=True)
    if metadata.get("excluded_pairs") or metadata.get("pairs_missing_mask_endpoint"):
        st.warning(
            f"Excluded pairs: {metadata.get('excluded_pairs', 0)}; mask layers missing a member in the ROI: {metadata.get('pairs_missing_mask_endpoint', 0)}. See the coverage tables in the export."
        )
    for entry in metadata["groups"]:
        st.write(entry["name"])
        for name in ("pairs.tsv", "pairs.npz", "pairs.vtk"):
            _download(path.parent / entry["directory"] / name, f"Download {entry['name']} · {name}")
    for name in ("pair_masks.h5", "pair_mask_layers.tsv", "excluded_pairs.tsv", "export.zip"):
        _download(path.parent / name, "Download " + name)
    if metadata.get("mobie"):
        from morphofeatures.analysis_ui import mobie_export_results

        report = path.parent / metadata["mobie"]
        mobie_export_results(json.loads(report.read_text()), report)
    st.caption(
        "NPZ endpoints/vectors use physical Z/Y/X micrometers; VTK lines use X/Y/Z micrometers for ParaView. HDF5 roles uses pair,Z,Y,X with physical spacing/origin attributes. See Analysis and MoBIE documentation for napari loading."
    )
