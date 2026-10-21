"""Guided physical-DINO experiments using the existing Workflow and Results pages."""

import json
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from morphofeatures.config import repository_root
from morphofeatures.configuration_editor import load_document
from morphofeatures.dino_experiments import experiment_defaults, resolve_paths
from morphofeatures.workspace_jobs import embedding_names
from morphofeatures.workspace_navigation import selector_key


def source_editor(draft, path, source, *, key):
    from morphofeatures.workflow_ui import field, validation_expander

    field(
        draft,
        path + ["source"],
        "Image source",
        "local",
        options=["local", "platybrowser", "ome_zarr"],
    )
    kind = source.get("source", "local")
    if kind == "platybrowser":
        from morphofeatures.analysis.platybrowser import DEFAULTS

        options = path + ["platybrowser"]
        for name, label, value in (
            ("project_url", "PlatyBrowser project URL", DEFAULTS["project_url"]),
            ("revision", "Pinned project revision", DEFAULTS["revision"]),
            ("dataset", "Published dataset", DEFAULTS["dataset"]),
            ("raw_source", "Published raw source", "raw"),
            ("segmentation_source", "Published segmentation source", "nuclei"),
            ("raw_level", "Raw resolution level", 1),
            ("segmentation_level", "Segmentation resolution level", 0),
        ):
            field(draft, options + [name], label, value)
        field(
            draft,
            options + ["object_ids"],
            "Object IDs used by annotations",
            "segmentation",
            options=["segmentation", "cell"],
        )
        field(
            draft,
            options + ["cache_directory"],
            "Remote chunk cache",
            str(repository_root() / "outputs/remote_cache/platybrowser"),
        )
    elif kind == "ome_zarr":
        for name in ("raw", "segmentation"):
            options = path + ["ome_zarr", name]
            field(draft, options + ["url"], f"{name.capitalize()} OME-Zarr URL", "")
            value = source.get("ome_zarr", {}).get(name, {})
            url = value.get("url", "")
            if st.button(
                f"Find {name} resolutions",
                key=key + name,
                help="Read the layer's metadata to list available image sizes and voxel spacings. This does not download the full image or start embedding.",
            ):
                try:
                    from morphofeatures.data.ome_zarr import discover_ome_zarr

                    st.session_state["dino-levels:" + url] = discover_ome_zarr(
                        url, source.get("remote_options"), image_index=value.get("image_index", 0)
                    )["levels"]
                except Exception as error:
                    st.error(f"Prepare data → {name} OME-Zarr URL: {error}")
            levels = st.session_state.get("dino-levels:" + url)
            if levels:
                st.dataframe(pd.DataFrame(levels))
            field(
                draft,
                options + ["level"],
                f"{name.capitalize()} resolution",
                0,
                options=list(range(len(levels))) if levels else None,
            )
            field(draft, options + ["channel"], f"{name.capitalize()} channel", 0)
            field(draft, options + ["timepoint"], f"{name.capitalize()} timepoint", 0)
        field(
            draft,
            path + ["remote_options", "cache_directory"],
            "Remote chunk cache",
            str(repository_root() / "outputs/remote_cache/ome-zarr"),
        )
    else:
        for name in ("raw", "segmentation"):
            field(draft, path + [name], f"{name.capitalize()} container / array", "")
            field(
                draft,
                path + [name + "_key"],
                f"{name.capitalize()} dataset key",
                None,
            )
            field(draft, path + [name + "_axes"], f"{name.capitalize()} stored axes", "zyx")
            for dimension, parameter in (("c", "channel"), ("t", "timepoint")):
                if dimension in source.get(name + "_axes", "zyx"):
                    field(
                        draft,
                        path + [name + "_" + parameter],
                        f"{name.capitalize()} {parameter}",
                        0,
                    )
            field(
                draft,
                path + [name + "_spacing_zyx"],
                f"{name.capitalize()} voxel spacing (Z, Y, X)",
                [1.0, 1.0, 1.0],
            )
            field(
                draft,
                path + [name + "_origin_zyx"],
                f"{name.capitalize()} origin (Z, Y, X)",
                [0.0, 0.0, 0.0],
            )
        field(draft, path + ["unit"], "Physical coordinate unit", "um", options=["um", "nm"])
    from morphofeatures.object_selection_ui import object_selection_fields

    object_selection_fields(draft, path, physical=True, limit_path=path[:3] + ["max_objects"])
    with validation_expander(draft, "Object selection and source bounds"):
        field(
            draft,
            path + ["roi"],
            "Segmentation ROI [[start Z,Y,X], [stop Z,Y,X]]",
            None,
        )
        field(
            draft,
            path + ["object_index"],
            "Existing object bounds table",
            None,
        )
        field(
            draft,
            path + ["id_mapping"],
            "Cell-to-segmentation ID mapping",
            None,
        )
        field(draft, path + ["object_ids"], "Only these object IDs", None)
        field(
            draft,
            path + ["data_version"],
            "Source data version",
            "1",
        )
    object_eligibility_controls(draft, path, source, key=key)


def object_eligibility_controls(draft, path, source, *, key):
    from morphofeatures.data.object_eligibility import assess_objects, eligibility_message
    from morphofeatures.workflow_ui import field, validation_expander

    with validation_expander(draft, "Object eligibility, boundaries and memory", expanded=True):
        st.caption(
            "These checks run before fine/coarse sampling. The memory limit applies to the entire object's segmentation box, including its background—not to the fine tile width or GPU batch size."
        )
        field(
            draft,
            path + ["max_object_voxels"],
            "Maximum object-box voxels (including halo)",
            256**3,
        )
        field(
            draft,
            path + ["boundary_policy"],
            "Boundary objects",
            "skip",
            options=["skip", "error", "allow_partial"],
            format_func={
                "skip": "Skip objects touching a volume or ROI edge",
                "error": "Stop run if an object touches an edge",
                "allow_partial": "Use the available part (mark as partial)",
            }.get,
        )
        field(
            draft,
            path + ["oversized_policy"],
            "Objects above the memory limit",
            "skip",
            options=["skip", "error"],
            format_func={
                "skip": "Skip and report the required size",
                "error": "Stop run and report the required size",
            }.get,
        )
        field(
            draft,
            path + ["on_object_error"],
            "Other object-reading or encoding errors",
            "skip",
            options=["skip", "error"],
            format_func={
                "skip": "Skip the object and record the reason",
                "error": "Stop run on the first object error",
            }.get,
        )
        budget = source.get("max_object_voxels", 256**3)
        if isinstance(budget, int) and budget > 0:
            st.caption(
                f"Configured limit: {budget:,} voxels. A binary mask of that size alone is {budget / 1024**3:.2f} GiB; source labels, foreground coordinates and sampling arrays require additional RAM. Raising this limit does not increase the SLURM memory request."
            )
        if source.get("boundary_policy") == "allow_partial":
            st.warning(
                "Fine and coarse features may describe only part of a nucleus. Reports and previews mark these objects as partial. This can affect biological comparisons; increasing an ROI cannot recover tissue absent from the original volume."
            )
        st.caption(
            "To check an earlier failed run without rescanning its volume, select its matching objects.tsv under Object selection and source bounds → Existing object bounds table. Use a table indexed on this exact segmentation grid."
        )
        stage = draft["document"]["stages"][path[2]]
        signature = json.dumps(
            [source, stage.get("max_objects", 100), stage.get("seed", 42)], sort_keys=True
        )
        if st.button(
            "Check object eligibility (no DINO)",
            key=key + ":eligibility",
            help="Check ROI/volume-edge contact and required mask-box sizes from existing object bounds. Without a bounds table, use an ROI of at most 128³ segmentation voxels for this interactive check. No DINO weights or raw image pixels are loaded; extraction can still fail on contrast, stale bounds or sampling cutoffs.",
        ):
            st.session_state.pop(key + ":eligibility-result", None)
            try:
                from morphofeatures.data.dino_views import index_source, validate_source

                resolved = validate_source(resolve_paths(source, repository_root()))
                if not resolved.get("object_index"):
                    roi = resolved.get("roi")
                    if (
                        roi is None
                        or np.prod(np.asarray(roi)[1] - np.asarray(roi)[0], dtype=object) > 128**3
                    ):
                        raise ValueError(
                            "Choose an existing objects.tsv bounds table, or an ROI of at most 128³ segmentation voxels. Larger scans run in the worker; a failed run's input/objects.tsv or cache objects.tsv can be reused if the segmentation grid is unchanged."
                        )
                resolved.update(
                    max_objects=stage.get("max_objects", 100), seed=stage.get("seed", 42)
                )
                with tempfile.TemporaryDirectory(prefix="dino-eligibility-") as folder:
                    objects = index_source(resolved, folder)
                    table, summary = assess_objects(resolved, objects)
                st.session_state[key + ":eligibility-result"] = (signature, table, summary)
            except (ValueError, OSError, KeyError) as error:
                st.error(
                    f"Prepare data → Object eligibility, boundaries and memory → Check object eligibility: {error}"
                )
        result = st.session_state.get(key + ":eligibility-result")
        if result and result[0] == signature:
            _, table, summary = result
            message = eligibility_message(summary)
            if not summary["eligible_objects"] or summary["error_objects"]:
                st.error(message)
            elif summary["skip_objects"] or summary["partial_objects"]:
                st.warning(message)
            else:
                st.success(message)
            st.caption(summary["interpretation"])
            st.dataframe(table, hide_index=True)
            st.download_button(
                "Download object eligibility table",
                table.to_csv(sep="\t", index=False),
                file_name="object_eligibility.tsv",
                mime="text/tab-separated-values",
                key=key + ":eligibility-download",
            )


def _preview(source, views, key, stride=14):
    from morphofeatures.data.dino_views import PhysicalObjects, index_source, validate_source

    chosen_id = st.number_input(
        "Object ID for input preview (0 selects first)",
        min_value=0,
        value=0,
        key=key + ":id",
        help="Enter an object ID from this dataset, or 0 to preview the first eligible object. This only chooses the preview; it does not change the experiment's object selection. The object must be available within the selected region or bounds table.",
    )
    if st.button(
        "Preview actual DINO inputs",
        key=key + ":preview",
        help="Read a few fine and coarse views of one object and display their contrast, mask and model patch grid. No DINO weights are loaded. For an interactive preview, supply an object-bounds table or a region of at most 128 × 128 × 128 voxels.",
    ):
        try:
            resolved = validate_source(resolve_paths(source, repository_root()))
            if not resolved.get("object_index"):
                roi = resolved.get("roi")
                if roi is None or np.prod(np.array(roi)[1] - np.array(roi)[0]) > 128**3:
                    raise ValueError(
                        "For this interactive preview, supply an object index or an ROI of at most 128³ voxels. Larger indexing runs in the worker."
                    )
            resolved.update(max_objects=1)
            if chosen_id:
                resolved["object_ids"] = [int(chosen_id)]
            with tempfile.TemporaryDirectory(prefix="dino-preview-") as temporary:
                objects = index_source(resolved, temporary)
                with PhysicalObjects(resolved, objects, views) as reader:
                    if objects.empty:
                        raise ValueError(
                            "No objects were found for this ID/ROI. Check the object ID, ID mapping and segmentation bounds."
                        )
                    obj = reader.prepare(objects.iloc[0].to_dict())
                    coverage_scope = obj.get("coverage_scope", "complete_object")
                    examples = []
                    summaries = []
                    for mode in ("fixed_tiles", "whole_label"):
                        before = len(reader.excluded_views)
                        for i, sample in enumerate(reader.planes(obj, mode)):
                            examples.append(sample)
                            if i >= 2:
                                break
                        summaries.append(
                            {
                                "mode": mode,
                                **reader.sampling_summary,
                                "excluded_preview_views": reader.excluded_views[before:],
                            }
                        )
            st.session_state[key + ":images"] = (
                json.dumps([source, views], sort_keys=True),
                examples,
                {
                    "coverage_scope": coverage_scope,
                    "sampling": summaries,
                    "raw_native_pixels_zyx": (
                        np.asarray(views.get("tile_size_um", 1.0))
                        / np.asarray(resolved["raw_spacing_zyx"])
                    ).tolist(),
                    "mask_native_pixels_zyx": (
                        np.asarray(views.get("tile_size_um", 1.0))
                        / np.asarray(resolved["segmentation_spacing_zyx"])
                    ).tolist(),
                },
            )
        except Exception as error:
            st.error(f"Prepare data → DINO input preview: {error}")
    saved = st.session_state.get(key + ":images")
    if saved and saved[0] == json.dumps([source, views], sort_keys=True):
        if len(saved) > 2:
            if saved[2].get("coverage_scope") == "partial_object":
                st.warning(
                    "Preview uses a partial object: fine and coarse images cover only the part available inside the source volume / ROI."
                )
            st.caption(
                "Tile width in native Z/Y/X pixels: raw "
                + str(saved[2]["raw_native_pixels_zyx"])
                + "; segmentation "
                + str(saved[2]["mask_native_pixels_zyx"])
            )
            for summary in saved[2].get("sampling", []):
                st.caption(
                    f"{summary['mode']}: {summary['eligible_tiles']} eligible tiles; "
                    f"{summary['selected_tiles']} selected. Preview displays up to three valid images, not the full experiment."
                )
                if not summary["eligible_tiles"]:
                    st.warning(
                        "Prepare data → Fine texture: no tiles pass the current 3D cutoff. "
                        "Lower the cutoff, reduce tile width, or change the grid placement."
                    )
                reasons = list(
                    dict.fromkeys(v["reason"] for v in summary.get("excluded_preview_views", []))
                )
                if reasons:
                    with st.expander("Why preview images were excluded"):
                        for reason in reasons:
                            st.write(reason)
        for image, mask, info in saved[1]:
            left, right = st.columns(2)
            left.image(
                image,
                clamp=True,
                caption=f"{info['mode']} · axis {info['axis']} · {info['pixel_size_um']:.4g} µm/pixel",
            )
            overlay = np.repeat(image[..., None], 3, axis=2)
            overlay[mask, 1] = np.maximum(overlay[mask, 1], 0.65)
            overlay[stride - 1 :: stride, :, :] = [0.1, 0.7, 0.9]
            overlay[:, stride - 1 :: stride, :] = [0.1, 0.7, 0.9]
            right.image(
                overlay,
                clamp=True,
                caption=f"Foreground {100 * info['foreground_fraction']:.1f}% · mask used for weighted tokens",
            )
        st.caption(
            "Shown images are normalized DINO inputs before grayscale replication/ImageNet channel scaling. Slice sampling is not exhaustive voxel coverage."
        )


def sampling_controls(draft, root):
    from morphofeatures.workflow_ui import field, validation_expander

    with validation_expander(
        draft, "Fine texture: patch size, nucleus content and placement", expanded=True
    ):
        st.caption(
            "A fine tile is a small 3D box. DINO sees selected 2D slices through that box. "
            "These settings are shared by Prepare data previews and the experiment run."
        )
        field(draft, root + ["tile_sizes_um"], "Fine tile widths to compare (µm)", [1.0])
        field(
            draft,
            root + ["views", "min_foreground_fraction"],
            "Minimum nucleus content per 3D fine tile (0–1)",
            0.01,
        )
        field(
            draft,
            root + ["views", "min_slice_foreground_fraction"],
            "Minimum nucleus content per 2D fine image (0–1)",
            0.0,
        )
        st.caption(
            "Enter 0.25 for 25%, 0.50 for 50%, or 0.75 for 75%. "
            "Both cutoffs apply to the selected nucleus, not neighboring nuclei. "
            "Higher cutoffs favor interior texture and can exclude nuclear edges or small nuclei."
        )
        choices = {
            "bbox_min": "Start at bounding-box corner (existing behavior)",
            "bbox_center": "Center a tile on the bounding-box midpoint",
            "centroid": "Center a tile on the nucleus foreground centroid",
        }
        field(
            draft,
            root + ["views", "tile_anchor"],
            "Position of the fine tile grid",
            "bbox_min",
            options=list(choices),
            format_func=choices.get,
        )
        field(draft, root + ["views", "max_tiles"], "Maximum fine tiles per nucleus", 64)
        field(draft, root + ["views", "exhaustive"], "Process all eligible tiles", False)
        field(
            draft,
            root + ["views", "max_candidate_tiles"],
            "Maximum candidate tiles per object (indexing safeguard)",
            250000,
        )
        if draft["document"]["stages"][root[2]].get("views", {}).get("exhaustive"):
            st.warning(
                "Process all eligible tiles is enabled: Maximum fine tiles per nucleus is ignored. "
                "The foreground cutoffs, candidate-grid safeguard and object memory limit still apply. "
                "For a bounded pilot, turn this off and choose a tile cap."
            )


def view_preparation_fields(draft, root):
    from morphofeatures.workflow_ui import field, validation_expander

    with validation_expander(draft, "Slice sampling and DINO input images", expanded=False):
        for name, label, default in (
            ("axes", "Slice axes (0=Z, 1=Y, 2=X)", [0, 1, 2]),
            ("fine_fractions", "Fine slice positions within foreground", [0.5]),
            (
                "coarse_fractions",
                "Coarse slice positions across the nucleus",
                [0.1, 0.3, 0.5, 0.7, 0.9],
            ),
            ("size", "DINO input image side (pixels)", 224),
            ("mask", "Mask everything outside the selected nucleus", True),
            ("percentiles", "Shared per-object intensity percentiles", [1, 99]),
        ):
            field(draft, root + ["views", name], label, default)


def experiment_form(draft, index, *, page):
    from morphofeatures.workflow_ui import (
        _replace_document,
        classifier_input_fields,
        field,
        projection_settings_fields,
        rerun,
        validation_expander,
    )

    stage = draft["document"]["stages"][index]
    root = ["document", "stages", index]
    if page == "Prepare data":
        st.caption(
            "Choose raw images and instance labels first, then configure the object selection, patches and slices below. Physical grids may differ when their coordinates align. Analyze contains backbone, feature-combination and evaluation settings."
        )
        for i, dataset in enumerate(stage["datasets"]):
            path = root + ["datasets", i]
            with validation_expander(
                draft, f"Dataset {i + 1}: {dataset.get('name', '')}", expanded=True
            ):
                field(draft, path + ["name"], "Dataset name", f"Dataset {i + 1}")
                load_path = st.text_input(
                    "Load source / inspection YAML",
                    key=f"dino-source-load:{index}:{i}",
                    help="Path to an existing source or inspection YAML containing image locations and spatial metadata. Load source settings copies these into this dataset's form; it does not load model weights or submit a job.",
                )
                if st.button("Load source settings", key=f"dino-source-button:{index}:{i}"):
                    try:
                        loaded = load_document(Path(load_path).expanduser())
                        from morphofeatures.analysis.mobie_export import export_source_defaults

                        source = (
                            loaded
                            if "raw" in loaded or "ome_zarr" in loaded
                            else export_source_defaults(loaded)
                        )
                        if "platybrowser" in source:
                            source["source"] = "platybrowser"
                        document = deepcopy(draft["document"])
                        document["stages"][index]["datasets"][i]["source"] = source
                        _replace_document(draft, document)
                        rerun()
                    except Exception as error:
                        st.error(f"Prepare data → Load source settings: {error}")
                source_editor(
                    draft,
                    path + ["source"],
                    dataset.setdefault("source", {}),
                    key=f"dino-source:{index}:{i}",
                )
                if len(stage["datasets"]) > 1 and st.button(
                    "Remove dataset", key=f"dino-remove:{index}:{i}"
                ):
                    document = deepcopy(draft["document"])
                    document["stages"][index]["datasets"].pop(i)
                    _replace_document(draft, document)
                    rerun()
        if st.button("Add independent dataset", key=f"dino-add:{index}"):
            document = deepcopy(draft["document"])
            dataset = deepcopy(experiment_defaults()["datasets"][0])
            dataset["name"] = f"Dataset {len(stage['datasets']) + 1}"
            document["stages"][index]["datasets"].append(dataset)
            _replace_document(draft, document)
            rerun()
        field(
            draft,
            root + ["max_objects"],
            "Objects per dataset (0 runs the full selection)",
            100,
        )
        sampling_controls(draft, root)
        view_preparation_fields(draft, root)
        for i, dataset in enumerate(stage["datasets"]):
            with st.expander(f"Preview DINO inputs: {dataset['name']}", expanded=True):
                scales = stage.get("tile_sizes_um", [1.0]) or [1.0]
                preview_scale = st.selectbox(
                    "Preview tile width (µm)",
                    scales,
                    key=f"preview-scale:{draft['id']}:{index}:{i}",
                    help="Inspect one of the tile widths configured above. This selects a preview, not another experiment setting. The preview reports how many native image pixels fit inside this width.",
                )
                preview_family = st.selectbox(
                    "Preview token grid",
                    ["DINOv2 (14 pixels)", "DINOv3 (16 pixels)"],
                    key=f"preview-token:{draft['id']}:{index}:{i}",
                    help="Select the overlay to inspect: 14-pixel model patches for DINOv2, 16 for DINOv3. This changes only the preview; choose the actual backbones on Analyze.",
                )
                _preview(
                    dataset["source"],
                    {**stage["views"], "tile_size_um": preview_scale},
                    f"dino-preview:{draft['id']}:{index}:{i}",
                    stride=14 if preview_family.startswith("DINOv2") else 16,
                )
        return
    st.caption(
        "Compare small-region texture (fine), whole-nucleus texture (coarse), and their combined vectors. "
        "For each scale, CLS summarizes the full image; foreground-weighted features emphasize image "
        "patches occupied by the nucleus. Both are saved so you can compare their biological usefulness."
    )
    with st.expander("Prepared data and view settings", expanded=True):
        st.write("Datasets: " + ", ".join(d["name"] for d in stage["datasets"]))
        from morphofeatures.data.object_selection import resolve_selection

        for dataset in stage["datasets"]:
            try:
                options = resolve_paths(dataset.get("source", {}), repository_root()).get("object_selection")
                _, selection = resolve_selection(options)
                if selection:
                    cap = stage.get("max_objects", 100)
                    planned = min(selection["requested"], cap) if cap else selection["requested"]
                    st.caption(f"{dataset['name']}: {selection['requested']:,} requested target IDs; at most {planned:,} objects after the cap. Availability in the source/ROI and quality checks can reduce this further.")
            except (ValueError, OSError) as error:
                st.error(f"Prepare data → {dataset['name']} → Target objects: {error}")
        st.caption(
            f"Objects per dataset: {stage.get('max_objects', 100) or 'full selection'} · Fine widths: {stage.get('tile_sizes_um', [1.0])} µm · Fine slice positions: {stage['views'].get('fine_fractions', [0.5])} · Coarse slice positions: {stage['views'].get('coarse_fractions', [0.1, 0.3, 0.5, 0.7, 0.9])}"
        )
        st.caption(
            "Sources, object limits, patch coverage and input-image settings are edited on Prepare data. These same settings feed the preview and the worker."
        )
        if st.button("Edit data and DINO views", key=f"dino-prepare:{draft['id']}:{index}"):
            st.session_state["workflow_step"] = "Prepare data"
            rerun()
    from morphofeatures.dino_defaults import backbone_defaults, backbone_fields

    for i, _backbone in enumerate(stage["backbones"]):
        with validation_expander(draft, f"DINO backbone {i + 1}", expanded=True):
            path = root + ["backbones", i]
            backbone_fields(draft, path)
            if len(stage["backbones"]) > 1 and st.button(
                "Remove backbone", key=f"dino-backbone-remove:{index}:{i}"
            ):
                document = deepcopy(draft["document"])
                document["stages"][index]["backbones"].pop(i)
                _replace_document(draft, document)
                rerun()
    if st.button("Add another DINO backbone", key=f"dino-backbone-add:{index}"):
        document = deepcopy(draft["document"])
        document["stages"][index]["backbones"].append(
            backbone_defaults(st.session_state.get("workspace_dino_defaults"), "dinov3")
        )
        _replace_document(draft, document)
        rerun()
    with validation_expander(
        draft, "Fine/coarse feature combination and model execution", expanded=True
    ):
        from morphofeatures.analysis_variants_ui import scaling_fields
        from morphofeatures.dino_experiments import (
            FUSION_COMBINATIONS,
            FUSION_METHODS,
            fusion_recipes,
        )

        st.markdown("**1. Build the combined feature vectors**")
        field(
            draft, root + ["combination_methods"], "Feature combination methods", ["balanced"],
            options=list(FUSION_METHODS), multiple=True, format_func=FUSION_METHODS.get,
            help="Choose raw, balanced, or both in the same run. Raw joins original feature values and ignores block weights. Balanced normalizes each block to unit length, multiplies it by the square root of its weight, then joins the blocks. Both preserve all feature dimensions and reuse the same DINO extraction.",
        )
        field(
            draft, root + ["additional_combinations"], "Additional feature combinations", [],
            options=list(FUSION_COMBINATIONS), multiple=True, format_func=lambda k: FUSION_COMBINATIONS.get(k, (str(k), []))[0],
            help="The primary combination is fine foreground + coarse CLS. Add other pairs or all four blocks here; each uses the combination methods selected above. All four gives 3072 dimensions with ViT-B. Original single-block outputs remain available. More combinations add analysis time, not DINO inference.",
        )
        st.caption("Fine/coarse describes the image field of view; CLS/foreground describes token pooling. Raw concatenation preserves the extracted feature values, after the configured DINO image/model processing.")
        path = root + ["views"]
        field(draft, path + ["batch_size"], "Images per model batch", 16)
        field(
            draft,
            root + ["fusion_weights"],
            "Fine / coarse distance weights",
            [0.5, 0.5],
        )
        field(
            draft,
            root + ["extra_pairings"],
            "Additional fine/coarse pairings (advanced)",
            [],
        )
        st.markdown("**2. Choose how those vectors enter analysis**")
        scaling_fields(draft, root, experiment=True)
        try:
            recipes = fusion_recipes(stage)
            st.dataframe(pd.DataFrame([
                {"Combined representation": r["name"], "Blocks": ", ".join(r["blocks"]),
                 "Method": FUSION_METHODS[r["combination"]],
                 "Weights": "Ignored (raw)" if r["combination"] == "raw" else str(r["weights"])}
                for r in recipes
            ]), hide_index=True)
        except ValueError as error:
            st.warning(str(error))
        if stage.get("transfer", {}).get("enabled"):
            st.caption("Scaling comparisons apply to within-dataset evaluations and joint maps. Label transfer retains its own reference preprocessing, including its initial whole-vector L2; it is run once per representation.")
        field(
            draft,
            root + ["views", "aggregation"],
            "Aggregate views per object",
            "mean",
            options=["mean", "max"],
        )
        field(
            draft, root + ["device"], "DINO compute device", "auto", options=["auto", "cpu", "cuda"]
        )
        field(
            draft,
            root + ["cache"],
            "Reusable embedding/view cache",
            str(repository_root() / "outputs/embedding_cache"),
        )
    from morphofeatures.dino_runtime_ui import runtime_fields

    runtime_fields(draft, root)
    projection_settings_fields(
        draft,
        root,
        umap_label="Generate UMAP as well as PCA",
        clusters_label="Embedding clusters",
        seed_label="Cohort, split and projection seed",
    )
    with validation_expander(draft, "Annotations and label transfer", expanded=True):
        field(
            draft,
            root + ["export_tsv"],
            "Write TSV feature tables in worker outputs",
            False,
        )
        field(draft, root + ["folds"], "Held-out validation folds", 5)
        field(draft, root + ["classify"], "Evaluate annotated objects within each dataset", True)
        if stage.get("classify"):
            from morphofeatures.analysis.classification import CLASSIFIER_NAMES

            field(
                draft,
                root + ["classifier_models"],
                "Within-dataset classifiers",
                ["logistic", "knn"],
                options=["logistic", "mlp", "knn"],
                multiple=True,
                format_func=lambda value: CLASSIFIER_NAMES.get(value, value),
            )
            classifier_input_fields(draft, root)
        for i, dataset in enumerate(stage["datasets"]):
            path = root + ["datasets", i]
            field(draft, path + ["annotations"], f"{dataset['name']}: class-label table", None)
            field(draft, path + ["label_column"], f"{dataset['name']}: class column", "auto")
            field(
                draft,
                path + ["group_column"],
                f"{dataset['name']}: specimen/acquisition column",
                None,
            )
            field(
                draft,
                path + ["baselines"],
                f"{dataset['name']}: saved embedding baselines",
                [],
            )
        path = root + ["transfer"]
        field(
            draft,
            path + ["enabled"],
            "Train a reference classifier and predict other datasets",
            False,
        )
        if stage.get("transfer", {}).get("enabled"):
            field(
                draft,
                path + ["reference"],
                "Labeled reference dataset",
                stage["datasets"][0]["name"],
                options=[d["name"] for d in stage["datasets"]],
            )
            transfer_classifier_fields(draft, path)
            transfer_condition_fields(draft, path)
            field(draft, path + ["min_confidence"], "Unknown if class score is below", 0.6)
            field(
                draft, path + ["distance_quantile"], "Reference-distance rejection quantile", 0.99
            )
            field(
                draft,
                path + ["label_mapping"],
                "Target-to-reference class name mapping",
                {},
            )
            st.caption(
                "CORAL estimates each target correction from its unlabeled features. Unknown flags are score/distance heuristics; source validation does not establish accuracy in another species."
            )


def transfer_condition_fields(draft, path):
    from morphofeatures.analysis.transfer import CONDITION_NAMES
    from morphofeatures.analysis.transfer_adapter import ADAPTER_DEFAULTS
    from morphofeatures.workflow_ui import field, validation_expander

    conditions = field(
        draft,
        path + ["conditions"],
        "Alignment conditions",
        ["uncorrected", "coral"],
        options=list(CONDITION_NAMES),
        multiple=True,
        format_func=lambda value: CONDITION_NAMES.get(value, value),
    )
    st.caption(
        "Select all five conditions for the full comparison. Every condition uses the same objects and reference validation folds. Reserve target annotations for the final evaluation; do not tune settings against them."
    )
    if {"adapter", "deep_coral"} & set(conditions):
        with validation_expander(draft, "Neural adapter training"):
            st.caption(
                "The neural conditions use full L2-normalized embedding vectors and their own linear classification head. Classical classifier/PCA/UMAP settings above do not apply to the adapter. DINO remains frozen. Both neural conditions share initialization, batches and training length; only the CORAL loss is enabled or disabled."
            )
            controls = (
                (
                    "hidden_dim",
                    "Adapter hidden width",
                    "Number of hidden neurons learning combinations of the saved embedding features. 128 is a modest starting size; larger networks need more labeled reference objects.",
                ),
                (
                    "latent_dim",
                    "Adapted embedding dimensions",
                    "Number of features exported for each object after adaptation. These vectors feed the neural classifier and joint PCA/UMAP. Use the same size in both neural conditions; this does not change the original DINO vectors.",
                ),
                (
                    "epochs",
                    "Adapter training epochs",
                    "Complete passes over the labeled reference objects, repeated for each validation fold and the final model. Both neural conditions use this fixed training length. Target labels never control stopping.",
                ),
                (
                    "batch_size",
                    "Adapter objects per training batch",
                    "Number of object vectors per supervised batch and, for Deep CORAL, per domain alignment batch. This counts objects, not image slices. At least two objects per domain are needed to estimate covariance; larger batches usually give steadier estimates.",
                ),
                (
                    "learning_rate",
                    "Adapter learning rate",
                    "Size of each neural weight update. Start at 0.001; reduce it if training loss becomes unstable. The same rate is used with and without CORAL.",
                ),
                (
                    "weight_decay",
                    "Adapter weight decay",
                    "Penalty on large neural weights to limit overfitting. 0.0001 is the starting value; zero disables this penalty. Keep it identical across the neural comparison.",
                ),
                (
                    "coral_weight",
                    "Deep CORAL loss weight",
                    "Strength of covariance alignment relative to reference classification loss. Start at 1 and inspect both losses in training_history.tsv. Larger values push datasets together more strongly and can erase useful biology. Zero reproduces the adapter without CORAL; target labels must not guide this choice.",
                ),
                (
                    "standardize",
                    "Standardize adapter input features",
                    "Scale each L2-normalized input feature using labeled reference training objects only. The same fitted scaling is applied to target objects. This is fitted again inside each validation fold and is independent of the classical classifier controls.",
                ),
            )
            for name, label, help_text in controls:
                field(
                    draft, path + ["adapter", name], label, ADAPTER_DEFAULTS[name], help=help_text
                )
            field(
                draft,
                path + ["adapter", "device"],
                "Adapter training device",
                "cpu",
                options=["cpu", "cuda", "auto"],
                help="CPU works without a GPU and is suitable for a small pilot on saved vectors. CUDA requires a GPU in the local machine or Slurm allocation. Auto uses a visible GPU when available. Complete cohorts stay in host memory; only batches move to the GPU.",
            )


def transfer_classifier_fields(draft, path):
    from morphofeatures.analysis.classification import CLASSIFIER_NAMES
    from morphofeatures.workflow_ui import classifier_input_fields, field, validation_expander

    classifier = field(
        draft,
        path + ["classifier"],
        "Reference classifier",
        "logistic",
        options=["logistic", "mlp", "knn"],
        format_func=lambda value: CLASSIFIER_NAMES.get(value, value),
    )
    with validation_expander(draft, "Reference classifier settings"):
        if classifier == "mlp":
            field(draft, path + ["hidden_dimensions"], "MLP hidden-layer widths", [64])
        elif classifier == "knn":
            field(draft, path + ["knn_k"], "Labeled reference neighbors", 5)
        else:
            field(draft, path + ["linear_c"], "Logistic regularization C", 1.0)
        if classifier != "knn":
            field(draft, path + ["max_iter"], "Maximum classifier iterations", 2000)
        field(
            draft,
            path + ["classifier_normalization"],
            "Feature scaling before classifier projection",
            "none",
            options=["none", "standardize"],
        )
        classifier_input_fields(draft, path)
        field(draft, path + ["folds"], "Reference validation folds", 5)


def artifact_stage_form(draft, index):
    from morphofeatures.workflow_ui import field, projection_settings_fields

    stage = draft["document"]["stages"][index]
    root = ["document", "stages", index]
    if stage["action"] == "fuse":
        st.caption("Raw concatenation preserves feature values and ignores weights. Balanced block L2 sets each block to unit length before weighting. Analysis scaling is chosen separately in the following analysis stage.")
        field(draft, root + ["name"], "Combined embedding name", "Fine + coarse")
        field(draft, root + ["embeddings"], "Saved feature blocks {name: path}", {},
              help="Give each block a name and its embedding file path (NPZ/NPY/TSV/CSV), not a folder. You can combine different view recipes or older and newer extractions of the same cells. In a label-transfer reference folder, original_embeddings.npz retains the original feature values; embeddings.npz may already be normalized.")
        earlier = [name for s in draft["document"]["stages"][:index] for name in embedding_names(s)]
        field(
            draft,
            root + ["from_embeddings"],
            "Include earlier embedding outputs",
            [],
            options=earlier,
            multiple=True,
        )
        field(
            draft,
            root + ["component_order"],
            "Feature-block order",
            [],
        )
        field(draft, root + ["weights"], "Ordered feature-block weights", [0.5, 0.5])
        field(draft, root + ["combination"], "Combination", "balanced", options=["balanced", "raw"])
        field(
            draft,
            root + ["assume_same_dataset"],
            "I confirm matching object identities when metadata is missing",
            False,
        )
        from morphofeatures.workflow_ui import validation_expander

        with validation_expander(draft, "Object identity across source files"):
            st.caption("Different source containers can describe the same specimen. Use this only when you have verified that equal IDs in every block refer to the same cells or nuclei. It does not translate IDs between cell and nucleus namespaces.")
            identity_path = root + ["identity_override"]
            confirmed = field(draft, identity_path + ["confirmed"],
                              "I verified matching objects across different source files", False,
                              help="Allows fusion when source fingerprints differ, including older N5 patches combined with newer volume-derived views. Keep off for independent specimens. The declaration and original metadata are saved with the combined vectors.")
            if confirmed:
                field(draft, identity_path + ["dataset"], "Shared specimen / dataset name", "",
                      help="A stable, specific name for this specimen, used with its ID namespace to identify the combined output. Use the same name for other combinations of these exact objects.")
                field(draft, identity_path + ["id_namespace"], "Common embedding ID namespace", "cell",
                      options=["cell", "nucleus", "object"],
                      help="The IDs actually stored in every embedding file. Nuclear images can carry cell IDs. Translate mismatched namespaces using a verified mapping before combining.")
                field(draft, identity_path + ["reason"], "How was object correspondence verified?", "",
                      help="Record the source specimen and mapping or export history establishing correspondence. Equal numeric IDs or the same DINO checkpoint alone are insufficient.")
        if st.button("Add raw/balanced comparison", key=f"fusion-compare:{draft['id']}:{index}"):
            from morphofeatures.workflow_ui import _replace_document, rerun
            from morphofeatures.workspace_state import with_fusion_comparison

            _replace_document(draft, with_fusion_comparison(draft["document"], index))
            rerun()
        st.caption("This adds the alternative fusion method and compares both on shared objects, with unchanged and whole-vector L2 analysis. Review the generated stages and adjust scaling/classifiers before submitting. Your source embedding files are reused.")
    else:
        field(draft, root + ["reference_model"], "Saved reference.json (optional)", None)
        if stage.get("reference_model"):
            st.caption(
                "The saved reference supplies the classical classifier settings and shared validation folds. Reference-to-target CORAL refits that classifier; neural conditions train their own adapter and linear head using the adapter controls below. Keep splits.tsv beside the saved reference."
            )
        field(
            draft,
            root + ["reference"],
            "Reference embedding and annotation settings",
            {"name": "Reference", "embedding": "", "annotations": "", "label_column": "auto"},
        )
        field(
            draft,
            root + ["targets"],
            "Target datasets and optional evaluation annotations",
            [{"name": "Target", "embedding": "", "annotations": None}],
        )
        transfer_classifier_fields(draft, root)
        transfer_condition_fields(draft, root)
        field(draft, root + ["min_confidence"], "Unknown below class score", 0.6)
        field(draft, root + ["distance_quantile"], "Reference-distance rejection quantile", 0.99)
        field(
            draft,
            root + ["assume_same_model"],
            "I confirm identical representations when metadata is missing",
            False,
        )
        projection_settings_fields(draft, root)
        from morphofeatures.provenance_ui import transfer_input_check

        transfer_input_check(stage)


def render_analysis(config, path):
    from morphofeatures.analysis_ui import joint_projection_results
    from morphofeatures.analysis_variants_ui import result_controls
    from morphofeatures.result_panels import analysis_panels

    path = Path(path)
    metadata = json.loads(path.read_text())
    from morphofeatures.result_browser_ui import focus

    path, metadata = result_controls(path, metadata, exploration=False)
    focus(path, metadata)
    if metadata["schema"] == "morphofeatures.joint_analysis.v1":
        joint_projection_results(config, metadata, path)
    else:
        analysis_panels(config, metadata, path)


def saved_combination_document(metadata, path, entries):
    """Prepare a reviewable fusion comparison using saved, ID-matched vectors only."""
    from morphofeatures.analysis.projection import PROJECTION_DEFAULTS
    from morphofeatures.workspace_state import with_fusion_comparison

    if len(entries) < 2:
        raise ValueError("Select at least two saved feature blocks.")
    identities = {(e["dataset"], e["backbone"], str(e["scale_um"])) for e in entries}
    if len(identities) != 1:
        raise ValueError("Choose blocks from one dataset, checkpoint and fine scale.")
    dataset = next((d for d in metadata.get("settings", {}).get("datasets", []) if d["name"] == entries[0]["dataset"]), {})
    scientific = metadata.get("settings", {})
    analysis = {k: v for k, v in scientific.items() if k in PROJECTION_DEFAULTS or k in {
        "umap", "classify", "folds", "classifier_models", "classifier_space", "classifier_dimensions",
        "classifier_standardize", "classifier_umap_neighbors", "classifier_umap_min_dist", "classifier_umap_epochs",
        "evaluation_pca", "linear_c", "knn_k", "max_iter", "hidden_dimensions",
    }}
    analysis.update(
        action="analyze", from_embedding="Combined", normalization="none", analysis_normalizations=["none", "l2"],
        annotations=dataset.get("annotations"), label_column=dataset.get("label_column", "auto"),
        group_column=dataset.get("group_column"),
    )
    document = {"stages": [{
        "action": "fuse", "name": "Combined", "combination": "balanced",
        "embeddings": {e["representation"]: str(Path(path).parent / e["path"]) for e in entries},
        "weights": [1 / len(entries)] * len(entries),
    }, analysis]}
    return with_fusion_comparison(document, 0)


def saved_combination_controls(config, metadata, path):
    from morphofeatures.dino_experiments import FEATURE_BLOCKS
    from morphofeatures.workflow_ui import navigate, use_draft
    from morphofeatures.workspace_state import new_draft

    groups = {}
    for entry in metadata.get("embeddings", []):
        if entry.get("representation") in {v[0] for v in FEATURE_BLOCKS.values()}:
            groups.setdefault((entry["dataset"], entry["backbone"], str(entry["scale_um"])), []).append(entry)
    if not groups:
        return
    with st.expander("Compare combinations from these saved vectors"):
        group = st.selectbox("Saved dataset / backbone / fine scale", list(groups),
                             format_func=lambda key: " · ".join(key), key="fusion-source:" + str(path),
                             help="Choose blocks extracted from the same dataset, DINO checkpoint and fine-patch width. Each dataset retains its own object IDs; fusion does not pair cells between datasets.")
        entries = {e["representation"]: e for e in groups[group]}
        names = st.multiselect(
            "Feature blocks to combine", list(entries),
            default=[n for n in ("Fine foreground", "Coarse CLS") if n in entries],
            key="fusion-blocks:" + str(path) + str(group),
            help="Select any two or more, including all four. This uses the saved matched high-dimensional vectors and their metadata; image sources and DINO weights are not read again.",
        )
        st.caption("Create a draft comparing raw and balanced fusion, each with no analysis scaling and whole-vector L2. You can adjust these conditions, annotations and classifiers in Analyze, then run locally or on SLURM.")
        if st.button("Create saved-vector comparison workflow", disabled=len(names) < 2, key="fusion-draft:" + str(path)):
            document = saved_combination_document(metadata, path, [entries[n] for n in names])
            draft = new_draft("import", document=document, source="Saved DINO combinations: " + str(path))
            draft["name"] = "Compare saved DINO combinations"
            use_draft(config, draft, "Analyze")
            navigate("Workflow")


def experiment_results(config, metadata, path):
    from copy import deepcopy

    from morphofeatures.artifact_availability import availability_details, selected_artifact
    from morphofeatures.result_browser_ui import focus, safe_choice
    from morphofeatures.result_catalog import report_choices, vector_artifacts

    metadata = deepcopy(metadata)
    st.info(metadata.get("interpretation", "Saved experiment outputs."))
    availability_details(vector_artifacts(metadata, path), key="experiment:" + str(path))
    if metadata.get("execution_state") == "partial":
        st.info("Partial experiment: completed outputs are available below. Later outputs may still be running or may have failed; inspect Runs for execution status.")
    if metadata.get("schema") == "morphofeatures.dino_experiment.v1":
        for category in ("comparisons", "joint_analyses", "transfers"):
            metadata[category] = report_choices(metadata.get(category, []), path.parent)
        matched = metadata.get("embeddings", [])
        original = metadata.get("original_embeddings", [])
        population = "Matched comparison cohort"
        if matched and original:
            population = safe_choice("Vector population", [population, "Original extraction population"], key="experiment-population:" + str(path))
        if original and (not matched or population == "Original extraction population"):
            metadata["embeddings"] = original
            st.caption("Original extraction vectors may cover different objects across methods. A new comparison will match its inputs explicitly.")
        metadata["embeddings"] = [e for e in metadata.get("embeddings", []) if
            (status := selected_artifact(path.parent / e["path"])).ready and status.kind == "vectors"]
    if metadata.get("object_selections"):
        with st.expander("Targeted object selection reports"):
            st.caption("Requested IDs are matched before the shared object cap. Later quality exclusions appear in each extraction's coverage/excluded tables; matched analysis cohorts may be smaller again.")
            for entry in metadata["object_selections"]:
                report_path = Path(entry["report"])
                if report_path.is_file():
                    summary = json.loads(report_path.read_text())
                    st.write(f"{entry['dataset']}: {summary['requested']:,} requested · {summary['matched']:,} available · {summary['selected']:,} selected after object cap · {summary['missing']:,} unavailable")
                table_path = Path(entry["table"])
                st.caption(f"ID-by-ID selection report: {table_path}")
                if table_path.is_file() and table_path.stat().st_size <= 200 * 1024**2:
                    st.download_button(f"Download {entry['dataset']} selection table", table_path.read_bytes(),
                                       file_name="selection.tsv", key="selection-download:" + str(table_path))
    if metadata.get("object_eligibility"):
        with st.expander("Object eligibility before DINO extraction"):
            from morphofeatures.data.object_eligibility import eligibility_message

            for entry in metadata["object_eligibility"]:
                summary = entry["summary"]
                message = f"{entry['dataset']}: {eligibility_message(summary)}"
                if summary["skip_objects"] or summary["partial_objects"]:
                    st.warning(message)
                else:
                    st.caption(message)
                st.caption(f"{summary['interpretation']} Report: {entry['table']}")
    from morphofeatures.provenance_ui import compatibility_details

    compatibility_details(metadata.get("input_compatibility", []))
    choices = []
    if metadata["schema"] == "morphofeatures.transfer.v1":
        from morphofeatures.analysis.classification import CLASSIFIER_NAMES, classifier_space
        from morphofeatures.analysis.transfer import CONDITION_NAMES
        from morphofeatures.analysis_ui import evaluation_results

        reference = metadata.get("classifier_settings", metadata.get("settings", {}))
        # Older reports keep the actual fitted settings in the reference artifact.
        reference_path = path.parent / metadata.get("reference_model", "reference/reference.json")
        if "classifier_settings" not in metadata and reference_path.is_file():
            reference = json.loads(reference_path.read_text()).get("settings", reference)
        st.caption(
            f"Reference classifier: {CLASSIFIER_NAMES.get(reference.get('classifier', 'logistic'), reference.get('classifier'))}; "
            f"whole-vector L2 input, additional scaling: {reference.get('classifier_normalization', 'none')}; "
            f"classification input: {classifier_space(reference)}; "
            f"dimensions: {reference.get('classifier_dimensions', reference.get('evaluation_pca') or 2) if classifier_space(reference) != 'embeddings' else 'all'}; "
            f"standardize selected dimensions: {reference.get('classifier_standardize', False)}. "
            "Reference scores use held-out reference objects, before target adaptation and joint plotting."
        )
        evaluation_results(
            metadata["source_validation"],
            "Reference cross-validation",
            key="reference-evaluation:" + str(path),
        )
        if not metadata.get("targets"):
            st.info("No completed target results are available in this report.")
            return
        selected = safe_choice(
            "Target dataset and correction",
            range(len(metadata["targets"])),
            format_func=lambda i: (
                metadata["targets"][i]["name"]
                + " · "
                + CONDITION_NAMES.get(
                    metadata["targets"][i]["condition"], metadata["targets"][i]["condition"]
                )
            ),
            key="transfer-artifact:" + str(path),
            help="Choose a target dataset and correction condition. Known target labels, when supplied, are used only for evaluation; assignment coverage is not an accuracy score.",
        )
        selected_target = metadata["targets"][selected]
        st.caption(
            "This condition used unlabeled target features during adaptation. Target labels were reserved for evaluation."
            if selected_target.get("target_features_used", selected_target["condition"] == "coral")
            else "This condition learned from reference objects only; target features were used only for prediction and plotting."
        )
        if selected_target["condition"] in {"adapter", "deep_coral"}:
            st.caption(
                "Classifier: neural adapter with a linear softmax head. Source validation refits the adapter inside each saved fold. Distances for unknown detection use L2-normalized adapted features."
            )
        if selected_target.get("source_validation") and selected_target["condition"] not in {
            "uncorrected",
            "coral",
        }:
            evaluation_results(
                selected_target["source_validation"],
                "This condition · reference cross-validation",
                key="condition-reference-evaluation:" + str(path) + ":" + str(selected),
            )
        evaluation_results(
            selected_target["metrics"],
            selected_target["name"]
            + " · "
            + CONDITION_NAMES.get(selected_target["condition"], selected_target["condition"])
            + " · target evaluation",
            key="target-evaluation:" + str(path) + ":" + str(selected),
        )
        folder = path.parent / metadata["targets"][selected]["directory"]
        focus(path, metadata, vectors={selected_target["name"] + " · " + selected_target["condition"]: str(folder / "embeddings.npz")})
        st.code(str(folder))
        history = folder / "training_history.tsv"
        if history.is_file():
            st.line_chart(
                pd.read_csv(history, sep="\t").set_index("epoch")[
                    ["classification_loss", "coral_loss", "total_loss"]
                ]
            )
            st.caption(
                "CORAL loss is displayed before multiplication by its configured weight. Training losses are diagnostics; they are not target accuracy."
            )
        for name in (
            "predictions.tsv",
            "embeddings.npz",
            "reference_embeddings.npz",
            "coral.npz",
            "condition.json",
            "adapter.json",
            "adapter_parameters.npz",
            "training_history.tsv",
        ):
            artifact = folder / name
            if selected_artifact(artifact).ready and artifact.stat().st_size <= 200 * 1024**2:
                st.download_button(
                    "Download " + name,
                    artifact.read_bytes(),
                    name,
                    key="transfer-download:" + str(artifact),
                )
        for target in metadata["targets"]:
            if not report_choices([target], path.parent, key="joint_analysis"):
                continue
            title = (
                target["name"]
                + " · "
                + CONDITION_NAMES.get(target["condition"], target["condition"])
            )
            choices.append((title, path.parent / target["joint_analysis"]))
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        **{k: t[k] for k in ("name", "condition")},
                        **{
                            k: v for k, v in t["metrics"].items() if not isinstance(v, (dict, list))
                        },
                    }
                    for t in metadata["targets"]
                ]
            )
        )
        comparison = path.parent / "condition_comparison.tsv"
        if comparison.is_file():
            st.download_button(
                "Download condition comparison",
                comparison.read_bytes(),
                comparison.name,
                key="condition-comparison:" + str(path),
            )
            st.caption(
                "Accuracy and balanced accuracy include unknown rejections. Candidate scores evaluate the proposed labels before rejection, helping distinguish classifier changes from changes in rejection coverage."
            )
    else:
        if metadata.get("fusion_recipes"):
            with st.expander("Feature combination recipes used"):
                st.dataframe(pd.DataFrame(metadata["fusion_recipes"]), hide_index=True)
        from morphofeatures.analysis.projection import NORMALIZATION_NAMES

        scalings = metadata.get("analysis_normalizations", ["l2"])
        st.caption("Analysis scaling evaluated: " + "; ".join(NORMALIZATION_NAMES.get(v, v) for v in scalings) + ". Each setting uses the shared object cohort and re-evaluates enabled classifiers on the same folds.")
        st.dataframe(
            pd.DataFrame(metadata["embeddings"]).drop(
                columns=["path", "original_path"], errors="ignore"
            )
        )
        if metadata.get("cohorts"):
            st.dataframe(pd.DataFrame(metadata["cohorts"])[["dataset", "selected", "matched"]])
        metric_path = path.parent / "metrics.tsv"
        summary_path = path.parent / "metrics_summary.tsv"
        if summary_path.is_file():
            st.caption(
                "Held-out score means and fold-to-fold standard deviations on the matched cohorts."
            )
            st.dataframe(pd.read_csv(summary_path, sep="\t"))
        if metric_path.exists() and metric_path.stat().st_size > 1:
            with st.expander("Individual held-out folds"):
                st.dataframe(pd.read_csv(metric_path, sep="\t"))
        for category in ("comparisons", "joint_analyses"):
            for item in metadata[category]:
                choices.append(
                    (
                        " · ".join(str(v) for k, v in item.items() if k != "path"),
                        path.parent / item["path"],
                    )
                )
        if metadata["transfers"]:
            with st.expander("Label-transfer experiments"):
                i = st.selectbox(
                    "Transfer representation",
                    range(len(metadata["transfers"])),
                    key=selector_key("experiment-transfer:" + str(path)),
                    format_func=lambda i: str(
                        {k: v for k, v in metadata["transfers"][i].items() if k != "path"}
                    ),
                    help="Choose the fine, coarse or combined representation whose label-transfer results you want to inspect. Each representation has its own classical/neural models and selected adaptation conditions.",
                )
                transfer_path = path.parent / metadata["transfers"][i]["path"]
                st.json(json.loads(transfer_path.read_text()))
                if st.checkbox(
                    "Show transfer evaluations and projections",
                    key=str(transfer_path),
                    help="Open reference cross-validation, confusion matrices and reference/target plots for this experiment. Target evaluation requires known target labels. Corrected and uncorrected results are shown separately so you can compare adaptation.",
                ):
                    experiment_results(config, json.loads(transfer_path.read_text()), transfer_path)
                    return
        if metadata["embeddings"]:
            st.markdown("**Saved embedding vectors**")
            i = safe_choice(
                "Vector artifact",
                range(len(metadata["embeddings"])),
                key="experiment-vector:" + str(path),
                format_func=lambda i: " · ".join(
                    str(v)
                    for k, v in metadata["embeddings"][i].items()
                    if k not in {"path", "original_path"}
                ),
                help="Choose a dataset and representation to download its full feature vectors with exact object IDs. Combined vectors contain the weighted fine and coarse features, not their PCA or UMAP coordinates.",
            )
            from morphofeatures.artifact_locations import artifact_path

            embedding = artifact_path(path.parent / metadata["embeddings"][i]["path"])
            focus(path, metadata, vectors={" · ".join(str(metadata["embeddings"][i].get(k, "")) for k in ("dataset", "representation", "backbone", "scale_um")): str(embedding)})
            st.caption(str(embedding))
            sidecar = embedding.with_suffix(".metadata.json")
            if selected_artifact(sidecar).ready and sidecar.stat().st_size <= 200 * 1024**2:
                st.download_button(
                    "Download vector metadata (keep beside NPZ)",
                    sidecar.read_bytes(),
                    sidecar.name,
                    key="experiment-vector-metadata:" + str(path),
                )
            if embedding.stat().st_size <= 200 * 1024**2:
                st.download_button(
                    "Download NPZ vectors and exact object IDs",
                    embedding.read_bytes(),
                    embedding.name,
                )
                import zipfile

                with zipfile.ZipFile(embedding) as archive:
                    small_table = sum(item.file_size for item in archive.infolist()) <= 32 * 1024**2
                saved_tsv = embedding.with_suffix(".tsv")
                if selected_artifact(saved_tsv).ready and saved_tsv.stat().st_size <= 200 * 1024**2:
                    st.download_button(
                        "Download worker TSV vectors", saved_tsv.read_bytes(), saved_tsv.name
                    )
                if st.checkbox(
                    "Prepare TSV vector download",
                    disabled=not small_table,
                    help="Convert a small embedding file into a tab-separated table with one row per object. For larger data, use the saved NPZ file or enable TSV export in the workflow's Analyze page; browser conversion is limited to avoid excessive memory use.",
                ):
                    from morphofeatures.data.io import load_embeddings

                    table = load_embeddings(embedding)
                    frame = pd.DataFrame(table.features)
                    frame.insert(0, "label_id", table.label_ids)
                    text = frame.to_csv(sep="\t", index=False)
                    if len(text.encode()) <= 200 * 1024**2:
                        st.download_button("Download TSV vectors", text, "embeddings.tsv")
                if not small_table:
                    st.caption(
                        "This table is too large to format in the browser. Enable worker TSV export or use the NPZ vectors."
                    )
            else:
                st.caption(f"This NPZ exceeds the 200 MiB browser download limit. Copy it directly from the worker: {embedding}")
    archive = path.parent / "export.zip"
    if selected_artifact(archive).ready:
        if archive.stat().st_size <= 200 * 1024**2:
            st.download_button(
                "Download complete experiment report",
                archive.read_bytes(),
                "experiment.zip",
                key="experiment-zip:" + str(path),
            )
        else:
            st.caption(
                f"Complete report archive: {archive.stat().st_size / 1024**2:,.1f} MiB, above the 200 MiB browser download limit. "
                f"Copy it directly from the worker: {archive}. Individual vector downloads remain available below that limit."
            )
    if choices:
        chosen = safe_choice(
            "Analysis to display",
            range(len(choices)),
            format_func=lambda i: choices[i][0],
            key="experiment-analysis:" + str(path),
            help="Select a within-dataset comparison or a joint map across datasets. Each representation has a separately fitted PCA/UMAP space; coordinates from different representations are not directly interchangeable.",
        )
        render_analysis(config, choices[chosen][1])
