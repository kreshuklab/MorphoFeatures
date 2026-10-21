"""Targeted object cohorts in the existing preparation/extraction forms."""

from copy import deepcopy

import streamlit as st


def object_selection_fields(draft, parent, *, physical=False, limit_path=None):
    from morphofeatures.config import repository_root
    from morphofeatures.data.object_selection import resolve_selection, resolve_selection_paths
    from morphofeatures.workflow_ui import (
        _clear_checks,
        _get,
        _put,
        field,
        rerun,
        touch,
        validation_expander,
    )

    path = parent + ["object_selection"]
    with validation_expander(draft, "Target objects", expanded=True):
        st.caption(
            "Choose which cells or nuclei to process using IDs already assigned to your data. An ID/annotation table is enough; class labels are not required. Existing ROI and object-quality checks still apply."
        )
        if st.button(
            "Use bundled Platynereis ground-truth IDs",
            key=str(path) + ":ground-truth",
            help="Load analysis/data/class_labels.tsv (cell IDs), retaining rows with a cell_type. This only changes object selection; it does not train a model or use labels to build features.",
        ):
            _put(
                draft,
                path,
                {
                    "mode": "table",
                    "ids_file": str(repository_root() / "analysis/data/class_labels.tsv"),
                    "id_column": "label_id",
                    "filter_column": "cell_type",
                    "filter_values": [],
                    "require_nonempty": True,
                    "id_space": "output",
                    "missing": "error",
                },
            )
            touch(draft)
            _clear_checks(draft)
            st.session_state["editor_epoch"] = st.session_state.get("editor_epoch", 0) + 1
            rerun()
        mode = field(
            draft,
            path + ["mode"],
            "Objects to process",
            "all",
            options=["all", "ids", "table"],
            format_func={
                "all": "All objects allowed by existing settings",
                "ids": "Enter an ID list",
                "table": "Use an ID / annotation table",
            }.get,
            help="Targeting happens before reading per-object images and DINO encoding. Locating objects may still require a segmentation index scan. All fine/coarse variants share the selected cohort. All preserves the existing ROI, ID-list and object-cap settings.",
        )
        if mode == "all":
            return
        if mode == "ids":
            field(
                draft,
                path + ["ids"],
                "Target object IDs",
                [],
                help="A YAML list such as [27641, 23159, 19347]. Use the exact integer IDs, not row positions. Duplicate IDs are deduplicated and reported; an empty list is an error.",
            )
        else:
            field(
                draft,
                path + ["ids_file"],
                "ID / annotation table (CSV or TSV)",
                "",
                help="Path accessible to both this UI and the job worker. The table needs an ID column; extra columns may be used to choose a category. Relative paths in the UI start at the repository root.",
            )
            field(
                draft,
                path + ["id_column"],
                "Column containing object IDs",
                "label_id",
                help="Column header holding exact IDs. IDs are read as text before conversion so large integer IDs are not rounded.",
            )
            field(
                draft,
                path + ["filter_column"],
                "Optional category / annotation column",
                None,
                optional=True,
                help="For example cell_type, specimen, or include. Leave empty to use every row. A filter never requires a particular biological annotation vocabulary.",
            )
            if _get(draft, path + ["filter_column"]):
                field(
                    draft,
                    path + ["require_nonempty"],
                    "Only rows with a nonempty annotation",
                    True,
                    help="Exclude blank values in the selected annotation column. Useful for selecting labeled cells from a table that also contains unlabeled cells.",
                )
                field(
                    draft,
                    path + ["filter_values"],
                    "Categories to include (empty means all)",
                    [],
                    help="A YAML list of exact category names, such as [neuron, muscle]. Leave [] to retain every category passing the nonempty check.",
                )
        if physical or _get(draft, path + ["id_space"], "output") != "output":
            field(
                draft,
                path + ["id_space"],
                "Match selected IDs against",
                "output",
                options=["output", "segmentation"] if physical else ["output"],
                format_func={
                    "output": "Workflow output IDs",
                    "segmentation": "Segmentation mask IDs",
                }.get,
                help="Workflow output IDs are the label_ids written to embeddings. PlatyBrowser can export cell IDs while reading nuclei through its mapping. Segmentation matches native mask values. This setting does not infer cell/nucleus correspondence.",
            )
        if (
            _get(draft, path + ["ids_file"])
            == str(repository_root() / "analysis/data/class_labels.tsv")
            and mode == "table"
        ):
            st.info(
                "The bundled ground-truth table uses CELL IDs. For physical DINO / distributed PlatyBrowser nuclei, select cell IDs under 'Object IDs used by annotations'. Single-crop preparation retains native segmentation IDs. If your crops or mask use NUCLEUS IDs, translate target IDs with the cell-to-nucleus mapping below."
            )
        st.caption(
            "Target IDs and input IDs must name the same objects in the same specimen. Equal numbers alone do not establish cell/nucleus correspondence. Selection translates IDs for matching; it does not rename the output embeddings."
        )
        with validation_expander(draft, "Translate target IDs (optional)"):
            field(
                draft,
                path + ["mapping_file"],
                "Selection ID mapping table",
                None,
                optional=True,
                help="Optional CSV/TSV mapping from the target table's IDs to the IDs being matched in this workflow. For cell → nucleus selection, use cells_to_nuclei.tsv. Leave empty when both already use the same namespace.",
            )
            if _get(draft, path + ["mapping_file"]):
                field(
                    draft,
                    path + ["mapping_from"],
                    "Mapping column for target-table IDs",
                    "label_id",
                    help="Column identifying objects in your target ID list/table, usually label_id for cell IDs.",
                )
                field(
                    draft,
                    path + ["mapping_to"],
                    "Mapping column for workflow / mask IDs",
                    "nucleus_id",
                    help="Column containing IDs expected by this workflow, usually nucleus_id when selecting nuclear masks or nucleus-indexed crops. Positive mappings must be one-to-one.",
                )
        field(
            draft,
            path + ["missing"],
            "If a target ID is unavailable",
            "error",
            options=["error", "skip"],
            format_func={
                "error": "Stop and explain missing IDs",
                "skip": "Skip unavailable IDs and report them",
            }.get,
            help="Missing mappings or IDs outside the input/ROI appear in selection.tsv. Skip still fails if nothing matches. Later quality exclusions remain in the usual preprocessing/extraction reports.",
        )
        if limit_path:
            cap = _get(draft, limit_path, 0)
            st.caption(
                f"Current object cap: {cap or 'no cap'}. It applies after targeting; for fine/coarse experiments the cap is shared by all datasets. Existing explicit ID lists also restrict the selection."
            )
            if cap and st.button(
                "Use every targeted object (set object cap to 0)", key=str(path) + ":uncap"
            ):
                _put(draft, limit_path, 0)
                touch(draft)
                _clear_checks(draft)
                st.session_state["editor_epoch"] = st.session_state.get("editor_epoch", 0) + 1
                rerun()
        if st.button(
            "Check target table / IDs",
            key=str(path) + ":check",
            help="Read only the selection and optional mapping table. No images, model weights, or segmentation scan. Source/ROI availability is checked when the object index is available.",
        ):
            try:
                document = {"object_selection": deepcopy(_get(draft, path))}
                resolve_selection_paths(document, repository_root())
                _, audit = resolve_selection(document["object_selection"])
                st.success(
                    f"{audit['requested']:,} unique target IDs from {audit['filtered_rows']:,} retained rows; {audit['duplicate_ids_removed']} duplicate IDs removed."
                )
                st.dataframe(audit["rows"][:50], hide_index=True)
                unmapped = sum(row["target_id"] is None for row in audit["rows"])
                if unmapped:
                    st.warning(
                        f"{unmapped:,} requested IDs have no positive mapping. Correct the mapping or choose to skip unavailable IDs."
                    )
                st.caption(
                    "This confirms the table contents, not availability in the selected input or ROI. Labels in the table are used only for filtering."
                )
            except (ValueError, OSError) as error:
                st.error(f"Target objects → {error}")
