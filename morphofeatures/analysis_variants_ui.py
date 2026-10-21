"""Named analysis combinations and interactive, saved Results exploration."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import streamlit as st

from morphofeatures.analysis.projection import NORMALIZATION_NAMES, PROJECTION_DEFAULTS
from morphofeatures.analysis.variants import (
    REPORT_SCHEMAS,
    VARIANT_KEYS,
    rename_result,
    start_variants,
    variant_history,
)
from morphofeatures.workspace_navigation import selector_key


def scaling_fields(draft, path, *, experiment=False):
    from morphofeatures.workflow_ui import _get, _replace_document, field, rerun

    stage = _get(draft, path)
    enabled = experiment or stage.get("analysis_normalizations") is not None
    if enabled:
        field(
            draft, path + ["analysis_normalizations"], "Analysis scaling methods to compare",
            [stage.get("normalization", "l2" if experiment else "standardize")],
            options=list(NORMALIZATION_NAMES), multiple=True, format_func=NORMALIZATION_NAMES.get,
            help="Select one or more. Each method gets its own PCA/UMAP, clusters and newly evaluated classifiers on identical objects and held-out folds. No analysis scaling preserves the saved vector values; L2 divides each complete vector by its length; standardize fits each feature's mean and variance (training folds only for classification). This does not change the saved fusion vectors.",
        )
        st.caption(
            "These are full analysis comparisons, including classifier evaluation. The separate additional "
            "plot/clustering combinations reuse classifier predictions. Balanced fusion has already normalized "
            "each block; selecting no analysis scaling does not undo that operation."
        )
        if stage.get("classifier_standardize"):
            st.info("Additional classifier standardization is enabled. Disable it for a classifier with no scaling at all; it is independent of the analysis scaling selected here.")
    label = "Use a single analysis scaling control" if enabled else "Compare multiple analysis scaling methods"
    if not experiment and st.button(label, key="scaling-mode:" + draft["id"] + ":" + str(path)):
        document = deepcopy(draft["document"])
        updated = document["stages"][path[-1]]
        if enabled:
            updated["normalization"] = (updated.pop("analysis_normalizations") or ["none"])[0]
        else:
            updated["analysis_normalizations"] = list(dict.fromkeys([updated.get("normalization", "standardize"), "none", "l2"]))
        _replace_document(draft, document)
        rerun()


def variant_fields(draft, path):
    from morphofeatures.workflow_ui import _clear_checks, _get, _put, field, touch

    stage = _get(draft, path)
    defaults = {
        key: stage.get(key, value) for key, value in {**PROJECTION_DEFAULTS, "umap": True}.items()
    }
    # These workflows deliberately use these plotting normalizations internally.
    # Adding a combination should start from their actual main map settings.
    if stage.get("action") in {"transfer", "dino_experiment"}:
        defaults["normalization"] = "none" if stage["action"] == "transfer" else (stage.get("analysis_normalizations") or [stage.get("normalization", "l2")])[0]
    variants = stage.setdefault("analysis_variants", [])
    st.caption(
        "Stack named combinations below. Each adds a projection/clustering result using the same "
        "saved vectors and plotted objects as the main analysis. Extraction, adaptation and "
        "classification run once. All variants are exploratory."
    )
    if stage.get("action") in {"dino_experiment", "transfer"}:
        st.caption(
            "These combinations apply to every representation map and target/transfer condition "
            "in this stage, so a large experiment can produce many additional plots."
        )
    if not isinstance(variants, list) or any(not isinstance(v, dict) for v in variants):
        field(draft, path + ["analysis_variants"], "Analysis variants (list)", [])
        st.warning("Use a list of mappings, each with a name and projection/clustering settings.")
        return

    def remove_variant(index):
        variants.pop(index)
        for key in list(st.session_state):
            if (
                isinstance(key, str)
                and key.startswith(f"guided:{draft['id']}:")
                and ".analysis_variants." in key
            ):
                del st.session_state[key]
        _clear_checks(draft)
        touch(draft)

    for i, variant in enumerate(variants):
        vp = path + ["analysis_variants", i]
        with st.container(border=True):
            st.markdown(f"**Combination {i + 1}: {variant.get('name', 'Unnamed')}**")
            field(
                draft,
                vp + ["name"],
                "Combination name",
                f"Variant {i + 1}",
                help="A unique description for these settings, such as PCA50 + K12. Used in Results and provenance; it does not rename files.",
            )
            for key, label, default, choices in (
                (
                    "normalization",
                    "Embedding normalization",
                    "standardize",
                    ["standardize", "l2", "none"],
                ),
                ("cluster_method", "Clustering method", "kmeans", ["kmeans", "leiden"]),
                ("clusters", "K-means clusters", 8, None),
                ("cluster_pca_components", "PCA components before clustering (0 = off)", 0, None),
                ("umap", "Include UMAP", True, None),
                ("umap_pca_components", "PCA components before UMAP (0 = off)", 0, None),
                ("neighbors", "UMAP / Leiden neighbors", 15, None),
                ("min_dist", "UMAP minimum distance", 0.0, None),
                ("umap_epochs", "UMAP epochs (null uses automatic)", 50, None),
                ("resolution", "Leiden resolution", 0.004, None),
                ("seed", "Projection / clustering random seed", 42, None),
            ):
                default = defaults.get(key, default)
                if key.endswith("pca_components"):
                    default = default or 0
                    if variant.get(key) is None:
                        _put(draft, vp + [key], 0 if key in variant else default)
                field(
                    draft,
                    vp + [key],
                    label,
                    default,
                    options=choices,
                    help=(
                        "Controls PCA/UMAP and clustering randomness for this combination. "
                        "The object cohort is fixed to the parent's plotted objects; this seed "
                        "does not resample objects or refit the classifier."
                    )
                    if key == "seed"
                    else None,
                )
            st.button(
                "Remove combination",
                key=f"remove-variant:{draft['id']}:{path}:{i}",
                on_click=remove_variant,
                args=(i,),
            )

    def add_variant():
        index = len(variants) + 1
        names = {v.get("name") for v in variants}
        while f"Variant {index}" in names:
            index += 1
        variants.append({"name": f"Variant {index}", **defaults})
        _clear_checks(draft)
        touch(draft)

    st.button(
        "Add analysis combination",
        key=f"add-variant:{draft['id']}:{path}",
        on_click=add_variant,
    )


def result_name_controls(path, metadata):
    from morphofeatures.workflow_ui import rerun

    st.subheader(metadata.get("display_name", path.stem.replace("_", " ").capitalize()))
    with st.expander("Rename this result"):
        name = st.text_input(
            "Result display name",
            metadata.get("display_name", path.stem),
            key="result-name:" + str(path),
            help="Changes the name shown in Results and provenance. Object IDs, run IDs and file paths remain stable; the rename history is retained.",
        )
        if st.button("Save result name", key="rename-result:" + str(path)):
            try:
                rename_result(path, name)
                rerun()
            except (OSError, ValueError) as error:
                st.error(str(error))
        if metadata.get("name_history"):
            st.dataframe(pd.DataFrame(metadata["name_history"]))


@st.fragment(run_every="2s")
def _progress(path):
    rows = variant_history(path)
    if not rows:
        return
    state = tuple((row["id"], row["state"]) for row in rows)
    key = "variant-progress:" + str(path)
    previous = st.session_state.get(key)
    st.session_state[key] = state
    if previous and previous != state and not any(s in {"queued", "running"} for _, s in state):
        st.rerun()
    st.dataframe(
        pd.DataFrame(
            [
                {key: row.get(key) for key in ("name", "state", "seconds", "error", "result")}
                for row in rows
            ]
        )
    )
    if any(row["state"] == "failed" for row in rows):
        st.warning(
            "Some combinations failed. Their errors are saved above; completed results remain available."
        )


def result_controls(path, metadata, *, exploration=True):
    """Select a revision before plotting/inspection so every downstream view agrees."""
    original = Path(path).resolve()
    rows = variant_history(original)
    failures = sum(row["state"] == "failed" for row in rows)
    if failures:
        st.warning(
            f"{failures} analysis combination(s) failed. Open the exploration section below "
            "for recorded errors; the original and completed combinations are available."
        )
    choices = {str(original): metadata.get("display_name", "Original analysis")}
    for row in rows:
        if row["state"] == "completed":
            child = Path(row["result"])
            if child.is_file():
                value = json.loads(child.read_text())
                choices[str(child)] = value.get("display_name", row["name"])
    if len(choices) > 1:
        selected = st.selectbox(
            "Saved analysis combination",
            list(choices),
            format_func=choices.get,
            key=selector_key("analysis-combination:" + str(original)),
        )
        path = Path(selected)
        metadata = json.loads(path.read_text())
    result_name_controls(path, metadata)
    if metadata.get("schema") not in REPORT_SCHEMAS or not exploration:
        return path, metadata
    with st.expander("Explore clustering and UMAP on these embeddings"):
        st.warning(
            "Re-analysis runs on the UI host and may take time or substantial memory for large "
            "datasets. The browser stays usable while a background process works. For larger "
            "batches, use Workflow with SLURM. Results are saved alongside this run."
        )
        key = "variant-editor:" + str(original) + ":" + str(path)
        if key not in st.session_state:
            options = {k: v for k, v in metadata.get("settings", {}).items() if k in VARIANT_KEYS}
            st.session_state[key] = {
                "id": "results-" + hashlib.sha256(key.encode()).hexdigest()[:16],
                "revision": 0,
                "document": {
                    "stages": [
                        {
                            "action": "analyze",
                            **options,
                            "analysis_variants": [{"name": "New combination", **options}],
                        }
                    ]
                },
            }
        draft = st.session_state[key]
        variant_fields(draft, ["document", "stages", 0])
        active = any(row["state"] in {"queued", "running"} for row in rows)
        if st.button(
            "Compute and save combinations",
            key="compute-variants:" + str(original),
            disabled=active,
        ):
            try:
                variants = deepcopy(draft["document"]["stages"][0]["analysis_variants"])
                start_variants(original, variants)
                st.rerun()
            except (ValueError, OSError) as error:
                st.error(str(error))
        st.caption(
            "These controls change plots and clusters only. Existing classifier scores and "
            "predicted labels keep their original classifier settings. Every combination uses "
            "the original result's plotted cohort, including dataset/native-ID correspondence."
        )
        _progress(original)
    st.download_button(
        "Download current result metadata",
        json.dumps(metadata, indent=2),
        path.name,
        key="current-result-metadata:" + str(path),
    )
    return path, metadata
