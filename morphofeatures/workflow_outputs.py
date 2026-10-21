"""User-facing output contracts and result entry points, shared by workflow pages.

Listing a run must not scan image volumes, load embedding arrays, or walk caches.
Use the worker's explicit artifacts and the saved report to discover outputs.
"""

from pathlib import Path

# Filenames are the public entry points written by the respective workers.
REPORT_ACTIONS = {
    "distillation.json": "distill",
    "experiment.json": "dino_experiment",
    "analysis.json": "analyze",
    "comparison.json": "compare",
    "joint_analysis.json": "joint_analyze",
    "transfer.json": "transfer",
    "bilateral.json": "bilateral",
    "context.json": "context",
    "fusion.json": "fuse",
    "mobie_export.json": "export_mobie",
    "bilateral_export.json": "export_bilateral",
    "volume_export.json": "export_labels",
    "preprocessing.json": "preprocess",
    "extraction.json": "extract",
}

OUTPUTS = {
    "distill": (
        "Embedding distillation",
        "`distillation.json` compares original vectors, PCA (centered SVD) and a supervised neural adapter. "
        "Each has full and held-out NPZ vectors, confusion figures and a linear-probe model. "
        "`split.tsv` records training/validation IDs; metrics and predictions use only validation objects. "
        "Reusable PCA/adapter parameters, exploratory maps and optional bilateral comparisons are saved beside the report.",
    ),
    "preprocess": (
        "Prepared data",
        "Crops or distributed patches, exact object IDs, masks and reusable data settings. "
        "`preprocessing.json` records preparation and exclusions; the saved YAML feeds a later training or embedding workflow. When Target objects is enabled, selection.tsv / selection.json explain requested and available IDs.",
    ),
    "train": (
        "MAE training",
        "Model checkpoints, resolved training settings and loss logs. Reuse a checkpoint together with its matching data/model configuration to extract embeddings.",
    ),
    "extract": (
        "Embedding extraction",
        "One high-dimensional vector per retained object in an NPZ embedding file, with a `.metadata.json` sidecar recording the model, input data and view settings. DINO targeting adds selection.tsv / selection.json with requested and available IDs. No projection is created unless an analysis stage is included.",
    ),
    "analyze": (
        "Embedding analysis",
        "`analysis.json`, ID-preserving `embeddings.npz`, `coordinates.tsv`, `object_labels.tsv`, and SVG/PNG projection figures. With classification enabled and usable annotations: held-out predictions, evaluation splits and classifier scores.",
    ),
    "compare": (
        "Representation comparison",
        "`comparison.json` and `report.md`, plus a `representation-*` folder for each method's matched embeddings, coordinates, labels and figures. With classification: common held-out folds and comparable scores. Each method has a separately fitted map.",
    ),
    "joint_analyze": (
        "Shared map across datasets",
        "`joint_analysis.json`, shared PCA/UMAP coordinates, clustering, figures and per-dataset tables retaining native IDs. Each representation recipe defines its own shared space.",
    ),
    "dino_experiment": (
        "DINO fine/coarse experiment",
        "Open `experiment.json` for the complete result. Save fine, coarse and combined NPZ vectors, including selected raw/balanced combinations. Each analysis scaling receives its own plots, clusters and classifier evaluation on matched objects. Input previews, coverage and exclusions explain sampling. Targeting adds dataset-N/input/selection.tsv and selection.json with requested, matched and capped cohorts. Held-out scores need annotations; joint maps need multiple datasets; label transfer must be enabled.",
    ),
    "fuse": (
        "Combined embeddings",
        "`embeddings.npz` and its metadata contain the actual combined feature vectors. `fusion.json` records block order, dimensions and weights; `component-*.npz` preserves inputs; `excluded.tsv` explains missing or invalid components. Projections require a following analysis stage.",
    ),
    "transfer": (
        "Label transfer",
        "`transfer.json`, a reusable reference classifier and validation records, and per-target/per-condition predictions, rejection reasons, embeddings and maps. Target accuracy is reported only when evaluation labels are supplied; enabled adaptation methods also save their parameters.",
    ),
    "bilateral": (
        "Bilateral-partner retrieval",
        "`bilateral.json`, `summary.tsv`, per-object retrieval ranks and plots comparing representations on matched candidate/search populations. Requested comparisons with earlier runs add comparison tables and figures.",
    ),
    "context": (
        "Spatial context features",
        "`context.json`, context embedding vectors and metadata, physical-neighbor graph tables, ID mappings and `coverage.tsv`. Optional downstream analysis produces maps and evaluation results.",
    ),
    "export_labels": (
        "Labels on the original volume",
        "`volume_export.json`, requested known-type, predicted-type and cluster label volumes in HDF5/Zarr, `label_lookup.tsv` and `object_mapping.tsv`. These are categorical masks, not raw images or embedding vectors.",
    ),
    "export_mobie": (
        "MoBIE project",
        "`mobie_export.json` identifies the project, dataset, object tables and views. Images are referenced or copied according to your export choice; referenced sources must remain accessible to viewers.",
    ),
    "export_bilateral": (
        "Bilateral 3D visualization",
        "`bilateral_export.json`, pair/position tables and exclusions, plus the selected 3D viewer files, label masks or MoBIE project. Preserve the original segmentation and ID mapping for spatial interpretation.",
    ),
}

START_ACTIONS = {
    "distill": ("distill",),
    "raw": ("preprocess", "train", "extract", "analyze"),
    "platybrowser": ("preprocess", "train", "extract", "analyze"),
    "ome_zarr": ("preprocess", "train", "extract", "analyze"),
    "preprocess": ("preprocess",),
    "crops": ("train", "extract", "analyze"),
    "demo": ("train", "extract", "analyze"),
    "checkpoint": ("extract", "analyze"),
    "dino": ("extract", "analyze"),
    "dino_experiment": ("dino_experiment",),
    "fusion": ("fuse", "analyze"),
    "transfer": ("transfer",),
    "embeddings": ("analyze", "compare"),
    "joint_embeddings": ("joint_analyze",),
    "published": ("analyze",),
    "bilateral": ("bilateral",),
    "context": ("context", "analyze"),
}

DINO_FILES = (
    (
        "experiment.json",
        "Start here: indexes datasets, model/scale combinations, vector files and analyses, with the submitted settings.",
    ),
    (
        "metrics.tsv / metrics_summary.tsv",
        "With classification: individual held-out folds / their mean and sample standard deviation. Scores are not computed from the plotted UMAP.",
    ),
    (
        "dataset-D/input/",
        "Shared selected objects, resolved source/ID mapping and eligibility checks before extraction.",
    ),
    (
        "dataset-D/matched_ids.tsv / excluded_from_comparison.tsv",
        "Objects shared across every requested representation, backbone and scale / missing methods by object.",
    ),
    (
        "MODEL-B/scale-S/dataset-D/extract/views/",
        "Original fine and coarse NPZ vectors and metadata. Some methods can retain more objects than the matched comparison.",
    ),
    (
        "…/extract/views/input_preview.png / .svg / preview-*.npz",
        "A bounded sample of the actual input images and masks; not every image embedded by the run.",
    ),
    (
        "…/extract/views/coverage.tsv / excluded.tsv",
        "Views and tiles used for each object / reasons objects or view modes could not be encoded. Sparse slices are not exhaustive voxel coverage.",
    ),
    (
        "…/extract/views/views.jsonl",
        "Per-view locations, physical size, normalization and sampling decisions. Can be several GB; stream line by line instead of loading it into a spreadsheet.",
    ),
    (
        "…/extract/views/extraction.json / *.metadata.json",
        "Extraction summary and reusable model/view/source provenance. Keep a vector's metadata beside its NPZ file.",
    ),
    (
        "…/extract/views/progress.json / performance.json / resume.json",
        "Progress, timing/cache/memory diagnostics and restart bookkeeping. These are not biological scores. Newer runs can also contain memory_plan.json.",
    ),
    (
        "…/fusion-F/",
        "Combined embeddings, original component copies, fusion settings and exclusions. Equal-weight balanced concatenation preserves all fine/coarse dimensions.",
    ),
    (
        "…/matched-R/embeddings.npz + embeddings.metadata.json",
        "Recommended vectors for comparing methods or reusing in another workflow: exact shared cohort and original high-dimensional features.",
    ),
    (
        "…/comparison/comparison.json / report.md",
        "Detailed settings, cohort checks, metrics and confusion matrices / a short readable comparison report.",
    ),
    (
        "…/comparison/representation-R/",
        "Analysis copy of matched embeddings; coordinates.tsv, object_labels.tsv and projection.svg/.png. With classification: predictions.tsv and splits.tsv.",
    ),
    (
        "…/comparison/coordinates.tsv / splits.tsv",
        "All representation maps in one table (separate fitted spaces) / the shared classification folds.",
    ),
    (
        "Joint analyses / transfers (when requested)",
        "Use the paths indexed by experiment.json. One dataset creates no joint map; disabled transfer creates no transfer predictions.",
    ),
    (
        "export.zip",
        "Worker-created report archive, including vectors, tables and figures; nested export.zip files are omitted. It does not bundle source volumes or model weights. Large archives stay on disk.",
    ),
)


def report_action(metadata):
    schema = metadata.get("schema", "")
    return {
        "morphofeatures.distillation.v1": "distill",
        "morphofeatures.dino_experiment.v1": "dino_experiment",
        "morphofeatures.analysis.v1": "analyze",
        "morphofeatures.comparison.v1": "compare",
        "morphofeatures.joint_analysis.v1": "joint_analyze",
        "morphofeatures.fusion.v1": "fuse",
        "morphofeatures.embedding.v1": "extract",
        "morphofeatures.dino_views.v1": "extract",
        "morphofeatures.preprocessing.v1": "preprocess",
        "morphofeatures.distributed_patches.v1": "preprocess",
        "morphofeatures.label_volumes.v1": "export_labels",
        **{
            f"morphofeatures.{name}.v1": action
            for name, action in (
                ("transfer", "transfer"),
                ("bilateral", "bilateral"),
                ("context", "context"),
                ("mobie_export", "export_mobie"),
                ("bilateral_export", "export_bilateral"),
            )
        },
    }.get(schema)


def ordered_artifacts(artifacts):
    """Surface registered report entry points before settings/logs; never scan disk."""
    paths = list(dict.fromkeys(str(p) for p in artifacts if p))
    return sorted(
        paths,
        key=lambda p: (
            0 if Path(p).name in REPORT_ACTIONS else 1 if Path(p).suffix in {".npz", ".npy"} else 2
        ),
    )


def expected_outputs(stages=None, *, start=None):
    actions = (
        START_ACTIONS.get(start, ()) if start else (stage.get("action") for stage in (stages or ()))
    )
    return [OUTPUTS[action] for action in dict.fromkeys(actions) if action in OUTPUTS]


def show_expected_outputs(stages=None, *, start=None):
    import streamlit as st

    items = expected_outputs(stages, start=start)
    if not items:
        return
    st.markdown(
        "**What this workflow will produce**\n\n"
        + "\n".join(f"- **{title}:** {description}" for title, description in items)
    )
    st.caption(
        "Outputs follow the stages you enable. Runs → Open results opens a saved report without repeating the computation. "
        "NPZ files contain object IDs and feature vectors; coordinates.tsv contains plotting positions."
    )


def show_result_outputs(metadata, path):
    """Explain saved artifacts without reading arrays or large view manifests."""
    import streamlit as st

    action = report_action(metadata)
    if action not in OUTPUTS:
        return
    path = Path(path)
    st.markdown("**Outputs and how to use them**")
    st.write(OUTPUTS[action][1])
    st.caption(
        f"Result entry point: {path} · Files live beside this report or in its indexed subfolders."
    )
    if action == "dino_experiment":
        with st.expander("Fine/coarse output files explained"):
            st.table(
                [
                    {"File or folder": name, "What it contains / when to use it": detail}
                    for name, detail in DINO_FILES
                ]
            )
            st.caption(
                "B, S, D, F and R are zero-based list indices, not biological IDs or replicates. "
                "Use the representation names below; indices can change when you add baselines or pairings. "
                "With several analysis scalings, comparison/representation-R indexes conditions and need not match matched-R. "
                "label_ids retain the selected object's namespace (cell or nucleus); keep the recorded mapping for volume inspection."
            )
        rows = []
        for item in metadata.get("embeddings", []):

            def display(value):
                if not value:
                    return "—"
                value = Path(value)
                try:
                    return str(value.relative_to(path.parent))
                except ValueError:
                    return str(value)

            rows.append(
                {
                    "Dataset": item.get("dataset"),
                    "Representation": item.get("representation"),
                    "Backbone": item.get("backbone"),
                    "Fine width (µm)": str(item.get("scale_um", "")),
                    "Matched vectors (reuse these)": display(item.get("path")),
                    "Original extraction or fusion": display(item.get("original_path")),
                }
            )
        if rows:
            with st.expander("Where each representation is saved"):
                st.dataframe(rows, hide_index=True, use_container_width=True)
        st.caption(
            "The vector download selector uses matched embeddings. Original extractions may include additional objects. "
            "Copy the matching .metadata.json alongside downloaded NPZ files to preserve model, view and source identity."
        )
