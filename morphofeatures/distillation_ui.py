"""Guided distillation settings and compact comparison results."""

from pathlib import Path

import pandas as pd
import streamlit as st


def stage_form(draft, index):
    from morphofeatures.analysis.distillation import defaults
    from morphofeatures.analysis.transfer_adapter import ADAPTER_DEFAULTS
    from morphofeatures.spatial_probe_ui import bilateral_fields
    from morphofeatures.workflow_ui import advanced_fields, field, validation_expander
    from morphofeatures.workspace_jobs import embedding_names

    root = ["document", "stages", index]
    stage = draft["document"]["stages"][index]
    initial = defaults()
    st.info("Compare the original vectors with PCA and a cell-type-trained neural adapter. Images and the original backbone stay unchanged. One reserved validation fold is excluded from fitting both reducers and all linear probes.")
    field(draft, root + ["name"], "Output name", "Distilled",
          help="Later stages can select this name for adapter vectors, or its .base, .pca and .adapter outputs.")
    earlier = [name for s in draft["document"]["stages"][:index] for name in embedding_names(s)]
    if earlier:
        field(draft, root + ["from_embedding"], "Use an earlier embedding output", "",
              options=["", *earlier], help="Leave blank to enter a saved file below. A named output connects this stage to an extraction, fusion or context stage in the same job.")
    if not stage.get("from_embedding") and not stage.get("from_extraction"):
        field(draft, root + ["embedding"], "Input embedding file", "",
              help="NPZ with label_ids and features, or a supported legacy embedding file. IDs must describe the same specimen and namespace as the annotations.")
    field(draft, root + ["annotations"], "Known cell-type table", "",
          help="TSV/CSV with unique label_id and known cell types. Missing/unknown labels are not training examples. Equal cell and nucleus numbers do not establish correspondence; map IDs before using this table.")
    field(draft, root + ["label_column"], "Cell-type column", "auto",
          help="Auto recognizes cell_type, label or cell_label. Enter your column name for other tables.")
    with validation_expander(draft, "Training and reserved validation objects", expanded=True):
        for key, label, default, help_text in (
            ("group_column", "Specimen or validation-group column (optional)", None, "Keep each specimen or independently defined group entirely within training or validation. Blank uses object-level splitting; neighboring or related objects may still share information."),
            ("folds", "Number of stratified partitions", 5, "Five partitions reserve approximately one fifth of labeled objects. Rare classes can reduce the effective number; each class needs at least two examples."),
            ("validation_fold", "Partition reserved for validation", 0, "Zero selects the first partition. Only this partition is scored; this workflow does not pool cross-validation models into one embedding space."),
            ("seed", "Reproducible split and training seed", 42, "Fix this to compare settings using identical objects. Repeated choices based on validation scores need an independent final test."),
            ("probe_c", "Linear probe inverse regularization (C)", 1.0, "A class-balanced logistic regression tests each representation on the same held-out objects. Larger C allows less regularization. Probe scaling is learned from training objects only."),
        ):
            field(draft, root + [key], label, default, optional=key == "group_column", help=help_text)
    with validation_expander(draft, "PCA / SVD baseline"):
        field(draft, root + ["pca_components"], "PCA dimensions", 64,
              help="Centered SVD keeps directions with the most training-set variation; these are not necessarily biological. The worker caps this at training objects minus one or input dimensions, and reports any adjustment.")
        field(draft, root + ["pca_standardize"], "Standardize features before PCA", True,
              help="Learn each feature's mean and standard deviation from training objects only. Disable to preserve relative input feature scales; PCA still centers its input.")
    with validation_expander(draft, "Neural adapter"):
        controls = (
            ("hidden_dim", "Hidden neurons", "Width of the intermediate nonlinear layer. Start at 128; larger networks can overfit a small labeled set."),
            ("latent_dim", "Exported adapter dimensions", "Number of learned features per object. Start at 64 to compare with a 64-component PCA baseline."),
            ("epochs", "Training epochs", "Fixed passes through training objects. Validation labels do not select an early-stopping point; 100 is a starting value."),
            ("batch_size", "Objects per batch", "Number of saved object vectors per neural update, not image slices. Larger batches use more training memory."),
            ("learning_rate", "Learning rate", "Step size for AdamW. Start at 0.001 and reduce it if training becomes unstable."),
            ("weight_decay", "Weight decay", "Regularize neural weights to reduce overfitting; 0.0001 is the default."),
            ("standardize", "Standardize adapter input features", "Scale using training-object statistics only. No automatic L2 normalization or CORAL is applied in this workflow."),
        )
        for key, label, help_text in controls:
            field(draft, root + ["adapter", key], label, ADAPTER_DEFAULTS[key], help=help_text)
        field(draft, root + ["adapter", "device"], "Training device", "cpu", options=["cpu", "cuda", "auto"],
              help="CPU is suitable for a small pilot on saved vectors. CUDA needs a local or SLURM GPU; auto chooses an available GPU.")
    with validation_expander(draft, "Projection and clustering"):
        from morphofeatures.workflow_ui import projection_settings_fields

        field(draft, root + ["analysis", "normalization"], "Map feature normalization", "none",
              options=["none", "standardize", "l2"], help="Changes maps and clusters only, not the reserved linear-probe evaluation or saved vectors.")
        projection_settings_fields(draft, root + ["analysis"])
    with validation_expander(draft, "Bilateral comparison"):
        field(draft, root + ["bilateral_population"], "Objects eligible for bilateral evaluation", "held_out",
              options=["held_out", "all"], format_func=lambda v: {"held_out": "Exclude all training objects", "all": "All objects (exploratory)"}[v],
              help="Held-out removes training objects from queries AND candidate search, including their known partner if that partner was trained on. All retains the full search population but includes supervised training objects. Compare methods using the same population, not scores from a different-sized published run.")
        bilateral_fields(draft, root + ["bilateral"], stage.setdefault("bilateral", {}), inline=True)
    with validation_expander(draft, "Additional imported settings"):
        advanced_fields(draft, root, set(initial) | {"from_embedding", "from_extraction", "bilateral"})


def results(config, metadata, path):
    from morphofeatures.dino_experiment_ui import render_analysis
    from morphofeatures.followup_ui import download_file
    from morphofeatures.workspace_navigation import selector_key

    st.info(metadata["interpretation"])
    for warning in metadata.get("warnings", []):
        st.warning(warning)
    section = st.radio("Distillation results", ["Scores", "Maps & inspection", "Bilateral probe", "Vectors & fitted models"],
                       horizontal=True, key=selector_key("distillation-section:" + str(path)))
    if section == "Scores":
        st.dataframe(pd.DataFrame(metadata["metrics"]), hide_index=True)
        validation = metadata["validation"]
        st.caption(f"{validation['training_objects']:,} training / {validation['validation_objects']:,} validation objects · {validation['grouping']}. These scores use the same standardized linear probe for every representation.")
        method = st.selectbox("Confusion matrix", list(metadata["representations"]),
                              format_func=lambda m: metadata["representations"][m]["name"], key="distill-confusion:" + str(path))
        entry = metadata["representations"][method]
        st.image(entry["confusion"])
        for suffix in (".png", ".svg"):
            download_file(Path(entry["confusion"]).with_suffix(suffix), "Download confusion matrix " + suffix)
        for name in ("metrics.tsv", "predictions.tsv", "split.tsv"):
            download_file(path.parent / name, "Download " + name)
    elif section == "Maps & inspection":
        render_analysis(config, Path(metadata["comparison"]))
    elif section == "Bilateral probe":
        if metadata.get("bilateral"):
            import json

            from morphofeatures.spatial_probe_ui import probe_results

            report = Path(metadata["bilateral"])
            st.caption("Search population: " + metadata["settings"]["bilateral_population"])
            probe_results(config, json.loads(report.read_text()), report)
        else:
            st.info("No bilateral probe was requested. Use Follow-up analysis to configure one from these saved vectors.")
    else:
        method = st.selectbox("Representation", list(metadata["representations"]), key="distill-vectors:" + str(path))
        entry = metadata["representations"][method]
        st.caption("Full vectors include training objects. Held-out vectors exclude every training ID and preserve the fitted coordinate system. Keep each .metadata.json file with its vectors.")
        for field in ("embedding", "held_out"):
            file = Path(entry[field])
            download_file(file, "Download " + field + " vectors")
            download_file(file.with_suffix(".metadata.json"), "Download " + field + " provenance")
        download_file(path.parent / "export.zip", "Download complete distillation results")
