"""Bundled published features and saved layouts, without silently refitting them."""

import numpy as np
import pandas as pd

from morphofeatures.analysis.relations import integer_ids, read_table
from morphofeatures.config import repository_root
from morphofeatures.representations import fingerprint_file
from morphofeatures.workflow_validation import FieldError


def references():
    root = repository_root()
    return {
        "MorphoFeatures (480 dimensions)": {
            "embedding": str(root / "analysis/data/morphofeatures_all_cells.npy"),
            "layout": str(root / "data_mobie/morphofeatures_clust_umap.tsv"),
        },
        "MorphoContextFeatures (200 dimensions)": {
            "embedding": str(
                root / "analysis/data/morphocontextfeatures_all_cells_agglomerated.npy"
            ),
            "layout": str(root / "data_mobie/morphocontextfeatures_clust_umap.tsv"),
        },
        "Manually defined features (140 dimensions)": {
            "embedding": str(root / "analysis/data/manually_defined_features.npy"),
        },
    }


def published_projection(table, settings):
    name = settings.get("published_reference")
    reference = references().get(name)
    if not reference or not reference.get("layout"):
        raise FieldError(
            "Choose a published feature set with a saved layout, or disable the saved layout option.",
            "published_reference",
            "use_published_layout",
        )
    from pathlib import Path

    if Path(settings["embedding"]).resolve() != Path(reference["embedding"]).resolve():
        raise FieldError(
            "Saved published coordinates belong to the selected published features. Restore that embedding file or disable the saved layout before analyzing your own features.",
            "embedding",
            "use_published_layout",
        )
    frame = read_table(reference["layout"])[["label_id", "cluster", "umap_1", "umap_2"]]
    frame["label_id"] = integer_ids(frame.label_id)
    for column in ("umap_1", "umap_2"):
        frame[column] = pd.to_numeric(frame[column])
    if frame.label_id.duplicated().any() or not set(frame.label_id) <= set(table.label_ids):
        raise FieldError(
            "Published layout contains duplicate IDs or IDs missing from its feature file; select newly computed analysis and check the reference files.",
            "published_reference",
        )
    if not np.isfinite(frame[["umap_1", "umap_2"]]).all().all():
        raise ValueError("Published coordinates contain nonfinite values")
    return frame.sort_values("label_id"), {
        "layout": "Saved published UMAP and clusters; no projection or clustering was fitted",
        "source": fingerprint_file(reference["layout"]),
        "reference": name,
        "feature_objects": len(table.label_ids),
        "projected_objects": len(frame),
        "objects_without_published_coordinates": [
            int(i) for i in sorted(set(table.label_ids) - set(frame.label_id))
        ],
    }
