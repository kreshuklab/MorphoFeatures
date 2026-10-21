"""Supervised embedding compression with a common, reserved validation split.

PCA is centered SVD, not a supervised method. Both reducers are fitted on the
same training objects. Exported vectors come from one fitted transform per
method; vectors from independently fitted validation folds are never mixed.
"""

import hashlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from morphofeatures.analysis.annotations import read_annotations
from morphofeatures.analysis.joint import extraction_metadata
from morphofeatures.analysis.transfer_adapter import fit_adapter, validate_adapter_settings
from morphofeatures.artifacts import write_json_atomic
from morphofeatures.data.io import export_embeddings, load_embeddings
from morphofeatures.representations import fingerprint_file
from morphofeatures.workflow_validation import FieldError, prefix_fields

METHODS = {"base": "Original", "pca": "PCA (centered SVD)", "adapter": "Supervised adapter"}


def defaults():
    return {
        "action": "distill", "name": "Distilled", "embedding": "", "annotations": "",
        "label_column": "auto", "group_column": None, "folds": 5, "validation_fold": 0,
        "seed": 42, "pca_components": 64, "pca_standardize": True,
        "adapter": {"latent_dim": 64, "epochs": 100, "standardize": True},
        "probe_c": 1.0, "bilateral_population": "held_out",
        "analysis": {"normalization": "none", "umap": True, "clusters": 8, "seed": 42},
    }


def output_names(stage):
    name = stage.get("name")
    return [name, *(f"{name}.{method}" for method in METHODS)] if name else []


def validate_distillation(settings, *, check_files=True):
    for key, minimum in (("pca_components", 2), ("folds", 2), ("validation_fold", 0), ("seed", 0)):
        value = settings.get(key, defaults()[key])
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise FieldError(f"Enter an integer of at least {minimum}.", key)
    if settings.get("validation_fold", 0) >= settings.get("folds", 5):
        raise FieldError("Validation fold must be smaller than the number of folds (first fold is 0).", "validation_fold")
    if not np.isfinite(float(settings.get("probe_c", 1))) or float(settings.get("probe_c", 1)) <= 0:
        raise FieldError("Linear probe C must be positive and finite.", "probe_c")
    validate_adapter_settings(settings.get("adapter", {}))
    if settings.get("bilateral_population", "held_out") not in {"held_out", "all"}:
        raise FieldError("Choose held_out or all for the bilateral search population.", "bilateral_population")
    if check_files and not Path(settings.get("annotations") or "").is_file():
        raise FieldError("Select a TSV/CSV with unique label_id and known cell types in the embedding's ID namespace.", "annotations")
    if check_files:
        with prefix_fields("annotations"):
            read_annotations(settings["annotations"], settings)
    from morphofeatures.analysis.projection import validate_projection_settings
    from morphofeatures.analysis.variants import validate_variants

    with prefix_fields("analysis"):
        validate_projection_settings(settings.get("analysis", {}))
        validate_variants(settings.get("analysis", {}))
    if settings.get("bilateral", {}).get("enabled"):
        from morphofeatures.analysis.bilateral import validate_bilateral

        with prefix_fields("bilateral"):
            validate_bilateral(settings["bilateral"], check_files=check_files, representation_count=3)


def prepare_split(table, settings):
    """Join exact native IDs; reserve a stratified (optionally grouped) fold."""
    from morphofeatures.representation_analysis import evaluation_splits

    annotations = read_annotations(settings["annotations"], settings)
    labeled = annotations.loc[annotations.known_label.notna()]
    rows = np.flatnonzero(np.isin(table.label_ids, labeled.label_id.to_numpy()))
    if not len(rows):
        raise FieldError("No known labels match these embeddings. Check the specimen and cell/nucleus ID namespace.", "annotations", "embedding")
    with prefix_fields("annotations"):
        y, groups, splits = evaluation_splits(table.label_ids[rows], annotations, settings)
    fold = settings.get("validation_fold", 0)
    if fold >= len(splits):
        raise FieldError(f"Only {len(splits)} folds are possible with these labels. Select validation_fold below {len(splits)}.", "validation_fold")
    train, test = splits[fold]
    if set(y[test]) - set(y[train]):
        raise FieldError("Some validation classes are absent from training. Revise specimen groups or the selected fold.", "group_column", "validation_fold")
    return annotations, rows, y, groups, train, test, len(splits)


def apply_pca(parameters, features):
    """Portable NumPy inference for the exported centered-SVD baseline."""
    return ((features - parameters["input_mean"]) / parameters["input_scale"] - parameters["pca_mean"]) @ parameters["components"].T


def _confusion_figure(matrix, classes, path, title):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    size = max(5, min(18, len(classes) * 0.45))
    fig, ax = plt.subplots(figsize=(size + 1, size))
    totals = matrix.sum(1, keepdims=True)
    image = ax.imshow(matrix / np.maximum(totals, 1), vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes, xlabel="Predicted type",
           ylabel="Known type", title=title)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    fig.colorbar(image, ax=ax, label="Fraction within known type", shrink=0.8)
    fig.tight_layout()
    for suffix in (".svg", ".png"):
        fig.savefig(path.with_suffix(suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


def distill_embeddings(settings, destination):
    """Fit once on training objects, export all objects and a disjoint probe cohort."""
    from morphofeatures.representation_analysis import compare, package_results

    settings = {**defaults(), **deepcopy(settings)}
    validate_distillation(settings)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    table = load_embeddings(Path(settings["embedding"])).sorted()
    annotations, rows, y, groups, train, test, folds = prepare_split(table, settings)
    train_rows, test_rows = rows[train], rows[test]
    x = np.asarray(table.features)  # Preserve the baseline's original values/dtype.
    dims = min(settings["pca_components"], len(train) - 1, x.shape[1])
    if dims < 2:
        raise FieldError("PCA needs at least three training objects and two input features.", "annotations", "pca_components")
    split_table = pd.DataFrame({
        "label_id": table.label_ids[rows], "known_label": y,
        "role": np.where(np.isin(np.arange(len(rows)), train), "train", "validation"),
        "group": groups if groups is not None else "",
    })
    split_table.to_csv(destination / "split.tsv", sep="\t", index=False)
    scaler = StandardScaler(with_mean=settings["pca_standardize"], with_std=settings["pca_standardize"])
    scaled = scaler.fit_transform(x[train_rows])
    pca = PCA(n_components=dims, svd_solver="randomized", random_state=settings["seed"]).fit(scaled)
    parameters = {
        "input_mean": scaler.mean_ if settings["pca_standardize"] else np.zeros(x.shape[1]),
        "input_scale": scaler.scale_ if settings["pca_standardize"] else np.ones(x.shape[1]),
        "pca_mean": pca.mean_, "components": pca.components_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }
    np.savez_compressed(destination / "pca_parameters.npz", **parameters)
    empty = np.empty((0, x.shape[1]), dtype=np.float32)
    adapter, history = fit_adapter(x[train_rows], y[train], empty, empty, settings["adapter"],
                                  seed=settings["seed"], with_coral=False)
    adapter.metadata.update(
        input_normalization="training-object standardization" if adapter.metadata["settings"]["standardize"] else "none",
        loss="class-balanced cross entropy; no CORAL loss or target dataset",
        training_split=str(destination / "split.tsv"),
        input_recipe=extraction_metadata(settings["embedding"]).get("representation_recipe"),
    )
    adapter.save(destination / "adapter_model", history)
    features = {"base": x, "pca": apply_pca(parameters, x), "adapter": adapter.transform(x)}
    source = fingerprint_file(settings["embedding"])
    source_metadata = extraction_metadata(settings["embedding"])
    train_ids = set(table.label_ids[train_rows])
    held_out = ~np.isin(table.label_ids, list(train_ids))
    if groups is not None:
        # Unlabeled cells in a training specimen are not independent validation
        # cells. Unknown groups cannot establish eligibility for a grouped probe.
        object_groups = annotations.set_index("label_id")[settings["group_column"]].reindex(table.label_ids)
        held_out &= object_groups.notna().to_numpy() & ~object_groups.astype(str).isin(set(groups[train])).to_numpy()
    report = {
        "schema": "morphofeatures.distillation.v1", "settings": settings, "source": source,
        "annotation_source": fingerprint_file(settings["annotations"]),
        "split": "split.tsv", "representations": {}, "metrics": [],
        "validation": {"kind": "one reserved stratified fold", "fold": settings["validation_fold"],
                       "effective_folds": folds, "training_objects": len(train), "validation_objects": len(test),
                       "held_out_population": int(held_out.sum()),
                       "grouping": settings.get("group_column") or "object-level (related-object leakage remains possible)"},
        "interpretation": (
            "One training split fits PCA, the neural adapter and each linear probe; validation objects "
            "are never used to fit these models. Every exported object uses the same fitted transform. "
            "All-object maps include training cells and are exploratory. Held-out bilateral retrieval "
            "removes training objects from both queries and search candidates (with groups, it also "
            "removes training groups and objects with unknown group membership); its search population "
            "differs from a full-specimen published probe. Repeated tuning on this split makes it "
            "a development set, not an independent final test."
        ),
        "warnings": [],
    }
    if dims != settings["pca_components"]:
        report["warnings"].append(f"PCA reduced from {settings['pca_components']} to {dims} components to fit the training cohort and input dimensions.")
    if source_metadata.get("supervision"):
        report["warnings"].append("The input vectors already carry fitted-representation exposure. This split protects the new reducer only; check the upstream training objects before claiming independent validation.")
    available_ids = set(table.label_ids)
    report["annotations_without_embeddings"] = [int(v) for v in annotations.label_id if v not in available_ids]
    classes = np.unique(y)
    predictions = []
    paths, evaluation_paths = {}, {}
    for method, values in features.items():
        folder = destination / method
        folder.mkdir(exist_ok=True)
        full = export_embeddings(folder / "embeddings.npz", table.label_ids, values)
        evaluation = export_embeddings(folder / "held_out.npz", table.label_ids[held_out], values[held_out])
        metadata = deepcopy(source_metadata)
        metadata.setdefault("schema", "morphofeatures.embedding.v1")
        if method != "base":
            parameter_file = destination / ("pca_parameters.npz" if method == "pca" else "adapter_model/adapter_parameters.npz")
            metadata["model"] = "embedding_distillation"
            parameter_source = fingerprint_file(parameter_file)
            digest = hashlib.sha256()
            with parameter_file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            parameter_source["sha256"] = digest.hexdigest()
            metadata["representation_recipe"] = {
                "method": method, "algorithm": "distillation.v1",
                "parent": source_metadata.get("representation_recipe", {}),
                "parameters": parameter_source,
            }
            metadata["supervision"] = {
                "labels_used": method == "adapter" or source_metadata.get("supervision", {}).get("labels_used", False),
                "training_ids": sorted(int(i) for i in train_ids),
                "split": str(destination / "split.tsv"),
                "warning": "These vectors were fitted on the recorded training objects. Ordinary cross-validation on all objects does not re-fit this representation; use held_out.npz or a new independent specimen for downstream validation.",
            }
            if source_metadata.get("supervision"):
                metadata["supervision"]["upstream"] = source_metadata["supervision"]
        metadata["distillation"] = {"report": str(destination / "distillation.json"), "method": method, "source": source}
        for file in (full, evaluation):
            exported_metadata = deepcopy(metadata)
            exported_metadata["distillation"]["population"] = "all" if file == full else "held_out"
            if file == evaluation and method != "base":
                exported_metadata["supervision"]["warning"] = (
                    "This file excludes this reducer's training objects (and training/unknown groups "
                    "for grouped splits). Upstream representation exposure and related-object leakage "
                    "must still be assessed; the reducer was not refitted within downstream classifier folds."
                )
            write_json_atomic(file.with_suffix(".metadata.json"), exported_metadata)
        paths[METHODS[method]], evaluation_paths[METHODS[method]] = str(full), str(evaluation)
        # A common linear probe isolates representation effects. Its scaling is
        # trained only on training rows, independently of plotting normalization.
        probe_scaler = StandardScaler().fit(values[train_rows])
        probe = LogisticRegression(C=settings["probe_c"], class_weight="balanced",
                                   max_iter=2000, random_state=settings["seed"])
        probe.fit(probe_scaler.transform(values[train_rows]), y[train])
        predicted = probe.predict(probe_scaler.transform(values[test_rows]))
        matrix = confusion_matrix(y[test], predicted, labels=classes)
        pd.DataFrame(matrix, index=classes, columns=classes).to_csv(folder / "confusion.tsv", sep="\t", index_label="known_label")
        _confusion_figure(matrix, classes, folder / "confusion", METHODS[method] + " · held-out linear probe")
        np.savez_compressed(folder / "linear_probe.npz", mean=probe_scaler.mean_, scale=probe_scaler.scale_,
                            coefficients=probe.coef_, intercept=probe.intercept_, classes=probe.classes_)
        report["metrics"].append({
            "representation": METHODS[method], "dimensions": values.shape[1],
            "balanced_accuracy": float(balanced_accuracy_score(y[test], predicted)),
            "macro_f1": float(f1_score(y[test], predicted, labels=classes, average="macro", zero_division=0)),
        })
        predictions.extend({"label_id": int(i), "known_label": known, "predicted_label": pred, "representation": METHODS[method]}
                           for i, known, pred in zip(table.label_ids[test_rows], y[test], predicted))
        report["representations"][method] = {
            "name": METHODS[method], "embedding": str(full), "held_out": str(evaluation),
            "objects": len(values), "held_out_objects": int(held_out.sum()), "dimensions": values.shape[1],
            "confusion": str(folder / "confusion.png"),
        }
    pd.DataFrame(predictions).to_csv(destination / "predictions.tsv", sep="\t", index=False)
    pd.DataFrame(report["metrics"]).to_csv(destination / "metrics.tsv", sep="\t", index=False)
    # Labels color maps only. Re-running generic CV on pre-fitted adapter vectors
    # would leak training labels, so proxy evaluation above is authoritative.
    analysis = {**settings["analysis"], "annotations": settings["annotations"], "label_column": settings["label_column"],
                "classify": False, "bilateral": {"enabled": False}}
    report["comparison"] = str(compare(paths, destination / "comparison", analysis))
    if settings.get("bilateral", {}).get("enabled"):
        from morphofeatures.analysis.bilateral import run_bilateral

        population = evaluation_paths if settings["bilateral_population"] == "held_out" else paths
        report["bilateral"] = str(run_bilateral(population, destination / "bilateral", settings["bilateral"]))
    result = write_json_atomic(destination / "distillation.json", report)
    package_results(destination)
    return result
