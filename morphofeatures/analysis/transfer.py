"""Matched label-transfer comparisons with classical and neural covariance alignment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from morphofeatures.analysis.annotations import read_annotations
from morphofeatures.analysis.classification import UnsupervisedUMAP, configured_classifier
from morphofeatures.analysis.joint import extraction_metadata, joint_analyze
from morphofeatures.analysis.projection import normalize_features
from morphofeatures.artifacts import write_json_atomic
from morphofeatures.data.io import export_embeddings, load_embeddings
from morphofeatures.representation_analysis import evaluation_splits, package_results
from morphofeatures.workflow_validation import FieldError, at_fields

CONDITION_NAMES = {
    "uncorrected": "Uncorrected · reference classifier",
    "coral": "Classical CORAL · target → reference",
    "coral_source_to_target": "Classical CORAL · reference → target",
    "adapter": "Neural adapter · without CORAL loss",
    "deep_coral": "Neural adapter · with Deep CORAL loss",
}


def validate_transfer_settings(settings):
    from morphofeatures.analysis.projection import validate_projection_settings
    from morphofeatures.analysis.variants import validate_variants
    from morphofeatures.workflow_validation import FieldError

    validate_projection_settings(settings)
    validate_variants(settings)
    conditions = settings.get("conditions", ["uncorrected", "coral"])
    if not isinstance(conditions, list) or not all(isinstance(v, str) for v in conditions):
        raise FieldError("Choose transfer conditions as a list", "conditions")
    if len(set(conditions)) != len(conditions):
        raise FieldError("Select each transfer condition only once", "conditions")
    if settings.get("classifier", "logistic") not in {"logistic", "linear", "mlp", "knn"}:
        raise FieldError("Choose logistic, mlp or knn", "classifier")
    if settings.get("classifier_normalization", "none") not in {"none", "standardize"}:
        raise FieldError("Choose none or standardize", "classifier_normalization")
    # Use the same parameter checks as the Classification Tool before a worker starts.
    try:
        _estimator(settings, 2)
    except (TypeError, ValueError) as error:
        raise FieldError(
            str(error),
            "classifier",
            "hidden_dimensions",
            "knn_k",
            "classifier_space",
            "classifier_dimensions",
            "classifier_umap_neighbors",
            "classifier_umap_min_dist",
            "classifier_umap_epochs",
        ) from error
    if not settings.get("conditions", ["uncorrected", "coral"]) or set(
        settings.get("conditions", ["uncorrected", "coral"])
    ) - set(CONDITION_NAMES):
        raise FieldError("Choose at least one supported transfer condition", "conditions")
    if {"adapter", "deep_coral"} & set(settings.get("conditions", [])):
        from morphofeatures.analysis.transfer_adapter import validate_adapter_settings

        validate_adapter_settings(settings.get("adapter", {}))
    if not 0 <= float(settings.get("min_confidence", 0.6)) <= 1:
        raise FieldError("Minimum class score must be between zero and one", "min_confidence")
    if not 0 < float(settings.get("distance_quantile", 0.99)) <= 1:
        raise FieldError("Distance quantile must be in (0, 1]", "distance_quantile")
    if not isinstance(settings.get("label_mapping", {}), dict):
        raise FieldError("Label-name mapping must be a dictionary", "label_mapping")


def validate_transfer_inputs(settings, *, allow_pending=False):
    """Check saved recipes before fitting any reference classifier or adapter."""
    from morphofeatures.analysis.recipe_compatibility import compare_recipes, compatibility_message

    reference = settings.get("reference", {})
    reference_field = ("reference", "embedding")
    if settings.get("reference_model"):
        reference_field = ("reference_model",)
        with at_fields(reference_field):
            info = json.loads(Path(settings["reference_model"]).read_text())
            metadata = info.get("extraction", {})
            reference = info.get("source", reference)
    elif allow_pending and reference.get("from_embedding") and not reference.get("embedding"):
        return []
    else:
        with at_fields(reference_field):
            path = Path(reference.get("embedding", ""))
            if not path.is_file():
                raise ValueError(f"Choose an existing reference embedding file: {path}")
            metadata = extraction_metadata(path)
    comparisons = []
    for i, target in enumerate(settings.get("targets", [])):
        target_field = ("targets", i, "embedding")
        if allow_pending and target.get("from_embedding") and not target.get("embedding"):
            continue
        with at_fields(target_field):
            path = Path(target.get("embedding", ""))
            if not path.is_file():
                raise ValueError(f"Choose an existing target embedding file: {path}")
            other = extraction_metadata(path)
        comparison = compare_recipes(
            metadata,
            other,
            reference.get("name", "Reference"),
            target.get("name", f"Target {i + 1}"),
        )
        comparison.update(
            reference_path=str(settings.get("reference_model") or reference.get("embedding")),
            target_path=str(path),
        )
        if comparison["differences"]:
            raise FieldError(compatibility_message(comparison), reference_field, target_field)
        if not comparison["verified"] and not settings.get("assume_same_model", False):
            raise FieldError(
                f"Model provenance is unavailable or incomplete for {comparison['reference']} / {comparison['target']}. "
                "Select the original extraction files with metadata, or confirm that these embeddings used the same model and feature settings. "
                "This confirmation applies only to missing records; it cannot override recorded differences.",
                reference_field,
                target_field,
                "assume_same_model",
            )
        comparisons.append(comparison)
    return comparisons


def coral_transform(reference, target):
    """Regularized covariance matching. Row vectors map target into reference coordinates."""
    if min(len(reference), len(target)) < 3:
        raise ValueError("CORAL needs at least three reference and three target objects")
    if reference.shape[1] != target.shape[1]:
        raise ValueError("Reference and target feature dimensions differ")

    def power(values, exponent):
        covariance = LedoitWolf().fit(values).covariance_
        eigenvalues, vectors = np.linalg.eigh(covariance)
        floor = max(float(np.trace(covariance) / covariance.shape[0]) * 1e-6, 1e-12)
        return (vectors * np.maximum(eigenvalues, floor) ** exponent) @ vectors.T

    transform = power(target, -0.5) @ power(reference, 0.5)
    target_mean, reference_mean = target.mean(0), reference.mean(0)
    return (target - target_mean) @ transform + reference_mean, {
        "matrix": transform,
        "target_mean": target_mean,
        "reference_mean": reference_mean,
    }


def _estimator(settings, train_count):
    return configured_classifier(
        settings.get("classifier", "logistic"),
        {**settings, "normalization": settings.get("classifier_normalization", "none")},
        train_count=train_count,
    )


def _distances(features, labels, reference, reference_labels):
    result = np.full(len(features), np.inf)
    for label in np.unique(labels):
        candidates = reference[reference_labels == label]
        if not len(candidates):
            continue
        indices = np.flatnonzero(labels == label)
        for start in range(0, len(indices), 256):
            rows = indices[start : start + 256]
            # Limit reference-side memory as well as query-side memory.
            for offset in range(0, len(candidates), 4096):
                result[rows] = np.minimum(
                    result[rows], cdist(features[rows], candidates[offset : offset + 4096]).min(1)
                )
    return result


def _saved_folds(path, ids):
    """Reuse the actual exported folds, including for a reopened reference artifact."""
    frame = pd.read_csv(path, sep="\t", dtype={"label_id": np.int64})
    lookup = {int(label): i for i, label in enumerate(ids)}
    folds, tested = [], []
    for _, block in frame.groupby("fold", sort=True):
        train, test = [
            np.asarray(
                [lookup[int(v)] for v in block.loc[block.role.eq(role), "label_id"]], dtype=int
            )
            for role in ("train", "test")
        ]
        if len(train) + len(test) != len(ids) or len(set(train) | set(test)) != len(ids):
            raise ValueError(
                "Saved reference splits do not partition the reference training cohort"
            )
        if "group" in block and block["group"].notna().any():
            if block["group"].isna().any() or set(block.loc[block.role.eq("train"), "group"]) & set(
                block.loc[block.role.eq("test"), "group"]
            ):
                raise ValueError(
                    "Saved reference splits must keep complete specimen groups separate"
                )
        tested.extend(test)
        folds.append((train, test))
    if sorted(tested) != list(range(len(ids))):
        raise ValueError("Saved reference splits must hold out each labeled object exactly once")
    return folds


def _classification_metrics(truth, predicted):
    order = sorted(set(truth))
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
        "per_class": classification_report(truth, predicted, output_dict=True, zero_division=0),
        "confusion_labels": order,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=order).tolist(),
        "evaluation_objects": len(truth),
    }


def _reference_thresholds(info, directory, quantile):
    """Apply an edited rejection quantile consistently when reusing a reference."""
    if quantile == float(info["settings"].get("distance_quantile", 0.99)):
        return info["thresholds"]
    path = Path(directory) / "oof_predictions.tsv"
    if not path.is_file():
        raise ValueError(
            "Changing the reference-distance quantile requires the saved oof_predictions.tsv beside reference.json"
        )
    frame = pd.read_csv(path, sep="\t", dtype={"known_label": str})
    return {
        str(label): float(
            np.quantile(frame.loc[frame.known_label.eq(label), "same_class_distance"], quantile)
        )
        for label in info["classes"]
    }


def fit_reference(spec, settings, destination, *, alignment_target=None, split_file=None):
    table = load_embeddings(spec["embedding"]).sorted()
    annotations = read_annotations(spec["annotations"], spec)
    labels = annotations.set_index("label_id").reindex(table.label_ids).known_label
    counts = labels.value_counts()
    retained = counts[counts >= max(2, int(settings.get("minimum_class_count", 2)))].index
    rows = np.flatnonzero(labels.isin(retained).to_numpy())
    if len(retained) < 2:
        raise ValueError(
            "Reference needs at least two classes with at least two annotated objects each"
        )
    values = normalize_features(table.features, "l2").astype(np.float64)
    fit_settings = {**settings, "group_column": spec.get("group_column")}
    y, groups, folds = evaluation_splits(table.label_ids[rows], annotations, fit_settings)
    if split_file is not None:
        folds = _saved_folds(split_file, table.label_ids[rows])
    oof_labels = np.empty(len(y), dtype=object)
    distances = np.full(len(y), np.nan)
    split_rows = []
    for fold, (train, test) in enumerate(folds):
        if set(y[train]) != set(y):
            raise ValueError(
                "Every reference fold must contain every class; revise specimen groups or folds"
            )
        x_train, x_test = values[rows[train]], values[rows[test]]
        if alignment_target is not None:
            # Held-out source objects cannot contribute to alignment or preprocessing.
            x_train, transform = coral_transform(alignment_target, x_train)
            x_test = (x_test - transform["target_mean"]) @ transform["matrix"] + transform[
                "reference_mean"
            ]
        classifier = _estimator(settings, len(train)).fit(x_train, y[train])
        oof_labels[test] = classifier.predict(x_test)
        distances[test] = _distances(x_test, y[test], x_train, y[train])
        split_rows.extend(
            {
                "label_id": int(table.label_ids[rows[i]]),
                "fold": fold,
                "role": role,
                "group": str(groups[i]) if groups is not None else None,
            }
            for role, indices in (("train", train), ("test", test))
            for i in indices
        )
    quantile = float(settings.get("distance_quantile", 0.99))
    if not 0 < quantile <= 1:
        raise ValueError("Distance quantile must be in (0, 1]")
    thresholds = {
        str(label): float(np.quantile(distances[y == label], quantile)) for label in np.unique(y)
    }
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if alignment_target is not None:
        values, transform = coral_transform(alignment_target, values)
        np.savez_compressed(
            destination / "coral.npz",
            matrix=transform["matrix"],
            reference_mean=transform["target_mean"],
            target_mean=transform["reference_mean"],
            input_mean=transform["target_mean"],
            output_mean=transform["reference_mean"],
        )
    model = _estimator(settings, len(rows)).fit(values[rows], y)
    pd.DataFrame(split_rows).to_csv(destination / "splits.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "label_id": table.label_ids[rows],
            "known_label": y,
            "predicted_label": oof_labels,
            "same_class_distance": distances,
        }
    ).to_csv(destination / "oof_predictions.tsv", sep="\t", index=False)
    estimator = model.steps[-1][1]
    params = {
        "features": values[rows],
        "all_features": values,
        "label_ids": table.label_ids,
        "training_label_ids": table.label_ids[rows],
        "labels": y.astype(str),
    }
    if hasattr(estimator, "coef_"):
        params.update(coef=estimator.coef_, intercept=estimator.intercept_)
    estimator_state = {}
    if settings.get("classifier") == "mlp":
        estimator_state = {
            "kind": "mlp",
            "layers": len(estimator.coefs_),
            "n_features_in": int(estimator.n_features_in_),
            "n_outputs": int(estimator.n_outputs_),
            "out_activation": estimator.out_activation_,
        }
        for i, (coef, intercept) in enumerate(zip(estimator.coefs_, estimator.intercepts_)):
            params[f"mlp_coef_{i}"] = coef
            params[f"mlp_intercept_{i}"] = intercept
    preprocessing, umap_preprocessing = {}, {}
    for name, step in model.steps[:-1]:
        if isinstance(step, UnsupervisedUMAP):
            state, raw, embedding = step.numeric_state()
            raw_key, embedding_key = name + "_raw", name + "_embedding"
            params[raw_key], params[embedding_key] = raw, embedding
            umap_preprocessing[name] = {**state, "raw": raw_key, "embedding": embedding_key}
            continue
        if step == "passthrough":
            continue
        attributes = {}
        for attribute in (
            "components_",
            "mean_",
            "explained_variance_",
            "explained_variance_ratio_",
            "singular_values_",
            "n_components_",
            "n_samples_",
            "n_features_in_",
            "noise_variance_",
            "scale_",
            "var_",
            "n_samples_seen_",
        ):
            if hasattr(step, attribute):
                key = "preprocessing_" + name + "_" + attribute
                params[key] = np.asarray(getattr(step, attribute))
                attributes[attribute] = key
        if attributes:
            preprocessing[name] = attributes
    original = export_embeddings(
        destination / "original_embeddings.npz", table.label_ids, table.features
    )
    write_json_atomic(
        original.with_suffix(".metadata.json"), extraction_metadata(spec["embedding"])
    )
    annotations.to_csv(destination / "annotations.tsv", sep="\t", index=False)
    np.savez_compressed(destination / "parameters.npz", **params)
    artifact = {
        "schema": "morphofeatures.reference.v1",
        "classes": estimator.classes_.tolist(),
        "settings": settings,
        "thresholds": thresholds,
        "normalization": "l2",
        "preprocessing": preprocessing,
        "umap_preprocessing": umap_preprocessing,
        "estimator": estimator_state,
        "annotations": "annotations.tsv",
        "source": spec,
        "extraction": extraction_metadata(spec["embedding"]),
        "metrics": {
            "accuracy": float(accuracy_score(y, oof_labels)),
            "balanced_accuracy": float(balanced_accuracy_score(y, oof_labels)),
            "macro_f1": float(f1_score(y, oof_labels, average="macro")),
            "grouping": "specimen" if groups is not None else "object-level",
            "per_class": classification_report(y, oof_labels, output_dict=True, zero_division=0),
            "confusion_labels": sorted(set(y)),
            "confusion_matrix": confusion_matrix(y, oof_labels, labels=sorted(set(y))).tolist(),
            "evaluation_objects": len(y),
            "effective_folds": len(folds),
        },
    }
    if alignment_target is not None:
        artifact["condition"] = "coral_source_to_target"
        artifact["metrics"]["target_features_used"] = True
        artifact["metrics"]["interpretation"] = (
            "Reference validation refits source-to-target alignment inside every source training fold; unlabeled target features participate."
        )
    write_json_atomic(destination / "reference.json", artifact)
    return load_reference(destination / "reference.json")


def load_reference(path):
    """Restore our numeric artifact without executing pickle contents."""
    path = Path(path)
    info = json.loads(path.read_text())
    if info.get("schema") != "morphofeatures.reference.v1":
        raise ValueError("Select a reference.json created by Transfer labels")
    with np.load(path.parent / "parameters.npz", allow_pickle=False) as archive:
        params = dict(archive)
    model = _estimator(info["settings"], len(params["labels"]))
    for name, attributes in info.get("preprocessing", {}).items():
        step = model.named_steps[name]
        for attribute, key in attributes.items():
            value = params[key]
            setattr(step, attribute, value.item() if value.ndim == 0 else value)
    for name, state in info.get("umap_preprocessing", {}).items():
        model.named_steps[name].restore_numeric(
            state, params[state["raw"]], params[state["embedding"]]
        )
    state = info.get("estimator", {})
    if state.get("kind") == "mlp":
        from sklearn.preprocessing import LabelBinarizer

        estimator = model.steps[-1][1]
        estimator.classes_ = np.array(info["classes"])
        estimator._label_binarizer = LabelBinarizer().fit(estimator.classes_)
        estimator.coefs_ = [params[f"mlp_coef_{i}"] for i in range(state["layers"])]
        estimator.intercepts_ = [params[f"mlp_intercept_{i}"] for i in range(state["layers"])]
        estimator.n_layers_ = state["layers"] + 1
        estimator.n_features_in_ = state["n_features_in"]
        estimator.n_outputs_ = state["n_outputs"]
        estimator.out_activation_ = state["out_activation"]
    elif "coef" in params:
        estimator = model.steps[-1][1]
        estimator.classes_ = np.array(info["classes"])
        estimator.coef_, estimator.intercept_ = params["coef"], params["intercept"]
        estimator.n_features_in_ = params["coef"].shape[1]
    else:
        values = params["features"]
        for _, step in model.steps[:-1]:
            if step != "passthrough":
                values = step.transform(values)
        model.steps[-1][1].fit(values, params["labels"])
    return info, params, model


def _fit_adapter_condition(info, params, target, settings, folder, split_file, *, with_coral):
    from morphofeatures.analysis.transfer_adapter import fit_adapter

    features, labels = params["features"], params["labels"]
    folds = _saved_folds(split_file, params["training_label_ids"])
    predicted = np.empty(len(labels), dtype=object)
    distances = np.empty(len(labels))
    options = settings.get("adapter", {})
    seed = int(info["settings"].get("seed", 42))
    for fold, (train, test) in enumerate(folds):
        adapter, _ = fit_adapter(
            features[train],
            labels[train],
            features[train],
            target,
            options,
            seed=seed + fold,
            with_coral=with_coral,
        )
        predicted[test] = adapter.classes_[adapter.predict_proba(features[test]).argmax(1)]
        # Unit-length latent distances are comparable across independently fitted folds.
        distances[test] = _distances(
            normalize_features(adapter.transform(features[test]), "l2"),
            labels[test],
            normalize_features(adapter.transform(features[train]), "l2"),
            labels[train],
        )
    quantile = float(settings.get("distance_quantile", 0.99))
    thresholds = {
        str(c): float(np.quantile(distances[labels == c], quantile)) for c in np.unique(labels)
    }
    adapter, history = fit_adapter(
        features,
        labels,
        params["all_features"],
        target,
        options,
        seed=seed,
        with_coral=with_coral,
    )
    adapter.metadata.update(
        reference_cohort=params["label_ids"].tolist(),
        training_label_ids=params["training_label_ids"].tolist(),
        thresholds=thresholds,
        distance_space="l2_adapter_latent",
    )
    adapter.save(folder, history)
    Path(folder / "splits.tsv").write_bytes(Path(split_file).read_bytes())
    pd.DataFrame(
        {
            "label_id": params["training_label_ids"],
            "known_label": labels,
            "predicted_label": predicted,
            "same_class_distance": distances,
        }
    ).to_csv(folder / "oof_predictions.tsv", sep="\t", index=False)
    metrics = {
        **_classification_metrics(labels, predicted),
        "effective_folds": len(folds),
        "grouping": info["metrics"]["grouping"],
        "target_features_used": adapter.metadata["target_features_used"],
        "interpretation": "Adapter is refitted inside each source training fold. Target labels are never used; the CORAL condition uses unlabeled target features.",
    }
    write_json_atomic(folder / "source_validation.json", metrics)
    return adapter, thresholds, metrics


def predict_transfer_condition(path, features):
    """Reproduce a fitted condition's vectors and class scores from raw saved embeddings.

    Keep condition.json and its model files within the exported transfer directory.
    This reuses the fitted target-cohort transform; it never adapts to new input rows.
    """
    from morphofeatures.analysis.transfer_adapter import NumericAdapter

    path = Path(path)
    info = json.loads(path.read_text())
    if info.get("schema") != "morphofeatures.transfer_condition.v1":
        raise ValueError("Select a condition.json created by Transfer labels")
    values = normalize_features(np.asarray(features), "l2").astype(np.float64)
    if info["condition"] in {"adapter", "deep_coral"}:
        model = NumericAdapter.load(path.parent / "adapter.json")
        values = model.transform(values)
        return values, model.probabilities_from_latent(values)
    if info["condition"] == "coral":
        with np.load(path.parent / "coral.npz", allow_pickle=False) as transform:
            values = (values - transform["target_mean"]) @ transform["matrix"] + transform[
                "reference_mean"
            ]
    _, _, model = load_reference(path.parent / info["classifier_artifact"])
    return values, model.predict_proba(values)


def _numeric_digest(*paths):
    digest = hashlib.sha256()
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            for name in sorted(arrays.files):
                value = arrays[name]
                digest.update(str((name, str(value.dtype), value.shape)).encode())
                digest.update(value.tobytes())
    return digest.hexdigest()


def transfer_labels(settings, destination):
    validate_transfer_settings(settings)
    compatibility = validate_transfer_inputs(settings)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if settings.get("reference_model"):
        info, params, model = load_reference(settings["reference_model"])
        if info.get("condition"):
            raise ValueError(
                "Use the original reference/reference.json as the reusable reference; adapted classifiers belong to a specific target condition"
            )
        import shutil

        (destination / "reference").mkdir(exist_ok=True)
        for name in (
            "reference.json",
            "parameters.npz",
            "annotations.tsv",
            "splits.tsv",
            "oof_predictions.tsv",
            "original_embeddings.npz",
            "original_embeddings.metadata.json",
        ):
            original = Path(settings["reference_model"]).parent / name
            if (
                original.exists()
                and original.resolve() != (destination / "reference" / name).resolve()
            ):
                shutil.copy2(original, destination / "reference" / name)
    else:
        info, params, model = fit_reference(
            settings["reference"], settings, destination / "reference"
        )
    reference_annotations = str(
        destination / "reference" / info.get("annotations", "annotations.tsv")
    )
    reference_path = export_embeddings(
        destination / "reference" / "embeddings.npz", params["label_ids"], params["all_features"]
    )
    write_json_atomic(reference_path.with_suffix(".metadata.json"), info["extraction"])
    confidence = float(settings.get("min_confidence", 0.6))
    if not 0 <= confidence <= 1:
        raise ValueError("Minimum confidence must lie between zero and one")
    distance_quantile = float(
        settings.get("distance_quantile", info["settings"].get("distance_quantile", 0.99))
    )
    reference_thresholds = _reference_thresholds(info, destination / "reference", distance_quantile)
    report = {
        "schema": "morphofeatures.transfer.v1",
        "reference_model": "reference/reference.json",
        "source_validation": info["metrics"],
        "reference_classifier": info["settings"].get("classifier", "logistic"),
        "classifier_settings": info["settings"],
        "settings": settings,
        "input_compatibility": compatibility,
        "targets": [],
        "interpretation": "Conditions share object cohorts and source validation folds. Classical CORAL and Deep CORAL use unlabeled target features; target labels are reserved for evaluation. Unknown flags are score/distance heuristics. Better mixing in a joint map does not establish biological correspondence.",
    }
    split_file = destination / "reference" / "splits.tsv"
    new_conditions = {"coral_source_to_target", "adapter", "deep_coral"}
    if new_conditions & set(settings.get("conditions", [])) and not split_file.is_file():
        raise ValueError(
            "This comparison needs splits.tsv beside the saved reference model to reuse its exact validation folds"
        )
    for index, target in enumerate(settings.get("targets", [])):
        table = load_embeddings(target["embedding"]).sorted()
        metadata = extraction_metadata(target["embedding"])
        original_artifact = export_embeddings(
            destination / f"target-{index}" / "original_embeddings.npz",
            table.label_ids,
            table.features,
        )
        write_json_atomic(original_artifact.with_suffix(".metadata.json"), metadata)
        original = normalize_features(table.features, "l2").astype(np.float64)
        if original.shape[1] != params["features"].shape[1]:
            raise FieldError(
                f"Reference and target feature dimensions differ: reference has {params['features'].shape[1]} columns; "
                f"{target.get('name', 'Target')} has {original.shape[1]}. Select the same backbone, pooling and fusion recipe. "
                "Use full feature vectors on both sides, not UMAP coordinates paired with embeddings.",
                "reference_model" if settings.get("reference_model") else "reference.embedding",
                ("targets", index, "embedding"),
            )
        for condition in settings.get("conditions", ["uncorrected", "coral"]):
            folder = destination / f"target-{index}" / condition
            folder.mkdir(parents=True, exist_ok=True)
            active_model, active_info = model, info
            reference_values = params["all_features"]
            distance_reference = params["features"]
            thresholds = reference_thresholds
            source_validation = info["metrics"]
            direction = "none"
            target_used = False
            model_artifact = "../../reference/reference.json"
            if condition == "coral":
                values, transform = coral_transform(params["all_features"], original)
                np.savez_compressed(
                    folder / "coral.npz",
                    **transform,
                    input_mean=transform["target_mean"],
                    output_mean=transform["reference_mean"],
                )
                direction, target_used = "target_to_reference", True
            elif condition == "uncorrected":
                values = original
            elif condition == "coral_source_to_target":
                active_info, adapted, active_model = fit_reference(
                    {
                        **info["source"],
                        "embedding": str(destination / "reference" / "original_embeddings.npz"),
                        "annotations": reference_annotations,
                        "label_column": "known_label",
                    },
                    {**info["settings"], "distance_quantile": distance_quantile},
                    folder / "model",
                    alignment_target=original,
                    split_file=split_file,
                )
                values = original
                reference_values, distance_reference = adapted["all_features"], adapted["features"]
                thresholds, source_validation = active_info["thresholds"], active_info["metrics"]
                (folder / "coral.npz").write_bytes((folder / "model" / "coral.npz").read_bytes())
                direction, target_used = "reference_to_target", True
                model_artifact = "model/reference.json"
            else:  # Validated neural conditions; their linear heads replace the reference classifier.
                active_model, thresholds, source_validation = _fit_adapter_condition(
                    info,
                    params,
                    original,
                    {**settings, "distance_quantile": distance_quantile},
                    folder,
                    split_file,
                    with_coral=condition == "deep_coral",
                )
                reference_values = active_model.transform(params["all_features"])
                distance_reference = normalize_features(
                    active_model.transform(params["features"]), "l2"
                )
                values = active_model.transform(original)
                direction, target_used = (
                    "shared_neural_space",
                    active_model.metadata["target_features_used"],
                )
                model_artifact = "adapter.json"
            scores = (
                active_model.probabilities_from_latent(values)
                if condition in {"adapter", "deep_coral"}
                else active_model.predict_proba(values)
            )
            classes = np.array(info["classes"])
            predicted = classes[scores.argmax(1)]
            distance_values = (
                normalize_features(values, "l2")
                if condition in {"adapter", "deep_coral"}
                else values
            )
            distance = _distances(distance_values, predicted, distance_reference, params["labels"])
            low = scores.max(1) < confidence
            far = distance > np.array([thresholds[str(c)] for c in predicted])
            rejected = low | far
            frame = pd.DataFrame(
                {
                    "label_id": table.label_ids,
                    "candidate_label": predicted,
                    "predicted_label": np.where(rejected, "unknown", predicted),
                    "prediction_confidence": scores.max(1),
                    "reference_distance": distance,
                    "rejected": rejected,
                    "rejection_reason": np.where(
                        low & far,
                        "low score; distant from reference",
                        np.where(low, "low score", np.where(far, "distant from reference", "")),
                    ),
                }
            )
            for j, label in enumerate(classes):
                frame["score:" + str(label)] = scores[:, j]
            metrics = {"assigned_fraction": float((~rejected).mean())}
            if target.get("annotations"):
                annotations = read_annotations(target["annotations"], target)
                frame = frame.merge(
                    annotations[["label_id", "known_label"]],
                    how="left",
                    on="label_id",
                    validate="one_to_one",
                )
                frame["target_label"] = frame["known_label"]
                mapping = settings.get("label_mapping", {})
                frame["known_label"] = frame.known_label.map(
                    lambda value, mapping=mapping: mapping.get(value, value)
                )
                annotated = frame.known_label.notna()
                if annotated.any():
                    truth = frame.loc[annotated, "known_label"].astype(str)
                    evaluation_truth = np.where(truth.isin(classes), truth, "unknown")
                    result_labels = frame.loc[annotated, "predicted_label"]
                    order = list(classes) + ["unknown"]
                    metrics.update(
                        accuracy=float(accuracy_score(evaluation_truth, result_labels)),
                        balanced_accuracy=float(
                            balanced_accuracy_score(evaluation_truth, result_labels)
                        ),
                        macro_f1=float(f1_score(evaluation_truth, result_labels, average="macro")),
                        per_class=classification_report(
                            evaluation_truth, result_labels, output_dict=True, zero_division=0
                        ),
                        confusion_matrix=confusion_matrix(
                            evaluation_truth, result_labels, labels=order
                        ).tolist(),
                        confusion_labels=order,
                        annotated_objects=int(annotated.sum()),
                        novel_target_classes=sorted(set(truth) - set(classes)),
                        candidate_accuracy=float(
                            accuracy_score(
                                evaluation_truth, frame.loc[annotated, "candidate_label"]
                            )
                        ),
                        candidate_balanced_accuracy=float(
                            balanced_accuracy_score(
                                evaluation_truth, frame.loc[annotated, "candidate_label"]
                            )
                        ),
                    )
            frame.to_csv(folder / "predictions.tsv", sep="\t", index=False)
            frame.to_csv(folder / "object_labels.tsv", sep="\t", index=False)
            embedding = export_embeddings(
                folder / "embeddings.npz", table.label_ids, values.astype(np.float32)
            )
            condition_reference_path = export_embeddings(
                folder / "reference_embeddings.npz",
                params["label_ids"],
                reference_values.astype(np.float32),
            )
            if settings.get("export_tsv", False):
                export_embeddings(
                    embedding.with_suffix(".tsv"), table.label_ids, values.astype(np.float32)
                )
                export_embeddings(
                    condition_reference_path.with_suffix(".tsv"),
                    params["label_ids"],
                    reference_values.astype(np.float32),
                )
            numeric_paths = (
                [folder / "adapter_parameters.npz"]
                if condition in {"adapter", "deep_coral"}
                else [folder / "coral.npz"]
                if condition in {"coral", "coral_source_to_target"}
                else [destination / "reference" / "parameters.npz"]
            )
            adaptation = {"condition": condition, "model_id": _numeric_digest(*numeric_paths)}
            condition_info = {
                "schema": "morphofeatures.transfer_condition.v1",
                "condition": condition,
                "name": CONDITION_NAMES[condition],
                "direction": direction,
                "target_features_used": target_used,
                "target_labels_used": False,
                "input_normalization": "l2",
                "classifier_artifact": model_artifact,
                "classes": classes.tolist(),
                "thresholds": thresholds,
                "min_confidence": confidence,
                "distance_quantile": distance_quantile,
                "distance_space": "l2_adapter_latent"
                if condition in {"adapter", "deep_coral"}
                else "aligned_full_features",
                "source_validation": source_validation,
                "reference_cohort": params["label_ids"].tolist(),
                "training_label_ids": params["training_label_ids"].tolist(),
                "target_cohort": table.label_ids.tolist(),
                "splits": "../../reference/splits.tsv",
                "adaptation": adaptation,
            }
            write_json_atomic(folder / "condition.json", condition_info)
            for artifact, original_metadata in (
                (embedding, metadata),
                (condition_reference_path, info["extraction"]),
            ):
                write_json_atomic(
                    artifact.with_suffix(".metadata.json"),
                    {
                        **original_metadata,
                        **({"adaptation": adaptation} if condition != "uncorrected" else {}),
                        "alignment": {
                            k: condition_info[k]
                            for k in (
                                "condition",
                                "direction",
                                "target_features_used",
                                "input_normalization",
                                "reference_cohort",
                                "target_cohort",
                            )
                        },
                    },
                )
            joint = joint_analyze(
                {
                    **{
                        k: settings[k]
                        for k in (
                            "cluster_method",
                            "neighbors",
                            "min_dist",
                            "umap_epochs",
                            "umap_pca_components",
                            "cluster_pca_components",
                            "resolution",
                            "unlabeled_opacity",
                        )
                        if k in settings
                    },
                    "datasets": [
                        {
                            "name": "Reference",
                            "embedding": str(condition_reference_path),
                            "annotations": reference_annotations,
                            "label_column": "known_label",
                        },
                        {
                            "name": "Target: " + target.get("name", "Target"),
                            "embedding": str(embedding),
                            "annotations": str(folder / "predictions.tsv")
                            if "known_label" in frame
                            else None,
                            "label_column": "known_label",
                        },
                    ],
                    "assume_same_model": settings.get("assume_same_model", False),
                    "normalization": "none",
                    "umap": settings.get("umap", True),
                    "clusters": settings.get("clusters", 8),
                    "seed": settings.get("seed", 42),
                },
                folder / "joint",
            )
            # Preserve transfer predictions in the native-ID target analysis used by inspection/MoBIE.
            native = folder / "joint" / "dataset-1"
            if (native / "coordinates.tsv").exists():
                coordinates = pd.read_csv(
                    native / "coordinates.tsv", sep="\t", dtype={"label_id": np.int64}
                )
                columns = [c for c in frame if c != "known_label"]
                coordinates = coordinates.drop(
                    columns=[c for c in columns if c != "label_id" and c in coordinates]
                )
                coordinates = coordinates.merge(
                    frame[columns], on="label_id", how="left", validate="one_to_one"
                )
                coordinates.to_csv(native / "coordinates.tsv", sep="\t", index=False)
                frame.merge(coordinates[["label_id", "cluster"]], on="label_id", how="left").to_csv(
                    native / "object_labels.tsv", sep="\t", index=False
                )
                from morphofeatures.analysis.visualization import (
                    label_palette,
                    save_projection_figure,
                )

                shared_path = folder / "joint" / "coordinates.tsv"
                shared = pd.read_csv(
                    shared_path, sep="\t", dtype={"label_id": np.int64, "source_label_id": np.int64}
                )
                target_rows = shared.dataset.eq("Target: " + target.get("name", "Target"))
                lookup = frame.set_index("label_id")
                for column in columns:
                    if column != "label_id":
                        shared.loc[target_rows, column] = shared.loc[
                            target_rows, "source_label_id"
                        ].map(lookup[column])
                shared.to_csv(shared_path, sep="\t", index=False)
                palette = label_palette(shared)
                for base, values in ((native, coordinates), (folder / "joint", shared)):
                    report_path = base / (
                        "analysis.json" if base == native else "joint_analysis.json"
                    )
                    report_metadata = json.loads(report_path.read_text())
                    report_metadata["class_colors"] = palette
                    write_json_atomic(report_path, report_metadata)
                    values.attrs.update(
                        class_colors=palette, cluster_colors=report_metadata["cluster_colors"]
                    )
                    for suffix in ("svg", "png"):
                        save_projection_figure(
                            values,
                            base / ("projection." + suffix),
                            "Label transfer · " + CONDITION_NAMES[condition],
                            settings,
                        )
            from morphofeatures.analysis.variants import run_configured_variants

            run_configured_variants(joint, settings)
            package_results(folder / "joint")
            report["targets"].append(
                {
                    "name": target.get("name", f"Target {index + 1}"),
                    "condition": condition,
                    "condition_name": CONDITION_NAMES[condition],
                    "target_features_used": target_used,
                    "source_validation": source_validation,
                    "directory": str(folder.relative_to(destination)),
                    "joint_analysis": str(Path(joint).relative_to(destination)),
                    "metrics": metrics,
                }
            )
            write_json_atomic(folder / "metrics.json", metrics)
    if not report["targets"]:
        raise ValueError("Select at least one target dataset")
    from morphofeatures.analysis.visualization import save_confusion_figures

    save_confusion_figures(
        info["metrics"], destination / "reference", "Reference · held-out predictions"
    )
    for target in report["targets"]:
        (destination / target["directory"] / "source_validation").mkdir(exist_ok=True)
        save_confusion_figures(
            target["source_validation"],
            destination / target["directory"] / "source_validation",
            f"{target['condition_name']} · reference held-out predictions",
        )
        save_confusion_figures(
            target["metrics"],
            destination / target["directory"],
            f"{target['name']} · {target['condition']} · target evaluation",
        )
    write_json_atomic(destination / "transfer.json", report)
    pd.DataFrame(
        [
            {
                "target": t["name"],
                "condition": t["condition"],
                "target_features_used": t["target_features_used"],
                **{
                    "source_" + k: v
                    for k, v in t["source_validation"].items()
                    if isinstance(v, (int, float))
                },
                **{
                    "target_" + k: v for k, v in t["metrics"].items() if isinstance(v, (int, float))
                },
            }
            for t in report["targets"]
        ]
    ).to_csv(destination / "condition_comparison.tsv", sep="\t", index=False)
    package_results(destination)
    return destination / "transfer.json"
