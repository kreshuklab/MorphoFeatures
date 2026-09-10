"""Reusable projections and matched, leakage-aware representation comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
)
from sklearn.preprocessing import StandardScaler

from morphofeatures.analysis.annotations import join_labels, read_annotations
from morphofeatures.analysis.classification import LEGACY_CELL_TYPES, configured_classifier

# Public aliases retained for callers; both UI routes use the same implementation.
from morphofeatures.analysis.projection import normalize_features as transformed
from morphofeatures.analysis.projection import project_embeddings as project_table
from morphofeatures.analysis.visualization import label_palette
from morphofeatures.analysis.visualization import save_projection_figure as save_figure
from morphofeatures.artifacts import write_json_atomic
from morphofeatures.data.io import export_embeddings, load_embeddings
from morphofeatures.representations import fingerprint_file


def package_results(destination):
    """Build the portable result archive inside the worker, not a UI callback."""
    import os
    import zipfile

    destination = Path(destination)
    archive = destination / "export.zip"
    temporary = destination / "export.zip.tmp"
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for item in sorted(destination.rglob("*")):
            if item.is_file() and item not in {archive, temporary}:
                bundle.write(item, item.relative_to(destination))
    os.replace(temporary, archive)
    return archive


def analyze(embedding, destination, settings):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    table = load_embeddings(embedding).sorted()
    frame, diagnostics = project_table(table.label_ids, table.features, settings)
    export_embeddings(destination / "embeddings.npz", table.label_ids, table.features)
    result = {
        "schema": "morphofeatures.analysis.v1",
        "settings": settings,
        "source": fingerprint_file(embedding),
        "diagnostics": diagnostics,
        "coordinates": "coordinates.tsv",
        "embedding": "embeddings.npz",
        "interpretation": "Projection and clustering are exploratory; biological predictive value is untested.",
    }
    labels = join_labels(table.label_ids, settings)
    if settings.get("annotations") and settings.get("classify", True):
        result["classification"] = classify_embedding(table, destination, settings)
        labels = pd.read_csv(
            destination / "object_labels.tsv", sep="\t", dtype={"label_id": np.int64}
        )
        result["interpretation"] = (
            "Projections are exploratory. Classification scores use held-out annotations and "
            "training-fold-only scaling; generalization to independent specimens depends on "
            "the grouping and representation pretraining exposure."
        )
    metadata = Path(embedding).with_suffix(".metadata.json")
    if metadata.exists():
        result["extraction"] = json.loads(metadata.read_text())
    frame = frame.merge(labels, on="label_id", how="left", validate="one_to_one")
    result["class_colors"] = frame.attrs["class_colors"] = label_palette(labels)
    labels.merge(frame[["label_id", "cluster"]], on="label_id", how="left").to_csv(
        destination / "object_labels.tsv", sep="\t", index=False
    )
    result["object_labels"] = "object_labels.tsv"
    frame.to_csv(destination / "coordinates.tsv", sep="\t", index=False)
    save_figure(frame, destination / "projection.svg", "Exploratory morphology", settings)
    path = write_json_atomic(destination / "analysis.json", result)
    package_results(destination)
    return path


def evaluation_splits(ids, annotations, settings):
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    column = (
        "known_label" if "known_label" in annotations else settings.get("label_column", "label")
    )
    group_column = settings.get("group_column")
    if column not in annotations:
        raise ValueError(f"Annotations require label column {column!r}")
    matched = annotations.set_index("label_id").reindex(ids)
    if matched[column].isna().any():
        raise ValueError("Evaluation requires labels for every evaluated object")
    y = matched[column].astype(str).to_numpy()
    folds = int(settings.get("folds", 5))
    if settings.get("fold_policy", "reduce") == "reduce":
        folds = min(folds, int(pd.Series(y).value_counts().min()))
    if folds < 2 or pd.Series(y).value_counts().min() < folds:
        raise ValueError(
            "Each class needs at least folds objects; reduce folds or provide more labels"
        )
    if group_column:
        if group_column not in matched or matched[group_column].isna().any():
            raise ValueError("group_column must exist and be populated for every evaluated object")
        groups = matched[group_column].astype(str).to_numpy()
        if len(np.unique(groups)) < folds:
            raise ValueError("There must be at least folds independent specimen/acquisition groups")
        split = list(
            StratifiedGroupKFold(
                folds, shuffle=True, random_state=int(settings.get("seed", 42))
            ).split(ids, y, groups)
        )
    else:
        groups = None
        split = list(
            StratifiedKFold(folds, shuffle=True, random_state=int(settings.get("seed", 42))).split(
                ids, y
            )
        )
    for train, test in split:
        if len(np.unique(y[train])) < 2 or not len(test):
            raise ValueError(
                "A fold has fewer than two training classes or no test objects; revise groups/folds"
            )
        if groups is not None and set(groups[train]) & set(groups[test]):
            raise ValueError("Specimen groups overlap between training and test data")
    return y, groups, split


def evaluate_features(features, y, splits, settings, *, predictions=None):
    from sklearn.neighbors import NearestNeighbors

    scores = []
    for fold, (train, test) in enumerate(splits):
        x_train, x_test = features[train], features[test]
        normalization = settings.get("normalization", "standardize")
        if normalization == "standardize":
            scaler = StandardScaler().fit(x_train)
            x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
        else:
            x_train, x_test = (
                transformed(x_train, normalization),
                transformed(x_test, normalization),
            )
        if settings.get("evaluation_pca"):
            dimensions = int(settings["evaluation_pca"])
            if dimensions > min(x_train.shape):
                raise ValueError(
                    "evaluation_pca exceeds a training fold's sample count or feature dimension"
                )
            reducer = PCA(dimensions, random_state=int(settings.get("seed", 42))).fit(x_train)
            x_train, x_test = reducer.transform(x_train), reducer.transform(x_test)
        k = min(int(settings.get("knn_k", 5)), len(train))
        if k < 1:
            raise ValueError("knn_k must be positive")
        neighbors = (
            NearestNeighbors(n_neighbors=k).fit(x_train).kneighbors(x_test, return_distance=False)
        )
        relevance = y[train][neighbors] == y[test, None]
        available = np.asarray([(y[train] == label).sum() for label in y[test]])
        row = {
            "fold": fold,
            "train_objects": len(train),
            "test_objects": len(test),
            "retrieval_precision_at_k": float(relevance.mean()),
            "retrieval_recall_at_k": float(np.mean(relevance.sum(1) / np.maximum(available, 1))),
            "k": k,
            "test_objects_with_class_absent_from_training": int((available == 0).sum()),
        }
        classes = [name for name in LEGACY_CELL_TYPES if name in set(y)]
        classes.extend(sorted(set(y) - set(classes)))
        encoded = np.asarray([classes.index(value) for value in y])
        for model_name in settings.get("classifier_models", ["logistic", "knn"]):
            name = "linear" if model_name == "logistic" else model_name
            estimator = configured_classifier(model_name, settings, train_count=len(train))
            estimator.fit(features[train], encoded[train])
            predicted = np.asarray(classes)[estimator.predict(features[test])]
            confidence = estimator.predict_proba(features[test]).max(axis=1)
            if predictions is not None:
                predictions.extend(
                    {
                        "fold": fold,
                        "classifier": name,
                        "row": int(index),
                        "actual": str(actual),
                        "predicted": str(value),
                        "confidence": float(probability),
                    }
                    for index, actual, value, probability in zip(
                        test, y[test], predicted, confidence
                    )
                )
            row[name + "_accuracy"] = float(accuracy_score(y[test], predicted))
            row[name + "_balanced_accuracy"] = float(balanced_accuracy_score(y[test], predicted))
        model = KMeans(
            n_clusters=min(int(settings.get("clusters", 8)), len(train)),
            n_init=10,
            random_state=int(settings.get("seed", 42)),
        ).fit(x_train)
        row["test_kmeans_ari"] = float(adjusted_rand_score(y[test], model.predict(x_test)))
        scores.append(row)
    return scores


def classify_embedding(table, destination, settings):
    """Evaluate one representation on ID-joined annotations and retain OOF evidence."""
    from sklearn.metrics import classification_report, confusion_matrix

    path = Path(settings["annotations"])
    annotations = read_annotations(path, settings)
    if (
        "label_id" not in annotations
        or annotations.label_id.isna().any()
        or annotations.label_id.duplicated().any()
    ):
        raise ValueError("Annotations need unique, nonmissing label_id values")
    column = "known_label"
    if column not in annotations:
        raise ValueError(f"Missing annotation column: {column}")
    annotated = set(annotations.loc[annotations[column].notna(), "label_id"])
    indices = np.flatnonzero(np.isin(table.label_ids, list(annotated)))
    if not len(indices):
        raise ValueError("No embedding object IDs match labeled annotations")
    matched_labels = annotations.set_index("label_id").loc[table.label_ids[indices], column]
    counts = matched_labels.value_counts()
    retained = counts[counts >= max(2, int(settings.get("minimum_class_count", 2)))].index
    indices = indices[matched_labels.isin(retained).to_numpy()]
    if len(retained) < 2:
        raise ValueError(
            "At least two matched cell types need two or more examples; disable classification to show labels only"
        )
    ids = table.label_ids[indices]
    y, groups, splits = evaluation_splits(ids, annotations, settings)
    predictions = []
    scores = evaluate_features(
        table.features[indices], y, splits, settings, predictions=predictions
    )
    split_rows = [
        {
            "fold": fold,
            "role": role,
            "label_id": int(ids[i]),
            "label": y[i],
            "group": groups[i] if groups is not None else None,
        }
        for fold, (train, test) in enumerate(splits)
        for role, rows in (("train", train), ("test", test))
        for i in rows
    ]
    pd.DataFrame(split_rows).to_csv(destination / "splits.tsv", sep="\t", index=False)
    for row in predictions:
        row["label_id"] = int(ids[row.pop("row")])
    pd.DataFrame(predictions).to_csv(destination / "predictions.tsv", sep="\t", index=False)
    classes = sorted(set(y))
    details = {}
    for name in dict.fromkeys(row["classifier"] for row in predictions):
        rows = [row for row in predictions if row["classifier"] == name]
        actual, predicted = [row["actual"] for row in rows], [row["predicted"] for row in rows]
        details[name] = {
            "classes": classes,
            "confusion_matrix": confusion_matrix(actual, predicted, labels=classes).tolist(),
            "per_class": classification_report(
                actual, predicted, labels=classes, output_dict=True, zero_division=0
            ),
        }
    score_frame = pd.DataFrame(scores)
    metrics = [
        key for key in score_frame if key not in {"fold", "train_objects", "test_objects", "k"}
    ]
    models = settings.get("classifier_models", ["logistic", "knn"])
    if not models:
        raise ValueError("Select at least one classifier, or disable classification")
    prediction_model = settings.get("prediction_model", models[0])
    if prediction_model not in models:
        raise ValueError("prediction_model must be one of classifier_models")
    chosen = "linear" if prediction_model == "logistic" else prediction_model
    labels = join_labels(table.label_ids, settings)
    labels["predicted_label"] = pd.Series(pd.NA, index=labels.index, dtype="string")
    labels["prediction_source"] = pd.Series(pd.NA, index=labels.index, dtype="string")
    labels["prediction_confidence"] = np.nan
    oof = pd.DataFrame([row for row in predictions if row["classifier"] == chosen]).set_index(
        "label_id"
    )
    for name, source in (("predicted_label", "predicted"), ("prediction_confidence", "confidence")):
        labels.loc[np.isin(table.label_ids, ids), name] = oof.loc[ids, source].to_numpy()
    labels.loc[np.isin(table.label_ids, ids), "prediction_source"] = "held_out_fold"
    unseen = np.flatnonzero(~np.isin(table.label_ids, ids))
    if len(unseen) and settings.get("predict_unlabeled", True):
        estimator = configured_classifier(prediction_model, settings, train_count=len(ids))
        ordered = [name for name in LEGACY_CELL_TYPES if name in set(y)]
        ordered.extend(sorted(set(y) - set(ordered)))
        encoded = np.asarray([ordered.index(value) for value in y])
        estimator.fit(table.features[indices], encoded)
        labels.loc[unseen, "predicted_label"] = np.asarray(ordered)[
            estimator.predict(table.features[unseen])
        ]
        labels.loc[unseen, "prediction_confidence"] = estimator.predict_proba(
            table.features[unseen]
        ).max(1)
        labels.loc[unseen, "prediction_source"] = "fit_on_labeled_objects"
    labels["prediction_classifier"] = prediction_model
    labels.to_csv(destination / "object_labels.tsv", sep="\t", index=False)
    return {
        "evaluated_objects": len(ids),
        "excluded_object_ids": [int(i) for i in table.label_ids if i not in set(ids)],
        "effective_folds": len(splits),
        "prediction_model": prediction_model,
        "prediction_protocol": "Evaluated objects use held-out predictions; other objects use a model fitted on all eligible labeled objects. Confidence is not calibrated.",
        "annotation_source": fingerprint_file(path),
        "grouping": settings.get("group_column")
        or "object-level; related-object leakage remains untested",
        "learned_preprocessing": "training fold only",
        "representation_training_exposure": "not inferred; assess pretraining and specimen overlap separately",
        "fold_metrics": scores,
        "mean_metrics": {key: float(score_frame[key].mean()) for key in metrics},
        "std_metrics": {key: float(score_frame[key].std(ddof=0)) for key in metrics},
        "classifiers": details,
        "predictions": "predictions.tsv",
        "splits": "splits.tsv",
    }


def compare(embeddings, destination, settings):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if len(embeddings) < 2:
        raise ValueError("Comparison requires at least two representations")
    tables = {name: load_embeddings(Path(path)).sorted() for name, path in embeddings.items()}
    extraction_metadata = {}
    source_signatures = []
    for name, path in embeddings.items():
        metadata_path = Path(path).with_suffix(".metadata.json")
        if metadata_path.exists():
            extraction_metadata[name] = json.loads(metadata_path.read_text())
            sources = extraction_metadata[name].get("sources", {})
            data_sources = {
                key: value
                for key, value in sources.items()
                if key in {"crops", "label_ids", "patches_container", "positions_container"}
            }
            if data_sources:
                source_signatures.append(json.dumps(data_sources, sort_keys=True))
    if len(set(source_signatures)) > 1:
        raise ValueError(
            "Embedding provenance refers to different target datasets; extract the same object source for each representation"
        )
    ids = np.asarray(
        sorted(set.intersection(*(set(t.label_ids) for t in tables.values()))), dtype=np.int64
    )
    if len(ids) < 3:
        raise ValueError("Fewer than three matched objects across representations")
    excluded = {
        name: sorted(int(i) for i in set(t.label_ids) - set(ids)) for name, t in tables.items()
    }
    annotation_excluded = []
    splits = None
    # Classification is shared with standalone analysis, including rare-class
    # filtering, ID-sorted fold assignment, and out-of-fold predictions.
    report = {
        "schema": "morphofeatures.comparison.v1",
        "settings": settings,
        "matched_objects": len(ids),
        "excluded_from_matching": excluded,
        "excluded_from_evaluation": annotation_excluded,
        "protocol": {
            "tuning_budget": "one fixed configuration per representation",
            "learned_preprocessing": "training fold only",
            "grouping": settings.get("group_column")
            or "object-level; related-object leakage remains untested",
            "dataset_identity": "verified from extraction provenance"
            if len(source_signatures) == len(tables)
            else "incomplete provenance; matching uses label_id",
            "representation_training_exposure": "not inferred; report whether the representation was trained on evaluation specimens",
        },
        "representations": {},
        "interpretation": "Evaluate biological tasks using held-out metrics; projections alone do not establish quality."
        if splits
        else "No labels supplied: only exploratory neighborhoods and clustering diagnostics; predictive biological value and generalization remain untested.",
    }
    frames = []
    for index, (name, table) in enumerate(tables.items()):
        sub = destination / f"representation-{index}"
        sub.mkdir(exist_ok=True)
        features = table.features[np.searchsorted(table.label_ids, ids)]
        export_embeddings(sub / "embeddings.npz", ids, features)
        frame, diagnostics = project_table(ids, features, settings)
        result = {
            "directory": sub.name,
            "source": fingerprint_file(embeddings[name]),
            "diagnostics": diagnostics,
            "dimensions": features.shape[1],
            "feature_bytes": features.nbytes,
        }
        metadata = Path(embeddings[name]).with_suffix(".metadata.json")
        if metadata.exists():
            result["extraction"] = json.loads(metadata.read_text())
        labels = join_labels(ids, settings)
        if settings.get("annotations") and settings.get("classify", True):
            from morphofeatures.data.contracts import EmbeddingTable

            result["classification"] = classify_embedding(
                EmbeddingTable(ids, features), sub, settings
            )
            result["evaluation"] = result["classification"]["fold_metrics"]
            labels = pd.read_csv(sub / "object_labels.tsv", sep="\t", dtype={"label_id": np.int64})
            if index == 0:
                (destination / "splits.tsv").write_bytes((sub / "splits.tsv").read_bytes())
                report["excluded_from_evaluation"] = result["classification"]["excluded_object_ids"]
                report["interpretation"] = (
                    "Matched representations use identical held-out folds. Projections remain exploratory."
                )
        frame = frame.merge(labels, on="label_id", how="left", validate="one_to_one")
        result["class_colors"] = frame.attrs["class_colors"] = label_palette(labels)
        labels.merge(frame[["label_id", "cluster"]], on="label_id", how="left").to_csv(
            sub / "object_labels.tsv", sep="\t", index=False
        )
        frame.to_csv(sub / "coordinates.tsv", sep="\t", index=False)
        save_figure(frame, sub / "projection.svg", name, settings)
        frame["representation"] = name
        frames.append(frame)
        report["representations"][name] = result
    pd.concat(frames).to_csv(destination / "coordinates.tsv", sep="\t", index=False)
    if settings.get("annotations"):
        report["annotation_source"] = fingerprint_file(settings["annotations"])
    lines = [
        "# Representation comparison",
        "",
        report["interpretation"],
        "",
        f"Matched objects: {len(ids)}. Objects without evaluation labels: {len(report['excluded_from_evaluation'])}.",
        "",
        "The same folds and one fixed downstream parameter set are used for every representation. "
        "Scaling and optional evaluation PCA are fitted on training folds only. "
        "Representation pretraining exposure must be assessed separately.",
        "",
        "| Representation | Dimensions | Extraction seconds | Training cost |",
        "|---|---:|---:|---|",
    ]
    for name, result in report["representations"].items():
        extraction = result.get("extraction", {})
        lines.append(
            f"| {str(name).replace('|', '/')} | {result['dimensions']} | {extraction.get('extraction_seconds', 'unavailable')} | {str(extraction.get('training_cost', 'unavailable')).replace('|', '/')} |"
        )
        if result.get("evaluation"):
            scores = pd.DataFrame(result["evaluation"])
            result["mean_metrics"] = {
                column: float(scores[column].mean())
                for column in scores
                if column not in {"fold", "train_objects", "test_objects", "k"}
            }
            result["std_metrics"] = {
                column: float(scores[column].std(ddof=0)) for column in result["mean_metrics"]
            }
    lines.extend(
        [
            "",
            "Full per-fold metrics, means, deviations, exclusions, settings, and provenance: comparison.json. "
            "Evaluation splits: splits.tsv when annotations are supplied. Figures and matched embeddings: representation-* directories.",
            "",
            "Protocol: " + json.dumps(report["protocol"], sort_keys=True),
        ]
    )
    (destination / "report.md").write_text("\n".join(lines) + "\n")
    path = write_json_atomic(destination / "comparison.json", report)
    package_results(destination)
    return path


def nearest_neighbors(embedding, label_id, k=8, normalization="standardize"):
    from sklearn.neighbors import NearestNeighbors

    table = load_embeddings(Path(embedding))
    selected = np.flatnonzero(table.label_ids == label_id)
    if not len(selected):
        raise ValueError(f"Object {label_id} is absent from this embedding")
    features = transformed(table.features, normalization)
    distances, indices = (
        NearestNeighbors(n_neighbors=min(k + 1, len(features)))
        .fit(features)
        .kneighbors(features[selected])
    )
    return (
        pd.DataFrame({"label_id": table.label_ids[indices[0]], "distance": distances[0]})
        .query("label_id != @label_id")
        .head(k)
    )
