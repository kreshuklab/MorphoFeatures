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
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler, normalize

from morphofeatures.analysis.projection import cluster_embeddings, compute_umap
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


def transformed(features, normalization):
    if normalization == "standardize":
        return StandardScaler().fit_transform(features)
    if normalization == "l2":
        return normalize(features)
    if normalization == "none":
        return np.asarray(features)
    raise ValueError("normalization must be standardize, l2, or none")


def project_table(ids, features, settings):
    if len(ids) < 3 or features.shape[1] < 2:
        raise ValueError("Analysis requires at least three objects and two features")
    seed = int(settings.get("seed", 42))
    data = transformed(features, settings.get("normalization", "standardize"))
    projection = PCA(n_components=2, random_state=seed).fit_transform(data)
    clusters = cluster_embeddings(
        data,
        method=settings.get("cluster_method", "kmeans"),
        n_clusters=int(settings.get("clusters", 8)),
        n_neighbors=int(settings.get("neighbors", 15)),
        resolution=float(settings.get("resolution", 0.004)),
        seed=seed,
    )
    frame = pd.DataFrame(
        {"label_id": ids, "cluster": clusters, "pca_1": projection[:, 0], "pca_2": projection[:, 1]}
    )
    if settings.get("umap", True):
        if len(ids) < 4:
            raise ValueError(
                "UMAP requires at least four objects; disable umap for smaller subsets"
            )
        reduced = compute_umap(
            data,
            n_neighbors=int(settings.get("neighbors", 15)),
            min_dist=float(settings.get("min_dist", 0.1)),
            seed=seed,
            n_epochs=settings.get("umap_epochs"),
        )
        frame["umap_1"], frame["umap_2"] = reduced.T
    diagnostics = {"clusters_observed": int(len(np.unique(clusters)))}
    if 1 < len(np.unique(clusters)) < len(ids):
        try:
            diagnostics["silhouette"] = float(
                silhouette_score(data, clusters, sample_size=min(2000, len(ids)), random_state=seed)
            )
        except ValueError as error:
            diagnostics["silhouette_unavailable"] = str(error)
    return frame, diagnostics


def save_figure(frame, destination, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["pca"] + (["umap"] if "umap_1" in frame else [])
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 5), squeeze=False)
    for axis, method in zip(axes[0], methods):
        axis.scatter(
            frame[f"{method}_1"], frame[f"{method}_2"], c=frame.cluster, s=12, cmap="tab20"
        )
        axis.set(xlabel=method.upper() + " 1", ylabel=method.upper() + " 2", title=title)
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)


def analyze(embedding, destination, settings):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    table = load_embeddings(embedding).sorted()
    frame, diagnostics = project_table(table.label_ids, table.features, settings)
    frame.to_csv(destination / "coordinates.tsv", sep="\t", index=False)
    export_embeddings(destination / "embeddings.npz", table.label_ids, table.features)
    save_figure(frame, destination / "projection.svg", "Exploratory morphology")
    result = {
        "schema": "morphofeatures.analysis.v1",
        "settings": settings,
        "source": fingerprint_file(embedding),
        "diagnostics": diagnostics,
        "coordinates": "coordinates.tsv",
        "embedding": "embeddings.npz",
        "interpretation": "Projection and clustering are exploratory; biological predictive value is untested.",
    }
    metadata = Path(embedding).with_suffix(".metadata.json")
    if metadata.exists():
        result["extraction"] = json.loads(metadata.read_text())
    path = write_json_atomic(destination / "analysis.json", result)
    package_results(destination)
    return path


def evaluation_splits(ids, annotations, settings):
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    column = settings.get("label_column", "label")
    group_column = settings.get("group_column")
    if column not in annotations:
        raise ValueError(f"Annotations require label column {column!r}")
    matched = annotations.set_index("label_id").reindex(ids)
    if matched[column].isna().any():
        raise ValueError("Evaluation requires labels for every evaluated object")
    y = matched[column].astype(str).to_numpy()
    folds = int(settings.get("folds", 5))
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


def evaluate_features(features, y, splits, settings):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

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
        for name, estimator in {
            "knn": KNeighborsClassifier(n_neighbors=k),
            "linear": LogisticRegression(
                C=float(settings.get("linear_c", 1.0)),
                max_iter=int(settings.get("max_iter", 2000)),
                random_state=int(settings.get("seed", 42)),
                class_weight="balanced",
            ),
        }.items():
            predicted = estimator.fit(x_train, y[train]).predict(x_test)
            row[name + "_accuracy"] = float(accuracy_score(y[test], predicted))
            row[name + "_balanced_accuracy"] = float(balanced_accuracy_score(y[test], predicted))
        model = KMeans(
            n_clusters=min(int(settings.get("clusters", 8)), len(train)),
            n_init=10,
            random_state=int(settings.get("seed", 42)),
        ).fit(x_train)
        row["test_cluster_ari"] = float(adjusted_rand_score(y[test], model.predict(x_test)))
        scores.append(row)
    return scores


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
    y = groups = splits = None
    evaluation_indices = np.arange(len(ids))
    if settings.get("annotations"):
        path = Path(settings["annotations"])
        annotations = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
        if "label_id" not in annotations or annotations.label_id.duplicated().any():
            raise ValueError("Annotations need unique label_id values")
        label_column = settings.get("label_column", "label")
        if label_column not in annotations:
            raise ValueError(f"Missing annotation column: {label_column}")
        annotated = set(annotations.loc[annotations[label_column].notna(), "label_id"])
        evaluation_indices = np.asarray([i for i, value in enumerate(ids) if value in annotated])
        evaluation_ids = ids[evaluation_indices]
        annotation_excluded = sorted(int(i) for i in set(ids) - annotated)
        y, groups, splits = evaluation_splits(evaluation_ids, annotations, settings)
        split_rows = []
        for fold, (train, test) in enumerate(splits):
            for role, indices in (("train", train), ("test", test)):
                split_rows.extend(
                    {
                        "fold": fold,
                        "role": role,
                        "label_id": int(evaluation_ids[i]),
                        "label": y[i],
                        "group": groups[i] if groups is not None else None,
                    }
                    for i in indices
                )
        pd.DataFrame(split_rows).to_csv(destination / "splits.tsv", sep="\t", index=False)
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
        frame.to_csv(sub / "coordinates.tsv", sep="\t", index=False)
        save_figure(frame, sub / "projection.svg", name)
        frame["representation"] = name
        frames.append(frame)
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
        if splits:
            result["evaluation"] = evaluate_features(
                features[evaluation_indices], y, splits, settings
            )
        report["representations"][name] = result
    pd.concat(frames).to_csv(destination / "coordinates.tsv", sep="\t", index=False)
    if settings.get("annotations"):
        report["annotation_source"] = fingerprint_file(settings["annotations"])
    lines = [
        "# Representation comparison",
        "",
        report["interpretation"],
        "",
        f"Matched objects: {len(ids)}. Objects without evaluation labels: {len(annotation_excluded)}.",
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
