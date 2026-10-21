"""CPU baselines for predicting a full binary gene panel from frozen embeddings.

This is multilabel prediction: each cell can express several genes. It is not
multiclass prediction of one of the observed profiles. Inputs must contain only
morphology features, never expression or expression-derived annotations.

The accompanying ``full_expression_prediction_tutorial.ipynb`` fits preprocessing
inside each training fold, preserves cell/nucleus identities, and evaluates spatial
as well as random grouped holdouts. This module also makes saved estimators portable
between notebook sessions in the same MorphoFeatures environment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, jaccard_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from threadpoolctl import threadpool_limits


def embedding_preprocessor(n_components=50, *, seed=42):
    """Training-centered, row-L2-normalized embeddings, optionally followed by PCA.

    Fit this only on training cells. PCA is not whitened; no UMAP coordinates or
    expression-derived features enter the classifier. ``None`` retains all feature
    dimensions. Scaling individual PCs is deliberately not done for these baselines.
    """
    steps = [StandardScaler(with_std=False), Normalizer()]
    if n_components is not None:
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be a positive integer or None")
        steps.append(PCA(n_components=n_components, svd_solver="randomized", random_state=seed))
    return make_pipeline(*steps)


class BinaryExpressionPredictor(BaseEstimator):
    """Predict one positive-class probability per gene, in input column order.

    ``predict_proba`` returns an (objects, genes) array, not a list of two-column
    arrays. Random forests and Extra Trees use native multioutput trees: splits are
    shared across genes. Logistic, histogram boosting and optional XGBoost fit one
    estimator per variable gene and package all outputs together. No expression
    target is supplied as a predictor to any of the models.

    Genes having only one class in training use a Laplace-smoothed training prior;
    they remain in the output and are flagged by ``constant_targets_``. In particular,
    a globally unobserved gene cannot be learned from this data.

    Defaults are modest exploratory CPU budgets, not tuned best-performing settings.
    Set ``workers`` explicitly on shared systems. Internal boosting early stopping
    is disabled so it cannot silently create a random validation split across a
    spatial holdout. Threshold 0.5 is fixed, not optimized using test labels.
    """

    def __init__(
        self,
        kind="extra_trees",
        *,
        workers=4,
        seed=42,
        trees=128,
        tree_depth=12,
        min_leaf=12,
        boosting_iterations=60,
        boosting_leaves=7,
        logistic_c=10.0,
        threshold=0.5,
    ):
        self.kind = kind
        self.workers = workers
        self.seed = seed
        self.trees = trees
        self.tree_depth = tree_depth
        self.min_leaf = min_leaf
        self.boosting_iterations = boosting_iterations
        self.boosting_leaves = boosting_leaves
        self.logistic_c = logistic_c
        self.threshold = threshold

    def _independent_estimator(self, column):
        if self.kind == "logistic":
            return LogisticRegression(C=self.logistic_c, max_iter=2000, random_state=self.seed)
        if self.kind == "hist_gradient_boosting":
            return HistGradientBoostingClassifier(
                max_iter=self.boosting_iterations,
                learning_rate=0.1,
                max_leaf_nodes=self.boosting_leaves,
                min_samples_leaf=max(20, self.min_leaf),
                max_bins=63,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=self.seed + column,
            )
        if self.kind == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise ImportError(
                    "Optional model 'xgboost' requires the xgboost package in this Python "
                    "environment. Other full-profile models do not require it."
                ) from exc
            return XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_estimators=self.boosting_iterations,
                max_depth=3,
                learning_rate=0.1,
                reg_lambda=1.0,
                n_jobs=1,
                random_state=self.seed + column,
            )
        raise ValueError(f"Unknown independent model: {self.kind}")

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True, dtype=np.float32)
        if y.ndim != 2 or not y.shape[1] or not np.isin(y, [0, 1]).all():
            raise ValueError("Targets must be a nonempty (objects, genes) binary matrix")
        if self.kind not in {
            "logistic", "random_forest", "extra_trees", "hist_gradient_boosting", "xgboost"
        }:
            raise ValueError(f"Unknown full-profile model: {self.kind}")
        if self.workers < 1 or not 0 < self.threshold < 1:
            raise ValueError("workers must be positive and threshold must lie strictly between 0 and 1")
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = y.shape[1]
        self.train_positive_ = y.sum(axis=0).astype(np.int64)
        self.prior_ = (self.train_positive_ + 1) / (len(y) + 2)
        self.constant_targets_ = (self.train_positive_ == 0) | (self.train_positive_ == len(y))
        self.estimators_ = None
        self.forest_ = None
        with threadpool_limits(limits=1):
            if self.kind in {"random_forest", "extra_trees"}:
                constructor = (
                    RandomForestClassifier if self.kind == "random_forest" else ExtraTreesClassifier
                )
                self.forest_ = constructor(
                    n_estimators=self.trees, max_depth=self.tree_depth,
                    min_samples_leaf=self.min_leaf, max_features="sqrt",
                    n_jobs=self.workers, random_state=self.seed,
                ).fit(X, y[:, 0] if self.n_outputs_ == 1 else y)
            else:
                def fit_column(j):
                    if self.constant_targets_[j]:
                        return None
                    return self._independent_estimator(j).fit(X, y[:, j])

                self.estimators_ = Parallel(n_jobs=self.workers, prefer="threads")(
                    delayed(fit_column)(j) for j in range(self.n_outputs_)
                )
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "prior_")
        X = check_array(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} input features, got {X.shape[1]}")
        result = np.broadcast_to(self.prior_, (len(X), self.n_outputs_)).copy()
        with threadpool_limits(limits=1):
            if self.forest_ is not None:
                arrays = self.forest_.predict_proba(X)
                classes = self.forest_.classes_
                if self.n_outputs_ == 1:
                    arrays, classes = [arrays], [classes]
                for j, (p, labels) in enumerate(zip(arrays, classes)):
                    if not self.constant_targets_[j]:
                        result[:, j] = p[:, np.flatnonzero(labels == 1)[0]]
            else:
                def predict_column(j):
                    estimator = self.estimators_[j]
                    if estimator is None:
                        return result[:, j]
                    return estimator.predict_proba(X)[:, np.flatnonzero(estimator.classes_ == 1)[0]]

                result = np.column_stack(Parallel(n_jobs=self.workers, prefer="threads")(
                    delayed(predict_column)(j) for j in range(self.n_outputs_)
                ))
        return result.astype(np.float32)

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(np.uint8)


def profile_scores(y, probabilities, gene_ids, *, threshold=0.5, minimum_class_size=50):
    """Per-gene and whole-panel held-out scores; rare/constant targets remain visible.

    Macro AP/AUC exclude constant evaluation targets (their discrimination is
    undefined). ``macro_AP_supported`` additionally requires at least 50 positives
    and negatives by default, for a less unstable companion result. Counts are
    descriptive; this filter never changes model training or exported predictions.
    Micro AP, Brier, log loss, F1, Jaccard and Hamming include the entire panel.
    """
    y = np.asarray(y)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 2 or y.shape != p.shape or y.shape[1] != len(gene_ids):
        raise ValueError("Targets, probabilities and gene order have inconsistent shapes")
    if len(y) == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("Evaluation requires a nonempty binary target matrix")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite and between zero and one")
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between zero and one")
    calls = p >= threshold
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    logloss = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).mean(axis=0)
    rows = []
    for j, gene in enumerate(gene_ids):
        positives = int(y[:, j].sum())
        negatives = len(y) - positives
        variable = positives > 0 and negatives > 0
        rows.append({
            "gene_id": str(gene), "positive_cells": positives, "negative_cells": negatives,
            "prevalence": float(y[:, j].mean()), "variable": variable,
            "supported": min(positives, negatives) >= minimum_class_size,
            "average_precision": average_precision_score(y[:, j], p[:, j]) if variable else np.nan,
            "roc_auc": roc_auc_score(y[:, j], p[:, j]) if variable else np.nan,
            "brier": float(np.mean((y[:, j] - p[:, j]) ** 2)),
            "log_loss_nats": float(logloss[j]),
            "f1_at_threshold": f1_score(y[:, j], calls[:, j], zero_division=0),
            "predicted_positive_cells": int(calls[:, j].sum()),
        })
    table = pd.DataFrame(rows)
    table["AP_above_prevalence"] = table.average_precision - table.prevalence
    supported = table.loc[table.supported]
    flat_y = y.ravel()
    summary = {
        "cells": len(y), "genes": y.shape[1], "variable_genes": int(table.variable.sum()),
        "supported_genes": int(table.supported.sum()), "threshold": threshold,
        "macro_AP": float(table.average_precision.mean()),
        "macro_AP_supported": float(supported.average_precision.mean()),
        "macro_roc_auc": float(table.roc_auc.mean()),
        "micro_AP": (
            float(average_precision_score(flat_y, p.ravel()))
            if 0 < flat_y.sum() < len(flat_y) else np.nan
        ),
        "brier": float(table.brier.mean()), "log_loss_nats": float(logloss.mean()),
        "micro_F1": float(f1_score(y, calls, average="micro", zero_division=0)),
        "macro_F1": float(f1_score(y, calls, average="macro", zero_division=0)),
        "mean_cell_Jaccard": float(jaccard_score(y, calls, average="samples", zero_division=0)),
        "hamming_loss": float(np.mean(y != calls)),
        "exact_profile_accuracy": float(np.mean(np.all(y == calls, axis=1))),
        "observed_genes_per_cell": float(y.sum(axis=1).mean()),
        "predicted_genes_per_cell": float(calls.sum(axis=1).mean()),
    }
    return summary, table
