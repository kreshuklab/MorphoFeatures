"""Cross-modal retrieval and predictive probes on genuinely held-out objects."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.preprocessing import normalize


def column_correlations(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    denominator = np.sqrt(np.sum(a * a, 0) * np.sum(b * b, 0))
    return np.divide(
        np.sum(a * b, 0), denominator, out=np.full(a.shape[1], np.nan), where=denominator > 1e-12
    )


def association_matrix(a, b):
    """Spearman correlations; undefined constant columns remain NaN."""
    a, b = rankdata(a, axis=0), rankdata(b, axis=0)
    a, b = a - a.mean(0), b - b.mean(0)
    denominator = np.sqrt(np.sum(a * a, 0)[:, None] * np.sum(b * b, 0)[None, :])
    return np.divide(
        a.T @ b, denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0
    )


def retrieval_metrics(morphology, expression, top_k=10):
    """Same-cell retrieval, both directions; analytical random ordering within ties.

    Search populations must be identical across compared methods. Exact-cell
    retrieval is strict: different cells with equivalent binary profiles remain
    different candidates. Uniform chance = min(k,n)/n.
    """
    if morphology.shape != expression.shape or len(morphology) < 2:
        raise ValueError("Aligned (same cells, same dimensions) matrices required")
    similarity = normalize(morphology) @ normalize(expression).T
    result = {
        "retrieval_candidates": len(similarity),
        "chance_recall_at_k": min(top_k, len(similarity)) / len(similarity),
    }
    for name, s in [("morph_to_expr", similarity), ("expr_to_morph", similarity.T)]:
        positive = np.diag(s)
        better = np.sum(s > positive[:, None] + 1e-10, axis=1)
        tied = np.sum(np.abs(s - positive[:, None]) <= 1e-10, axis=1)
        rank = 1 + better + (tied - 1) / 2
        result[f"{name}_recall_at_k"] = float(np.mean(np.clip((top_k - better) / tied, 0, 1)))
        result[f"{name}_median_rank"] = float(np.median(rank))
        result[f"{name}_median_rank_fraction"] = float(np.median(rank / len(s)))
        result[f"{name}_mean_tie_size"] = float(tied.mean())
    return result


class GeneDecoder:
    """The same ridge-penalized logistic probe for every morphology representation."""

    def __init__(self, c=1.0, min_train_class=3):
        self.c, self.min_train_class = c, min_train_class

    def fit(self, x, y):
        self.prior_ = (y.sum(0) + 1) / (len(y) + 2)
        self.estimators_ = []
        for j in range(y.shape[1]):
            if min(y[:, j].sum(), len(y) - y[:, j].sum()) < self.min_train_class:
                self.estimators_.append(None)
            else:
                self.estimators_.append(LogisticRegression(C=self.c, max_iter=1000).fit(x, y[:, j]))
        return self

    def predict_proba(self, x):
        p = np.broadcast_to(self.prior_, (len(x), len(self.prior_))).copy()
        for j, estimator in enumerate(self.estimators_):
            if estimator is not None:
                p[:, j] = estimator.predict_proba(x)[:, 1]
        return p


def gene_scores(truth, probabilities, train_prior, genes, min_test_class=10):
    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    prior = np.clip(train_prior, 1e-6, 1 - 1e-6)
    loss = -(truth * np.log(p) + (1 - truth) * np.log1p(-p)).mean(0)
    prior_loss = -(truth * np.log(prior) + (1 - truth) * np.log1p(-prior)).mean(0)
    rows = []
    for j, gene in enumerate(genes):
        positives = int(truth[:, j].sum())
        valid = 0 < positives < len(truth)
        rows.append(
            {
                "gene": str(gene),
                "positive_test_cells": positives,
                "test_prevalence": positives / len(truth),
                "supported": min(positives, len(truth) - positives) >= min_test_class,
                "average_precision": average_precision_score(truth[:, j], p[:, j])
                if valid
                else np.nan,
                "roc_auc": roc_auc_score(truth[:, j], p[:, j]) if valid else np.nan,
                "log_loss": loss[j],
                "log_loss_gain_bits": (prior_loss[j] - loss[j]) / np.log(2),
                "brier": np.mean((truth[:, j] - p[:, j]) ** 2),
            }
        )
    frame = pd.DataFrame(rows)
    selected = frame.loc[frame.supported]
    summary = {
        "supported_test_genes": len(selected),
        "macro_AP": selected.average_precision.mean(),
        "macro_AP_minus_prevalence": (selected.average_precision - selected.test_prevalence).mean(),
        "macro_ROC_AUC": selected.roc_auc.mean(),
        "mean_log_loss_gain_bits": frame.log_loss_gain_bits.mean(),
        "mean_brier": frame.brier.mean(),
    }
    return frame, summary


def probe_expression_to_morphology(z_train, z_test, morphology_train, morphology_test):
    model = Ridge(alpha=len(z_train) * 0.01).fit(z_train, morphology_train)
    prediction = model.predict(z_test)
    mse = float(np.mean((prediction - morphology_test) ** 2))
    prior_mse = float(np.mean((morphology_train.mean(0) - morphology_test) ** 2))
    return model, {
        "expr_to_morph_R2": r2_score(morphology_test, prediction),
        "expr_to_morph_MSE_gain_vs_train_mean": 1 - mse / prior_mse,
    }
