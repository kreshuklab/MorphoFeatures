"""Small paired models; every transform can use one modality independently.

MOFA2 has no native inductive encoder. MOFARidge therefore explicitly trains
two ridge heads to predict the training MOFA factors. These are approximations,
not MOFA posterior inference, and the benchmark names the composite method.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSCanonical
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, normalize


class MorphologyPreprocessor:
    def __init__(self, components=40, seed=42):
        self.components, self.seed = components, seed

    def fit(self, x):
        x = np.asarray(x, float)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite morphology")
        self.keep_ = np.ptp(x, axis=0) > 0
        if not self.keep_.any():
            raise ValueError("No variable morphology features in training")
        self.mean_ = x[:, self.keep_].mean(0)
        a = normalize(x[:, self.keep_] - self.mean_)
        self.pca_ = PCA(
            n_components=min(self.components, len(x) - 1, a.shape[1]),
            svd_solver="randomized",
            random_state=self.seed,
        ).fit(a)
        self.pc_keep_ = self.pca_.explained_variance_ > max(
            1e-12, self.pca_.explained_variance_.max() * 1e-8
        )
        self.scaler_ = StandardScaler().fit(self.pca_.transform(a)[:, self.pc_keep_])
        return self

    def transform(self, x):
        a = normalize(np.asarray(x)[:, self.keep_] - self.mean_)
        return self.scaler_.transform(self.pca_.transform(a)[:, self.pc_keep_])


class ExpressionFilter:
    def fit(self, y):
        if not np.isin(y, [0, 1]).all():
            raise ValueError("Expected binary expression")
        self.keep_ = np.ptp(y, axis=0) > 0
        if not self.keep_.any():
            raise ValueError("No variable expression features in training")
        return self

    def transform(self, y):
        return np.asarray(y, float)[:, self.keep_]


def inverse_sqrt(covariance, ridge):
    if ridge <= 0:
        raise ValueError("CCA ridge must be strictly positive")
    scale = np.trace(covariance) / len(covariance)
    eig, vectors = np.linalg.eigh(covariance + ridge * max(scale, 1e-12) * np.eye(len(covariance)))
    return (vectors / np.sqrt(np.maximum(eig, 1e-14))) @ vectors.T


class LinearAlignment:
    """Ridge CCA or PLSCanonical, with training-only score scaling per modality."""

    def __init__(self, kind="rcca", factors=10, ridge=0.1):
        self.kind, self.factors, self.ridge = kind, factors, ridge

    def fit(self, x, y, **_kwargs):
        self.x_mean_, self.y_mean_ = x.mean(0), y.mean(0)
        a, b = x - self.x_mean_, y - self.y_mean_
        rank = min(self.factors, a.shape[1], b.shape[1], len(a) - 1)
        if self.kind == "rcca":
            xx, yy, xy = a.T @ a / (len(a) - 1), b.T @ b / (len(a) - 1), a.T @ b / (len(a) - 1)
            ix, iy = inverse_sqrt(xx, self.ridge), inverse_sqrt(yy, self.ridge)
            u, self.regularized_singular_values_, vt = np.linalg.svd(
                ix @ xy @ iy, full_matrices=False
            )
            self.x_weights_, self.y_weights_ = ix @ u[:, :rank], iy @ vt.T[:, :rank]
        elif self.kind == "pls":
            model = PLSCanonical(n_components=rank, scale=False, algorithm="svd").fit(x, y)
            self.x_weights_, self.y_weights_ = model.x_rotations_, model.y_rotations_
        else:
            raise ValueError(self.kind)
        zx, zy = a @ self.x_weights_, b @ self.y_weights_
        sign = np.where(np.sum(zx * zy, axis=0) < 0, -1, 1)
        self.y_weights_ *= sign
        self.x_scale_ = np.maximum(zx.std(0), 1e-8)
        self.y_scale_ = np.maximum(zy.std(0), 1e-8)
        return self

    def transform_morphology(self, x):
        return ((x - self.x_mean_) @ self.x_weights_) / self.x_scale_

    def transform_expression(self, y):
        return ((y - self.y_mean_) @ self.y_weights_) / self.y_scale_

    def expression_loadings(self):
        return self.y_weights_ / self.y_scale_


class MOFARidge:
    """Native MOFA2 training + separate, frozen ridge projections into its factors.

    Binary input remains binary. The Gaussian morphology input is already
    training-centered/scaled. MOFA weights/factors and head fit diagnostics are
    exported. A common factor scaling preserves the MOFA coordinate system.
    """

    def __init__(self, factors=10, iterations=300, head_ridge=0.01, seed=42):
        self.factors, self.iterations = factors, iterations
        self.head_ridge, self.seed = head_ridge, seed

    def fit(self, x, y, *, output=None, cell_ids=None, genes=None):
        try:
            from mofapy2.run.entry_point import entry_point
        except ImportError as exc:
            raise ImportError(
                "MOFA2 needs mofapy2==0.7.3; use the shared-space environment"
            ) from exc
        entry = entry_point()
        entry.set_data_options(center_groups=False, scale_views=False, use_float32=False)
        entry.set_data_matrix(
            [[x.copy()], [y.copy()]],
            likelihoods=["gaussian", "bernoulli"],
            views_names=["morphology_pcs", "binary_expression"],
            groups_names=["paired_training"],
            samples_names=[[str(i) for i in (cell_ids if cell_ids is not None else range(len(x)))]],
            features_names=[
                [f"morphology_pc_{i + 1}" for i in range(x.shape[1])],
                list(genes) if genes is not None else [f"gene_{i}" for i in range(y.shape[1])],
            ],
        )
        entry.set_model_options(factors=self.factors, spikeslab_weights=True, ard_weights=True)
        entry.set_train_options(
            iter=self.iterations,
            convergence_mode="medium",
            seed=self.seed,
            gpu_mode=False,
            verbose=False,
            quiet=True,
            freqELBO=5,
            dropR2=None,
            save_interrupted=False,
        )
        entry.build()
        entry.run()
        nodes = entry.model.getNodes()
        self.training_factors_ = nodes["Z"].getExpectation().copy()
        self.weights_ = [a["E"].copy() for a in nodes["W"].getExpectations()]
        self.training_stats_ = entry.model.getTrainingStats()
        if not np.isfinite(self.training_factors_).all():
            raise ValueError("Nonfinite MOFA factors; inspect the native training log")
        if output is not None:
            entry.save(str(Path(output) / "native_mofa.hdf5"), save_data=False)
        z = self.training_factors_
        self.factor_mean_, self.factor_scale_ = z.mean(0), np.maximum(z.std(0), 1e-8)
        self.x_head_ = Ridge(alpha=len(x) * self.head_ridge).fit(x, z)
        self.y_head_ = Ridge(alpha=len(y) * self.head_ridge).fit(y, z)
        return self

    def transform_morphology(self, x):
        return (self.x_head_.predict(x) - self.factor_mean_) / self.factor_scale_

    def transform_expression(self, y):
        return (self.y_head_.predict(y) - self.factor_mean_) / self.factor_scale_

    def expression_loadings(self):
        # Generative weights, distinct from the ridge encoder's coefficients.
        return self.weights_[1]


class PairedNeural:
    """Small CPU towers: paired InfoNCE + within-view reconstruction.

    Fixed epochs; no test-based early stopping. The expression decoder uses
    Bernoulli cross entropy. This is a pilot, not a pretrained multimodal model.
    """

    def __init__(self, factors=10, epochs=60, hidden=64, seed=42, threads=2):
        self.factors, self.epochs, self.hidden = factors, epochs, hidden
        self.seed, self.threads = seed, threads

    def fit(self, x, y, **_kwargs):
        import torch
        from torch import nn
        from torch.nn import functional as F

        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        self.x_net_ = nn.Sequential(
            nn.Linear(x.shape[1], self.hidden), nn.GELU(), nn.Linear(self.hidden, self.factors)
        )
        self.y_net_ = nn.Sequential(
            nn.Linear(y.shape[1], self.hidden), nn.GELU(), nn.Linear(self.hidden, self.factors)
        )
        self.x_decoder_, self.y_decoder_ = (
            nn.Linear(self.factors, x.shape[1]),
            nn.Linear(self.factors, y.shape[1]),
        )
        self.y_mean_ = y.mean(0)
        tx, ty = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        cy = ty - torch.tensor(self.y_mean_, dtype=torch.float32)
        params = [
            p
            for net in (self.x_net_, self.y_net_, self.x_decoder_, self.y_decoder_)
            for p in net.parameters()
        ]
        optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(self.seed)
        self.history_ = []
        for epoch in range(self.epochs):
            order = torch.randperm(len(x), generator=generator)
            totals = []
            for ix in order.split(256):
                if len(ix) < 2:
                    continue
                zx, zy = self.x_net_(tx[ix]), self.y_net_(cy[ix])
                similarity = F.normalize(zx, dim=1) @ F.normalize(zy, dim=1).T / 0.2
                target = torch.arange(len(ix))
                alignment = (
                    F.cross_entropy(similarity, target) + F.cross_entropy(similarity.T, target)
                ) / 2
                reconstruction = F.mse_loss(
                    self.x_decoder_(zx), tx[ix]
                ) + F.binary_cross_entropy_with_logits(self.y_decoder_(zy), ty[ix])
                loss = alignment + reconstruction
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                totals.append([loss.item(), alignment.item(), reconstruction.item()])
            self.history_.append([epoch + 1, *np.mean(totals, axis=0)])
        for net in (self.x_net_, self.y_net_, self.x_decoder_, self.y_decoder_):
            net.eval()
        return self

    def transform_morphology(self, x):
        import torch

        with torch.no_grad():
            return self.x_net_(torch.tensor(x, dtype=torch.float32)).numpy()

    def transform_expression(self, y):
        import torch

        with torch.no_grad():
            return self.y_net_(torch.tensor(y - self.y_mean_, dtype=torch.float32)).numpy()
