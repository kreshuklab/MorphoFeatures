"""Read saved results; make pedagogical figures without refitting shared models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_figure(fig, directory, name):
    for extension in ("png", "svg"):
        fig.savefig(directory / f"{name}.{extension}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_figures(output):
    output = Path(output)
    config = json.loads((output / "provenance.json").read_text())["config"]
    settings = config["analysis"]
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "font.size": 9,
        }
    )
    scores = pd.read_csv(output / "tables/scores.tsv", sep="\t")
    for morph in config["inputs"]["morphology"]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
        names = ["morphology_direct", *settings["methods"]]
        for scheme, color, shift in zip(settings["schemes"], ["#2166ac", "#d6604d"], [-0.12, 0.12]):
            subset = (
                scores.loc[(scores.morphology == morph) & (scores.scheme == scheme)]
                .set_index("method")
                .reindex(names)
            )
            for ax, metric, title in zip(
                axes,
                ["morph_to_expr_recall_at_k", "macro_AP", "mean_log_loss_gain_bits"],
                [
                    f"Find matched expression among candidates: recall@{settings['retrieval_k']}",
                    "Predict genes from morphology: average precision",
                    "Predict genes: log-loss gain over prior (bits)",
                ],
            ):
                ax.scatter(
                    subset[metric], np.arange(len(names)) + shift, c=color, s=35, label=scheme
                )
                ax.set_yticks(range(len(names)), names)
                ax.set_title(title, fontsize=9)
                ax.grid(axis="x", alpha=0.2)
        axes[0].axvline(
            scores.chance_recall_at_k.dropna().median(),
            color="gray",
            linestyle="--",
            label="chance (median)",
        )
        axes[2].axvline(0, color="gray", linestyle="--")
        axes[0].legend(fontsize=7, loc="best")
        fig.suptitle(f"{morph}: fixed held-out cells; higher is better. No significance claim.")
        save_figure(fig, directory, f"{morph}_heldout_performance")
        for scheme in settings["schemes"]:
            methods = [m for m in settings["methods"] if m != "rcca_shuffled"]
            fig, axes = plt.subplots(
                len(methods), 2, figsize=(10, 3 * len(methods)), squeeze=False, layout="constrained"
            )
            for row, method in enumerate(methods):
                path = output / "models" / morph / scheme / method / "heldout_embeddings.npz"
                with np.load(path, allow_pickle=False) as f:
                    n = len(f["label_ids"])
                    ix = np.sort(
                        np.random.default_rng(settings["seed"]).choice(
                            n, min(settings["plot_cells"], n), replace=False
                        )
                    )
                    xm, xe, burden = (
                        f["morphology_xy"][ix],
                        f["expression_xy"][ix],
                        f["positive_calls"][ix],
                    )
                limits = np.vstack([xm, xe])
                for ax, xy, view in zip(
                    axes[row], [xm, xe], ["Morphology alone", "Expression alone"]
                ):
                    artist = ax.scatter(
                        *xy.T, c=burden, cmap="viridis", s=7, linewidths=0, rasterized=True
                    )
                    ax.set(
                        xlim=(limits[:, 0].min(), limits[:, 0].max()),
                        ylim=(limits[:, 1].min(), limits[:, 1].max()),
                    )
                    ax.set_title(f"{method}: {view}")
                    ax.set_xticks([])
                    ax.set_yticks([])
                fig.colorbar(
                    artist, ax=list(axes[row]), shrink=0.7, label="Positive spatial gene calls"
                )
            fig.suptitle(
                f"{morph} / {scheme}: same held-out cells; one training-fitted PCA display per row\nColor is observed expression for inspection only; morphology coordinates do not use it.",
                fontsize=10,
            )
            save_figure(fig, directory, f"{morph}_{scheme}_shared_pca")
            if settings.get("run_umap", False):
                import umap

                shown = [m for m in settings["umap_methods"] if m in methods]
                fig, axes = plt.subplots(
                    len(shown), 2, figsize=(10, 3 * len(shown)), squeeze=False, layout="constrained"
                )
                for row, method in enumerate(shown):
                    path = output / "models" / morph / scheme / method / "heldout_embeddings.npz"
                    with np.load(path, allow_pickle=False) as f:
                        n = len(f["label_ids"])
                        ix = np.sort(
                            np.random.default_rng(settings["seed"]).choice(
                                n, min(settings["plot_cells"], n), replace=False
                            )
                        )
                        joined = np.vstack(
                            [f["morphology_features"][ix], f["expression_features"][ix]]
                        )
                        xy = umap.UMAP(
                            n_neighbors=min(30, len(joined) - 1),
                            min_dist=0.2,
                            metric="cosine",
                            init="random",
                            random_state=settings["seed"],
                            n_jobs=1,
                        ).fit_transform(joined)
                        count = len(ix)
                        burden = f["positive_calls"][ix]
                        np.savez_compressed(
                            path.parent / "display_umap.npz",
                            label_ids=f["label_ids"][ix],
                            nucleus_ids=f["nucleus_ids"][ix],
                            morphology_xy=xy[:count],
                            expression_xy=xy[count:],
                        )
                    for ax, coords, view in zip(
                        axes[row],
                        [xy[:count], xy[count:]],
                        ["Morphology alone", "Expression alone"],
                    ):
                        artist = ax.scatter(
                            *coords.T, c=burden, cmap="viridis", s=7, linewidths=0, rasterized=True
                        )
                        ax.set(
                            xlim=(xy[:, 0].min(), xy[:, 0].max()),
                            ylim=(xy[:, 1].min(), xy[:, 1].max()),
                        )
                        ax.set_title(f"{method}: {view}")
                        ax.set_xticks([])
                        ax.set_yticks([])
                    fig.colorbar(
                        artist, ax=list(axes[row]), shrink=0.7, label="Positive spatial gene calls"
                    )
                fig.suptitle(
                    f"{morph} / {scheme}: one joint UMAP per row, split into two views\nUMAP sees held-out points for display only; no validation score uses these coordinates.",
                    fontsize=10,
                )
                save_figure(fig, directory, f"{morph}_{scheme}_shared_umap")
