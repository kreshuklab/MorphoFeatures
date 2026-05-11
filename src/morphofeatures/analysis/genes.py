"""Gene-expression plotting utilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_umap_and_ids(cluster_file_name: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load cell IDs and UMAP coordinates from a clustering TSV."""

    labels_matrix = np.asarray(pd.read_csv(cluster_file_name, sep="\t"))
    cell_ids = labels_matrix[:, 0].astype(int)
    umap_embedding = labels_matrix[:, -2:]
    return cell_ids, umap_embedding


def get_genes(gene_file: str | Path, chosen_ids: np.ndarray) -> pd.DataFrame:
    """Load gene-expression rows for selected cell IDs."""

    genes_frame = pd.read_csv(gene_file, sep="\t")
    return genes_frame[genes_frame["label_id"].isin(chosen_ids)]


def plot_gene(embedding: np.ndarray, genes: pd.DataFrame, gene_name: str, save_dir: str | Path | None = None) -> None:
    """Plot one gene expression vector on a UMAP embedding."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Gene plotting requires matplotlib.") from exc

    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, c=1 - genes[gene_name], alpha=1, cmap="autumn", vmin=0, vmax=1)
    plt.title(gene_name)
    figure = plt.gcf()
    figure.set_size_inches(16, 12)
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(save_dir) / f"{gene_name}.png")
        plt.close()
    else:
        plt.show()


def main(argv: list[str] | None = None) -> None:
    """Run the gene plotting CLI."""

    parser = argparse.ArgumentParser(description="Plot gene expression on a UMAP.")
    parser.add_argument("clustering_file", type=Path)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--one-gene", type=str, default="")
    parser.add_argument("--genes-file", type=Path, default=Path("analysis/data/gene_expression.tsv"))
    args = parser.parse_args(argv)

    ids, umap_embedding = load_umap_and_ids(args.clustering_file)
    gene_table = get_genes(args.genes_file, ids)
    if args.one_gene:
        plot_gene(umap_embedding, gene_table, args.one_gene, args.save_path)
    else:
        for gene_name in list(gene_table):
            if gene_name != "label_id":
                plot_gene(umap_embedding, gene_table, gene_name, args.save_path)


if __name__ == "__main__":
    main()
