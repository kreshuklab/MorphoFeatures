import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_umap_and_ids(cluster_file_name):
    labels_matrix = np.array(pd.read_csv(cluster_file_name, sep='\t'))
    cell_ids = labels_matrix[:, 0]
    u_emb = labels_matrix[:, -2:]
    return cell_ids.astype('int'), u_emb


def get_genes(gene_file, chosen_ids):
    genes_df = pd.read_csv(gene_file, sep='\t')
    genes = genes_df[genes_df['label_id'].isin(chosen_ids)]
    return genes


def plot_gene(emb, genes, gene_name, save_dir=False):
    plt.scatter(emb[:, 0], emb[:, 1], s=5, c=1-genes[gene_name],
                alpha=1, cmap='autumn', vmin=0, vmax=1)
    plt.title(gene_name)
    figure = plt.gcf()
    figure.set_size_inches(16, 12)
    if save_dir:
        plt.savefig(os.path.join(save_dir, '{}.png'.format(gene_name)))
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recluster a single label with more resolution')
    parser.add_argument('clustering_file', type=str,
                        help='path to the clustering file')
    parser.add_argument('--save_path', type=str, default='',
                        help='a path (folder) to save gene plots')
    parser.add_argument('--one_gene', type=str, default='',
                        help='the name of one gene to plot (otherwise plots all)')
    args = parser.parse_args()

    ids, umap_emb = load_umap_and_ids(args.clustering_file)

    genes_file = 'data/gene_expression.tsv'
    gene_table = get_genes(genes_file, ids)

    if args.one_gene:
        plot_gene(umap_emb, gene_table, args.one_gene, args.save_path)
    else:
        for g in list(gene_table):
            plot_gene(umap_emb, gene_table, g, args.save_path)
