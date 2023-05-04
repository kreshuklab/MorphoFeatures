import argparse
import pickle
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import umap
from umap_and_clustering import get_data, analyze, save_labels


def recluster_label(curr_labels, lbl2recluster, res=0.05, nn=20):
    is_lbl = curr_labels == lbl2recluster
    fit_umap = umap.UMAP(n_neighbors=nn, metric='euclidean', min_dist=0.0, n_components=2)
    _ = fit_umap.fit_transform(embedding[is_lbl])
    net_graph = nx.from_scipy_sparse_matrix(fit_umap.graph_)
    G = ig.Graph.from_networkx(net_graph)
    part = leidenalg.find_partition(G, leidenalg.CPMVertexPartition, resolution_parameter=res)
    part_labels = np.array(part.membership) + 1
    plot_labels = np.zeros_like(curr_labels)
    plot_labels[is_lbl] = part_labels
    plt.scatter(umap_emb[:, 0], umap_emb[:, 1], c=plot_labels, cmap='tab10')
    plt.show()
    upd_labels = np.copy(curr_labels)
    upd_labels[is_lbl] = part_labels + np.max(curr_labels)
    print('Before')
    analyze(curr_labels, types, ids, neighbors_dict)
    print('After')
    analyze(upd_labels, types, ids, neighbors_dict)
    return upd_labels


def load_cl_labels(cluster_file_name):
    labels_matrix = np.array(pd.read_csv(cluster_file_name, sep='\t'))
    labels = labels_matrix[:, -3]
    u_emb = labels_matrix[:, -2:]
    return labels, u_emb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recluster a single label with more resolution')
    parser.add_argument('embedding_file', type=str,
                        help='path to the embedding file')
    parser.add_argument('clustering_file', type=str,
                        help='path to the clustering file')
    parser.add_argument('label', type=int,
                        help='label to recluster')
    parser.add_argument('--save_path', type=str, default=None,
                        help='a path to save umap and clustering')
    parser.add_argument('--clust_nn', type=int, default=20,
                        help='number of nearest neighbors for clustering')
    parser.add_argument('--clust_res', type=float, default=0.05,
                        help='resolution for clustering')
    args = parser.parse_args()

    dict_file = 'data/bilateral_neighbors.pkl'
    with open(dict_file, 'rb') as f:
        neighbors_dict = pickle.load(f)

    label_file = 'data/types_and_intensity_corr.tsv'
    embedding, types, type_names, ids = get_data(args.embedding_file, label_file)

    labels, umap_emb = load_cl_labels(args.clustering_file)

    new_labels = recluster_label(labels, args.label, nn=args.clust_nn, res=args.clust_res)

    if args.save_path is not None:
        save_labels(ids, new_labels, umap_emb, args.save_path)
