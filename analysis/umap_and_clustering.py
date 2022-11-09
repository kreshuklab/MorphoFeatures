import argparse
import os
import pickle
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


big_merges_ids = [19150, 21994, 19465, 27204, 18324, 28660, 12279, 20136,
                  13911, 18402, 27925, 17365, 28685, 23288, 19856, 25523,
                  16801, 18146, 22422, 28158, 28058, 30556, 27641, 17343, 18437]


def get_data(emb_file, label_file):
    all_embed = np.load(emb_file)
    all_ids = all_embed[:, 0]
    all_embed = all_embed[:, 1:]
    label_df = pd.read_csv(label_file, sep='\t')
    to_select = (~label_df['label_id'].isin(big_merges_ids)) & (label_df['is_corr'])
    selected_ids = np.array(label_df['label_id'][to_select])
    type_names = list(label_df.cell_label.unique())
    lbl2id = {lbl: idx for idx, lbl in enumerate(type_names)}
    cell_types = label_df[to_select].cell_label.map(lbl2id)
    selected_embed = all_embed[[np.where(all_ids == i)[0][0] for i in selected_ids]]
    scaled_data = StandardScaler().fit_transform(selected_embed)
    return scaled_data, cell_types, type_names, selected_ids


def get_umap(emb, neib=15, metric='euclidean', min_dist=0.0, n_components=2):
    fit_umap = umap.UMAP(n_neighbors=neib, metric=metric,
                         min_dist=min_dist, n_components=n_components)
    return fit_umap.fit_transform(emb)


def plot_types(emb, cell_types, type_names):
    colors = sns.color_palette("Set1")
    colors = [colors[i] for i in [4, 1, 3, 0, 2, 5, 6]]
    for tp in np.unique(cell_types):
        tp_data = emb[cell_types == tp]
        if tp == 0:
            plt.scatter(tp_data[:, 0], tp_data[:, 1], color='grey', alpha=0.2, label=type_names[tp])
        else:
            plt.scatter(tp_data[:, 0], tp_data[:, 1], color=colors[tp-1], label=type_names[tp])
    plt.legend(loc='upper left', fontsize=15)
    plt.show()


def umap_leiden_clust(emb, nn=20, res=0.004):
    fit_umap = umap.UMAP(n_neighbors=nn, metric='euclidean', min_dist=0.0, n_components=2)
    _ = fit_umap.fit_transform(emb)
    net_graph = nx.from_scipy_sparse_matrix(fit_umap.graph_)
    G = ig.Graph.from_networkx(net_graph)
    part = leidenalg.find_partition(G, leidenalg.CPMVertexPartition, resolution_parameter=res)
    cl_labels = np.array(part.membership)
    analyze(cl_labels, types, ids, neighbors_dict)
    show_types_in_clusters(cl_labels, types, type_names)
    plt.scatter(umap_emb[:, 0], umap_emb[:, 1], c=cl_labels, cmap='Spectral')
    plt.show()
    return cl_labels


def show_types_in_clusters(cl_lbls, cell_types, type_names):
    clusters = np.unique(cl_lbls)
    type_distr = np.zeros((np.max(clusters) + 1, len(type_names)))
    for clust in clusters:
        clust_types = cell_types[cl_lbls == clust]
        un_types, type_counts = np.unique(clust_types, return_counts=True)
        type_distr[clust][un_types] = type_counts
    type_distr_df = pd.DataFrame(data=type_distr[:, 1:].astype(int), columns=type_names[1:])
    print(type_distr_df)


def analyze(cl_lbls, cell_types, cell_ids, nbrs_dict):
    print('Resulted in {} clusters'.format(len(np.unique(cl_lbls))))
    labl_true, labl_pred = cell_types[cell_types != 0], cl_lbls[cell_types != 0]
    homogen = metrics.homogeneity_score(labl_true, labl_pred)
    print('Homogeneity score {:0.3f}'.format(homogen))
    lbl_dict = {idx: lbl for idx, lbl in zip(cell_ids, cl_lbls)}
    in_same_clust = []
    for idx in cell_ids:
        lbl = lbl_dict[idx]
        if lbl == -1: continue
        if idx not in nbrs_dict: continue
        nbr_ids = nbrs_dict[idx]
        nbr_ids = [n for n in nbr_ids if n in cell_ids]
        in_same_clust.append(np.any([lbl_dict[n] == lbl for n in nbr_ids]))
    bil_mean = np.mean(in_same_clust)
    print('Bilateral score {:0.3f}'.format(bil_mean))
    return np.mean([bil_mean, homogen])


def plot_labels_separately(cl_lbls):
    for i in np.unique(cl_lbls):
        print(i)
        plt.scatter(umap_emb[:, 0], umap_emb[:, 1],
                    c=(0.5, 0.5, 0.5), s=0.1, alpha=0.5)
        plt.scatter(umap_emb[cl_lbls == i, 0], umap_emb[cl_lbls == i, 1],
                    c='red', s=10)
        plt.show()


def save_labels(ids, cl_lbls, u_emb, out_name):
    labels_df = pd.DataFrame({'label_id': ids, 'cluster': cl_lbls, 'bool': np.ones_like(ids)})
    df2save = labels_df.pivot(index='label_id', columns='cluster')['bool'].fillna(0)
    df2save['cluster'] = cl_lbls
    df2save['umap_1'], df2save['umap_2'] = u_emb[:, 0], u_emb[:, 1]
    df2save.to_csv(out_name, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot a umap, cluster the embedding')
    parser.add_argument('embedding_file', type=str,
                        help='path to the embedding file')
    parser.add_argument('--save_path', type=str, default=None,
                        help='a path to save umap and clustering')
    parser.add_argument('--plot_labels_separately', type=int, default=0,
                        help='plot each label separately on the umap')
    parser.add_argument('--umap_nn', type=int, default=15,
                        help='number of nearest neighbors for umap')
    parser.add_argument('--clust_nn', type=int, default=20,
                        help='number of nearest neighbors for clustering')
    parser.add_argument('--clust_res', type=float, default=0.004,
                        help='resolution for clustering')
    args = parser.parse_args()

    label_file = 'data/types_and_intensity_corr.tsv'
    dict_file = 'data/bilateral_neighbors.pkl'
    with open(dict_file, 'rb') as f:
        neighbors_dict = pickle.load(f)

    embedding, types, type_names, ids = get_data(args.embedding_file, label_file)

    print('Plotting umap')
    umap_emb = get_umap(embedding, neib=args.umap_nn)
    plot_types(umap_emb, types, type_names)

    print('Clustering')
    labels = umap_leiden_clust(embedding, nn=args.clust_nn, res=args.clust_res)
    if args.plot_labels_separately:
        plot_labels_separately(labels)

    if args.save_path is not None:
        save_labels(ids, labels, umap_emb, args.save_path)
