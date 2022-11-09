import argparse
import os
import pickle
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from log_regress import reorder


def calculate_distances(embed, cell_ids, cell_nbrs, cosine=False, use_one_side=False):
    normed_embed = StandardScaler().fit_transform(embed)
    if cosine:
        precom_dist = cosine_similarity(normed_embed)
        nn_nbrs = NearestNeighbors(n_neighbors=normed_embed.shape[0],
                                   algorithm='auto', metric='precomputed',
                                   n_jobs=8).fit(2 - precom_dist)
        indices = nn_nbrs.kneighbors(2 - precom_dist, return_distance=False)
    else:
        nn_nbrs = NearestNeighbors(n_neighbors=normed_embed.shape[0],
                                   algorithm='ball_tree', metric='euclidean',
                                   n_jobs=8).fit(normed_embed)
        indices = nn_nbrs.kneighbors(normed_embed, return_distance=False)
    all_distances = get_nearest_index(indices, cell_ids, cell_nbrs, use_one_side)
    return all_distances[all_distances >= 0], cell_ids[all_distances >= 0]


def get_nearest_index(nn, ids, nbrs, one_side=False):
    ids = ids.astype('int')
    is_side1_ids = is_side1[ids]
    nearest_index = []
    for n, idx in enumerate(ids):
        if idx not in nbrs:
            nearest_index.append(-1)
            continue
        pot_nbrs = [i for i in nbrs[idx] if np.any(ids == i)]
        if not pot_nbrs:
            nearest_index.append(-1)
            continue
        pot_nbrs_ids = [np.where(ids == i)[0][0] for i in pot_nbrs]
        if one_side:
            cell_on_side1 = is_side1[idx]
            # potential symm neighbors on the same side - cell is central
            if not np.all(is_side1[pot_nbrs] == ~cell_on_side1):
                nearest_index.append(-1)
                continue
            is_nn_other_side = ~is_side1_ids[nn[n]] if cell_on_side1 else is_side1_ids[nn[n]]
            nn_other_side = nn[n][is_nn_other_side]
            min_dist = np.min([np.where(nn_other_side == i)[0][0] for i in pot_nbrs_ids])
        else:
             min_dist = np.min([np.where(nn[n] == i)[0][0] for i in pot_nbrs_ids])
        nearest_index.append(min_dist)
    return np.array(nearest_index)



def get_embed(f_name):
    if f_name.endswith('h5'):
        with h5py.File(f_name, 'r') as f:
            all_embed = f['embed'][:]
            all_ids = f['label_ids'][:]
    elif f_name.endswith('tsv'):
        morph_stats = pd.read_csv(f_name, sep='\t')
        all_ids = morph_stats.label_id
        all_embed = morph_stats.drop(columns=['label_id'])
    elif f_name.endswith('np'):
        all_embed = np.loadtxt(f_name)
        all_ids = all_embed[:, 0]
        all_embed = all_embed[:, 1:]
    elif f_name.endswith('npy'):
        all_embed = np.load(f_name)
        all_ids = all_embed[:, 0]
        all_embed = all_embed[:, 1:]
    if not np.all(np.sort(all_ids) == all_ids):
        all_embed, all_ids = reorder(all_embed, all_ids)
    return all_embed, all_ids


def merge_embeds(emb_files):
    for f in emb_files:
        print(os.path.split(os.path.split(f)[0])[1],
              os.path.split(f)[1])
    if len(emb_files) == 1:
        all_embs, ids = get_embed(emb_files[0])
    else:
        embs_ids = [get_embed(f) for f in emb_files]
        all_ids = np.array([i[1] for i in embs_ids])
        assert np.sum(np.abs(all_ids - all_ids[0])) == 0
        ids = all_ids[0]
        all_embs = np.column_stack([i[0] for i in embs_ids])
    return all_embs, ids


def plot_many_dists(dist_list, names_list, mv=200):
    for dist, name in zip(dist_list, names_list):
        plt.hist(dist, bins=len(dist), label=name, cumulative=True,
                 density=True, histtype='step', lw=2)
    plt.ylim(0.2, 1)
    plt.xlim(0, mv)
    plt.legend(loc='lower right', fontsize=20)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the bilateral pairs difference')
    parser.add_argument('features_files', type=str, nargs='+',
                        help='path to file/s with features to analyze')
    parser.add_argument('--features_files2', type=str, nargs='+', default=0,
                        help='another set of features to analyze')
    parser.add_argument('--plot', type=int, default=0,
                        help='plot the distance distribution')
    parser.add_argument('--save_dist', type=str, default=None,
                        help='path to save distance for each cell')
    parser.add_argument('--cosine', type=int, default=0, choices=[0, 1],
                        help='use cosine distance instead of euclidean')
    parser.add_argument('--one_side', type=int, default=0, choices=[0, 1],
                        help='calculate on the other side only')
    args = parser.parse_args()

    dict_file = 'data/bilateral_neighbors.pkl'
    with open(dict_file, 'rb') as f:
        nbrs_dict = pickle.load(f)

    loc_file = 'data/distance_from_midline_cells_1_0_1.tsv'
    is_side1 = np.insert(np.array(pd.read_csv(loc_file, sep='\t')['side']), 0, False)

    distances = []
    emb, ids = merge_embeds(args.features_files)
    nbr_dist, filt_ids = calculate_distances(emb, ids, nbrs_dict, args.cosine, args.one_side)
    distances.append(nbr_dist)
    print("Mean distance: ", int(np.mean(nbr_dist)))
    print("Median distance: ",  int(np.median(nbr_dist)))

    if args.features_files2:
        emb2, ids2 = merge_embeds(args.features_files2)
        nbr_dist2, _ = calculate_distances(emb2, ids2, nbrs_dict, args.cosine, args.one_side)
        distances.append(nbr_dist2)
        print(int(np.mean(nbr_dist2)), int(np.std(nbr_dist2)), int(np.median(nbr_dist2)))

    if args.plot:
        plot_many_dists(distances, [str(i) for i in range(len(distances))], mv=args.plot)

    if args.save_dist is not None:
        tosave = np.column_stack([filt_ids, nbr_dist])
        hdr = 'label_id\tneighbor_distance'
        np.savetxt(args.save_dist, tosave, delimiter='\t', header=hdr, comments='')
