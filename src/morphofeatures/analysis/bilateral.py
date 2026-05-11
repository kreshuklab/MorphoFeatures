"""Bilateral-neighbor analysis for morphology embeddings."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from morphofeatures.data.embeddings import merge_embedding_tables, read_embedding_table


def get_nearest_index(
    nearest_neighbors: np.ndarray,
    ids: np.ndarray,
    neighbors_by_cell: Mapping[int, Sequence[int]],
    side_mask: np.ndarray,
    one_side: bool = False,
) -> np.ndarray:
    """Return each cell's nearest bilateral-neighbor rank."""

    ids = ids.astype(int)
    is_side1_ids = side_mask[ids]
    nearest_index = []
    selected_ids = set(int(label_id) for label_id in ids)
    id_to_row = {int(label_id): index for index, label_id in enumerate(ids)}

    for row_index, cell_id in enumerate(ids):
        if int(cell_id) not in neighbors_by_cell:
            nearest_index.append(-1)
            continue
        potential_neighbors = [int(neighbor) for neighbor in neighbors_by_cell[int(cell_id)] if int(neighbor) in selected_ids]
        if not potential_neighbors:
            nearest_index.append(-1)
            continue
        potential_neighbor_rows = [id_to_row[neighbor] for neighbor in potential_neighbors]
        if one_side:
            cell_on_side1 = side_mask[int(cell_id)]
            if not np.all(side_mask[potential_neighbors] == ~cell_on_side1):
                nearest_index.append(-1)
                continue
            is_neighbor_other_side = ~is_side1_ids[nearest_neighbors[row_index]] if cell_on_side1 else is_side1_ids[nearest_neighbors[row_index]]
            neighbor_order = nearest_neighbors[row_index][is_neighbor_other_side]
            min_distance = np.min([np.where(neighbor_order == neighbor_row)[0][0] for neighbor_row in potential_neighbor_rows])
        else:
            min_distance = np.min([np.where(nearest_neighbors[row_index] == neighbor_row)[0][0] for neighbor_row in potential_neighbor_rows])
        nearest_index.append(min_distance)
    return np.asarray(nearest_index)


def calculate_distances(
    embeddings: np.ndarray,
    cell_ids: np.ndarray,
    cell_neighbors: Mapping[int, Sequence[int]],
    side_mask: np.ndarray,
    cosine: bool = False,
    use_one_side: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate bilateral-neighbor nearest-neighbor ranks."""

    try:
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise ImportError("Bilateral analysis requires scikit-learn.") from exc

    mean = embeddings.mean(axis=0, keepdims=True)
    std = embeddings.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    normed_embeddings = (embeddings - mean) / std
    if cosine:
        precomputed_distance = 2 - cosine_similarity(normed_embeddings)
        nearest_model = NearestNeighbors(
            n_neighbors=normed_embeddings.shape[0],
            algorithm="auto",
            metric="precomputed",
            n_jobs=8,
        ).fit(precomputed_distance)
        indices = nearest_model.kneighbors(precomputed_distance, return_distance=False)
    else:
        nearest_model = NearestNeighbors(
            n_neighbors=normed_embeddings.shape[0],
            algorithm="ball_tree",
            metric="euclidean",
            n_jobs=8,
        ).fit(normed_embeddings)
        indices = nearest_model.kneighbors(normed_embeddings, return_distance=False)
    distances = get_nearest_index(indices, cell_ids, cell_neighbors, side_mask, use_one_side)
    valid = distances >= 0
    return distances[valid], cell_ids[valid]


def get_embed(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one embedding file."""

    table = read_embedding_table(path)
    return table.features, table.ids


def merge_embeds(embedding_files: Sequence[str | Path]) -> tuple[np.ndarray, np.ndarray]:
    """Merge multiple embedding files without scaling."""

    table = merge_embedding_tables(embedding_files, scale=False)
    return table.features, table.ids


def load_side_mask(path: str | Path) -> np.ndarray:
    """Load a label-indexed boolean side mask from a TSV file."""

    return np.insert(np.asarray(pd.read_csv(path, sep="\t")["side"], dtype=bool), 0, False)


def plot_many_dists(distance_list: Sequence[np.ndarray], names: Sequence[str], max_value: int = 200) -> None:
    """Plot cumulative bilateral-distance distributions."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires matplotlib.") from exc

    for distances, name in zip(distance_list, names, strict=True):
        plt.hist(distances, bins=len(distances), label=name, cumulative=True, density=True, histtype="step", lw=2)
    plt.ylim(0.2, 1)
    plt.xlim(0, max_value)
    plt.legend(loc="lower right", fontsize=20)
    plt.show()


def main(argv: list[str] | None = None) -> None:
    """Run bilateral-neighbor analysis from the command line."""

    parser = argparse.ArgumentParser(description="Calculate bilateral-pair embedding distances.")
    parser.add_argument("features_files", type=Path, nargs="+")
    parser.add_argument("--features-files2", type=Path, nargs="+", default=None)
    parser.add_argument("--plot", type=int, default=0)
    parser.add_argument("--save-dist", type=Path, default=None)
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--one-side", action="store_true")
    parser.add_argument("--neighbors-file", type=Path, default=Path("analysis/data/bilateral_neighbors.pkl"))
    parser.add_argument("--side-file", type=Path, default=Path("analysis/data/distance_from_midline_cells_1_0_1.tsv"))
    args = parser.parse_args(argv)

    with args.neighbors_file.open("rb") as handle:
        neighbors_by_cell = pickle.load(handle)
    side_mask = load_side_mask(args.side_file)
    distances = []
    embeddings, ids = merge_embeds(args.features_files)
    neighbor_distance, filtered_ids = calculate_distances(embeddings, ids, neighbors_by_cell, side_mask, args.cosine, args.one_side)
    distances.append(neighbor_distance)
    print("Mean distance:", int(np.mean(neighbor_distance)))
    print("Median distance:", int(np.median(neighbor_distance)))

    if args.features_files2:
        embeddings2, ids2 = merge_embeds(args.features_files2)
        neighbor_distance2, _ = calculate_distances(embeddings2, ids2, neighbors_by_cell, side_mask, args.cosine, args.one_side)
        distances.append(neighbor_distance2)
        print(int(np.mean(neighbor_distance2)), int(np.std(neighbor_distance2)), int(np.median(neighbor_distance2)))

    if args.plot:
        plot_many_dists(distances, [str(index) for index in range(len(distances))], max_value=args.plot)

    if args.save_dist:
        to_save = np.column_stack([filtered_ids, neighbor_distance])
        np.savetxt(args.save_dist, to_save, delimiter="\t", header="label_id\tneighbor_distance", comments="")


if __name__ == "__main__":
    main()
