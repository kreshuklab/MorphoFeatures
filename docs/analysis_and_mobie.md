# Analysis and MoBIE export

## Standard workflow

1. Load label-first embeddings.
2. Align IDs across groups and concatenate.
3. Standardize features for downstream distance/model workflows.
4. Compute UMAP and Leiden or K-means labels.
5. Export `label_id`, cluster, and UMAP columns as TSV.

```bash
python -m morphofeatures project \
  --embedding analysis/data/morphofeatures_all_cells.npy \
  --cluster-method leiden --output outputs/morphofeatures_clust_umap.tsv
```

The output columns are `label_id`, `cluster`, `umap_1`, and `umap_2`. Existing `data_mobie` tables retain the paper's one-hot cluster columns, color annotations, and published projections.

## MoBIE table contract

- Tab-separated UTF-8 file.
- First identifier column named `label_id`.
- One row per segmentation label.
- Numeric feature columns use stable names.
- Cluster/color columns retain their existing names when reproducing published views.

`export_embeddings` writes compatible feature/projection TSVs. Join new columns to an existing MoBIE table by `label_id`, never by row position.

## Other bundled analyses

- `python -m analysis.bilateral_neighbour_analysis`: bilateral partner ranking.
- `python -m analysis.plot_genes`: gene values on a saved UMAP.
- `python -m analysis.recluster_label`: higher-resolution Leiden subclustering.

These wrappers now resolve bundled data relative to the repository module. Plotting commands may open an interactive Matplotlib window; use their save arguments for headless runs.
