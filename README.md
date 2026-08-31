# MorphoFeatures

MorphoFeatures learns morphology embeddings from segmented 3D electron-microscopy volumes. The restored repository supports the published analysis artifacts, the original shape and texture objectives, and a small masked-autoencoder (MAE) pathway for new experiments.

The original representation concatenates six 80-dimensional groups:

1. cell shape
2. nucleus shape
3. coarse cell texture
4. coarse nucleus texture
5. fine cell texture
6. fine nucleus texture

The result is a label-first `(n_cells, 481)` array: column 0 is `label_id`, and columns 1 through 480 are features. Fine-texture patch features are averaged per cell. Neighbor aggregation can be reduced to 200 features and exported as a label-first MorphoContextFeatures array.

## What works now

- Installable package with one `morphofeatures` CLI.
- Exact validation of all bundled NumPy and MoBIE artifacts.
- Deterministic logistic-regression evaluation, UMAP, Leiden, and K-means workflows.
- Explicit volume, mapping, metadata, embedding, and MoBIE data contracts.
- Configurable data roots with no private scratch path.
- Restored DeepGCN point-cloud loaders, shape training, and full-batch inference.
- Maintained 3D texture autoencoder with NT-Xent, reconstruction, and L2 bottleneck losses.
- CPU, single-GPU, multi-GPU, and CPU checkpoint loading.
- Optional WandB logging, disabled by default.
- Pragmatic 3D MAE with the same train, encode, aggregate, export, and evaluate interface.
- Deterministic synthetic 3D fixture generator and automated smoke tests.
- Streamlit workspace for analysis, feature building, configuration, and documentation.

Raw PlatyBrowser EM volumes, original training configs, and original checkpoints are not bundled. Published downstream analysis is reproducible immediately; full model retraining requires external raw and segmentation volumes.

## Quick start

Create an environment with Python 3.9 or newer, then install the analysis and test dependencies:

```bash
python -m pip install -e ".[analysis,dev]"
python -m morphofeatures validate
python -m pytest -q
```

Run deterministic class prediction on the bundled 480-dimensional embeddings:

```bash
python -m morphofeatures classify --folds 5 --seed 42
```

Run a small UMAP and K-means smoke workflow without requiring igraph:

```bash
python -m morphofeatures project --subset 256 --umap-epochs 50 \
  --cluster-method kmeans --clusters 8 --output outputs/smoke_projection.tsv
```

For the published Leiden workflow, install the `analysis` extra and use `--cluster-method leiden`.

## Installation groups

```bash
# Published-data analysis
python -m pip install -e ".[analysis]"

# Restored DeepGCN and texture training
python -m pip install -e ".[legacy-training]"

# 3D masked autoencoder
python -m pip install -e ".[modern-training]"

# Streamlit workflow workspace
python -m pip install -e ".[ui]"

# Development and tests
python -m pip install -e ".[analysis,dev]"
```

CUDA is optional. Install the Torch build matching the host CUDA driver before installing the training groups. Historical N5/BDV input can use `pybdv` and `zarr`; exact old z5 containers may additionally require `z5py` from conda-forge.

See [installation](docs/installation.md) for environment details.

## Repository data

Bundled and immediately reproducible:

- `analysis/data/morphofeatures_all_cells.npy`: `(11382, 481)`
- `analysis/data/morphocontextfeatures_all_cells_agglomerated.npy`: `(10391, 201)`
- `analysis/data/manually_defined_features.npy`: `(11348, 141)`
- `analysis/data/class_labels.tsv`: 390 curated class labels
- `analysis/data/bilateral_neighbors.pkl`: bilateral candidates
- `analysis/data/gene_expression.tsv`: gene-expression analysis table
- `data_mobie/*.tsv`: feature, projection, cluster, and prediction tables

Required for training but not bundled:

- raw 3D EM volume
- cell segmentation
- nucleus segmentation
- cell-to-nucleus mapping
- cell and nucleus bounding-box/anchor tables

All spatial arrays and metadata use `z, y, x` coordinate order. Resolution values are physical voxel sizes in micrometers unless a config states another unit. See [data preparation](docs/data_preparation.md).

## Main commands

```text
morphofeatures validate        validate published artifacts
morphofeatures classify        logistic-regression cross-validation
morphofeatures project         UMAP plus Leiden or K-means
morphofeatures combine         concatenate aligned feature groups
morphofeatures context         aggregate neighbor features
morphofeatures synthetic       create a small 3D fixture
morphofeatures shape-train     train DeepGCN shape embeddings
morphofeatures shape-encode    export shape embeddings
morphofeatures texture-train   train coarse/fine texture embeddings
morphofeatures texture-encode  export or aggregate texture embeddings
morphofeatures mae-train       train a 3D masked autoencoder
morphofeatures mae-encode      export MAE embeddings
morphofeatures doctor          report optional runtime capabilities
morphofeatures ui              open the Streamlit workspace
```

Every command also works as `python -m morphofeatures ...` from the repository root. Legacy scripts remain runnable with `python -m analysis.log_regress`, `python -m analysis.umap_and_clustering`, and their documented arguments.

## New embeddings

Generate a deterministic fixture:

```bash
python -m morphofeatures synthetic outputs/synthetic --seed 7
```

Train the small MAE smoke configuration and export embeddings:

```bash
python -m morphofeatures mae-train --config configs/smoke.yaml \
  --output outputs/mae/checkpoint.pt
python -m morphofeatures mae-encode --config configs/smoke.yaml \
  --checkpoint outputs/mae/checkpoint.pt --output outputs/mae/embeddings.npy
```

The smoke config generates random crops when `data.crops` is absent. Full experiments set `data.crops` to an `(n, z, y, x)` or `(n, c, z, y, x)` NumPy array and preserve label IDs in the export step.

For legacy training, see [training new embeddings](docs/training_new_embeddings.md). For MAE comparisons, see [modern MAE workflow](docs/modern_mae_workflow.md).

## Reproducibility

The published arrays and tables are preserved unchanged. New outputs follow the same `label_id` convention, deterministic seeds are exposed by CLI/config, and checkpoints store model state without `DataParallel` prefixes. The full status and validation boundaries are recorded in [the reproducibility report](docs/reproducibility_report.md).

This repository accompanies the [MorphoFeatures paper](https://www.biorxiv.org/content/10.1101/2022.05.07.490949v1). Repository documents and publications describe scientific context; runtime behavior is defined by the package code, configs, tests, and data contracts.
