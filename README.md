# MorphoFeatures

MorphoFeatures is a reusable pipeline for learning and analyzing cell morphology representations from volumetric EM data and shape point clouds. The repository contains:

- texture feature extraction from raw cell/nucleus crops,
- shape feature extraction with a DeepGCN point-cloud encoder,
- downstream embedding analysis with UMAP, Leiden clustering, bilateral-neighbor scoring, gene overlays, and cell-type classification,
- paper/export data tables in `analysis/data/` and `data_mobie/`.

The code has been reorganized into an installable `src/` package with typed modules, configuration files, command-line entry points, and tests.

## What Data Goes In?

The **embedding matrix is not the raw input data**. It is the intermediate feature table produced after running either the texture encoder or the shape encoder.

There are three practical entry points:

| Goal | Input you provide | Command/module that creates embeddings | Output |
| --- | --- | --- | --- |
| Analyze the paper features | Existing file such as `analysis/data/morphofeatures_all_cells.npy` | none; embeddings are already precomputed | clustering/classification/plots |
| Extract texture features from new EM data | raw EM volume, cell segmentation, nucleus segmentation, cell-to-nucleus table, cell/nucleus metadata tables | `morphofeatures-texture predict ...` after training/loading a texture model | `avg_encoded.np` or averaged patch embeddings |
| Extract shape features from new shape data | point-cloud arrays in `.npz` or `.npy` form, with optional per-point features and cell IDs | `morphofeatures-shape embed ...` after training/loading a shape model | `.npy` embedding matrix |

The standard embedding format is:

```text
label_id  feature_0  feature_1  ...  feature_n
```

So if you only want to run downstream analysis, start with an embedding file. If you want to create embeddings for a new dataset, start with either texture inputs or shape inputs and run the corresponding encoder first.

## Installation

Core utilities only require NumPy, pandas, and PyYAML:

```bash
pip install -e .
```

Install optional extras for the full workflows:

```bash
pip install -e ".[analysis]"
pip install -e ".[shape]"
pip install -e ".[texture]"
```

Some original training dependencies, such as Inferno/Neurofire/z5py/libigl, may require platform-specific installation. The lightweight config, embedding, and table utilities are importable without those optional packages.

## Basic Usage

Validate and train a texture encoder from an experiment directory containing `train_config.yml` and `data_config.yml`:

```bash
morphofeatures-train texture path/to/experiment --dry-run
morphofeatures-train texture path/to/experiment --devices 0
```

Generate texture embeddings:

```bash
morphofeatures-texture predict path/to/experiment --devices 0
```

Train a shape model:

```bash
morphofeatures-train shape --config configs/shape_train.yaml --dry-run
morphofeatures-train shape --config configs/shape_train.yaml
```

Generate shape embeddings:

```bash
morphofeatures-shape embed --config configs/shape_inference.yaml --save-to runs/shape_embeddings.npy
```

Run analysis:

```bash
morphofeatures-analysis classify analysis/data/morphofeatures_all_cells.npy
morphofeatures-analysis cluster analysis/data/morphofeatures_all_cells.npy --save-path runs/clusters.tsv
```

## Configuration

Example configs live in `configs/`:

- `default.yaml` defines shared project conventions.
- `texture_model_train.yaml` shows the model/loss/optimizer config copied to `train_config.yml`.
- `texture_train.yaml` and `texture_predict.yaml` show how to configure raw volumes, segmentation tables, transforms, and dataloaders.
- `shape_train.yaml` configures point-cloud arrays, DeepGCN, optimizer, loss, and training cadence.
- `shape_inference.yaml` shows how to point inference at a trained shape checkpoint.
- `analysis.yaml` centralizes paths and analysis parameters.

Embedding files use the standard MorphoFeatures format: first column `label_id`, remaining columns numeric feature values.

## Repository Map

```text
src/morphofeatures/
  config/      YAML loading and typed config schemas
  data/        embedding, table, split, and volume helpers
  texture/     texture datasets, loaders, trainer, inference
  shape/       point-cloud datasets, loaders, trainer, inference, DeepGCN
  analysis/    clustering, classification, bilateral metrics, genes
  cli/         command-line dispatchers
configs/       example YAML configs
tests/         lightweight regression and smoke tests
notebooks/     tutorial notebook
analysis/data/ paper-facing analysis inputs
data_mobie/    MoBIE-compatible exported tables
```

See `ARCHITECTURE.md` for implementation details and `notebooks/tutorial.ipynb` for an end-to-end usage walkthrough.

For the specific question of how to obtain new encoder checkpoints, see `ENCODER_TRAINING.md`.
