# MorphoFeatures Architecture

## Overview

MorphoFeatures is split into reusable package layers:

```text
configs/*.yaml
      |
      v
morphofeatures.config
      |
      v
morphofeatures.data  ->  texture / shape model workflows  ->  embedding files
      |                                                   |
      +-------------------- analysis workflows <----------+
```

The central contract is an embedding table whose first column is `label_id` and whose remaining columns are feature dimensions. Texture and shape workflows both produce this format, and all analysis modules consume it.

## Input Contracts

The repository has two raw feature-extraction pipelines and one downstream analysis pipeline:

```text
texture raw inputs  ->  texture encoder  ->  embedding matrix  ->  analysis
shape raw inputs    ->  shape encoder    ->  embedding matrix  ->  analysis
precomputed paper embeddings ------------------------------->  analysis
```

The **embedding matrix** is therefore an output of feature extraction, not the first raw data object for the full pipeline.

Texture extraction expects:

- raw EM intensity volume in a z5/n5-compatible container,
- cell segmentation volume, typically referenced by a BigDataViewer XML,
- nucleus segmentation volume, also typically referenced by a BigDataViewer XML,
- `cells_to_nuclei.tsv` mapping cell label IDs to nucleus label IDs,
- cell and nucleus metadata TSVs with `label_id`, `bb_min_z/y/x`, `bb_max_z/y/x`, and `anchor_z/y/x` columns.

Shape extraction expects:

- a `.npz` file with `points` shaped like `(cells, points, xyz)` or `(cells, xyz, points)`,
- optional `features`, for example xyz plus normals, shaped like `(cells, points, channels)` or `(cells, channels, points)`,
- optional `ids` containing one label ID per cell.

Analysis expects:

- one or more embedding files where column 0 is `label_id` and columns 1..n are numeric features,
- optional label/metadata files for supervised evaluation, bilateral scoring, or gene overlays.

## Package Layout

```text
src/morphofeatures/
  config/
    loading.py        YAML loading and recursive default merging
    schema.py         typed dataclass schemas for common settings
  data/
    embeddings.py     embedding I/O, sorting, validation, merging
    splits.py         reproducible train/validation splits
    tables.py         TSV and cell-to-nucleus mapping helpers
    volumes.py        lazy z5py/pybdv volume access
  texture/
    datasets.py       cell crop and texture patch datasets
    loaders.py        config-driven dataloader factory
    transforms.py     Inferno transform construction
    trainer.py        Inferno/Neurofire training orchestration
    inference.py      per-cell/per-patch embedding generation
  shape/
    datasets.py       point-cloud array datasets
    loaders.py        train/validation/inference dataloaders
    trainer.py        DeepGCN metric-learning training loop
    inference.py      checkpoint loading and embedding export
    mesh_utils.py     OFF, mesh graph, and k-hop helpers
    network/          DeepGCN implementation
    augmentations/    simple NumPy transforms and ARAP deformation
  analysis/
    clustering.py     UMAP, Leiden clustering, MoBIE TSV export
    classification.py logistic-regression cell-type evaluation
    bilateral.py      bilateral-neighbor nearest-rank scoring
    genes.py          gene-expression UMAP overlays
    reclustering.py   high-resolution reclustering of one label
  cli/
    texture.py        `morphofeatures-texture`
    shape.py          `morphofeatures-shape`
    analysis.py       `morphofeatures-analysis`
    train.py          `morphofeatures-train`
  training/
    encoders.py       high-level validation and launch layer for encoder training
```

## Data Flow

1. **Texture features**
   - `texture.loaders.CellLoaders` reads YAML config, opens raw/cell/nucleus volumes, loads cell metadata, and builds datasets.
   - `texture.datasets` extracts nucleus-centered raw crops or fixed-radius high-resolution texture patches.
   - `texture.trainer` wraps the original Inferno/Neurofire trainer.
   - `texture.inference` writes per-cell `.np` embeddings or patch-level `.z5` predictions plus averaged patch features.

2. **Shape features**
   - `shape.datasets.load_shape_arrays` loads point clouds from `.npz` or separate NumPy arrays.
   - `shape.loaders` creates contrastive two-view batches for training and simple batches for inference.
   - `shape.trainer.ShapeTrainer` trains DeepGCN and fixes the original trainer issues: it uses `self.config`, trains on `train_loader`, treats schedulers as optional, and validates without gradient tracking.
   - `shape.inference.generate_embeddings` processes all batches, not only the first one.

3. **Encoder training orchestration**
   - `training.encoders` provides a lightweight validation and launch layer for new shape and texture encoders.
   - `morphofeatures-train shape --dry-run` and `morphofeatures-train texture --dry-run` report expected checkpoint locations before importing optional GPU dependencies.
   - The training layer delegates to `shape.trainer.ShapeTrainer` and `texture.trainer.train_texture_model` for actual model fitting.

4. **Analysis**
   - `data.embeddings` normalizes embedding I/O and ID alignment.
   - `analysis.clustering` loads labeled embeddings, computes UMAP, runs Leiden clustering, evaluates homogeneity/bilateral scores, and exports cluster tables.
   - `analysis.classification` merges embeddings and evaluates cell-type predictability.
   - `analysis.bilateral` calculates where bilateral counterparts appear in embedding nearest-neighbor lists.
   - `analysis.genes` overlays gene expression on UMAP coordinates.

## Design Choices

- Optional heavy dependencies are imported lazily, so lightweight modules remain usable without the full GPU/scientific stack.
- YAML config is the public interface for paths, constants, model settings, and run parameters.
- Backward-compatible module names remain for old imports such as `morphofeatures.texture.cell_loader` and `morphofeatures.shape.data_loading.loader`.
- Analysis functions no longer rely on globals; every function receives explicit inputs.
- Existing exported data directories are preserved.

## Testing

The test suite focuses on behavior that can be validated without optional training dependencies:

- recursive config loading,
- embedding sorting, merging, and standard matrix output,
- analysis helper behavior without hidden globals,
- texture dataset crop geometry with mocked volumes,
- shape array loading from `.npz`.

Run:

```bash
python -m unittest discover -s tests
```
