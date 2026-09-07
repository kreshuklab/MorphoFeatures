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

- [Dataset workspace workflows](docs/workspace_workflows.md): persistent MAE configuration,
  local/SLURM pipelines, cached MAE/DINO embeddings, matched comparisons, bounded instance
  preprocessing, and mesh intensity visualization/export.
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
- Guided Streamlit workspace: connected pipeline stages, saved drafts, exact run review,
  local/Slurm submission, and shared Runs and Results browsers.
- SLURM-safe experiment previews/submission, SQLite job persistence, scheduler refresh,
  bounded log tails, structured metrics, and explicit artifact inspection.
- Ordered, visualization-rich notebooks for contracts, CPU MAE training/encoding, biological
  analysis, and an optional bounded real-data N5 workflow.

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

## Workflow UI

```bash
python -m morphofeatures ui
```

Start in **Workflow** with raw volumes, prepared data, a checkpoint, or saved
embeddings. Configure the relevant stages, then use **Review & run** to inspect
the exact settings and command before running locally or submitting to Slurm.
**Save draft** preserves edits for later; **Save dry-run bundle** saves a run
snapshot without executing it. Follow progress in **Runs** and reopen outputs
in **Results**. Specialist workflows and sweeps remain in **Tools**.

See [the workspace guide](docs/workspace_workflows.md) and
[UI design notes](docs/ui_redesign.md).

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

# Executable notebook collection
python -m pip install -e ".[analysis,modern-training,notebooks]"

# Development and tests
python -m pip install -e ".[analysis,dev]"
```

CUDA is optional. Install the Torch build matching the host CUDA driver before installing the training groups. Historical N5/BDV input can use `pybdv` and `zarr`; exact old z5 containers may additionally require `z5py` from conda-forge.

See [installation](docs/installation.md) for environment details.

## End-to-end notebooks

Start Jupyter from the repository root and follow [the notebook guide](notebooks/README.md):

1. [Data preparation and contracts](notebooks/01_data_preparation_and_contracts.ipynb)
2. [CPU MAE training and encoding](notebooks/02_cpu_mae_training_and_encoding.ipynb)
3. [Biological analysis and interpretation](notebooks/03_biological_analysis_and_interpretation.ipynb)
4. [Real Platynereis MAE workflow](notebooks/04_real_platynereis_mae_workflow.ipynb)

The second notebook is the complete automated CPU workflow on segmented synthetic cells and
shows crop masks, token masks, reconstruction, losses, checkpoints, and embeddings. The third
uses bundled published embeddings and metadata. The fourth uses a canonical YAML plus explicit
notebook overrides to inspect, train, reconstruct, encode, and analyze 11,382 indexed real
Platynereis nuclei through lazy N5 reads. Its output is accurately scoped as nucleus-derived
texture. See the [legacy code/data inventory](docs/legacy_workspaces_inventory.md), the earlier
[whole-volume ROI audit](docs/platyneris_data_inventory.md), and [notebook workflows](docs/notebooks.md).
Generated files go below the configured output root.

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
morphofeatures n5-inventory    inspect N5 metadata without reading chunks
morphofeatures shape-train     train DeepGCN shape embeddings
morphofeatures shape-encode    export shape embeddings
morphofeatures texture-train   train coarse/fine texture embeddings
morphofeatures texture-encode  export or aggregate texture embeddings
morphofeatures mae-train       train a 3D masked autoencoder
morphofeatures mae-encode      export MAE embeddings
morphofeatures mae-sweep-prepare  resolve a bounded MAE ablation manifest
morphofeatures mae-sweep-compare  compare sweep losses and curves
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

The smoke config generates random crops when `data.crops` is absent. Full experiments set
`data.crops` to an `(n, z, y, x)` or `(n, c, z, y, x)` NumPy array and `data.label_ids` to one
finite, unique segmentation ID per crop. Smoke-only IDs default to `1..n` (zero remains
background).

For segmentation-masked crops, also set `data.loss_masks` to `(n,z,y,x)` or `(n,1,z,y,x)`
foreground masks. The MAE then computes masked-patch reconstruction loss only inside the target
cell/nucleus, avoiding a background-dominated MSE while leaving inference inputs unchanged.

The generic crop MAE checkpoint contract is `position-aware-3d-v2`: masking replaces voxel-patch
content but retains explicit `z,y,x` position. The audited nucleus-patch N5 route uses the separate
`grouped-nucleus-patches-v3` contract. There, one token is a complete stored `32³` texture patch,
one sample is a spatial group from one cell-associated nucleus, and the target is a downsampled
hidden patch. Checkpoints cannot cross these contracts. Structured metrics report model MSE and a
visible-patch mean baseline; judge a run by held-out improvement, reconstructions, and embedding QC.

For the indexed real-patch route, start from the portable
[`mae_nucleus_patches_template.yaml`](configs/mae_nucleus_patches_template.yaml) or the separate
site example under `configs/sites/`. It validates N5 keys, `(z,y,x)` resolution, parent IDs,
grouped splits, lazy central patch groups, whole-patch masking, checkpoint resume, and label-first
nucleus-level export. The site config contains external cluster paths but no copied data,
credentials, or institutional scheduler values.

For legacy training, see [training new embeddings](docs/training_new_embeddings.md). For MAE
comparisons, see [modern MAE workflow](docs/modern_mae_workflow.md). To adapt new inputs, follow
[mapping a new dataset](morphofeatures/howto_new_dataset.md).

## SLURM workspace

The Streamlit lifecycle is:

```text
validate inputs → configure → preview → submit → monitor → encode → inspect → analyze/export
```

`Workflow → Review & run` previews the exact worker configuration, command and Slurm
script. Review does not reserve a run ID. Saving a dry-run bundle creates an immutable
snapshot without execution. `Runs` reads the shared SQLite registry, refreshes jobs through
`squeue`/`sacct`, displays metrics and logs, and passes completed artifacts into a new workflow.
Only **Run locally** or **Submit to Slurm** launches work; navigation never submits jobs.
Published shape/texture workflows retain their existing adapters under `Tools`.

For real-data MAE training, `Tools` also provides the existing bounded, allowlisted soft grid. The default
one-at-a-time mode can compare learning rates/schedulers, linear versus 3D ResNet patch encoders,
embedding and reconstruction dimensions, and normalization without hiding settings in callbacks.
Each variant is a normal persistent job; optional encoding is queued with an `afterok` dependency.
See `configs/mae_nucleus_soft_grid.example.yaml` and the real-data notebook for metric,
reconstruction, and shallow cell-type-probe comparisons.

Start with the placeholder-only [profile example](configs/slurm_profiles.example.yaml) and the
[SLURM workspace guide](docs/slurm_workflow.md), including its mapping from a conventional cluster
batch script to structured profile fields. No live SLURM success is implied by this
repository's fake-backend and dry-run tests.

## Reproducibility

The published arrays and tables are preserved unchanged. New outputs follow the same `label_id` convention, deterministic seeds are exposed by CLI/config, and checkpoints store model state without `DataParallel` prefixes. The full status and validation boundaries are recorded in [the reproducibility report](docs/reproducibility_report.md).

This repository accompanies the [MorphoFeatures paper](https://www.biorxiv.org/content/10.1101/2022.05.07.490949v1). Repository documents and publications describe scientific context; runtime behavior is defined by the package code, configs, tests, and data contracts.
