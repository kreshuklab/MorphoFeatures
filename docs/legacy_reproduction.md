# Legacy analysis reproduction

No external volume is required for this workflow.

## Validate distributed artifacts

```bash
python -m morphofeatures validate --json
```

Expected arrays:

- MorphoFeatures: `(11382, 481)`
- agglomerated MorphoContextFeatures: `(10391, 201)`
- manually defined features: `(11348, 141)`

## Class prediction

```bash
python -m morphofeatures classify \
  --embedding analysis/data/morphofeatures_all_cells.npy \
  --labels analysis/data/class_labels.tsv --folds 5 --seed 42
```

The restored run standardizes features, uses shuffled stratified folds with a fixed seed, and reports fold scores plus the complete confusion matrix. The historical wrapper remains available:

```bash
python -m analysis.log_regress analysis/data/morphofeatures_all_cells.npy \
  --train_data_file analysis/data/class_labels.tsv --seed 42
```

## UMAP and clustering

Lightweight smoke workflow:

```bash
python -m morphofeatures project --subset 256 --umap-epochs 50 \
  --cluster-method kmeans --output outputs/projection.tsv
```

Published-style graph clustering:

```bash
python -m morphofeatures project --cluster-method leiden \
  --resolution 0.004 --output outputs/leiden_projection.tsv
```

Leiden requires `umap-learn`, `python-igraph`, and `leidenalg`. Exact coordinates and labels can differ across library versions and hardware even with fixed seeds; bundled MoBIE projection/cluster tables are the published reference.

## Neighbor context

```bash
python -m morphofeatures context \
  --embedding analysis/data/morphofeatures_all_cells.npy \
  --neighbors analysis/data/bilateral_neighbors.pkl \
  --features 200 --output outputs/context.npy
```

This maintained command averages available neighbor and self features, then applies feature agglomeration. The bundled context array remains the reference for the paper's exact historical neighborhood construction.

## Scientific reproduction limits

Published feature arrays and MoBIE tables are preserved as reference artifacts.
Full historical retraining additionally requires the original raw EM,
segmentations, meshes, preprocessing state, configurations, and checkpoints.
The historical Neurofire architecture and serialized Inferno trainer are not
recoverable here: the maintained native PyTorch texture autoencoder preserves
the objective and 80-dimensional output contract, without claiming identical
historical weights or model fidelity.

Bounded synthetic and real-data checks establish software integration, not
biological generalization or equivalence to the original training experiments.
The [whole-volume ROI inventory](platyneris_data_inventory.md) retains the
31-cell CPU pilot's source mapping, observed losses, ID checks, and saved output
locations. The [grouped nucleus inventory](legacy_workspaces_inventory.md)
records the current patch lineage and sampling contract. Optional-library
availability and test totals depend on the environment; use the maintained test
suite and `morphofeatures validate` to check a particular installation.
