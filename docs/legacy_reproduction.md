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
  --neighbors /path/to/physical_neighbor_mapping.pkl \
  --features 200 --output outputs/context.npy
```

This legacy command averages available neighbor and self features, then applies
feature agglomeration. It does **not** concatenate the focal vector and therefore
does not implement the complete paper context recipe. The UI retains its mean/max
options under Tools. Use **Workflow → Build spatial context from embeddings** for
the focal-vector-plus-neighborhood-mean construction and explicit coverage checks.
The bundled `bilateral_neighbors.pkl` contains evaluation candidates, **not**
physical contacts; it must not be used to build the context features being tested.

**Workflow → Explore published MorphoFeatures / MorphoContextFeatures** loads the
bundled vectors and their saved UMAP/cluster assignments without fitting them again.
The 200-dimensional context array is the historical reference; the original contact
graph and fitted dimensionality reducer are not bundled. Newly generated vectors
should therefore be described as a new context experiment, not exact reproduction.
See [the workflow guide](workspace_workflows.md#bilateral-retrieval-and-spatial-context)
and [the runnable example](../configs/workspace_bilateral_context.example.yaml).

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
