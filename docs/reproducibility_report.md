# Reproducibility report

## Restored functionality

- Package installation and optional dependency groups.
- Root-independent CLI/config path handling.
- Published NumPy and MoBIE artifact validation.
- Deterministic class prediction and small-subset projection/clustering tests.
- Label-first embedding I/O, group alignment, patch averaging, neighbor aggregation, and feature agglomeration.
- Shape point-cloud loading, mesh utilities/preprocessing, fixed DeepGCN training, and complete inference iteration.
- Native 3D texture training/inference without active Inferno or Neurofire dependencies.
- Portable device/checkpoint behavior and opt-in WandB.
- Transformer-based 3D MAE, synthetic fixture, tests, and Streamlit workflow workspace.

## Fixed defects

The private scratch root, forced `.cuda()` calls, hard-coded GPU IDs, missing shape modules, global trainer config, validation-as-training loader, undefined scheduler, first-batch inference break, stale scikit-learn warning API, stale NetworkX sparse API, cwd-relative paths, and non-package imports were removed or replaced.

## Published-data validation

Observed locally:

```text
morphofeatures_all_cells.npy                     (11382, 481) PASS
morphocontextfeatures_all_cells_agglomerated.npy (10391, 201) PASS
manually_defined_features.npy                    (11348, 141) PASS
MoBIE tables with label_id                       12/12 PASS
```

The bundled-label logistic-regression smoke test executes on 390 labels. The final local suite reported `11 passed, 2 skipped`; skips were the unavailable Torch MAE test and the broken local UMAP runtime.

## Commands

```bash
python -m morphofeatures validate
python -m morphofeatures classify --folds 5 --seed 42
python -m morphofeatures project --subset 256 --cluster-method kmeans \
  --output outputs/smoke_projection.tsv
python -m pytest -q
```

Training and inference commands are documented in `training_new_embeddings.md` and `modern_mae_workflow.md`.

## Known limitations

- External raw EM, cell/nucleus segmentations, original meshes, and original checkpoints are required for full retraining.
- The exact historical Neurofire architecture and serialized Inferno trainer are not recoverable from the repository; a maintained 3D autoencoder preserves the scientific objectives and 80-dimensional contract.
- Exact published UMAP/Leiden coordinates may vary with dependency versions; bundled MoBIE tables are authoritative.
- Local validation could not execute Torch/DeepGCN/MAE, igraph/Leiden, or N5/BDV integration because those optional packages and raw data are absent.
- The local installed UMAP is unusable because `llvmlite.dll` cannot load; the suite records this as an optional skip.
