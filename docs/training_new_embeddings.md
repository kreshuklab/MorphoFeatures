# Training new embeddings

## Shape branch

Prepare point clouds/normals or meshes and a manifest described in `data_preparation.md`. Edit `configs/shape_example.yaml`, then run:

```bash
python -m morphofeatures shape-train --config configs/shape_example.yaml
python -m morphofeatures shape-encode --config configs/shape_inference_example.yaml \
  --output outputs/shape_embeddings.npy
```

The loader samples a fixed number of points, centers/scales geometry, rotates and anisotropically scales training views, and returns adjacent contrastive pairs. The DeepGCN hidden output is 80-dimensional when `model.kwargs.out_channels: 80`.

`shape.preprocessing.mask_to_mesh` provides marching-cubes extraction, Taubin smoothing, optional simplification, normal repair, and mesh export. Watertightness depends on segmentation topology; inspect and repair problematic masks before training.

## Coarse/fine texture branches

Each experiment directory contains `train_config.yml`, `data_config.yml`, and later `test_config.yml` or `test_config_patches.yml`. Start from `configs/texture_train_example.yaml` and `configs/texture_data_example.yaml`.

```bash
python -m morphofeatures texture-train experiments/coarse_cell --device auto
python -m morphofeatures texture-encode experiments/coarse_cell --device auto
```

Fine patches can be exported and averaged:

```bash
python -m morphofeatures texture-encode experiments/fine_cell \
  --patches --aggregate --device auto
```

The maintained model combines NT-Xent contrastive loss, voxel reconstruction MSE, and L2 bottleneck regularization. Cell and nucleus masks are trained separately by setting `other.only_nucl` or `other.remove_nucl`.

## Combine groups

After independently training cell/nucleus shape, coarse texture, and fine texture groups:

```bash
python -m morphofeatures combine group1.npy group2.npy group3.npy \
  group4.npy group5.npy group6.npy --output outputs/morphofeatures.npy
```

The command sorts label IDs, verifies exact alignment, and refuses mismatched groups.

## Structured run metrics

Shape, texture, and MAE training preserve console output and optional WandB logging while also
appending JSON Lines events to `metrics.jsonl` beside the experiment/checkpoint. Each complete
line is independently readable during a running job and records timestamps, epoch/step, training
loss, validation loss when available, learning rate, checkpoints, and final/controlled-failure
events. The Streamlit monitor ignores only a partially written last line.

For SLURM previews, snapshots, persistence, dependencies, monitoring, and safe resubmission, see
`docs/slurm_workflow.md`.

## MAE crop and foreground-mask contract

For MAE training, `data.crops` contains fixed finite intensity inputs and `data.label_ids`
contains one unique segmentation ID per crop. Segmentation-masked cell/nucleus windows should
also provide `data.loss_masks`; this keeps random patch masking but restricts reconstruction MSE
to target-object voxels. The saved crop manifest should record bounds/center, crop coverage,
foreground fraction, resolution, normalization, source datasets, and selection exclusions.

Notebook 02 demonstrates the full crop contract on segmented synthetic cells. Notebook 04 uses
the indexed real N5 adapter for 11,382 cell-associated nuclei, including lazy reads, group splits,
raw/masked alignment, quick training, SLURM preview, per-ID encoding, and annotation joins. The
[legacy workspace inventory](legacy_workspaces_inventory.md) records inspected code/data,
licenses, duplicates, QC, and uncertainties; the earlier [ROI inventory](platyneris_data_inventory.md)
documents volume-to-crop preparation. Neither implies exact historical recovery.
