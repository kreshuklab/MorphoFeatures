# Modern 3D MAE workflow

The modern pathway tokenizes 3D crops with a strided convolution, keeps patch content separate
from a separable `z,y,x` positional encoding, masks a configurable fraction of patch-content
tokens, and reconstructs voxel patches with contextualized Transformer tokens. Position is added
after content masking, so every hidden location remains distinguishable. Training loss is computed
only on masked patches. Mean encoded tokens form the cell/patch embedding.

The generic crop architecture is versioned as `position-aware-3d-v2`. Checkpoints made before this
marker used a position-blind masking operation: masking replaced both content and position, forcing
every hidden token in a sample toward the same prediction. Those checkpoints cannot be resumed or
encoded safely and are rejected with an instruction to retrain from scratch.

## Smoke run

```bash
python -m pip install -e ".[modern-training]"
python -m morphofeatures mae-train --config configs/smoke.yaml \
  --output outputs/mae/checkpoint.pt
python -m morphofeatures mae-encode --config configs/smoke.yaml \
  --checkpoint outputs/mae/checkpoint.pt --output outputs/mae/embeddings.npy
```

Without `data.crops`, the smoke config generates eight deterministic random crops. For real data, set:

```yaml
data:
  crops: /data/project/coarse_cell_crops.npy
  label_ids: /data/project/coarse_cell_label_ids.npy
  loss_masks: /data/project/coarse_cell_masks.npy
mae:
  architecture_version: position-aware-3d-v2
  input_shape: [64, 64, 64]
  patch_size: [8, 8, 8]
  embedding_dim: 80
  encoder_depth: 4
  encoder_heads: 8
  mask_ratio: 0.75
training:
  epochs: 100
  batch_size: 8
  learning_rate: 0.0001
```

Crop dimensions must be divisible by patch size. Train cell/nucleus and coarse/fine modalities separately to preserve the six-group comparison.

### Lazy indexed N5 patches

For a fine nucleus-texture group, `data.source: n5_masked_patches` switches the same CLI to the
`grouped-nucleus-patches-v3` adapter. Start from `configs/mae_nucleus_patches_template.yaml`. It
validates one `uint8 (n,z,y,x)` patch store plus an `int64 (n,4)` `(label_id,z,y,x)` index, opens N5
lazily per worker, and splits parent IDs before loading spatial groups. One model token is one
complete stored texture patch; one sample is up to `group_size` central patches from one nucleus.
The encoder sees visible patch content plus relative coordinates, and the decoder predicts an
`8³` downsample of each hidden `32³` patch. The exported row is the nucleus-level encoder feature,
not an average of independently encoded patches and not the full six-group representation.

The audited EMBL path and `quick`/`full` profiles are in
`configs/sites/mae_platynereis_nuclei_embl.yaml`; keep equivalent site paths external elsewhere.
See `notebooks/04_real_platynereis_mae_workflow.ipynb` and
`docs/legacy_workspaces_inventory.md` for data meaning and preprocessing evidence.

### Linear versus 3D ResNet patch encoders

The grouped nucleus model has two spatial scales. First, `mae.patch_encoder` turns each complete
local `32³` patch into one token. Then the Transformer relates those tokens across their physical
`z,y,x` positions within a nucleus.

- `linear` flattens all voxels and applies one dense projection. It is the v3 baseline and remains
  the default; existing v3 linear checkpoints keep compatible state-dict keys.
- `resnet3d` uses a stride-two 3D convolution, configurable residual stages, GroupNorm,
  hierarchical downsampling, global average pooling, and a final projection. Its convolutional
  receptive fields impose a local-texture bias before the group Transformer. GroupNorm avoids
  depending on the variable number of visible patches as a BatchNorm batch.

```yaml
mae:
  patch_encoder: resnet3d
  resnet_channels: [16, 32, 64]
  resnet_blocks: [1, 1, 1]
```

This is a controlled architectural ablation, not an assumption that ResNet features are better.
Compare held-out loss against the visible-neighbor baseline, matched reconstructions, embedding
stability, and the same downstream task. A changed patch encoder requires a newly trained
checkpoint; it cannot resume a linear-encoder checkpoint.

### Learning-rate schedules and bounded soft grids

Real-data training accepts `training.scheduler: constant|cosine|step`, with `warmup_epochs`,
`min_learning_rate`, `scheduler_step_size`, and `scheduler_gamma`. Scheduler state is included in
portable checkpoints and restored on resume.

### Live progress and early stopping

The real N5 trainer writes append-only structured events for metadata validation, index loading,
ID splitting, patch selection, split-manifest creation, data-loader/model readiness, and training.
During an epoch, `batch_progress` records the phase, batch count, running loss, groups and patches
processed, throughput, elapsed time, and an ETA. The interval is bounded and configurable:

```yaml
training:
  progress_interval_batches: 25  # 0 disables within-epoch events
  early_stopping:
    enabled: false
    patience: 10
    min_delta: 0.000001
    restore_best: true
```

Early stopping monitors held-out `validation_loss`. An epoch counts as improved only when the loss
falls by more than `min_delta`; training stops after `patience` consecutive completed epochs
without that improvement. When enabled, every new best is atomically saved to
`paths.best_checkpoint`. With `restore_best: true`, the final checkpoint contains the matching
best model, optimizer, scheduler, scaler, epoch, and step state rather than a mixture of best model
weights and later optimizer state. The feature is disabled by default because patience is a
scientific choice, not merely a runtime optimization.

The maintained trainer currently uses exactly one CUDA device. Requesting multiple GPUs does not
parallelize it: no DataParallel or DistributedDataParallel path is configured. For the lazy N5
workflow, first benchmark one GPU and inspect batch throughput and utilization; multiple workers
can already make storage I/O the limiting resource. Multi-GPU support would require distributed
sampling by nucleus, cross-rank metric reduction, rank-zero checkpoint/event writing, and careful
control of aggregate N5 traffic.

`configs/mae_nucleus_soft_grid.example.yaml` defines a strict, bounded search over learning rate,
scheduler, patch encoder, embedding dimension, reconstruction shape, mask ratio, weight decay, and
normalization. `one_at_a_time` saves a baseline plus one changed setting per run; `cartesian` is
available but still subject to `max_runs` and an absolute 64-run limit. Arbitrary configuration
paths and commands are rejected.

```bash
morphofeatures mae-sweep-prepare \
  --config configs/sites/mae_platynereis_nuclei_embl.yaml \
  --profile quick \
  --sweep configs/mae_nucleus_soft_grid.example.yaml

morphofeatures mae-sweep-compare \
  --sweep outputs/sweeps/nucleus-local-texture-ablation \
  --output outputs/sweeps/nucleus-local-texture-ablation/comparison/metrics.tsv \
  --plot outputs/sweeps/nucleus-local-texture-ablation/comparison/loss_curves.png \
  --labels analysis/data/class_labels.tsv --classifier-model logistic
```

Preparation never submits. Submit the prepared manifest explicitly through Streamlit or the
guarded notebook cell. The comparison module also provides matched held-out reconstruction grids;
these are opt-in because every checkpoint and real N5 data must be loaded.

`loss_masks` is optional for backward compatibility, but recommended for segmentation-masked
cell/nucleus crops. It has shape `(n,z,y,x)` or `(n,1,z,y,x)`. Token locations are still masked
randomly; voxel MSE is accumulated only where a masked token overlaps foreground. At least one
foreground token is included in each sample's MAE mask. Without this weighting, mostly empty
windows can produce deceptively small losses by rewarding background reconstruction.

## Preprocessing compared with the legacy texture route

The legacy coarse loader uses authoritative physical bounds and cell-to-nucleus mappings, centers
on the mapped nucleus, masks raw intensity to cell/nucleus, and makes paired augmented views for
the contrastive-plus-reconstruction objective. The legacy fine loader samples multiple
high-resolution texture patches and averages their embeddings per cell.

An MAE migration should preserve those biological sampling decisions where possible. For generic
crops, the pretext task is random internal voxel-patch masking. For the audited N5 store, the
verified unit is instead a spatial sequence of complete texture patches, so v3 masks whole patches.
Neither route removes
mapping, physical resolution, coarse/fine scale, label-aligned aggregation, or augmentation QC.
Save model crops as finite float32 data, save foreground masks separately, split label IDs before
training, and record crop coverage and normalization. For the indexed fine-texture route, retain
patch coordinates and return one sequence embedding under the original `label_id`.

When `data.label_ids` is absent, encoding assigns deterministic smoke IDs `1..n`; real experiments
should provide one finite, unique, integer-valued segmentation label per crop. Training appends
preprocessing, batch progress, epoch/train/validation/learning-rate, early-stopping, and
checkpoint/final events to `metrics.jsonl` beside the checkpoint (or the path selected by the
SLURM run environment). Epoch events also contain the MSE
of a predictor that fills hidden foreground with each sample's visible-foreground mean and the
relative improvement over that baseline. A small normalized MSE is not useful unless it beats this
baseline and held-out reconstructions preserve spatial structure.

For v3 reconstruction QC, compare the hidden and predicted patch at the common configured
downsampled scale. With `norm_pix_loss: true`, the loss is standardized per target patch and a
no-information value is near one; the QC helper reverses that normalization with the target's
mean/std only for visualization. Reconstruction is a pretext-task diagnostic, not generated
high-resolution EM or evidence that stochastic ultrastructure can be inferred exactly.

## Compare against legacy embeddings

Use identical label IDs and train/evaluation splits. Export both methods with column 0 as `label_id`, then run the same commands:

```bash
python -m morphofeatures classify --embedding outputs/legacy.npy --seed 42 \
  --output-dir outputs/legacy_classifier
python -m morphofeatures classify --embedding outputs/mae.npy --seed 42 \
  --model logistic --class-weight balanced --output-dir outputs/mae_classifier
python -m morphofeatures project --embedding outputs/legacy.npy --output outputs/legacy_umap.tsv
python -m morphofeatures project --embedding outputs/mae.npy --output outputs/mae_umap.tsv
```

Report embedding dimension, crop/patch resolution, mask ratio, seed, training cells, and checkpoint. MAE and contrastive loss scales are not directly comparable; compare downstream metrics and biological consistency.

The classifier joins embeddings and curated labels by `label_id`, permits a partial ID overlap,
drops classes with fewer than the configured minimum examples, and standardizes within each
cross-validation fold. `--model mlp --hidden-dimensions 64` provides a shallow nonlinear probe.
Saved outputs include out-of-fold predictions, a confusion matrix, per-class recall, and a JSON
summary. Especially for the quick 1,000-nucleus cohort, small/imbalanced label overlap makes this
illustrative rather than a biological benchmark. Cross-validation within one animal does not
establish independent-animal generalization.

The shared `EmbeddingMethod` interface exposes `train`, `encode_cells`, `aggregate_patches`, `export_embeddings`, and `evaluate_embeddings` for future self-supervised methods.
