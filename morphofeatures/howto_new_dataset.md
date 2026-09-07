# Mapping a new dataset onto MorphoFeatures

This guide starts from the data decisions that must be made before choosing a model. The primary
worked example is `notebooks/04_real_platynereis_mae_workflow.ipynb`, which consumes local,
nucleus-masked EM patches. The same contracts apply to new organisms and acquisition campaigns,
but the biological unit, physical resolution, mask producer, and split groups must be stated for
each dataset.

## First decide what one feature row represents

Write one sentence before preparing arrays, for example:

> One row represents the nucleus mapped to cell segmentation `label_id`, summarized from local
> nucleus-masked EM texture patches.

Other legitimate choices include a complete cell shape, complete nucleus shape, coarse cell
texture, or fine cell texture. Do not call a nucleus-only result “complete MorphoFeatures.” The
historical 480-dimensional representation concatenates six separately trained 80-dimensional
groups after exact ID alignment:

1. cell shape;
2. nucleus shape;
3. coarse cell texture;
4. coarse nucleus texture;
5. fine cell texture;
6. fine nucleus texture.

Fine patch features are aggregated per cell. All joins use `label_id`, never row position.

## Required source provenance

Record immutable versions or checksums for:

- raw volume and pyramid level;
- cell segmentation;
- nucleus segmentation;
- cell-to-nucleus mapping;
- cell/nucleus metadata tables and any exclusion list;
- crop/patch producer and configuration;
- voxel resolution in explicit `(z,y,x)` order;
- animal, batch, sample, and acquisition identifiers when available.

Segmentations must contain integer labels with zero reserved for background. Manually inspect
size outliers and likely merge errors; pieces attached to more than roughly 25% of the original
cell can substantially change texture and shape features. Preserve proofreading and inclusion
versions rather than editing source volumes in place.

## Option A: use an indexed N5 patch store

The maintained real-data MAE accepts:

| Input | Required contract |
|---|---|
| patch dataset | N5 `uint8`, shape `(n_patches,z,y,x)`, fixed spatial shape |
| positions dataset | N5 `int64`, shape `(n_patches,4)`, columns `(label_id,z,y,x)` |
| IDs dataset | N5 integer vector containing each unique positive parent ID exactly once |
| ordering | position rows grouped in increasing `label_id`; one row per patch row |
| coordinate convention | explicit integer `(z,y,x)` centers with a documented grid resolution |

Use `morphofeatures n5-inventory` to inspect metadata without visiting patch chunks:

```bash
morphofeatures n5-inventory /path/to/patches.n5 /path/to/positions.n5 \
  --output outputs/inventory/new_dataset.json
```

If the patch store contains a different dtype, rank, independent mask, or non-center coordinate
semantics, write a versioned converter rather than weakening the validator. A converter should
write a manifest linking each output row to its input label and coordinate.

## Option B: start from aligned raw and segmentation volumes

For coarse cell/nucleus experiments, prepare fixed, patch-divisible `(z,y,x)` crops from aligned
raw and label volumes. Store the target binary mask independently when possible. Crop by a
documented anchor such as the mapped nucleus center, and save:

- crop array `(n,1,z,y,x)` or the format required by the selected legacy loader;
- one positive integer `label_id` per sample;
- loss masks when foreground-aware MAE training is intended;
- physical origin/resolution and crop bounds;
- rejected IDs with reasons such as border truncation or insufficient coverage.

For fine texture, generate many local patches but preserve the parent ID for every patch so that
splitting and aggregation happen at the biological unit. The older Platynereis ROI example in
`docs/platyneris_data_inventory.md` demonstrates a volume-to-crop route; the indexed patch store
in `docs/legacy_workspaces_inventory.md` is the maintained primary example.

## Create a site YAML from the portable template

Copy `configs/mae_nucleus_patches_template.yaml` outside version control or into a clearly named
site-config directory. Fill at least:

```yaml
data:
  source: n5_masked_patches
  patches_container: /path/to/patches.n5
  patches_key: patches
  positions_container: /path/to/positions.n5
  positions_key: positions
  ids_key: ids
  modality: EM
  biological_unit: nucleus
  axes: zyx
  resolution_zyx_um: [0.025, 0.020, 0.020]
  position_resolution_zyx_um: [0.100, 0.080, 0.080]
  patch_shape_zyx: [32, 32, 32]
  group_size: 200
  position_stride_zyx: [8, 8, 8]
  mask_mode: nonzero
  normalization: dtype

mae:
  architecture_version: grouped-nucleus-patches-v3
  input_shape: [32, 32, 32]
  reconstruction_shape: [8, 8, 8]
  patch_encoder: linear       # or resnet3d
  resnet_channels: [16, 32, 64]
  resnet_blocks: [1, 1, 1]
  norm_pix_loss: true

training:
  learning_rate: 0.001
  scheduler: constant         # constant, cosine, or step
  warmup_epochs: 0
  progress_interval_batches: 25
  early_stopping:
    enabled: false            # opt in only after choosing a defensible patience
    patience: 10
    min_delta: 0.000001
    restore_best: true

paths:
  run_dir: /writable/output/new_dataset/nucleus_texture
```

The position-to-patch resolution ratio must be integral for the audited center transform. The
Platynereis example uses factor four on every axis; do not reuse that factor unless your producer
does. Optional `qc_raw_*`, center radius, and scale fields enable read-only raw/masked alignment
views and never change the training source.

Use a compact notebook override only for an intentional run-local change:

```python
overrides = {
    "seed": 42,
    "paths": {"run_dir": "/writable/output/new_dataset/quick_001"},
}
config = resolve_real_mae_config(
    Path("/path/to/site.yaml"), profile="quick", overrides=overrides
)
config.save()  # durable resolved snapshot used by CLI and SLURM
```

Do not maintain a second notebook-only configuration schema.

## Inspect before training

At minimum, verify and visualize:

- array keys, shapes, dtypes, chunking, compression, and axes;
- exact patch/position row equality;
- finite, positive, integer-valued, unique parent IDs;
- exact equality between the IDs vector and unique position IDs;
- coordinate bounds and a sample reconstructed against raw source when available;
- intensity range/distribution before and after normalization;
- non-empty masks and foreground-fraction distribution;
- duplicate full `(label_id,z,y,x)` positions and sampled duplicate patch content;
- smallest/largest parent patch counts and extreme occupancy cases;
- configured flips/rotations on real examples.

An already masked patch does not necessarily contain an exact independent mask: using `value !=
0` conflates outside-mask background with true zero raw intensity. Prefer a stored binary mask for
new preparations when feasible and document the fallback when it is not.

## Split to answer the scientific question

Never split repeated texture patches independently. Split unique parent IDs first. When several
animals, batches, acquisitions, or correlated anatomical groups exist, group by the highest unit
needed for the claimed generalization. A held-out nucleus from the same animal does not establish
cross-animal generalization.

Save the resolved split manifest. Before training, assert that parent-ID intersections among
train, validation, and test are empty. Keep the test set untouched during hyperparameter/model
selection.

## Normalization and augmentation

`dtype` normalization maps integer dtype limits to `[0,1]` and matches the audited legacy `uint8 /
255` behavior. `foreground_percentile` and `foreground_zscore` are available for controlled
comparisons. Report the choice because it changes the reconstruction target and loss scale.

For grouped-patch v3, flips and rotations are intentionally disabled: every local patch and its
relative physical coordinate would have to receive the same transform, and anisotropic axes must
not be exchanged. Add a coherent group transform only after visual and biological review.

Patch size should cover a biologically meaningful local texture scale. Larger patches are not
automatically better: they consume memory and can mix organelles/textures. Report physical patch
extent, complete-patch masking unit, reconstruction downsample, mask ratio, and group size.

## Train, resume, and evaluate

Use `quick` to validate reads, gradients, metrics, checkpoint reload, and label-first encoding.
Use a new run directory for a changed configuration. Resume only from an explicit checkpoint and
preserve its resolved YAML and metric history.

Generic crop MAEs declare `position-aware-3d-v2`; the indexed nucleus-patch route declares
`grouped-nucleus-patches-v3`. A checkpoint must exactly match the selected model contract. Do not
resume v3 from a v1/v2 checkpoint: the biological sampling unit and reconstruction target differ.

Within v3, `linear` and `resnet3d` are alternative local patch encoders. The linear baseline sees
all patch voxels through one dense projection. The ResNet uses hierarchical 3D convolutional
receptive fields and therefore adds a local-texture inductive bias before the nucleus-level
Transformer. Changing the encoder is a new experiment and needs a new checkpoint; it does not
change the parent-ID split, whole-patch masking unit, or label-first output contract.

For small ablations, copy `configs/mae_nucleus_soft_grid.example.yaml` and use
`one_at_a_time` to vary one allowlisted field relative to the base profile. This is usually easier
to interpret than a Cartesian search and preserves the untouched test set for final evaluation.
Preparing a manifest does not submit jobs. Use a new sweep name, inspect every generated script,
then explicitly submit through Streamlit or the guarded notebook cell. Compare raw curves and
matched reconstructions first; after dependent encoding, run the same ID-joined shallow probe or
other prespecified downstream analysis for each variant.

For a serious run, inspect all of the following:

- training and held-out validation loss;
- visible-patch mean baseline loss and improvement over that baseline;
- complete hidden-patch reconstructions over several IDs and checkpoints;
- empty/replacement counts and input occupancy;
- embedding finiteness, dimension, unique labels, and exact expected coverage;
- distribution and stability of the one-per-parent sequence embeddings;
- stability over seeds/checkpoints;
- biological annotations joined by ID;
- prespecified classification, retrieval, neighborhood, or clustering diagnostics.

A completed SLURM job, falling training loss, clean UMAP, or high classifier score alone does not
prove a biological discovery. Check technical covariates, leakage, class imbalance, spatial
autocorrelation, and independent validation.

## SLURM profile and security

Cluster setup entries are argument lists and accept only reviewed `module load/purge/use` or
`source /path/to/activation-script` operations. Free-form shell text is rejected. Configure
partition, account, QoS, time, memory, CPUs, and GPUs in a private site profile; never commit
credentials or institution-specific accounts as defaults.

Preview command and script, then submit explicitly through Streamlit or the guarded notebook cell.
The job registry, resolved config, stdout/stderr, metrics, checkpoints, embedding metadata, and
run metadata remain below the configured output root. Use an `afterok` dependency for encoding
after successful training rather than blocking Jupyter or Streamlit.

## Shape and legacy texture alternatives

The MAE is one texture option, not a universal replacement. Shape workflows need point clouds or
meshes with physical coordinate provenance. Legacy coarse/fine texture workflows need their
volume, mask, mapping, and patch-position configuration. If comparing methods, keep biological
cohort, resolution, crop/patch policy, split, aggregation, and downstream evaluation fixed.

When no suitable proxy task exists, a larger embedding can be trained and reduced later, but do
not choose dimension solely from an attractive projection. The historical dimension of 80 is the
appropriate interoperability choice when building one of the six original groups.
