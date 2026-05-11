# Training New MorphoFeatures Encoders

An **encoder** is a neural network architecture plus trained weights. This repository contains the architectures and training code, but no pretrained checkpoint files. To obtain an encoder for a new dataset, you either train one or bring in a checkpoint from a previous run.

The full flow is:

```text
raw input data -> train encoder -> checkpoint -> run inference -> embedding matrix -> analysis
```

## Shape Encoder

Use the shape encoder when your input for each cell is a point cloud, usually sampled from a cell or nucleus surface.

### Input Format

The easiest input is a `.npz` file with:

- `points`: array shaped `(cells, points, 3)` or `(cells, 3, points)`
- `features`: optional array shaped `(cells, points, channels)` or `(cells, channels, points)`, for example coordinates plus normals
- `ids`: optional array with one cell label ID per row

If `features` is omitted, the model uses `points` as features.

### Train

First validate the config without launching GPU training:

```bash
morphofeatures-train shape --config configs/shape_train.yaml --dry-run
```

Then train:

```bash
morphofeatures-train shape --config configs/shape_train.yaml
```

Training writes checkpoints to:

```text
<experiment_dir>/checkpoints/
```

The best checkpoint is named like:

```text
best_ckpt_iter_<step>.pt
```

### Generate Embeddings

After training, copy `configs/shape_inference.yaml` and set `model.checkpoint` to the checkpoint file, then run:

```bash
morphofeatures-shape embed --config path/to/shape_inference.yaml --save-to runs/shape_embeddings.npy
```

That output is the embedding matrix consumed by analysis.

## Texture Encoder

Use the texture encoder when your input is volumetric EM data plus cell and nucleus segmentations.

### Input Format

Texture training needs:

- raw EM intensity volume in z5/n5-compatible format,
- cell segmentation volume, typically referenced by BigDataViewer XML,
- nucleus segmentation volume, also usually a BigDataViewer XML,
- `cells_to_nuclei.tsv` mapping cell label IDs to nucleus label IDs,
- cell metadata TSV with `label_id`, `bb_min_z/y/x`, `bb_max_z/y/x`, `anchor_z/y/x`,
- nucleus metadata TSV with the same metadata columns.

### Train

The texture trainer follows the original project-directory style. A run directory should contain:

```text
my_texture_run/
  train_config.yml   # model, loss, optimizer, epochs
  data_config.yml    # data paths, transforms, dataloaders
```

You can start from:

```text
configs/texture_model_train.yaml -> my_texture_run/train_config.yml
configs/texture_train.yaml       -> my_texture_run/data_config.yml
```

Validate without launching training:

```bash
morphofeatures-train texture my_texture_run --dry-run
```

Train:

```bash
morphofeatures-train texture my_texture_run --devices 0
```

Training writes checkpoints to:

```text
my_texture_run/Weights/
```

### Generate Embeddings

After training:

```bash
morphofeatures-texture predict my_texture_run --devices 0
```

This writes:

```text
my_texture_run/avg_encoded.np
```

For fine texture patch encoders, you can save and aggregate patch features:

```bash
morphofeatures-texture predict my_texture_run --devices 0 --save-patches --aggregate-patches
```

## What the Embedding Matrix Looks Like

Both encoder families produce the same downstream format:

```text
label_id  feature_0  feature_1  ...  feature_n
```

Each row is one cell. Column 0 identifies the cell, and the remaining columns are learned morphology features.

## Practical Advice

- Start with `--dry-run`; it catches missing config fields before importing heavy training dependencies.
- Use the shape pipeline first if you already have meshes or point clouds. It has fewer external volume-data assumptions.
- Use the texture pipeline only once raw volumes, segmentations, and metadata tables are aligned.
- Keep separate encoders for cell shape, nucleus shape, coarse texture, and fine texture if you want to reproduce the paper-style decomposition.
