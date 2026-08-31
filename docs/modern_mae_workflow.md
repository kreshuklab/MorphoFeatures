# Modern 3D MAE workflow

The modern pathway tokenizes 3D crops with a strided convolution, masks a configurable fraction of tokens, encodes masked and visible tokens with a Transformer, and reconstructs voxel patches. Training loss is computed only on masked patches. Mean encoded tokens form the cell/patch embedding.

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
mae:
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

## Compare against legacy embeddings

Use identical label IDs and train/evaluation splits. Export both methods with column 0 as `label_id`, then run the same commands:

```bash
python -m morphofeatures classify --embedding outputs/legacy.npy --seed 42
python -m morphofeatures classify --embedding outputs/mae.npy --seed 42
python -m morphofeatures project --embedding outputs/legacy.npy --output outputs/legacy_umap.tsv
python -m morphofeatures project --embedding outputs/mae.npy --output outputs/mae_umap.tsv
```

Report embedding dimension, crop/patch resolution, mask ratio, seed, training cells, and checkpoint. MAE and contrastive loss scales are not directly comparable; compare downstream metrics and biological consistency.

The shared `EmbeddingMethod` interface exposes `train`, `encode_cells`, `aggregate_patches`, `export_embeddings`, and `evaluate_embeddings` for future self-supervised methods.
