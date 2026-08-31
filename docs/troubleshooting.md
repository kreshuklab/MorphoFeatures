# Troubleshooting

## Optional dependency reported missing

Run `python -m morphofeatures doctor`. Install only the group needed by the command. Core validation and class prediction do not require Torch, CUDA, WandB, igraph, or private data.

## UMAP imports but llvmlite fails

UMAP depends on a compatible Numba/llvmlite pair. Create a fresh environment and reinstall the `analysis` group. A broken shared library is treated as an unavailable optional dependency, and UMAP tests skip with the underlying error.

## CUDA requested but unavailable

Use `device: cpu` or `--device cpu`. `auto` selects CUDA only when Torch reports it available. Checkpoints always load through `map_location` and are portable to CPU.

## DataParallel checkpoint key errors

Use `load_checkpoint` from `morphofeatures.training_runtime`. It removes a uniform `module.` prefix and loads into the unwrapped model. Architecture mismatches still fail intentionally.

## N5 or BDV input cannot open

For BDV XML install `pybdv`. For maintained stores prefer Zarr. Historical z5 N5 containers require `z5py`, commonly installed through conda-forge. Set explicit `raw_dataset`, `cell_dataset`, and `nucleus_dataset` keys in texture data config.

## Empty or mismatched embeddings

Column 0 must contain unique integer label IDs. Group concatenation sorts IDs and rejects any mismatch. Patch aggregation groups by IDs and does not assume contiguous patch rows.

## Raw-data training cannot reproduce the paper exactly

The repository does not include original raw EM volumes, exact training configs, checkpoints, or all preprocessing intermediates. Use bundled arrays for exact downstream reference and record new preprocessing/model choices as a new experiment.
