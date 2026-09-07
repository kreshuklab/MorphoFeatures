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

## Real nucleus MAE has a very small loss but smooth gray reconstructions

Check the resolved `mae.architecture_version`. The audited N5 arrays are already local `32³`
texture patches; they must use `grouped-nucleus-patches-v3`, where a sample groups patches from
one parent and masking hides complete patches. The superseded v2 real-data route hid internal
blocks inside one local patch and could lower raw `[0,1]` MSE by predicting conditional means.
Its checkpoints are intentionally rejected by v3. With `norm_pix_loss: true`, expect losses near
one initially and inspect predictions at the configured `8³` target scale. Even a correct MAE may
produce blurry reconstructions because stochastic ultrastructure is not exactly predictable;
held-out baseline improvement and downstream embedding validation are both required.

## SLURM submission or accounting fails

Save a dry-run preview first and inspect the exact script. `sbatch` must be available on the host
running Streamlit. Validate partition/account/QoS/resource values against local documentation.
Submission errors are retained in the registry. Monitoring queries `squeue` first and `sacct` for
jobs no longer active; accounting may be delayed or purged, in which case the normalized state is
`unknown` while explicit logs and artifacts remain accessible. See `docs/slurm_workflow.md`.
