# Installation

## Supported environment

Use Python 3.9 or newer. A fresh virtual environment is recommended because UMAP/Numba, Torch, torch-cluster, and igraph have compiled components.

```bash
python -m venv .venv
.venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e ".[analysis,dev]"
```

On Linux/macOS, activate with `source .venv/bin/activate`.

Verify the install:

```bash
python -m morphofeatures doctor
python -m morphofeatures validate
python -m pytest -q
```

Install and open the local workflow workspace:

```bash
python -m pip install -e ".[ui]"
python -m morphofeatures ui
```

The workspace opens at `http://localhost:8501`. Use `--port` when that port is occupied and `--headless` on remote machines.

Install the end-to-end notebooks (including the CPU MAE smoke workflow):

```bash
python -m pip install -e ".[analysis,modern-training,notebooks]"
jupyter lab
```

Start Jupyter from the repository root. See [the notebook guide](../notebooks/README.md).

## Training environments

```bash
python -m pip install -e ".[legacy-training]"
python -m pip install -e ".[modern-training]"
```

Install a CPU or CUDA Torch wheel appropriate for the machine first. `torch-cluster` must match the Torch and CUDA versions. WandB is separate and inactive unless config contains `wandb.enabled: true`:

```bash
python -m pip install -e ".[wandb]"
```

For historical BDV XML and N5 stores, the maintained path uses `pybdv` plus `zarr`. Exact legacy z5 containers may require:

```bash
conda install -c conda-forge z5py
```

Inferno and Neurofire are no longer required by active training. Old serialized Inferno Trainer objects remain environment-specific and should be exported to a plain state dict before migration.

## Configuration overrides

`configs/default.yaml` resolves paths against the repository. Two environment variables are supported for deployment-level overrides:

- `MORPHOFEATURES_DATA_ROOT`: external volume/table root.
- `MORPHOFEATURES_OUTPUT_ROOT`: output/checkpoint root.

Explicit YAML paths take precedence for individual files and datasets. CUDA and WandB are never selected silently.

The Streamlit scheduler backend additionally needs `sbatch`; monitoring uses `squeue` and
`sacct` when available. Their absence leaves dry-run, registry, logs, metrics, artifacts, and
local analysis usable. Configure placeholder-free cluster profiles as described in
`docs/slurm_workflow.md`; profiles contain no credentials.
