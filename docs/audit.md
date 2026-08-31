# Repository audit

## Scope and inventory

The pre-restoration repository contained 20 Python files, one one-line README, no package metadata, no tests or configs, three published NumPy feature matrices, analysis TSV/pickle inputs, and 12 MoBIE tables. The attached paper PDF is untracked and was treated only as scientific context.

The original code paths were:

- `morphofeatures/shape`: DeepGCN network, geometric augmentations, broken trainer, broken inference.
- `morphofeatures/texture`: PlatyBrowser-specific volume loader, datasets, Inferno/Neurofire trainer, z5 predictor.
- `analysis`: class prediction, UMAP/Leiden, reclustering, bilateral ranking, gene plots.
- `analysis/data`: published feature and annotation artifacts.
- `data_mobie`: MoBIE-ready feature and downstream tables.

## Findings

- Package initializers, metadata, dependency declarations, configuration, entrypoints, tests, and reproducibility documentation were absent.
- `texture/cell_loader.py` embedded `/scratch/zinchenk/cell_match/data/platy_data` and fixed PlatyBrowser paths/dataset keys.
- Active texture imports required Inferno, Neurofire, z5py, and pybdv before any command could start.
- `shape.data_loading.loader` and `shape.utils` were missing.
- Shape training used a global `config`, trained on `val_loader`, called an undefined scheduler, forced CUDA/DataParallel, and imported WandB unconditionally.
- Shape inference used only the first batch, loaded checkpoints without `map_location`, and converted device tensors directly to NumPy.
- Analysis used removed scikit-learn and NetworkX APIs, mutable global class lists, global workflow state, nondeterministic CV/UMAP, script-relative imports, and cwd-relative data paths.
- Texture prediction forced `.cuda()`, assumed GPU IDs, and mixed `.np`, `.npy`, and `.z5` behavior.
- Original raw volumes, training configs, checkpoints, mesh preprocessing scripts, and private infrastructure are not bundled.

## Decisions

- Preserve all published artifacts and label-first file conventions unchanged.
- Make NumPy/pandas/scikit-learn/PyYAML the dependency-light core.
- Load UMAP/Leiden, Torch, WandB, Textual, mesh, and legacy volume libraries only when their workflows are requested.
- Replace active Inferno/Neurofire training with a maintained native Torch 3D autoencoder. Keep historical N5/BDV readers as lazy optional adapters.
- Restore shape loaders around explicit point-cloud/mesh manifests and reimplement OFF/graph/preprocessing utilities.
- Standardize runtime behavior in `training_runtime.py`: `device=auto`, no forced CUDA, optional DataParallel, map-location checkpoint loading, normalized state-dict keys.
- Add a common embedding-method contract and a pragmatic transformer-based 3D MAE.
- Anchor all default paths to config or module location, never shell cwd.
- Provide Leiden when optional dependencies are present and deterministic K-means for lightweight smoke validation.

## Validation boundary

Validated locally: published shapes/IDs, all MoBIE headers, bundled-label logistic regression, contracts, synthetic generation logic, aggregation, CLI validation, and syntax. UMAP is installed in the local legacy Anaconda environment but its `llvmlite.dll` is broken, so the UMAP test skips there. Torch, igraph, Leiden, trimesh, zarr, z5py, and pybdv are absent locally; their tests or commands report the missing optional group. Raw-volume model fidelity cannot be validated without external EM data and original checkpoints.
