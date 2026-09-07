# MorphoFeatures notebooks

Run Jupyter from the repository root and execute the notebooks in order:

1. `01_data_preparation_and_contracts.ipynb` — visual synthetic raw/label/mask inspection, `z, y, x`, physical resolution, label-first embeddings, QC, and deterministic splits (about 1 minute; core plus plotting dependencies).
2. `02_cpu_mae_training_and_encoding.ipynb` — visual crop preparation, MAE patch masking, foreground-aware CPU training, reconstruction, checkpoint reload, and encoding (about 1–3 minutes; `modern-training` and `notebooks`).
3. `03_biological_analysis_and_interpretation.ipynb` — bundled six-group analysis, UMAP/PCA, clustering, classification, context, and MoBIE export (about 3–10 minutes; `analysis` and `notebooks`).
4. `04_real_platynereis_mae_workflow.ipynb` — real indexed nucleus-patch metadata, raw/masked QC, grouped splits, linear/3D-ResNet patch encoders, quick MAE training/reconstruction/encoding, bounded soft-grid SLURM previews/comparison, a shallow cell-type probe, and cautious legacy comparison (conda `z5py` required; runtime depends on shared-filesystem I/O).

MAE notebooks use the `position-aware-3d-v2` checkpoint contract and write to versioned run
directories. Unversioned checkpoints from the earlier position-blind implementation are rejected;
start a fresh run rather than reusing or resuming them.

Install everything used by the collection with:

```bash
python -m pip install -e ".[analysis,modern-training,notebooks]"
jupyter lab
```

Generated files are written below the configured output root. Notebook 3 falls back to PCA with an explicit message when UMAP is unavailable. Shape and legacy texture training are documented but not executed because their raw volumes, meshes, exact historical configs, and checkpoints are not bundled.

Notebook 4 uses `configs/sites/mae_platynereis_nuclei_embl.yaml` when its external paths are
mounted. Else it resolves `configs/mae_nucleus_patches_template.yaml` from the two
`MORPHOFEATURES_NUCLEUS_*` environment variables. Without either, real-data cells skip cleanly.
See [`docs/legacy_workspaces_inventory.md`](../docs/legacy_workspaces_inventory.md) for the audited
lineage, keys, resolution evidence, QC, storage costs, and remaining questions.

The narrative cell sources for notebooks 01 and 02 live in `build_tutorial_notebooks.py`.
Notebook 4 is canonical in notebook form; `build_real_mae_notebook.py` only validates it and
clears execution state, avoiding a second stale copy of its audited narrative.
