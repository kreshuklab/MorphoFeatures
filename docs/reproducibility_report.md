# Reproducibility report

## Restored functionality

- Package installation and optional dependency groups.
- Root-independent CLI/config path handling.
- Published NumPy and MoBIE artifact validation.
- Deterministic class prediction and small-subset projection/clustering tests.
- Label-first embedding I/O, group alignment, patch averaging, neighbor aggregation, and feature agglomeration.
- Shape point-cloud loading, mesh utilities/preprocessing, fixed DeepGCN training, and complete inference iteration.
- Native 3D texture training/inference without active Inferno or Neurofire dependencies.
- Portable device/checkpoint behavior and opt-in WandB.
- Transformer-based 3D MAE, synthetic fixture, tests, and Streamlit workflow workspace.
- Ordered, visualization-rich data-contract, CPU MAE, biological-analysis, and optional real-N5
  notebooks, with automated in-order CPU execution of the synthetic training workflow.
- Safe workflow commands, structured SLURM profiles/scripts, dry-run/fake/real backends,
  scheduler-state normalization, afterok dependencies, and confirmed cancellation.
- Persistent WAL-mode SQLite jobs, immutable snapshots, provenance, duplicate protection,
  JSONL metrics, bounded log tails, and explicit artifacts.

## Fixed defects

The private scratch root, forced `.cuda()` calls, hard-coded GPU IDs, missing shape modules, global trainer config, validation-as-training loader, undefined scheduler, first-batch inference break, stale scikit-learn warning API, stale NetworkX sparse API, cwd-relative paths, and non-package imports were removed or replaced.

## Published-data validation

Observed locally:

```text
morphofeatures_all_cells.npy                     (11382, 481) PASS
morphocontextfeatures_all_cells_agglomerated.npy (10391, 201) PASS
manually_defined_features.npy                    (11348, 141) PASS
MoBIE tables with label_id                       12/12 PASS
```

The bundled-label logistic-regression smoke test executes on 390 labels. The current local suite
reported `39 passed, 1 skipped`; the optional UMAP test skipped because `umap-learn` is not
installed. Torch was available, so the MAE unit tests and complete CPU notebook workflow
executed. The revised notebook prepared four segmented synthetic cells, trained with an explicit
foreground loss mask, saved/reloaded a checkpoint, encoded 16 features per label, and validated
finite unique label-first output.

## Bounded real-data validation

The permitted local Platynereis N5 folder was inventoried read-only. A bounded raw `s3` ROI and
aligned Paintera fragment `s2` ROI were mapped through the curated filtered/unmerged node labels,
yielding 31 fixed `32³` crops after completeness, size, and ≥90% coverage checks. The
`MorphoFeats_dev` environment had a CUDA PyTorch build but no visible GPU in this session, so the
10-epoch run used CPU. Foreground-aware train/validation loss decreased from approximately
0.400/0.439 to 0.198/0.215. Checkpoint reload and encoding produced a finite label-first
`(31,33)` array with exact input-ID preservation.

All 18 code cells of the real-data notebook executed successfully against the permitted root. An
executed copy with rendered QC figures was saved below the ignored output root. The complete
source inventory, N5 keys, ROI, observed counts, rejected candidates, and interpretation limits
are in `platyneris_data_inventory.md`.

## Commands

```bash
python -m morphofeatures validate
python -m morphofeatures classify --folds 5 --seed 42
python -m morphofeatures project --subset 256 --cluster-method kmeans \
  --output outputs/smoke_projection.tsv
python -m pytest -q
```

The complete configured Ruff run was also executed. It reports existing repository-wide style
debt (primarily import ordering, f-string and typing-modernization rules, including rules that
conflict with the retained Python 3.9 annotation style). A focused `E`, `F`, and `B` safety check
over every changed Python/test file, with the configured `E501` exclusion, passes. No broad legacy
style rewrite was performed for this workflow change.

Training and inference commands are documented in `training_new_embeddings.md` and `modern_mae_workflow.md`.

## Known limitations

- External raw EM, cell/nucleus segmentations, original meshes, and original checkpoints are required for full retraining.
- The exact historical Neurofire architecture and serialized Inferno trainer are not recoverable from the repository; a maintained 3D autoencoder preserves the scientific objectives and 80-dimensional contract.
- Exact published UMAP/Leiden coordinates may vary with dependency versions; bundled MoBIE tables are authoritative.
- Local validation executed the bounded N5/MAE route, but not a full DeepGCN or legacy texture
  experiment: `torch_cluster` and the authoritative full cell/nucleus MoBIE tables were absent.
  igraph and Leiden were also unavailable.
- UMAP is not installed in the current environment; the optional test records the skip and the
  biological notebook uses an explicit PCA fallback.
- No live SLURM cluster was available. Submission, refresh, dependencies, failure, and cancellation
  were validated with fake and dry-run backends; scheduler-command success is not claimed.
- The local PyTorch build includes CUDA, but `torch.cuda.is_available()` was false and NVML could
  not initialize in this session; no GPU execution is claimed.
