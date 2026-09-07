# End-to-end notebooks

Launch Jupyter from the repository root so package imports and repository-relative configs are
unambiguous:

```bash
python -m pip install -e ".[analysis,modern-training,notebooks]"
jupyter lab
```

Execute the notebooks under `notebooks/` in numeric order.

## 1. Data preparation and contracts

`01_data_preparation_and_contracts.ipynb` generates the deterministic fixture under the
configured output root and pauses to visualize aligned raw/cell/nucleus slices, one masked cell,
physical bounds, label QC, and the label-first feature matrix. It validates `z, y, x`, mapping and
metadata IDs, finite/unique embeddings, missing-ID diagnostics, and deterministic ID-based
splits. It uses only dependency-light scientific packages and normally takes less than a minute.

Replace fixture paths through YAML or `MORPHOFEATURES_DATA_ROOT`; do not hard-code private
cluster paths in the notebook. Real pyramid arrays must provide matching dataset keys and
physical resolutions.

## 2. CPU MAE training and encoding

`02_cpu_mae_training_and_encoding.ipynb` creates masked crops from segmented synthetic cells,
shows the raw/mask/model-input distinction, visualizes MAE token masks, trains three small CPU
epochs with foreground-aware reconstruction loss, plots structured metrics, reloads a portable
checkpoint, shows a composite reconstruction, encodes four cells, and validates the label-first
array. It normally takes 1–3 minutes. The test suite executes these code cells in order on CPU
when Torch and notebook tooling are installed.

Scaling requires externally prepared crops and IDs. Train cell/nucleus and coarse/fine
experiments separately. Shape and legacy texture examples are documented rather than executed:
they require external meshes or registered volumes and optional compiled libraries. Fine texture
patches remain aggregated per cell.

## 3. Biological analysis and interpretation

`03_biological_analysis_and_interpretation.ipynb` loads all six bundled 80-dimensional groups,
enforces identical label sets, combines them to 480 features, standardizes distance/model inputs,
runs UMAP (or an explicit PCA fallback) and deterministic K-means, reports stratified
classification and per-class recall/confusion, compares selected groups under the same protocol,
constructs neighbor context, and writes MoBIE-compatible tables. Typical runtime is 3–10 minutes.

Projection is exploratory, clustering proposes parameter-dependent partitions, classification
measures prediction on the available annotations, and context encodes a chosen neighborhood
definition. None alone establishes biological mechanism or a discrete cell type. Validate
findings against registered imagery, segmentation QC, spatial/anatomical covariates, independent
specimens, and domain knowledge.

## 4. Real Platynereis MAE workflow

`04_real_platynereis_mae_workflow.ipynb` is the primary real-data tutorial. It selects either the
portable YAML template or the separate EMBL site config and saves one validated resolved config.
It discovers N5 metadata without walking chunks, validates 2,506,460 patch/index rows and 11,382
parent IDs, visualizes optional unmasked raw beside stored nucleus-masked patches, quantifies
quality, constructs leakage-safe ID splits, and explains why complete stored patches—not internal
voxel blocks—are the correct tokens for this audited nucleus sequence.

The `quick` profile reads a bounded subset, trains a grouped model, plots metrics/reconstructions,
reloads a portable checkpoint, encodes one spatial group per ID, and exports label-first nucleus-
texture embeddings. It explains and exposes both the checkpoint-compatible linear patch encoder
and a hierarchical 3D ResNet patch encoder. A bounded soft-grid section previews one-at-a-time
architecture/optimization ablations, safe SLURM scripts, raw metric comparisons, and opt-in
matched reconstruction panels. Nothing is submitted without explicit boolean and phrase guards.

The notebook also joins the real embeddings to curated cell types by `label_id` and evaluates a
linear or shallow-MLP probe with fold-local standardization, out-of-fold confusion, per-class
recall, and explicit quick-cohort/sample-size caveats. The `full` profile uses all IDs, 80
dimensions, and a 200-epoch starting budget. A full job must be assessed from held-out loss,
reconstructions, embedding QC, checkpoint/seed stability, and the same ID-joined downstream
tasks—not completion alone.

The source lineage, licenses, duplicate candidates, N5 metadata, bounded QC, storage costs, and
unresolved author questions are in the [legacy workspace inventory](legacy_workspaces_inventory.md).
The earlier [whole-volume ROI audit](platyneris_data_inventory.md) remains a useful crop-extraction
reference. If external paths are absent, real-data cells skip clearly and never substitute
synthetic biology.

All generated outputs are separate from bundled published artifacts. Exact historical retraining
is not reproducible without the original raw EM, preprocessing state, exact configurations, and
checkpoints.
