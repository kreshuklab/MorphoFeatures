# Bring your own instance segmentation

The maintained workspace now connects configuration editing, MAE training,
embedding extraction, analysis, representation comparisons, preprocessing,
and mesh inspection. Scientific processing lives in shared Python modules;
the UI submits the same pipeline that the CLI and SLURM execute.

Install the optional runtime in your chosen environment, then launch:

```bash
python -m pip install -e '.[workspace]'
python -m morphofeatures ui
```

N5 grouped-patch workflows additionally require `z5py`, usually installed with
conda. DINO requires an official local repository checkout and backbone weights;
its upstream dependencies must be available in the worker environment. No DINO
code or weights are downloaded implicitly. The mesh page needs `trimesh` and
`plotly`, both included in the `workspace` extra.

## Guided workspace

The main destinations are **Workflow**, **Runs**, and **Results**. **Tools** keeps
published shape/texture workflows, feature assembly, classification/projection,
MAE sweeps, meshes and data inspection available. **Workspace settings** selects
one output directory shared by drafts, submissions and result lookup. Workspace
defaults are distinct from the scientific settings loaded into a workflow.

In **Workflow**, choose your starting point: raw aligned volumes, prepared crops
or grouped N5 patches, an existing checkpoint, saved embeddings, or an explicit
synthetic demonstration. Preparation, training, extraction and analysis share
one draft. The stage navigator shows what will run; moving between its steps
never starts processing. Existing artifacts let you enter at later stages.

Essential settings are visible in each stage. Expand advanced sections for
architecture, grouped data, extraction, evaluation and execution details. Full
pipeline YAML remains available for custom fields, repeated stages and named
representation comparisons. An edited YAML buffer must be applied explicitly;
if the form changes, reload the buffer before applying it. Invalid YAML leaves
the current draft intact.

### Load, edit and save settings

**Preview settings to load** reads a training configuration or pipeline YAML.
The preview names the source and stages. **Load settings into draft**, or
**Replace current draft settings**, applies it. Editing the path alone does not
load a file. Training and extraction also offer **Load model / data settings**
for replacing the model/data configuration of a single stage.

The site reference `configs/sites/mae_platynereis_nuclei_embl.yaml` is supported.
Its active profile is resolved before editing, so edits are not overwritten by
`profiles.full` during submission. Alternative profiles and additional fields
are retained. Checkpoint resume is an explicit path in advanced training
settings and is supported by grouped N5 MAE; whole-crop MAE rejects resume.
Loading a configuration or cloning a run does not implicitly resume training.

**Save draft** persists the document and UI settings under
`<workspace>/.morphofeatures/drafts/<draft_id>.json`. Saved drafts can be reopened
from **Start** after restarting the app. Unsaved changes survive navigation in
the same session. Concurrent saves detect conflicting versions. **Download
pipeline YAML** exports the runnable document without changing its source file.
Linked crop shapes are derived consistently for export and submission.

**Change which stages run** can stop a workflow early, append extraction or
analysis, replace training with an existing model, or turn standalone analysis
into a comparison. Stage changes can be undone. Completed outputs can be chosen
from a stage's artifact selector or from **Runs → Use this output in a new
workflow**. Crops, IDs, masks, model settings and checkpoints are carried forward
where the recorded output supplies them.

### Review and submit

Open **Review & run** to name the run and choose local execution or Slurm.
Slurm resources are configured once for the whole workflow; cluster presets
are separate from training profiles. Ordered stages run sequentially in one
allocation and share its resources.

**Review run** validates the configuration and renders the resolved worker YAML,
command and Slurm script without reserving a run ID or creating a registry
entry. The preview shows the stage order, directories and changes from loaded
settings. Changing settings invalidates the review.

After review, **Run locally** starts a detached process on the app host;
**Submit to Slurm** calls `sbatch` and shows the returned job ID. **Save dry-run
bundle** saves the configuration/script and a dry-run record without launching
work. Saving a dry run reserves its run ID: clone it with **Use settings for a
new run** to make a later submission under a fresh ID.

The destination is `<workspace>/experiments/<run_id>/workspace`. Training
checkpoint, log, split and metadata paths are scoped to that run.
`submitted_settings.yaml` preserves submitted settings; `job.yaml` contains the
resolved worker document, and each trained model has `resolved_config.yaml`.
Existing run directories cannot be overwritten. Invalid inputs and stale reviews
are rejected before launch; double submissions cannot allocate the same run.

See [the UI design notes](ui_redesign.md) for the navigation mapping, state
contract and implementation rationale.

## Local jobs, CLI, and SLURM

A runnable small CPU example trains on eight synthetic crops, extracts their
embeddings, and saves PCA/clustering results:

```bash
python -m morphofeatures workspace-submit \
  --config configs/workspace_pipeline.example.yaml \
  --output-root outputs/workspace --run-id smoke-001 --execution local
```

Local submission returns immediately and runs a separate Python process.
**Runs** shows persisted state, stage results, logs, and metrics. Use
**Refresh progress and logs** while a job runs. Training reports batch/epoch
events; extraction reports object or batch progress; preprocessing reports
scanned blocks and object outcomes. Exceptions mark the stage/job failed and
stop subsequent stages. Local CPU thread counts follow `slurm.cpus`; local
processes are intended for workstation or compute node use. Resource
reservations and shell/module setup apply only to SLURM.

For a concrete reviewable script and immutable configuration without execution:

```bash
python -m morphofeatures workspace-submit \
  --config configs/workspace_pipeline.example.yaml \
  --output-root outputs/workspace --run-id preview-001 --execution dry-run
```

Edit the example's `slurm` mapping for your cluster (partition, account, time,
memory, CPUs/GPUs, interpreter, and optional setup). Submit with a fresh run ID:

```bash
python -m morphofeatures workspace-submit \
  --config configs/workspace_pipeline.example.yaml \
  --output-root outputs/workspace --run-id cluster-001 --execution slurm
```

`--dependency <job_id>` adds an existing SLURM `afterok` dependency. Ordered
stages in one job work locally and on SLURM; their individual state is visible.
`from_preprocessing`, `from_training`, and `from_extraction` consume the preceding
stage of that type. A comparison's `from_embeddings` names earlier extraction
stages. Independent stages currently execute sequentially within a job and
share its resource allocation.

The exact worker command can also run a saved dry-run job in the foreground:

```bash
python -m morphofeatures workspace-run \
  --config outputs/workspace/experiments/preview-001/workspace/job.yaml
```

Relative paths in CLI pipeline YAML are resolved against its directory. UI
paths are resolved against the repository. Launch from the environment that
contains the desired dependencies, or set `slurm.python_executable`. Source
checkout availability and data paths must be shared with the compute nodes.
The SQLite registry and existing SLURM renderer/scheduler are reused.

## Extract, analyze, reopen, export

In **Workflow → Analyze → Extract embeddings**, load the model configuration,
identify the target crops or N5 objects, and set the checkpoint. A linked training
stage supplies its checkpoint and data automatically.
For MAE, the architecture must match the checkpoint. Extraction preserves the
object IDs supplied by the data source. A whole-crop checkpoint and a grouped
N5 checkpoint have different input contracts and are not interchangeable.

The common embedding artifact is NPZ with separate `label_ids` (`int64`) and
`features` arrays. Existing label-first NPY and TSV/CSV inputs remain supported.
NPZ and TSV preserve IDs above `2**53`; exporting these IDs to a float-based
label-first NPY is rejected. Each extraction has a metadata JSON containing
model identity, checkpoint digest, configuration, source fingerprints, settings,
exclusions, runtime, embedding size, process peak RSS, package versions, and
CUDA peak allocation when available. RSS is a process-lifetime peak, not an
isolated stage measurement. Training time is recorded separately when training
and extraction are linked; external training cost can be supplied explicitly.

Choose a shared `cache` directory to reuse extraction. Fingerprints include the
checkpoint SHA256 and data file path/size/mtime; small files are hashed. Large
volume contents are not exhaustively hashed. Keep datasets immutable, and
increment `data_version` after in-place N5/Zarr edits or other changes that may
not change container metadata. This is a reuse cache, not an archival integrity
guarantee. Analyses can be repeated directly from cached embeddings.

**Analyze** configures standardization/L2/no scaling, PCA, optional UMAP,
neighbors, minimum distance, epochs, seed, and K-means or optional Leiden.
Install the `analysis` extra for Leiden. PCA and clustering are always saved;
disable UMAP for very small inputs. The worker exports `analysis.json`,
`coordinates.tsv`, matched `embeddings.npz`, and `projection.svg`.

**Results** loads registered local/SLURM outputs or an explicit
`analysis.json`, `comparison.json`, or embedding file. No inference is repeated.
Plots show IDs in tooltips. Comparison panels share ID-based selection: click
the first plot to highlight corresponding objects across panels. Neighbor
inspection uses the representation's feature space. Supply a preprocessing
config for orthogonal object previews. Coordinate/cluster tables and a ZIP of
the saved report, figures, embeddings, metadata, and splits can be exported.

## DINO comparisons and interpretation

Supported official ViT backbone entry points are:

| Family | Supported entry points | Backbone checkpoint |
|---|---|---|
| DINOv2 | `dinov2_vits14`, `dinov2_vitb14`, and their `_reg` variants | Matching official backbone state dictionary |
| DINOv3 | `dinov3_vits16`, `dinov3_vitb16` | Matching official backbone state dictionary obtained through Meta's access process |

The adapter builds the local architecture with `pretrained=False`, then strictly
loads the supplied state dictionary (`state_dict` wrapping is also accepted).
Classifier heads, teacher/student training bundles, adapters, ConvNeXt, and
unlisted variants are not accepted. See the official
[DINOv2 model repository](https://github.com/facebookresearch/dinov2) and
[DINOv3 checkpoint instructions](https://github.com/facebookresearch/dinov3#pretrained-models).

`configs/workspace_comparison.example.yaml` extracts MAE and DINO representations
for one target configuration, then compares the named outputs. Replace paths,
remove unavailable representations, and submit through the CLI or load it from
**Workflow → Start**. For already cached files, start from saved embeddings and
choose **Change which stages run → Compare multiple saved representations**.

The 3D-to-2D procedure is explicit and configurable:

1. Read a grayscale ZYX crop. For N5, use the same central usable patch groups
   selected by the MAE loader, with the configured split and group size.
2. Use the saved instance mask when present. Legacy N5 masking uses the reader's
   `nonzero`/`all` convention; nonzero masking cannot distinguish an actual zero
   intensity voxel from background.
3. Normalize each crop/patch using foreground percentiles (configurable bounds),
   integer dtype limits, or already-unit-scaled input. Percentile-constant objects
   retain a foreground/background distinction. Optionally mask the background
   to zero. This normalization uses only that input object, not population labels.
4. Take configured fractional slices along chosen axes (`0=z, 1=y, 2=x`), with
   index `round(fraction * (size - 1))`. Defaults give three slices on each axis.
   Grayscale is repeated across RGB channels. Empty individual slices remain
   in the aggregation; an entirely empty object mask is excluded with a reason.
5. Bilinearly resize with antialiasing to a square, either stretching or
   letterboxing. Normalize channels with ImageNet mean `(0.485,0.456,0.406)` and
   standard deviation `(0.229,0.224,0.225)`. Input size must be divisible by 14
   for DINOv2 or 16 for DINOv3. Slice geometry uses voxel axes; anisotropy is not
   automatically resampled to isotropic physical resolution.
6. Select normalized CLS features or the mean normalized spatial patch tokens
   (register tokens are excluded). Aggregate all selected views/patches by mean
   or max into one vector per object. View batches bound GPU memory.

Comparisons use the intersection of unique object IDs, export exclusions, and
reject conflicting target provenance when present. Without metadata, identity
can only be checked by IDs; this limitation is reported. Labels are joined by
`label_id`, never by row position. Unannotated objects remain in exploratory
plots but are explicitly excluded from supervised evaluation.

With labels, each representation receives identical stratified folds, fixed
KNN/logistic settings, and the same optional PCA dimension. Set `group_column`
to specimen/acquisition to keep related objects together. Standardization and
evaluation PCA fit training folds only. Reported measures are KNN and balanced
accuracy, logistic linear-probe accuracy, retrieval precision/recall at K from
held-out queries against training objects, and test ARI from K-means fitted to
training features. Per-fold metrics, means/deviations, label/group splits,
exclusions, extraction costs and configurations are saved in the report.
There is one fixed downstream setting per representation, with no test-set
tuning. Cluster diagnostics and projection plots use all matched objects and
are explicitly exploratory, separate from held-out evaluation.

These evaluation splits do not prove that the representation itself was trained
without exposure to test specimens. Record MAE training provenance and distinguish
transductive evaluation from testing on a held-out specimen. Without annotations,
predictive biological value and generalization remain untested; attractive
projections or silhouette scores do not establish a better representation.

## Standardized preprocessing

**Workflow → Prepare data** and CLI `action: preprocess` call the same function. Accepted
sources are HDF5, N5, Zarr, and NPY with explicit dataset keys, axis order,
channel, shared voxel spacing, origin, and unit. The implementation validates
spatial shapes, keys/channels, axes (including HDF5 `DIMENSION_LABELS` when
present), integer labels, ROI bounds, and declared raw/segmentation grid
agreement. It cannot establish registration from pixel intensities; supply
already aligned grids. Explicit differing spacings/origins are rejected.

Scan memory is bounded by `block_shape`. Object extents/counts/centroids are
accumulated in SQLite, avoiding a dense index proportional to the largest ID.
Extraction holds one fixed crop at a time; final arrays are written with
memory maps. Temporary per-object files are consolidated and removed. Disk
space is needed for the temporary patches and final arrays simultaneously.

Zero label is background. `object_ids`, `max_objects`, and `min_voxels` control
selection. The ROI is half-open `[start_zyx, stop_zyx]`. Centers use bounding-box
midpoints or floored centroids. `crop_shape` fixes the output shape;
`oversized=skip` is the default, while `clip` records truncation.
`boundary=pad` pads beyond the ROI with background; `skip` rejects such crops.
Objects touching ROI boundaries are skipped by default because their extent
may be incomplete; `roi_boundary=allow` retains and marks them as truncated.
Only `segmentation == object_id` contributes raw intensity. Normalization is
shared with existing crop preparation (`dtype`, `percentile`, `none`), followed
by setting background to the configured output value.

`objects.tsv` records every encountered object's status, reason, ID, output row,
extent, centroid, crop start, physical/voxel center, padding, and truncation.
Absent requested IDs are reported. Empty or invalid crops are counted as failed;
the job can complete with other valid objects, while zero valid objects fails.
`preprocessing.json` records inputs, conventions, settings and totals.
Voxel index zero refers to the first voxel center; coordinate = origin +
index * spacing. Outputs are float32 `crops.npy`, exact boolean `masks.npy`,
int64 `label_ids.npy`, and a directly usable `mae_config.yaml`.

These are whole-object/cropped-object inputs for `position-aware-3d-v2` MAE.
They are not a conversion into the existing grouped N5 `v3` sampling objective;
keep grouped checkpoints paired with their original grouped input contract.

The provided bounded Clytia example can be run without processing the full
volume or launching a full training run:

```bash
python -m morphofeatures workspace-submit \
  --config configs/sites/workspace_clytia_smoke.yaml \
  --output-root outputs/workspace --run-id clytia-smoke-001 --execution local
```

## Inspected reference data and validation scope

Read-only inspection on 2026-09-07 established:

| Container | Actual data |
|---|---|
| `raw_patches_masked.n5` | `patches`: `(2506460,32,32,32)`, uint8, NumPy/Z5 order; N5 storage dimensions reverse this order |
| `abs_crop_centers_radius4_only_nucl.n5` | `positions`: `(2506460,4)`, int64; `ids`: `(11382,)`, int64 |
| Both supplied Clytia HDF5 files | `exported_data`: `(1500,900,2260,1)`, `zyxc`; raw uint8, segmentation uint64 |

The first indexed rows are `[3,116,1888,2662]`, `[3,116,1896,2654]`,
`[3,116,1896,2662]`, and `[3,116,1904,2654]`. The first raw patch spans 0–209,
with 52.34375% zeros. The current N5 reader interprets columns as
`label_id,z,y,x`; the site config declares patch spacing `(0.025,0.020,0.020)` µm
and position spacing `(0.100,0.080,0.080)` µm. Axis/physical-unit semantics come
from the existing reader and site configuration, not from the filenames or
bare N5 dimension attributes. Clytia attributes identify axes but supply no
reliable physical spacing. The smoke example therefore uses declared voxel
units instead of inventing micrometers.

Executed checks include configuration/profile round-tripping, real Streamlit
navigation retention and dashboard page loading, sparse/large ID preservation,
bounded preprocessing and mask alignment, a one-epoch train→extract→analyze
pipeline, group-separated comparison evaluation, detached local job completion,
and mesh sampling/export/reload. Existing MAE/real-MAE/crop/CLI/registry/SLURM
tests also pass. The Clytia `64³` ROI produced IDs 2,47,188,190; one other object
had an empty center crop and was reported failed, and five were omitted by the
object limit. All four crops are partial objects, suitable only for an
integration check. Training and embedding generation consumed them directly.

The official pretrained DINOv2 ViT-S/14 backbone also encoded these four crops
using three orthogonal central views at 224 pixels. Its embeddings were matched
to MAE IDs and passed through PCA, UMAP and clustering; the comparison was
saved for reopening. The source checkout was commit
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`; the checkpoint digest and exact
settings are retained with the smoke artifact. This verifies integration,
not biological representation quality. Other DINO variants are implemented
through the same official interfaces but were not tested with pretrained weights.

The full test invocation completed with **79 passed and 2 skipped**; both skips
were notebook checks requiring the unavailable `nbformat` package. After moving
ZIP preparation into the worker, all **14 workspace tests** passed again. Ruff
checks on the new modules/tests and `git diff --check` passed. Optional mesh
libraries were installed in an isolated `/tmp` dependency directory for testing,
not into the existing development environment.

No full training, whole-volume preprocessing, or live SLURM submission was used
for validation. SLURM rendering, argument construction and scheduler behavior
are tested with dry runs and existing scheduler fixtures. Actual cluster setup,
GPU execution, and supplied pretrained DINOv3 weights require validation in the
deployment environment. A killed local process that cannot write its final
status (e.g. host loss/SIGKILL) may require log inspection; this is not a durable
distributed task queue.

## Meshes

**Tools → Mesh inspection** loads triangle meshes, preserves available object
IDs or accepts a supplied ID, and displays a Plotly surface. An affine
`mesh_to_world_xyz` transforms mesh XYZ vertices into the units shared with raw
spacing/origin (ZYX). Raw samples use nearest-voxel or trilinear interpolation
at each vertex, with tiled reads and no implicit normal-direction averaging.
Outside-volume vertices receive NaN. Sampling again always starts from the
loaded source mesh, avoiding repeated application of the affine.

Export supports PLY, OBJ and GLB. PLY includes float intensity attributes;
formats also receive grayscale vertex colors where supported. An accompanying
`.vertices.tsv` is authoritative for exact int64 IDs and numerical intensities,
including NaN, with vertex indices and XYZ positions. A metadata companion
records the transform, sampling method and coordinate conventions. Keep both
companions with the mesh for lossless reopening. Large interactive meshes are
not automatically decimated; prepare an appropriate display mesh if needed.
