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

In **Workflow**, choose your starting point: raw aligned volumes, **Just Run
Preprocessing**, prepared crops or grouped N5 patches, **DINO features from
prepared objects**, an existing checkpoint, saved embeddings, or an explicit
synthetic demonstration. Preparation, training, extraction and analysis share
one draft. The stage navigator shows what will run; moving between its steps
never starts processing. Existing artifacts let you enter at later stages.
Only steps included in the draft appear in the navigator. The preparation page
also offers **Just Run Preprocessing** to keep its preparation stage and go
directly to review; the stage-change history lets you undo this choice.

Essential settings are visible in each stage. Expand advanced sections for
architecture, grouped data, extraction, evaluation and execution details. Full
pipeline YAML remains available for custom fields, repeated stages and named
representation comparisons. An edited YAML buffer must be applied explicitly;
if the form changes, reload the buffer before applying it. Invalid YAML leaves
the current draft intact.

Training exposes effective defaults even when a YAML file omits them. **MAE
encoder and input geometry**, **MAE decoder and reconstruction**, **Optimization
and data loading**, and **Validation, early stopping and resume** group related
controls. Grouped MAE exposes decoder width, depth, heads, target shape, and
normalized reconstruction loss. Crop MAE has a fixed two-layer MLP decoder with
an adjustable width, so transformer decoder heads and depth do not apply.
Setting tooltips describe their meaning and constraints rather than YAML paths.

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
Resource controls group partition/time/memory and CPU/GPU/account/QoS into rows.
**Mail notifications** provides event selection and an email address; configure
both to request notifications. Topology, dependencies, Python interpreter,
module setup, thread binding, local caches, and job-context logging are under
**Advanced scheduler and runtime settings**. Extra tasks or nodes do not enable
distributed training in the current single-worker pipeline.

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

For maintenance, keep the editable document separate from Streamlit widget
state (`workspace_state.py`), and keep review and submission tied to the same
fingerprinted `WorkspacePlan` (`workspace_jobs.py`). Preserve unfamiliar YAML
fields and inactive profiles, reject stale YAML buffers, and invalidate reviews
when settings change. Scientific computation belongs in shared worker modules,
not UI callbacks. Interaction and submission contracts are covered by
`tests/test_guided_workspace.py`; use temporary workspaces and fake schedulers
for regression checks. Independent allocations per stage and cross-job artifact
dependencies remain separate future work; current stages share one allocation.

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
The central workflow and Tools use `analysis.projection.project_embeddings`
and the same classifier pipeline in `analysis.classification`. New projection
defaults are eight K-means clusters, 15 neighbors, minimum distance 0, 50 UMAP
epochs, and all objects. Existing YAML values remain explicit overrides. Match
the subset, normalization, seed and UMAP settings when comparing plots; a
different projection appearance is not evidence of improved classification.
Install the `analysis` extra for Leiden. PCA and clustering are always saved;
disable UMAP for very small inputs. The worker exports `analysis.json`,
`coordinates.tsv`, matched `embeddings.npz`, `object_labels.tsv`, and `projection.svg`.

**Results** loads registered local/SLURM outputs or an explicit
`analysis.json`, `comparison.json`, or embedding file. No inference is repeated.
Plots show exact IDs in tooltips. Cluster, known-type and predicted-type panels
share identical PCA/UMAP coordinates and axis ranges. **Unlabeled point opacity**
fades objects without a known type in every panel; known/predicted types share
one color palette. Each graph has a separate legend below its plot, with wrapped
class names. **Graphs to export** selects all panels together or just clusters,
known types, or predicted types; **Prepare projection SVG / PNG** generates that
selection with its own legend.

Click, box-select or lasso points in any panel. **Add selected points to comparison**
retains those IDs; repeat elsewhere on the map to collect distant cells. Switch
**Comparison mode** to **Chosen objects** to edit the list, search for additional
IDs, or compare those cells directly. The slice table shows up to 25 chosen
objects and the complete ID/annotation table remains downloadable. Each
representation has a separate comparison list. **Nearest neighbors** retains the
original workflow: **Number of neighbors** controls up to 24 surrounding objects. **Find
neighbors in** distinguishes normalized embedding-space neighbors from neighbors
in the displayed PCA/UMAP plane; projection distances can distort the original
relationships. **Show selected object and neighbors** generates a table of
XY/XZ/YZ center slices, with IDs, distances, known/predicted types and clusters.
The image grid uses up to four objects per block, shared plane labels, and a
separate annotation table with wrapped class names. Source and contrast details
appear once in the caption. Download the graphical table as an SVG with editable
text or a 300 dpi PNG, and its IDs/distances as TSV. Contrast is scaled per object
and the selected maximum view side bounds pixel reads.

Input settings are recovered from extraction provenance, or supplied through
**Input data YAML for object inspection**. Prepared views support NumPy, HDF5,
N5 crops, and representative grouped N5 patches. Original views use the raw
volume and bounding boxes referenced by the preprocessing configuration. Grouped
N5 original views require `qc_raw_container`, `qc_raw_key`, and
`position_to_raw_scale_zyx`; they center a bounded raw field on the mean patch
position. A representative patch or bounded field is explicitly labeled and
does not imply the entire nucleus is visible. Coordinate/label tables and the
saved worker report archive remain available for export.

An inspection-only YAML can omit all training/model settings: use
`schema: morphofeatures.inspection.v1`, a `data` mapping, and optional
`inspection.mesh` and `inspection.validation` mappings. The ready-to-load
[Platynereis inspection example](../configs/sites/inspection_platynereis_nuclei_embl.example.yaml)
uses the paths from the site training configuration, including its existing
`cells_to_nuclei.tsv` mapping: the patch index uses cell IDs while the candidate
Paintera segmentation stores nucleus IDs. A scan on 2026-09-10 found 11,381 of
11,382 mapped patch-index IDs in `s2`; nucleus 11486 was absent at that level.

For grouped N5 slice views, `data.source`, `patches_container`, `patches_key`,
`positions_container`, `positions_key`, and `ids_key` identify the stored patches
and their parent IDs. Original raw views additionally need `qc_raw_container`,
`qc_raw_key`, and `position_to_raw_scale_zyx`. For prepared whole-object crops,
use `data.crops`, `data.label_ids`, optional `data.loss_masks`, and optional
`data.preprocessing`; container arrays need their matching `*_key` fields.
File paths are relative to the YAML location; N5/HDF5 dataset keys are separate
strings, not paths appended to the container filename.

`data.foreground_mask_container` / `foreground_mask_key` identify an optional
foreground map, with `foreground_mask_kind: binary` or `foreground_scores`.
These fields document the source and prevent its accidental use as instance IDs;
they do not convert a foreground map into per-cell masks or change MAE training.
The Platynereis `volumes/nuclei/foreground` is a whole-volume uint8 score map with
sampled values spanning 0–255. Thresholding could produce a binary foreground
mask, but neither scores nor binary foreground encode nucleus identities.
In contrast, `data.loss_masks` for prepared object inspection has shape
`(N,Z,Y,X)` or `(N,1,Z,Y,X)` and one aligned `data.label_ids` entry per row.

### Stream PlatyBrowser inspection data

In **Results**, select **Inspection data location → PlatyBrowser streaming**.
Enter the GitHub project URL, published dataset version and metadata revision,
then click **Connect to PlatyBrowser**. Alternatively, load
[inspection_platybrowser_streaming.example.yaml](../configs/sites/inspection_platybrowser_streaming.example.yaml)
in **Input data YAML for object inspection** and connect. This example requires
no local patch, position or whole-volume files. Existing embeddings are still
loaded through the normal Results input.

The connection downloads metadata and object tables only. **Show chosen objects**
or **Show selected object and neighbors** streams bounded original raw views;
**Render selected meshes** streams the nucleus-label chunks covering that batch.
Graphical tables and mesh exports work as for local data. Download the source
settings using **Download inspection YAML**; the connected Git revision is pinned
in that YAML. Editing connection settings hides the previous results until the
new settings are connected.

For the existing grouped N5 embeddings, choose **Embedding IDs refer to → cell**
to use the project's published `cells_to_nuclei.tsv`. Choose **nucleus** only when
embedding IDs already identify nuclei. The resolver follows versioned metadata
links, reads the published physical object bounds and derives voxel bounds for the
selected segmentation grid. Mesh archives record the source URLs, revision, levels,
table checksum, mapped nucleus ID and observed label voxel count.

Under **Streaming resolution and cache**, raw level **3** and nucleus level **0**
give the aligned `[0.100, 0.080, 0.080]` µm ZYX grid for PlatyBrowser dataset `1.0.1`.
Raw and mesh levels can be selected independently: both use physical coordinates.
Larger level numbers reduce detail; small objects can disappear when downsampled.
The existing mesh count, voxel budget and surface-detail controls still apply.
Published bounds cannot be reused after manually changing the source/grid;
change the level in the streaming controls and reconnect instead.

Downloads use a persistent cache (default 1024 MiB) with eviction of old entries.
Small derived catalogs are retained separately under `catalogs/` for provenance.
The example stores both beneath `outputs/remote_cache/platybrowser`; the cache path,
download-size limit and network timeout are editable. Change **Cache version** and
reconnect to fetch updated hosted data or rebuild a corrupt cached download. A
Git commit pins metadata; it does not make the separately hosted image bytes immutable.

The background check compares **all embedding IDs against the published table**,
and clearly reports that no full voxel scan was performed. Mesh loading verifies
the selected label in the actual streamed voxels. Missing chunks (HTTP 404) use
N5's zero/background convention; authentication, network and server failures are
reported as errors. Streaming supports scalar 3D HTTP(S) N5 volumes with gzip/raw
compression and axis-aligned BDV coordinates. It does not require an S3 account
or credentials for the public PlatyBrowser data. Remote OME-Zarr and rotated image
registrations are not supported by this reader.

### Interactive cell and nucleus surfaces

Under Results, expand **3D cell / nucleus surfaces** for either comparison mode.
**Maximum meshes rendered at once** (1–25, default 6) is independent of the number
of chosen points. **Meshes to display** selects the current batch without removing
objects from the comparison list. **Preview faces per mesh** limits display detail;
a total limit of 200,000 preview faces also applies. Surface loading happens only
when **Render selected meshes** is clicked, and missing objects are reported by ID.

Choose one of three sources:

- **Instance segmentation:** HDF5, N5, Zarr or NumPy instance labels with explicit
  axes, voxel spacing, origin and units. Preprocessing fills these settings when
  available. Supply `objects.tsv` or a CSV/TSV with `label_id`, `bbox_min_z`,
  `bbox_min_y`, `bbox_min_x`, `bbox_max_z`, `bbox_max_y`, `bbox_max_x`. Bounds are
  voxel coordinates on this segmentation grid, with exclusive maxima. An optional
  `segmentation_id` column maps embedding IDs to different instance IDs. Small
  volumes can be used without an index. The background ID check can build bounds
  for large volumes. Mesh loading itself rejects regions exceeding its voxel budget.
- **Prepared foreground masks:** uses explicit `data.loss_masks` and `data.label_ids`.
  Preprocessing metadata recovers world coordinates; otherwise meshes have
  crop-local coordinates. Grouped intensity patches alone do not define a complete
  cell/nucleus surface and require a segmentation or existing mesh files.
- **Existing mesh files:** a directory containing one `ID.ply`, `ID.obj`, `ID.glb`,
  or `ID.stl` per object, or a CSV/TSV with `label_id` and `mesh_path`. Paths in the
  table are relative to that table. Declare the common coordinate unit.

Generated surfaces use marching cubes with the selected **Surface sampling step**.
Larger steps affect both the preview and extracted mesh files. Boundary contact
is flagged because a region may contain only part of the object; padding closes
the surface at that boundary. No surface is inferred by thresholding raw intensity.

Drag and zoom the interactive 3D panels. Objects are centered for display using a
common spatial scale. The camera button downloads the current interactive view;
**Prepare mesh comparison SVG / PNG** uses the explicit azimuth/elevation controls
for a consistent graphical comparison. **Prepare mesh files** packages the displayed
batch as PLY, OBJ or GLB, together with lossless NPZ vertex/triangle arrays and a
manifest containing exact object IDs, coordinate units and extraction provenance.
The archive retains source coordinates and geometry before preview simplification.
Change the displayed batch to export other chosen cells. Install the `workspace`
extra for the interactive UI, mesh I/O and marching-cubes dependencies.

**Check object IDs in the background** starts a cancellable check when a valid
inspection source is selected. It compares every ID in the current embedding
artifact, including objects outside the projected/selected subset, with the
actual inspection source IDs. Equal counts with different IDs are a mismatch.
Extra source IDs are reported as a possible embedded subset; missing embedded
IDs are listed in the downloadable JSON report. Binary/score volumes are reported
as not comparable, rather than counting gray values as objects.

Local instance volumes are scanned in bounded blocks on a background thread. The check
also caches voxel-space bounds for mesh loading. Prepared masks are checked for
array shape, row count and their ID vector; their pixels are not exhaustively
scanned. Remote N5 checks use the supplied/published object table and volume metadata,
never a full remote scan. Mesh-file checks compare IDs from the directory/table and file existence.
Optional `inspection.mesh.id_mapping` uses a CSV/TSV with `label_id` and
`segmentation_id` (or `nucleus_id`, as in `cells_to_nuclei.tsv`) for explicit
correspondence. Label IDs and positive target IDs must be unique; zero targets
mean unassigned/background and are reported as missing. Paintera fragment IDs may
need conversion to final segment IDs before use; the check reports the IDs actually
stored in the selected voxel dataset.

Results and bounds are cached beneath `outputs/inspection_checks` (or the configured
output root). **Recheck object IDs** forces a fresh scan after in-place edits;
`inspection.validation.data_version` also invalidates the cache. Cancelling leaves
no successful partial result. Matching numeric IDs still does not prove spatial
alignment or biological identity; confirm the coordinate grid and inspect representative
raw/segmentation views. Interactive legends inherit the UI background and text
colors; downloadable publication figures retain their white background.

### Known-label overlays and classification

Set **Annotation table** and **Label column** in Analyze. `auto` detects
`cell_type`, `label`, or `cell_label`. Blank labels and the configurable
`unlabeled_values` (including `None`, `unknown`, and `unlabeled`) remain unassigned.
CSV/TSV IDs are parsed exactly, including integer-valued legacy strings such as
`3.0`; they are never joined through floating-point IDs. Turn **Evaluate
classifiers** off to overlay sparse labels without requiring enough examples
for cross-validation.

Choose logistic regression, MLP, and/or KNN. Logistic regression uses the Tool's
balanced class-weight default; MLP uses its ReLU network and default hidden
width `[64]`. Both routes sort matched objects by ID before assigning folds,
so annotation row order cannot change the evaluation. Classes below
`minimum_class_count` are excluded from fitting/evaluation but retain their
known labels in plots. `fold_policy: reduce` matches the Tool's fold-count
reduction; `strict` rejects insufficient examples. Grouped evaluation additionally
requires enough independent specimen/acquisition groups. Scaling and optional
evaluation PCA are fitted on training folds only.

**Classifier for label plots and volume export** selects the predicted-type
layer. Eligible labeled objects use out-of-fold predictions. Other objects can
receive predictions from a model fitted on all eligible labeled objects.
`object_labels.tsv` records `prediction_source`, classifier and confidence so
these cases remain distinguishable. Confidence is not calibrated. Fitted
predictions on new objects are not additional validation evidence.

```yaml
- action: analyze
  from_extraction: true
  umap: true
  cluster_method: kmeans       # or leiden
  clusters: 8
  normalization: standardize
  subset: 0                   # plotting only; 0 uses all objects
  neighbors: 15
  min_dist: 0.0
  umap_epochs: 50
  resolution: 0.004           # Leiden CPM resolution
  seed: 42
  annotations: /path/to/cell_types.tsv
  label_column: cell_type
  group_column: null
  classify: true
  classifier_models: [logistic, mlp, knn]
  minimum_class_count: 2
  folds: 5
  fold_policy: reduce
  class_weight: balanced
  linear_c: 1.0
  hidden_dimensions: [64]
  max_iter: 2000
  knn_k: 5
  evaluation_pca: null
  prediction_model: logistic
  predict_unlabeled: true
  unlabeled_opacity: 0.15
  input_config: /path/to/mae_config.yaml
```

### Export labels onto the original segmentation

In Results, expand **Map labels back to the original segmentation** and click
**Create volume export workflow**. The next form selects the instance volume,
dataset key, axes/channel, requested layers and HDF5/Zarr v2/Zarr v3 storage.
Preprocessing provenance fills in the original segmentation and coordinate
metadata when available. Review and run locally or on Slurm using the usual
workflow controls; whole-volume relabeling is performed in the worker.

Each layer (`known_label`, `predicted_label`, `cluster`) is a uint32 categorical
volume in Z/Y/X order on the full original grid. Cluster zero is assigned a
positive export category; export value zero always means background or
unassigned. Unmatched segmentation objects remain zero, and embedding objects
absent from the volume are reported. `label_lookup.tsv` maps category values to
names and colors. `object_mapping.tsv` retains exact original IDs, assigned
values and prediction provenance. These tables also distinguish background
from objects without annotations, although both display as zero in that layer.

**Also write RGB color volumes** adds uint8 Z/Y/X/RGB datasets using the plot
palette. Use those for baked-in colors, or open categorical datasets as label
images in Fiji/ilastik/napari with the appropriate HDF5/Zarr reader and lookup.
Readers do not all automatically apply custom color attributes. Both category
and RGB layers are written in bounded blocks, without changing the source
segmentation. HDF5 uses gzip; Zarr uses its backend compression. Zarr v3 works
with zarr-python 3 or a recent z5py supporting the v3 format.

Embedding IDs must identify the same instances as the source segmentation. To
map nucleus IDs to different cell IDs, supply a CSV/TSV `id_mapping` with unique
`label_id` and `segmentation_id` columns. Missing or ambiguous many-to-one
assignments are rejected rather than combining conflicting cell types.

```yaml
stages:
  - action: export_labels
    labels: /path/to/analysis/object_labels.tsv
    segmentation: /path/to/original_instances.h5
    segmentation_key: exported_data
    segmentation_axes: zyxc
    segmentation_channel: 0
    layers: [known_label, predicted_label, cluster]
    output_format: h5         # h5, zarr2, or zarr3
    include_rgb: true
    block_shape: [64, 64, 64]
    spacing_zyx: [1, 1, 1]
    origin_zyx: [0, 0, 0]
    unit: voxel
    id_mapping: null
    mapping_column: segmentation_id
```

## DINO comparisons and interpretation

For a standalone DINO run, choose **DINO features from prepared objects** in
Workflow, or **Tools → DINO features → Create DINO features workflow**. Provide
prepared intensity crops, original object IDs, and aligned foreground masks,
or select a completed preparation run. Instance-label values are not image
intensities. Supply the official local backbone checkout and matching pretrained
weights, then choose view axes/slices, normalization, resize policy, feature
token, and aggregation. Review and run locally or on Slurm as with MAE.

Existing indexed N5 patches can be used directly. In the DINO extraction stage,
expand **Load model / data settings**, enter the same grouped N5 YAML used for
MAE, and click **Load settings into this stage**. For the audited Platynereis
data, use `configs/sites/mae_platynereis_nuclei_embl.yaml`. This selects
`data.source: n5_masked_patches` and pairs `raw_patches_masked.n5` (key `patches`)
with `abs_crop_centers_radius4_only_nucl.n5` (keys `positions` and `ids`). Do not
enter these grouped patches directly under **Prepared crops (.npy / .h5 / .n5)**,
which expects one crop per object and explicit container dataset keys. Loading
data settings retains the DINO checkpoint and view choices;
the MAE architecture/training sections do not initiate MAE training. The chosen
profile controls object selection and group size, so use the same sampling and
IDs when comparing embeddings from different models. DINO produces one vector
per parent nucleus ID, not one per stored patch. No additional crop conversion
is needed for these indexed N5 inputs.

Extraction feeds the analysis stage automatically. PCA, optional UMAP, clusters,
and ID-linked neighbors are available in Results. **Classification check against
known labels** accepts an optional CSV/TSV annotation table with unique
`label_id`, a class column, and an optional specimen/acquisition column. Linear
and k-nearest-neighbor classifiers use held-out folds with training-fold-only
scaling and optional PCA. Results include fold metrics, per-class scores,
confusion matrices, and exportable `predictions.tsv` and `splits.tsv`. Unlabeled
objects remain in the projection and are explicitly excluded from evaluation.
Each class needs enough examples for the selected fold count; grouped checks
also need enough independent groups. Pretraining exposure and related-object
leakage must be assessed separately.

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

With labels, each representation receives identical stratified folds, the same
selected classifier settings, and optional PCA dimension. Set `group_column`
to specimen/acquisition to keep related objects together. Standardization and
evaluation PCA fit training folds only. Reported measures are KNN and balanced
accuracy, logistic linear-probe accuracy, retrieval precision/recall at K from
held-out queries against training objects, and test ARI from K-means fitted to
training features. Per-fold metrics, means/deviations, label/group splits,
exclusions, extraction costs and configurations are saved in the report.
There is one fixed downstream setting per representation, with no test-set
tuning. Cluster diagnostics and projection plots use all matched objects (or
the explicitly selected projection subset) and
are explicitly exploratory, separate from held-out evaluation.

These evaluation splits do not prove that the representation itself was trained
without exposure to test specimens. Record MAE training provenance and distinguish
transductive evaluation from testing on a held-out specimen. Without annotations,
predictive biological value and generalization remain untested; attractive
projections or silhouette scores do not establish a better representation.

## Standardized preprocessing

In **Prepare data**, enter the volume paths, dataset keys, axis orders, and
channels, then click **Inspect data dimensions**. The table shows stored shape,
spatial Z/Y/X shape, dtype, and channel. Changing a source setting invalidates
that inspection. Set inclusive ROI starts and exclusive stops, or choose
**Use full volume ROI**. **Show ROI image preview** displays raw intensity,
instance colors, and an alignment overlay at a selected axis/slice. A preview
reads at most a central 512 × 512 plane; larger ROIs are clearly marked as
cropped previews. It does not load the entire volume or scan all object IDs.
Preview counts refer only to IDs intersecting that slice.

Preprocessing-only runs save crops, IDs, masks, and `mae_config.yaml`. Continue
from that configuration in Runs to train MAE or extract DINO features. The
earlier real-data pilot under `outputs/notebooks/04_platynereis_mae/` is retained
as historical evidence; it is separate from the current grouped tutorial.

Select **Prepare data → Output format** to choose storage:

| Setting | Array outputs | Storage |
|---|---|---|
| `npy` (default) | `crops.npy`, `masks.npy`, `label_ids.npy` | Separate memory-mapped NumPy arrays |
| `h5` | `crops.h5` | One HDF5 file, requiring `h5py` |
| `n5` | `crops.n5/` | One N5 container directory, requiring `z5py` |

HDF5 and N5 contain `crops` (float32, N/Z/Y/X), `masks` (same shape), and
`label_ids` (int64, one per object), with gzip compression. Masks are boolean
in HDF5/NumPy and exact 0/1 uint8 in N5. Crop chunks contain one object and at
most 64 voxels along each spatial axis. Container attributes record axes,
spacing, origin, units, normalization and background; `objects.tsv` and
`preprocessing.json` remain alongside the container. Temporary patches are
removed after consolidation, and container exports do not retain duplicate
NumPy outputs.

In a pipeline YAML, set `output_format: h5` or `output_format: n5` on the
`action: preprocess` stage. Omitted settings retain the NumPy default. The
generated `mae_config.yaml` supplies all paths and keys, for example:

```yaml
data:
  source: masked_crops
  crops: /path/to/00-preprocess/crops.h5
  crops_key: crops
  label_ids: /path/to/00-preprocess/crops.h5
  label_ids_key: label_ids
  loss_masks: /path/to/00-preprocess/crops.h5
  loss_masks_key: masks
```

Linked stages inherit this automatically. For a separate run, load the generated
YAML in the MAE or DINO stage using **Load model / data settings**. Results object
previews accept the same YAML. DINO reads individual crops lazily; crop MAE keeps
its existing in-memory training behavior. Selecting N5 storage preserves the
whole-object crop layout; it does not turn these arrays into grouped N5 patches
with a separate positions index.

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
memory maps or compressed container chunks. Temporary per-object files are
consolidated and removed. Disk space is needed for the temporary patches and
final arrays simultaneously.

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
index * spacing. Every storage format preserves float32 crop intensities,
exact foreground masks and int64 IDs, and supplies a directly usable `mae_config.yaml`.

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
