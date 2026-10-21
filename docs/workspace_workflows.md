# Bring your own instance segmentation

The maintained workspace now connects configuration editing, MAE training,
embedding extraction, analysis, representation comparisons, preprocessing,
and mesh inspection. Scientific processing lives in shared Python modules;
the UI submits the same pipeline that the CLI and SLURM execute.

## Distill embeddings using known cell types

Choose **Workflow → Distill embeddings with cell types (adapter / PCA)**, or open
saved results and choose **Follow-up analysis → Distill embeddings**. Start with
one specimen's saved MAE, DINO or MorphoFeatures vectors and a TSV/CSV containing
`label_id` and a cell-type column. The annotation IDs must be in the same namespace
as the embeddings: cell IDs and nucleus IDs are not interchangeable. Unlabeled
objects can receive embeddings but do not train the classifier.

This workflow asks whether the information useful for predicting cell type also
helps other tasks. It compares three representations:

| Representation | What happens | What it can tell you |
| --- | --- | --- |
| Original | Keep the complete input vector unchanged. | Baseline performance. |
| PCA (centered SVD) | Optionally standardize features; fit principal directions on training objects and retain the chosen number of dimensions. | Whether compression of high-variance information is enough. PCA does not use class labels to choose directions. |
| Supervised adapter | Fit a Linear → ReLU → Linear encoder and a cell-type classification head, using class-balanced cross entropy. Export the encoder's latent vectors. | Whether class-relevant combinations help; they may also discard useful within-type variation. |

This is **not Deep CORAL**. There is no target dataset, covariance-alignment loss,
image re-embedding or DINO/MAE fine-tuning. The adapter shares its implementation
with the supervised neural condition of label transfer. Default hidden/latent
widths are 128/64, with 100 fixed training epochs. Start with a short pilot.
PCA defaults to 64 dimensions; a small training cohort automatically caps that
number, with the actual dimension reported. Use the same effective dimension
for the adapter when you want a dimension-matched comparison.

**Validation:** a reproducible stratified split reserves one partition. Five
partitions usually means 80% training and 20% validation; rare classes may reduce
the effective partition count. `validation_fold: 0` reserves the first partition.
Only that partition is scored; this is not five-fold averaged cross-validation.
An optional `group_column` keeps specimens or independently defined groups
entirely on one side of the split. Without it, scores are object-level, so related
or neighboring cells may still share information. Both reducers, their input
scaling and each evaluation classifier fit only the training objects. Neither
validation labels nor validation features fit the reducer. The common evaluation
classifier is a class-balanced logistic regression with training-fitted
standardization; its held-out balanced accuracy, macro-F1, predictions and
confusion matrices compare representation quality under that particular probe.
The adapter's own training-head accuracy is not substituted for this comparison.

All exported objects use **one fitted transform per method**. We never concatenate
vectors from separately trained folds: their coordinate axes would not be
interchangeable. The fitted model is not subsequently refitted on all labels.
Repeatedly choosing parameters from these validation scores makes this a
development set; reserve independent data for a final performance claim.

### Testing bilateral retrieval

Enable the optional bilateral comparison in the distillation stage, or select
both **Distill embeddings** and **Compare bilateral-partner retrieval** in a
follow-up draft. Keep **Probe original, PCA and adapter together after
distillation** checked to compare the new outputs.

The default **Exclude all training objects** removes adapter-training IDs from
both queries and the search population, before the existing cell/nucleus mapping
and matched-candidate probe. Unannotated objects remain eligible for object-level
splits. With specimen/group splitting, training groups and objects with unknown
group membership are also excluded. A known
bilateral partner that was used for training is also removed; missing partners
and ties are reported by the existing probe. This is a different search population
from the published whole-animal analysis: compare the three methods on this same
population, not directly against an old whole-animal score. Published embeddings
can be added in a subsequent bilateral workflow; that workflow re-matches all
search populations. The alternative **All objects (exploratory)** preserves the
full population but includes supervised training cells. Better cell-type
classification does not guarantee better bilateral ranking: the adapter may
compress distinct cells of the same type together.

Bilateral variants retain their existing preprocessing options. If you select
standardization or PCA there, that unsupervised step is fitted to the matched
search population, as in the existing retrieval probe. Choose no additional
scaling/PCA to evaluate the exported geometry directly. Neither option uses
candidate partners to fit the adapter.

### Output files and reuse

| File | Meaning |
| --- | --- |
| `distillation.json` | Main Results entry point, settings, fitting split, metrics and paths. |
| `split.tsv` | Exact native IDs and known labels, training/validation roles, optional groups. |
| `metrics.tsv`, `predictions.tsv` | Common linear-probe performance and predictions on validation objects only. |
| `base/`, `pca/`, `adapter/embeddings.npz` | All-object vectors with native IDs, including training objects. |
| Each method's `held_out.npz` | Same fitted vectors after removing training IDs (and training/unknown groups for a grouped split). Includes eligible validation and unannotated objects. Prefer these for a new evaluation. |
| Each vector's `.metadata.json` | Inherited source/inspection identity, fitted transformation fingerprint and training exposure. Keep it beside the NPZ. |
| Each method's `confusion.tsv`, `confusion.png/.svg`, `linear_probe.npz` | Validation confusion counts/figures and numeric fitted classifier parameters (scaler, classes, coefficients, intercept). |
| `pca_parameters.npz` | Input scaler, PCA center and components, explained-variance ratios. |
| `adapter_model/` | `adapter.json`, numeric `adapter_parameters.npz`, and `training_history.tsv`. No pickled neural model is required for inference. |
| `comparison/` | Exploratory maps/clusters on all objects, colored by known labels. Classifier evaluation is deliberately not repeated on already fitted vectors. |
| `bilateral/` | Optional matched retrieval ranks, coverage and plots for the chosen search population. |
| `export.zip` | Complete portable result bundle; archives over the browser limit remain downloadable from the worker filesystem. |

Reusable configuration: [`workspace_distillation.example.yaml`](../configs/workspace_distillation.example.yaml).
After a named extraction stage, replace `embedding` with
`from_embedding: YourExtractionName`. Later stages can reference `Distilled.base`,
`Distilled.pca`, `Distilled.adapter`; `Distilled` alone aliases the adapter vectors.
These names refer to all-object vectors. For separate follow-up validation, select
the held-out files in Results. Generic classifier cross-validation on all-object
adapter vectors does **not** refit the adapter and can reuse supervised information;
the metadata and Results warn about this.

For Python inference with the saved parameters, use
`analysis.distillation.apply_pca(parameters, features)` or
`analysis.transfer_adapter.NumericAdapter.load(path_to_adapter_json).transform(features)`.
Preserve the input feature order and recipe. Independently fitted PCA/adapter
models do not define a shared space: common-space checks distinguish their fitted
parameter fingerprints.

## Follow-up analysis and Results navigation

Results now separates **Explore results**, **Follow-up analysis** and **Files &
provenance**. Ordinary analyses further separate Projection, Classification,
Bilateral probe, Objects & meshes and Exports. Choose which representations to
display rather than rendering every map at once; point selections are retained
for object inspection. Existing classifiers, meshes, downloads, MoBIE export,
renaming and analysis-variant settings remain available in these sections.

**Follow-up analysis** accepts multiple saved representations and multiple tasks:
analysis/classification, distillation, bilateral retrieval, spatial context,
feature fusion, shared maps and label transfer. **Create follow-up workflow draft**
copies the selected artifact paths and parent-report provenance into an editable
Workflow. It does not submit a job, overwrite results or re-extract images.
Review the specimen identities, annotations, additional graph/candidate inputs
and resource settings before submission. Fusion compares blocks of the same
objects; joint maps and transfer use independent datasets with compatible recipes.

Existing in-place clustering/UMAP variants are still under Follow-up analysis
(and the follow-up expander on nested experiment maps). They retain their existing
background computation and provenance behavior. Runs has one main result-opening
button for the selected completed report. Other files and workflow continuation
actions are grouped in **Other saved outputs and workflow inputs**.

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

**Runs:** select a row in the table to open its details, logs and output actions.
The Workflow column describes the submitted stages, such as **DINOv2 embeddings
→ PCA + UMAP & clustering**, **Label transfer**, or **DINOv2 fine/coarse
comparison**. It reads the saved job configuration, including for older runs;
the internal worker name is not the workflow description. Filter by run ID,
display name or workflow. A missing configuration is identified explicitly.

**Back** in the sidebar returns to the previous page, selected run, result,
analysis combination or transfer condition. History is limited to this browser
session. It does not undo settings edits, cancel jobs or remove results. Switching
the workspace output directory starts a new navigation history so selections do
not refer to another workspace's registry.

**Workspace settings:** edit the repository, data, analysis-table, MoBIE-table
and output directories directly. You can also set the default seed for interactive
tools, the device offered by the published workflow tools, DINO code and weight
locations, cluster resource presets, and optional shared dataset locations. Existing
scientific workflows retain their own sources, seeds, devices and resources.

- **Load workspace YAML** accepts a file on the app host or an uploaded YAML and
  copies it into the editor. Apply it separately to activate it. Pipeline YAMLs
  with `stages` belong in Workflow, not in this editor.
- **Apply workspace settings** changes the active session. Changing the output
  directory switches the run registry and draft storage without moving files.
- **Download workspace YAML** exports the current editor settings to your
  computer. **Save workspace YAML** writes the same settings on the app host;
  replacing an existing file requires selecting **Replace existing settings
  file**. Neither action starts a job or activates unapplied settings.

Exports use resolved paths and retain additional fields from the loaded defaults.
Another user can load the file, adjust paths, then apply it. Source files, models
and image data are not copied. Relative paths in a server-side import follow the
usual rule: `paths.repo_root` is relative to that YAML, and other paths are relative
to the configured repository. Uploaded files have no original directory, so use
absolute paths (as in our exports); otherwise their base is the app repository.
Environment path overrides are reflected in the initial defaults; explicit UI
edits take precedence in the session. Save/export settings to reuse them after
restarting. The CLI's environment-variable overrides still apply when it loads
the exported file.

### DINO repository and weight defaults

In **Workspace settings → DINO backbone repositories and weights**, configure
DINOv2 and DINOv3 separately. The **backbone repository** is the local official
code checkout containing `hubconf.py`. Select a **default backbone** and enter
its matching checkpoint under **weights for …**. An optional **weights directory**
lets you enter a filename instead of its complete path. The directory is not
searched automatically: register and non-register models, and different model
sizes, require their own matching weights.

You can remember several checkpoints: select each architecture, enter its file,
then return to the architecture you want offered by default. **Apply workspace
settings** makes these locations available to new DINO extraction and fine/coarse
workflows, including DINO extraction added after preprocessing. Switching the
backbone inside a workflow selects its saved checkpoint, or leaves the checkpoint
box empty when none is configured. All per-run paths remain editable. Existing
nonempty paths in a draft or imported pipeline are retained; **Use saved DINO
locations for this backbone** explicitly replaces them with the current defaults.

Save or download the workspace YAML to reuse these defaults after restarting or
share them with a colleague. Paths must be accessible to the worker that runs the
job; setting a path does not copy or download weights. Every submitted workflow
saves its concrete paths, so later workspace changes do not alter previous runs.
For example, the following belongs in a **workspace defaults** YAML:

```yaml
dino:
  dinov2:
    model_repository: /path/to/dinov2
    weights_directory: /path/to/weights
    variant: dinov2_vitb14_reg
    checkpoints:
      dinov2_vitb14_reg: dinov2_vitb14_reg4_pretrain.pth
  dinov3:
    model_repository: /path/to/dinov3
    variant: dinov3_vits16
    checkpoints:
      dinov3_vits16: /path/to/matching-dinov3-small-checkpoint.pth
```

### Cluster resource presets

In **Workspace settings → Cluster resource presets**, choose **Edit presets in
this workspace** to create reusable resource templates. Select **Preset to edit**
and enter the partition, time limit, memory, CPUs, GPUs and optional account/QoS
provided by your cluster administrators. Under **Preset mail notifications**, select
the events you want and supply your email address. **Preset environment and
advanced resources** contains node/task counts, GPU request syntax, the worker's
Python interpreter, module/environment setup and runtime cache options. Help
icons explain each setting. Presets reserve resources; adding GPUs or nodes does
not itself parallelize a pipeline.

- **Add, copy or rename presets** lets you keep separate templates, for example
  `cpu-preview` and `gpu-long`. Duplicating copies all resources and notifications.
  Removing a preset only removes its template from the editor.
- **Import existing cluster presets** accepts an uploaded YAML or the file path
  in **Cluster resource presets YAML**. Click **Import presets into editor** to
  replace the editor's presets with a validated copy; an upload takes precedence
  over the path. The original file is never edited. Both a `profiles:` mapping
  and a bare mapping of preset names are accepted.
- **Apply workspace settings** makes the presets available in this session.
  In **Workflow → Review & run**, choose SLURM, expand **Load a cluster resource
  preset**, select your preset and click **Use cluster resource preset**. This
  copies its settings into that draft, where you can adjust them and review the
  generated script. Updating a template does not change resources already copied
  into drafts or submitted jobs. Published workflow tools also offer these presets.
- **Download cluster presets YAML** exports a standalone preset file.
  **Download workspace YAML** and **Save workspace YAML** include the presets
  inside the workspace settings, so recipients do not need the original preset
  file. Adjust site-specific partitions, accounts, email addresses and interpreter
  paths before using a colleague's presets. Save/export does not activate changes.

To continue using a separate file, select **Use a presets YAML file** and enter
its path, then apply the workspace settings. Review & run loads that file; it
also offers **Another YAML file** when workspace presets are active. Old workspace
YAMLs containing only `slurm.profiles` keep their existing file-based behavior.
Embedded presets are stored under `slurm.resource_presets`; `slurm.preset_source`
selects `workspace` or `file`. Validation identifies the preset and offending
field and prevents applying or exporting invalid active presets. Checks validate
the resource syntax; your cluster's scheduler determines available partitions,
accounts, quotas and maximum resources at submission.

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

**Check this page** validates the current Prepare data, Train or Analyze page.
**Continue to…** runs the same check and stays on the page if there is an error.
You can still use the page navigator to edit another step, including settings
needed by a linked stage. **Just Run Preprocessing** checks the preparation page
before advancing to review.

Messages identify the page, numbered stage, section and field where the validator
can identify one. For example:
**Train → Stage 2: Train → MAE decoder and reconstruction → Decoder dim**.
The error also appears beside the affected field. At final review, **Go to Train**
(or the relevant page) returns to its controls and opens the affected section.
Slurm errors similarly identify controls such as
**Review & run → Mail notifications → Mail address (optional)**.
Failures that cannot be attributed to one field are reported against their stage
rather than assigned to an unrelated box.

Checks report the first backend failure in each checked stage; fix those fields
and check again to reveal any remaining issues. Editing settings clears previous
check messages and invalidates an existing review. Invalid YAML typed into an
individual field stays visible when returning to that page, until corrected.
Page checks use the same validators as final review, but do not recheck another
page's data sources. Checks may read configured files, metadata and input indices;
they do not train models, extract embeddings, allocate a run or submit a job.

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

## Independent datasets in one shared PCA/UMAP

Choose **Workflow → Starting point → Shared space across independent datasets**.
This answers “where do cells from different specimens fall in the same map?”
Use **Compare representations** to compare different models on the same objects;
that operation matches IDs and fits a separate map per representation.

1. Give each dataset a unique name and select its embedding file. NPZ is preferred:
   it preserves exact integer object IDs separately from numeric features.
   IDs may repeat across datasets. Use **Add another dataset** for more inputs.
2. If embeddings are not ready, choose **Add extraction for this dataset**.
   Configure the first extraction's MAE/DINO model, checkpoint and prepared inputs
   before adding others. Additional extractions copy its model settings and
   require their own data paths. Use the same frozen checkpoint for every dataset.
   Independently trained models do not share a feature basis just because their
   vectors have the same length. Previously configured, named extraction stages
   can also be selected with **Use an earlier extraction**.
3. Optionally attach a separate annotation table and inspection YAML to each
   dataset. Annotation `label_id` values refer only to that dataset. The same cell
   type name receives the same color throughout the shared plot. Labels annotate
   the map; they do not affect fitting. This operation does not run a classifier.
4. Choose normalization, clustering and optional UMAP. **Maximum projected objects
   per dataset** defaults to zero (all). A positive cap reproducibly samples each
   dataset before fitting and can reduce the influence of a much larger specimen.
   Small datasets contribute all their objects. Unsampled objects retain their
   embeddings and annotations but have no shared coordinates or clusters.
5. **Check this page**, then **Review & run**. Use a GPU allocation if preceding
   extraction needs one. Projection alone runs on CPU. Inputs must fit in worker
   memory; the cap reduces fitting cost, but input embedding tables are still read.

The worker checks recorded checkpoint hashes, model family and relevant feature
settings, and requires equal feature dimensions. Keep each extraction's
`.metadata.json` sidecar beside its embeddings. The `embeddings.npz` copy beside
an `analysis.json` is also supported. For external files without provenance,
**I confirm identical models/settings when metadata is missing** records your
assumption as unverified. It cannot override a known mismatch. Linked extractions
are checked again once their files exist, before joint fitting. Matching weights
cannot establish that voxel spacing, crop size, intensity preparation or
acquisition conditions are biologically comparable; keep these consistent when
interpreting dataset differences.

**Results** opens `joint_analysis.json` and shows the same coordinates colored by
dataset, shared cluster and, when supplied, known cell type. Each graph has its
own legend and SVG/PNG download, with a combined figure also available. Points
selected in the shared map are routed to their dataset's original IDs under
**Inspect or export one dataset**. Image/mesh inspection and nearest neighbors in
that section use the selected dataset's own source. The joint map can compare
specimens visually; it does not load meshes from several volumes into one gallery.

The result folder contains:

- `coordinates.tsv`: shared PCA/UMAP, clusters, dataset names, `source_label_id`
  and an internal unique `label_id` for every projected object.
- `object_index.tsv`: the complete dataset/original-ID mapping and a `projected`
  flag, including unsampled objects.
- `embeddings.npz`: pooled features for projected objects, indexed by internal
  IDs. Do not use those internal IDs as segmentation labels.
- `dataset-0/analysis.json`, `dataset-1/analysis.json`, etc.: ordinary reports
  with **original dataset IDs**, full embeddings, annotations and the shared
  coordinates and cluster IDs. No projection is refitted for these reports.
  Use these for inspection, label-volume export and **Export this analysis to
  MoBIE**, selecting the matching raw/segmentation source for each dataset.
- `projection.svg`, per-dataset figures, provenance and `export.zip` containing
  the complete joint result.

Normalization, PCA, UMAP and clustering are fitted once on pooled sampled rows.
Adding a dataset and rerunning refits the map, so coordinates can change; this is
not a frozen reference model with an out-of-sample `transform`. Separation may
reflect cell biology or preparation/acquisition differences. A mixed map alone
does not establish classification accuracy or batch correction.

Load [workspace_joint_datasets.example.yaml](../configs/workspace_joint_datasets.example.yaml)
for saved embeddings or use its commented extraction template. In a pipeline
that trains once, both extraction stages can instead use `from_training: true`;
give the second extraction its own `config` with the second dataset's inputs.
Do not retrain between those extractions when sharing one model.

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

**PCA before UMAP (optional).** Previously the displayed PCA and UMAP were
independent projections of the normalized embedding vectors. That remains the
default. With UMAP enabled, set **PCA components before UMAP (0 = off)** to a
positive number, such as 50, to use this sequence:

`embedding vectors → chosen feature normalization → PCA components → 2D UMAP`

PCA summarizes the vectors before UMAP finds neighborhoods. This can make a
large embedding easier to process and suppress some variation, but it can also
remove useful biological differences. Compare the same objects with this option
off and on. The PCA variance percentage measures retained variation, not
biological accuracy. The separate two-dimensional PCA plot is still generated
from the original normalized features; its coordinates are not the input to UMAP.

The control is available in **Analyze**, **Compare representations**, **Shared
space across datasets**, **Compare DINO fine/coarse representations**, **Transfer
labels**, and **Tools → Projection and clustering**. The workflow YAML setting is:

```yaml
umap: true
umap_pca_components: 50  # null or 0 keeps the original direct-to-UMAP behavior.
```

The actual count is capped to the feature width or number of plotted objects
minus one, whichever is smaller. Results and saved report diagnostics record the
requested/actual count, fitted object count and variance retained after
normalization. In shared maps, PCA is fitted **once on the pooled selected
objects**, not separately for each dataset. Each representation in a comparison
has its own PCA and UMAP fits.

In transfer results the option applies after the chosen CORAL or neural
transformation, to that condition's reference and target vectors together. It is
off by default for every condition, including DeepCORAL; its already compact
adapted vectors may not need further reduction. To use different plotting
settings for individual conditions, select that condition in Results and use
the exploration controls described below. No DINO extraction or adapter
retraining is needed.

This plotting setting does not change exported embedding vectors, the existing
clustering input, or classifier preprocessing and scores. Classification on
PCA/UMAP still has its separate training-fold controls. MoBIE exports retain the
resulting coordinates. From the command line, use
`morphofeatures project --umap-pca-components 50 ...`; Tools and this command also
save a companion `*_projection.json` with projection provenance.

**PCA before clustering (independent option).** By default K-means clusters the
normalized embedding vectors: `standardize` scales each feature, `l2` scales each
object vector, and `none` uses the stored values. K-means does **not** use the
displayed 2D PCA or UMAP coordinates. Set **PCA components before clustering
(0 = off)** to, for example, 50 to cluster the first 50 PCA components instead.
This is independent of **PCA components before UMAP**; either, both, or neither
can be enabled. The same option applies before constructing a Leiden graph.
Results records the actual component count and retained variance. Silhouette is
measured in the selected clustering space, so compare it together with that
setting. Changing clusters does not change existing predicted cell types.

**Several combinations in one submission.** Open **Additional analysis
combinations** on Analyze, then **Add analysis combination**. Give each row a
recognizable name and choose its normalization, clustering method/count,
clustering PCA, UMAP PCA, neighborhood size, minimum distance and seed. The main
analysis is retained, and each additional combination creates its own saved
result. This works for ordinary analysis, matched representation comparisons,
shared-dataset maps, fine/coarse DINO experiments and label-transfer maps. In a
fine/coarse experiment, each representation map receives these combinations; in
transfer, each target/condition map receives them. Review shows the combinations
before local or SLURM submission. More combinations take more analysis time but
do not repeat extraction, classifier fitting or adaptation.

For **Transfer labels** and **Compare DINO fine/coarse representations**, set the
main map in **Analyze → Clustering, PCA and UMAP**, which opens expanded. You do
not need to add a combination first. Choose K-means and its cluster count, or
Leiden and its resolution. Both PCA controls, UMAP neighbors, minimum distance,
epochs, seed and unlabeled-point opacity are available here. UMAP-specific
controls appear when UMAP is enabled; clustering remains available when it is
off. The settings apply to each representation/transfer-condition map.

To make UMAP groups look tighter, lower **UMAP minimum distance** (for example,
from 0.5 to 0.1). This does not change cluster assignments or predicted cell
types. To change the number of groups, use the **K-means cluster count**, or
**Leiden resolution** (higher generally means more groups). **UMAP / Leiden
neighbors** changes the neighborhood considered by UMAP and Leiden, independently
of K-means. Classifier PCA/UMAP inputs are configured separately in the
classification section. The main seed also controls validation and, in DINO
experiments, cohort selection; use additional combinations or Results to vary
only projection/clustering seeds while retaining the original fitted results.

For example, add this to an analysis stage (or a fine/coarse or transfer stage):

```yaml
normalization: standardize
cluster_method: kmeans
clusters: 8
cluster_pca_components: null  # Main result: cluster full normalized vectors.
umap: true
umap_pca_components: null     # Main result: direct UMAP.
analysis_variants:
  - name: PCA50 for clustering and UMAP
    cluster_pca_components: 50
    umap_pca_components: 50
  - name: Twelve clusters with broader UMAP neighborhoods
    clusters: 12
    cluster_pca_components: 50
    umap_pca_components: 50
    neighbors: 30
    min_dist: 0.2
```

Omitted variant fields inherit the main analysis settings; `0`/`null` explicitly
disables either PCA step. The UI fills in a complete combination when adding it.
Every variant uses the **same exact plotted object cohort** as its parent,
including any original subset or per-dataset cap. Changing the variant seed
changes projection/clustering randomness, not cohort membership. Variants do
not tune classifiers against validation scores. Their saved classifier scores
and predictions are reused from the parent and retain its original classifier
settings, even if plot normalization changes.

**Explore an existing result without submitting another job.** In Results, open
an analysis, comparison or shared map (including a selected map inside a DINO or
transfer result). Expand **Explore clustering and UMAP on these embeddings**,
edit/add combinations, and click **Compute and save combinations**. A separate
background process on the UI host computes them sequentially while the browser
remains usable. Large embeddings can require substantial time and memory; use
Workflow with SLURM for larger batches. Closing the browser does not cancel an
already launched computation. The progress table records queued, running,
completed or failed states. A failed combination keeps its error and does not
discard successful combinations or overwrite the original result.

Choose **Saved analysis combination** to display the new coordinates and clusters,
then inspect objects, export figures or launch MoBIE export from that selected
result. Joint results preserve dataset names and original object IDs, including
their image/mesh inspection configuration. For combinations computed later,
use the selected result's own downloads; the original ZIP remains the snapshot
created with the original analysis.

Each parent has an `analysis_variants.json` index. Every combination lives under
`variants/<stable-id>/`, with its own report, coordinates, object labels,
SVG/PNG figures, export ZIP, and `variant.json` execution record. Embedding files
are hard-linked within a filesystem (copied when linking is unavailable), so
they remain the original feature vectors; treat saved scientific artifacts as
immutable. New results are attached to the existing run's registry when its
`job.yaml` and registry are available; standalone results still retain their
local index. The provenance graph connects parent and derived results and shows
settings and recorded errors. The small current metadata download includes the
latest display-name history.

**Rename completed work.** In Runs, use **Rename completed run → Run display
name → Save run name**. In Results, use **Rename this result**; this also works
for an individual saved combination. Renaming updates the displayed name and
provenance history. Permanent run IDs, scheduler IDs, filenames, dependencies
and volume/object IDs remain stable. Original submission configurations and
already-created ZIP snapshots retain their historical content.

### Stop a running job

In **Runs**, select the job's row. The **Stop job** controls are directly below
its name, above provenance and execution details. They apply to that one job;
selecting several rows for deletion does not stop all of them.

- **SLURM:** check **Confirm cancellation of this SLURM job**, then click
  **Cancel SLURM job**. This sends `scancel` for the allocation. Use **Refresh
  SLURM** to confirm it has stopped. The control remains available if a worker
  failed but SLURM still reports a live allocation, and a pending request can be
  resent.
- **Local:** check **Confirm stopping this local job**, then click **Stop local
  job**. This sends `SIGTERM` to the worker's process group, including its data
  loaders. Use **Refresh local job status** to check termination. If processes
  remain stuck, enable **Force kill local worker (SIGKILL)**, confirm again and
  click **Force kill local job**.

Local stopping is available on the Linux/macOS host where the job was launched.
The UI checks the launch host, process creation time, command and working
directory before sending a signal. New jobs record these details in `process.json`.
Older PID-only launch records can still be used if the live worker's command,
directory and launch time match. A recycled PID or a worker on another host is
not signalled; the UI explains the mismatch.

Stopping keeps already-written files. A file interrupted during writing may be
incomplete, and stopping does not guarantee a final checkpoint. Local jobs stay
**cancelling** while their worker group is alive; confirmed stops become
**cancelled**. Pending SLURM cancellations also remain protected from deletion.
Only after stopping is confirmed can you delete the run and its results.

### Delete finished runs and their results

In **Runs**, use the checkboxes at the left of the table to select one or more
rows. For one run, open **Delete finished run and results**. For several, open
**Delete N selected runs and results**. The most recently added row opens below
for inspection; the deletion section always names the complete selection.
Filtering removes hidden rows from the selection, so check the selection again
after changing a filter. Multiple job rows sharing a run ID count as one run.

1. Click **Review deletion** (or **Review deletion of selected runs**). This checks
   every job sharing the selected run IDs, counts the files, and looks for
   registered runs that depend on these outputs.
   No files are removed during review.
2. Read the **Are you sure?** prompt. For multiple runs, it gives the total number
   of folders, files and their size. Expand each **Files and dependencies** section
   for every registered job/stage, the exact directory and its top-level contents.
   A large run may take time to scan. The size is logical file size; hard-linked copies
   elsewhere can reduce the actual disk space recovered.
3. Check the confirmation box, then click **Permanently delete run** or
   **Permanently delete N runs**. **Keep run(s)** cancels without changing files.
   All selected runs are checked again before deletion starts. A changed file,
   active job or changed selection requires another review and confirmation;
   no selected folder is removed if this check fails.

Deletion removes the entire `outputs/experiments/<run-id>/` directory: settings
snapshots, logs, prepared crops, trained checkpoints, embeddings, plots, exports,
and interactive analysis variants stored there. This applies to **all jobs with
that run ID**, even if you selected just one row. Anything else you manually put
inside that directory is also removed. There is no undo.

Source images, pretrained weights, shared embedding/remote caches, custom output
files and other artifacts **outside the run folder are kept**. The review lists
known external references. These may be shared with other runs, so a reference
in the registry is not treated as permission to delete them. Symlinks are unlinked
without following their targets; a target inside another selected run is removed
as part of that run. Automatic deletion refuses a symlinked run root or a nested
mounted directory. Deleting an experiment does not clean unrelated notebook
outputs or the workspace-wide caches.

Queued/running jobs, unknown job states and active interactive analyses block
deletion. For a cancelled SLURM job, **Refresh SLURM** must confirm that cancellation
has finished; a cancellation request alone is insufficient. Finished downstream
runs outside the selection are listed as a warning: their files remain, but links to the deleted inputs
and recomputation using those inputs will break. Active downstream dependencies
block deletion. These checks cover saved references in this workspace; references
from notebooks, other workspaces or external applications cannot all be detected.

**A failed job still looks running.** Opening Runs or Results, and reviewing a
deletion, now checks saved worker outcomes against the registry. A final
`status.json` with its completion time (or a recorded registry exit status and
completion time) supersedes an old `RUNNING`/`PENDING` observation. The table labels
that observation as outdated and provenance retains it. The same check applies
to dependent runs: a failed downstream job no longer blocks removal of its inputs.
Its surviving references are still shown in the deletion warning.

**Refresh SLURM** also checks finished jobs whose scheduler state was never
finalized. Original worker errors and partial results survive the refresh, even
if old accounting records have been purged. If a later scheduler check confirms
that the allocation is still running, completing or requeued, deletion remains
blocked until SLURM confirms it has stopped. A timeout or forced kill may leave
the worker's progress file unfinished; use **Refresh SLURM** for that case.
Age alone or an error printed in a log does not prove a process has stopped.

After deletion, the run disappears from **Runs** and its registered outputs from
**Results**. Stale navigation pointers are cleared. A small deletion audit remains
in the SQLite registry, preserving job IDs and dependency history without retaining
the deleted results. Use a new run ID for subsequent experiments.

If file removal is interrupted or encounters a permission error, the run stays
listed as `deleting` or `deletion_failed`. Its files have been moved to a private
cleanup directory below `outputs/.morphofeatures/deleted_runs/`. The UI reports
that location. Resolve the permission problem if necessary, then use **Review
deletion** again to remove the remaining files. This recovery directory is not
a backup or an undo facility; cleanup may already have removed some files.
For a batch, successful deletions disappear from the table and any unfinished
cleanup remains listed with its own error. Select the remaining runs and review
again to retry. A filesystem cleanup error does not undo the other deletions.

### Finding and reusing pipeline outputs

**What this workflow will produce**, immediately below the input checklist, lists
the outputs of the selected entry point. Once you create or import a draft, it
follows the actual enabled stages. For a completed run, select its row in **Runs**
and use **Open results · …** under **Completed results**. These buttons open the
registered reports directly, without selecting through settings and log files.
The existing **Continue from an output** selector remains available and lists
report entry points first. A stale scheduler state does not hide completed worker
results. A missing report is shown explicitly rather than silently recomputed.

**Results** loads registered local/SLURM outputs or an explicit `experiment.json`,
`analysis.json`, `comparison.json`, `joint_analysis.json`, `transfer.json`, spatial
probe/export report, or embedding file. For a DINO experiment, start with the
top-level **`experiment.json`** to see all representations and downloads together.
The **Outputs and how to use them** section explains the selected result; fine/coarse
experiments also list their file layout and exact original/matched vector paths.
No inference is repeated.

| Pipeline stage | What to retain and where to start |
| --- | --- |
| Prepare data | Prepared crops/patches and masks, object IDs, reusable data YAML and `preprocessing.json`. Distributed patches also record per-patch and coverage tables. |
| Train MAE | Checkpoints, resolved model/data configuration and training metrics. Weights alone do not describe the input layout. |
| Extract MAE/DINO | Embedding NPZ and matching `.metadata.json` sidecar; extraction reports and any requested physical-view diagnostics. |
| Analyze / compare | `analysis.json` / `comparison.json`, embedding copies, coordinates, object labels and figures. Classification adds predictions, held-out splits and metrics. |
| Shared dataset map | `joint_analysis.json`, shared coordinates/figures and per-dataset outputs retaining native IDs. |
| DINO fine/coarse | `experiment.json`; extraction, fusion, matched vectors and comparisons. See the detailed output map below. |
| Combine vectors | `embeddings.npz`, `embeddings.metadata.json`, `fusion.json`, original component copies and `excluded.tsv`. |
| Transfer labels | `transfer.json`, reusable reference artifacts, source validation and per-target/per-condition vectors, predictions and maps. Adaptation adds transforms; accuracy requires known target labels. |
| Bilateral retrieval | `bilateral.json`, `summary.tsv`, per-object ranks, ID mappings, comparison plots and any requested comparisons to earlier runs. |
| Spatial context | `context.json`, context vectors and metadata, graph tables, ID mappings and `coverage.tsv`; optional downstream analysis. |
| Label-volume / bilateral / MoBIE export | Export manifest, category/pair tables, requested volumes/viewer files or MoBIE project. Referenced images remain external unless explicitly copied. |

**Which file should I use next?** An NPZ contains `label_ids` and the full
`features` matrix. A `coordinates.tsv` contains low-dimensional plotting positions
and related labels; it is not a replacement for the feature vectors. An
`object_labels.tsv` links object IDs to known/predicted types and clusters for
inspection or spatial export. IDs must be joined explicitly, with the recorded
cell-to-nucleus mapping where required. Keep the embedding metadata alongside
the NPZ when reusing or sharing vectors. The DINO result page offers both downloads.

Worker-created `export.zip` files bundle result files and figures, not original
image volumes, remote data caches or backbone repositories/weights. They are
snapshots: later interactive analyses have their own saved outputs. Browser
downloads are limited to **200 MiB** per file; larger files remain on the worker
and the UI gives their paths for direct copying. Optional outputs appear only
when the relevant stage/data is present; a successful extraction alone does not
imply that classification or label transfer was run.

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

### Stream other OME-Zarr sources and preview volumes

Choose **Workflow → Start → Stream OME-Zarr raw and segmentation**, or load
[workspace_ome_zarr.example.yaml](../configs/workspace_ome_zarr.example.yaml).
Enter either or both image-group URLs in **Prepare data**. For WEBKNOSSOS, use the
per-layer Zarr link from its sharing/export dialog; these links can expire or be
revoked ([WEBKNOSSOS documentation](https://docs.webknossos.org/webknossos/data/zarr_stream.html)).
Reusable repository examples leave URLs blank so private share links are not published.

- Click **Find raw resolutions** / **Find segmentation resolutions** to populate
  a selector showing each array path and physical voxel spacing. The YAML `level`
  is the **zero-based index in the metadata list**, not the path or magnification:
  a WEBKNOSSOS list `1, 2, 4, 8, 16, 32, 64` has indices `0…6`.
- Choose channels and timepoints, then **Connect OME-Zarr sources**. Connection
  and **Inspect data dimensions** read metadata only. NGFF axes and transforms
  determine the displayed Z,Y,X shape, voxel spacing and origin. Stored `cxyz`,
  `tczyx` and other permutations are transposed correctly on read.
- Enable **Show ROI image preview**. Choose raw, segmentation, or an alignment
  overlay when both grids match. Select the slice axis and position. Either layer
  also works alone, including local HDF5/N5/NumPy/Zarr volumes. Different grids
  can be previewed separately; the UI disables their overlay. Previews read a
  central window of at most 512×512 pixels and use percentile contrast for raw
  display; integer instance IDs remain exact. **Use a central 64-voxel ROI** is a
  convenient starting region, and explicit bounds can select any other location.

Before preparing crops, supply both raw and integer instance-label volumes on
matching dimensions, spacing, origin and units. Generic OME-Zarr sources have no
published instance catalog, so preprocessing scans **only an explicit ROI**.
`max_scan_voxels` defaults to 128³ and cannot exceed 256³; `max_objects` limits crop
extraction after the ROI scan. A null or oversized ROI is rejected before a job
is submitted. Zero is segmentation background. Check ROI boundary and cropping
policies before increasing the sample size; objects can intersect ROI edges.
The resulting local HDF5/N5/NPY crops feed the same MAE, DINO and analysis stages
as other Workflow sources. Percentile normalization is appropriate for raw
floating-point images whose intensities are outside [0,1].

The reader uses **Zarr-Python** (and its codecs) over the existing cached HTTP
transport. Zarr-Python 2 reads Zarr v2 sources; Zarr-Python 3 reads v2 and v3,
including NGFF 0.5 metadata. No `webknossos`, `ome-zarr`, `fsspec` or `aiohttp`
installation is required. Install the existing `workspace` extra if Zarr is absent;
use Zarr-Python 3 in both UI and worker environments for v3 sources. Chunk/shard
size and per-read byte/chunk limits apply; very large shards require a different
source/level. HTTP(S) image groups with standard multiscale metadata, numeric
3D arrays, optional channel/time axes, and axis-aligned scale/translation transforms
are supported. Cloud-specific `s3://` URLs, login/header-authenticated endpoints,
2D images, and rotated/sheared registrations require a different access path or
conversion. Missing chunks follow the declared Zarr fill value; authentication
and network errors remain errors.

The cache and YAML retain the selected URL, array path, grid and channel/timepoint.
Change the cache version after updating an unfinished hosted segmentation. Preview
settings alone do not prove that two layers label the same biological specimen.

### Train and embed streamed PlatyBrowser objects

In **Workflow → Start**, choose **Stream PlatyBrowser raw and segmentation**, or
load [workspace_platybrowser.example.yaml](../configs/sites/workspace_platybrowser.example.yaml).
The example prepares eight real nuclei into HDF5, trains a small crop MAE for one
epoch, extracts its embeddings and runs PCA/clustering. It starts no job until
submitted through **Review & run**.

1. In **Prepare data**, enter the project, dataset version and metadata revision.
   Click **Load available PlatyBrowser datasets**, select the raw intensity image
   and instance segmentation, then **Connect Workflow to PlatyBrowser**.
   Source names can also be entered directly or supplied in YAML as
   `platybrowser.raw_source` and `platybrowser.segmentation_source`.
2. Select matching resolutions. Dataset `1.0.1`'s `raw` level **3** and `nuclei`
   level **0** have the same physical grid. Connection validates dimensions,
   spacing and origin. Sources with different grids or rotated/sheared
   registrations require local resampling first. Segmentations need a published
   default TSV containing object IDs and physical bounding boxes. Region labels
   yield region crops; select an instance segmentation for cell/nucleus embeddings.
3. Inspect dimensions and use the bounded raw/label overlay to check an ROI.
   `roi: null` selects published objects throughout the volume without scanning
   all segmentation voxels. **Specific object IDs** selects stored segmentation
   IDs, and **Maximum objects** limits the number of successfully prepared crops.
   Start small before selecting all objects. The worker measures each selected
   object in its actual label voxels; absent/coarsened-away objects and read-limit
   failures are recorded in `objects.tsv`.
4. Expand **Choose training and embedding stages**. Choose **Train a new MAE and
   embed**, **Embed with an existing MAE**, or **No MAE**; optionally include DINO.
   Applying the choices creates an analysis stage for each embedding method and
   optionally a joint comparison. With no models selected, only preprocessing runs.
5. Configure training under **Train**, and checkpoints/DINO views under
   **Analyze → Extract embeddings**. Existing MAE checkpoints must match the crop
   architecture, crop shape and token size. Grouped N5 patch checkpoints require
   their original input layout; the Workflow does not convert them into crop models.
   DINOv2/v3 use the existing local pretrained-weight and backbone-repository fields.
6. Review the resolved, pinned source settings and submit locally or through Slurm.
   The worker host needs outbound HTTPS access to GitHub metadata and the image
   server, plus access to the configured cache and output directories.

Preprocessing streams bounded object windows into reusable local NPY/HDF5/N5
crops, masks and IDs. Training and both extractors use the same prepared objects;
training epochs do not repeatedly stream the original volume. The crop size,
object count, persistent download cache and per-object read budget are independent
controls. `oversized: clip` truncates larger objects to the crop; use larger crops
or `skip` when complete objects are needed. Large tissue regions/cells can exceed
the read budget even when the final crop is small; use coarser aligned grids.

**IDs remain the stored segmentation IDs**: the nuclei source produces nucleus-ID
embeddings, while the cells source produces cell-ID embeddings. Annotation tables
for classification/overlays must match that namespace. Cell-ID annotations from the
existing grouped nuclear patches require conversion with the published
`cells_to_nuclei.tsv` before use with newly prepared nucleus-ID embeddings.
The generated `mae_config.yaml` retains the inspection source and spatial metadata
for subsequent raw previews and meshes in Results. The run's `preprocessing.json`
records the published revision, source URLs, grid, table checksum and crop policies.
A metadata commit does not freeze the separately hosted image bytes; change the
cache version when those bytes change.

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

## Distributed fine patches for DINO

Choose **Prepare data → Object preparation layout → distributed_patches** to save
several small patches from each nucleus/cell instead of a single crop. Start with
**Raw image and instance segmentation**, **Just Run Preprocessing**, or either
streaming source. Existing single-crop preparation remains the default.

What you should have:

- A raw image and an **instance** segmentation: one distinct integer ID per object,
  with zero as background. A binary foreground mask alone cannot separate objects.
- Measured voxel sizes, coordinate origins and units. Both volumes must describe
  the same physical frame, though their resolutions may differ. ROI bounds are
  always in **segmentation voxels**, with an exclusive stop.
- Storage for the patches and masks. To embed them, also supply a local frozen
  DINOv2/v3 checkpoint and the matching official model checkout.

Use [workspace_distributed_patches.example.yaml](../configs/workspace_distributed_patches.example.yaml)
as a reusable backend/UI example. Replace its paths and example voxel sizes with
measured values. Existing PlatyBrowser and OME-Zarr source selectors work with
this layout; connect after selecting the layout and resolution levels. Remote
scanning without an object table requires an explicit bounded ROI. Whole-object
crop settings such as `center`, `oversized`, `boundary`, `normalization` and
`background` do **not** govern distributed patches; the patch controls below do.

### Choose physical size and sampling

| UI setting / YAML | Meaning and starting value |
| --- | --- |
| Fine patch sampling / `patches.sampling` | `physical_resampled` (default) fits a physical box onto a chosen output grid. `native_voxels` directly cuts original raw voxels without spatial interpolation or smoothing. |
| Fine patch width / `patches.tile_size_um` | Cubic field of view in µm for physical mode. **1 µm** is an experimental starting point; use the same physical width across datasets. Ignored and hidden in native mode. |
| Saved/native fine patch shape / `crop_shape` | Default **32³**. Physical mode requires equal sides, maximum 128³; 1 µm at 32³ gives 31.25 nm per saved voxel. Native mode allows unequal sides, each 1–128: these are exact counts of raw voxels. |
| Minimum object content / `patches.min_foreground_fraction` | Fraction of the entire **3D** tile occupied by this object, default **0.5**. 0.25 permits more boundary tiles; 1.0 keeps only full interior tiles. Nearby objects do not count as foreground. |
| Patch grid placement / `patches.tile_anchor` | `centroid` centers one tile on mean foreground position; `bbox_center` uses the bounds midpoint; `bbox_min` starts at the lower bounds corner. The non-overlapping grid extends across the bounds. |
| Maximum fine patches / `patches.max_tiles` | Default **64**. Start near the chosen center, then repeatedly choose the eligible tile farthest from previously chosen centers. This covers separated regions reproducibly. 200 is another sampling budget, not a requirement of the original method or a GPU batch size. |
| Save every eligible patch / `patches.exhaustive` | Default false. When true, save every eligible lattice tile and ignore `max_tiles`. Foreground filtering and safety limits still apply. |
| Object contrast / `patches.percentiles` | Default **[1,99]**. Estimate contrast from at most 65,536 deterministic foreground samples **once per object** and share it across that object's patches. |
| `max_objects`, `object_ids`, `seed` | Select objects independently of the patch cap. Zero means all objects; null/empty IDs means no ID filter. A seeded cohort is selected before extraction; skipped objects are not replaced. |
| `max_object_voxels` | Maximum segmentation bounding box including a one-voxel halo, default 256³, configurable up to one billion voxels. Oversized objects are skipped with a reason, never silently cropped. Raising it also needs sufficient RAM for label arrays, masks and foreground coordinates; SLURM memory does not increase automatically. |
| `patches.max_candidate_tiles` | Maximum lattice size before foreground filtering, default 250,000. Increase patch width if the grid is too large. |

Eligibility integrates foreground voxel volume over each physical tile. The mask
saved on the chosen grid uses nearest-neighbor sampling; its measured fraction
can therefore differ slightly from the eligibility fraction. Thresholding happens
**before** resampling. Tiny structures can disappear on a coarse saved grid;
empty patches are excluded and reported. If a tile spans only a few native
segmentation voxels, its mask can be blocky: select a finer segmentation level
when available. The UI shows native voxel counts for both sources and flags
fewer than four segmentation voxels per patch side. Raw intensities use linear interpolation
with a Gaussian prefilter when downsampling in physical mode. Saved pixels are float32 in [0,1];
background is zero, accompanied by an explicit mask so real zero-valued foreground
remains foreground. Only this object's intensities are retained.

### Keep patches in native voxel space

Choose **Fine patch sampling → Native raw voxels — do not resample**. Set the
native patch shape in Z/Y/X voxel counts. For example, `[32, 32, 32]` at
`[25, 10, 10]` nm spans **0.8 × 0.32 × 0.32 µm**. Coordinate units and spacing
describe existing pixels; entering them does not itself resample an image.

The native grid snaps the chosen anchor to raw voxel boundaries, then distributes
non-overlapping patches through the object. The saved intensities come from direct
raw array reads. Per-object contrast normalization and masking still apply, so
the saved values are normalized float32 rather than an unchanged copy of source
gray levels. A segmentation at a different resolution is mapped to those raw
voxel centers with nearest neighbors. Physical metadata remains necessary for
that alignment and for saved patch positions.

Native sampling preserves the acquired voxel grid, including its anisotropy.
Equal voxel counts in two datasets need not span equal physical tissue sizes;
use physical mode when a fixed physical field of view across datasets matters.
Native patch dimensions, effective spacing and grid algorithm are saved in the
representation recipe. Existing physical-mode configurations keep their behavior.
DINO still resizes **2D slices** to its image canvas for inference; letterbox
resizing uses physical aspect ratio for native patches. That does not resample
the saved 3D arrays.

To select native mode in the example YAML:

```yaml
crop_shape: [32, 32, 32] # Raw voxel counts; rectangular shapes also work.
patches:
  sampling: native_voxels
  min_foreground_fraction: 0.5
  tile_anchor: centroid
  max_tiles: 64
  exhaustive: false
  percentiles: [1, 99]
```

### Preview one object from a large volume

1. Optionally enter an **Existing object bounds table** (`objects.tsv`) from an
   earlier preparation, physical DINO extraction or preview of this exact grid.
   This provides an immediate spatial lookup without scanning the segmentation.
2. Enter a **Preview object ID**, or leave it blank to choose an eligible object
   reproducibly from the current selection. This ID changes the preview only.
3. Click **Preview distributed fine patches**. Lookup and reading run in the
   background, with progress and cancellation; the UI remains usable. If bounds
   are unavailable, a one-time scan locates IDs in the selected segmentation ROI
   using bounded 64³ reads. Large local ROIs are allowed. A label ID alone cannot
   reveal where its voxels are, so this first scan can take time; it is not
   guaranteed to be quick for a whole large volume. Remote scans still require
   an explicit ROI within `max_scan_voxels`.
4. Inspect the selected object's XY/XZ/YZ context and up to six actual output
   patches, with mask contours and physical aspect ratios. The overview is
   reduced for display; it does not change saved patch resolution. Object-box
   memory and complete-object checks still apply. Increase the memory limit only
   when RAM permits; the preview never silently crops an oversized object.
5. Use **Use preview bounds for preparation** to reuse the lookup in the worker.
   Bounds are cached under `outputs/preview_indices` (or the active workspace).
   They are reused across patch widths, thresholds and preview IDs. Changes to
   the segmentation file metadata, grid, ROI, mapping or `data_version` invalidate
   that lookup. Cancelled scans never become complete cached indices.

Only one object's raw context/patches are read; the full raw volume is never
loaded. Previewing does not start DINO, training or a SLURM job. The preview is a
sample, not a guarantee that every object passes; inspect the saved coverage
report after preparation. Changing grid placement or thresholds changes which
locations qualify. Objects missing from the ROI need a different ROI or source;
allowing partial coverage cannot recover missing tissue.

### Run, reuse and compare

Select **Include DINO embeddings** under **Choose training and embedding stages**,
then apply the choices. Configure weights and feature pooling on **Analyze**.
The generated stages use one center slice per axis per patch: up to 192 DINO
images per object at the default 64-patch cap. Slice axes/fractions and model image
batch size remain configurable. Empty masked slices are omitted. Valid view
vectors are averaged equally (or combined by maximum), yielding **one vector per
native object ID**, not one vector per patch. Averaging loses the spatial layout
between patches and can dilute rare structures; it is a texture summary.

For a preparation-only job, select **Just Run Preprocessing**. Later, in a DINO
workflow, load its **data_config.yaml** under **Load model / data settings**.
Loading this layout selects stored views, `normalization: unit` and letterbox
resizing: the input already has shared per-object contrast. Applying a second
per-patch percentile normalization would change that contrast and is rejected.
Choose CLS, unweighted patch tokens or foreground-weighted tokens as usual.

This saved format currently feeds **DINO**, not the single-crop MAE. It does not
train a grouped MAE or reproduce the original MorphoFeatures encoder. Existing
`n5_masked_patches` MAE/DINO inputs keep their existing reader and behavior. New
patches are physical lattice samples, so even a 32³ grid and 200-patch cap do not
imply the same recipe as the historical Platynereis patches.

Compare single-object and distributed inputs using the same checkpoint, physical
scale, object cohort, classifier and held-out folds. To combine fine and coarse
representations, run a coarse extraction on those IDs and feed the two saved
embedding files into **Combine embeddings**; this preprocessing layout itself
produces only fine patches. **Compare DINO fine/coarse representations** remains
the direct-volume experiment that samples physical 2D planes without this
intermediate saved 3D grid, and its defaults differ. Do not assume the two
extraction recipes are interchangeable.

### Saved files and backend contract

NPY output writes separate arrays. HDF5/N5 output puts **all arrays** into
`crops.h5` / `crops.n5`; `data_config.yaml` records their explicit keys:

| Array | Contents |
| --- | --- |
| `crops` | `(Npatches,Z,Y,X)` float32 normalized intensity. |
| `masks` (`data.loss_masks`) | Matching binary foreground mask (uint8 in N5). |
| `label_ids` | Unique int64 native object IDs, length Nobjects. |
| `patch_offsets` | int64 length Nobjects+1; patches for object i occupy `[offsets[i], offsets[i+1])`. |
| `patch_label_ids` | Repeated int64 parent ID for every patch row, checked against offsets. |
| `positions_um` | Float64 `(Npatches,3)` physical Z,Y,X centers. IDs never pass through floats. |

`objects.tsv` records selected object bounds in segmentation voxels and any mapped
segmentation IDs. `patches.tsv` records saved/excluded patches, bounds and centers
in µm, normalization limits and resampled foreground fractions. `coverage.tsv`
records eligible, selected and saved patch counts plus object-level exclusions.
`preprocessing.json` retains resolved sources, algorithm version, settings and
summary counts; the generated configuration retains the original volume and mesh
inspection information. These files are written even when no object succeeds,
so the failure has a concrete report to inspect.

Objects touching a volume/ROI boundary, exceeding the mask budget, lacking usable
contrast, or lying outside the raw coverage are skipped. A source index with
incomplete bounds is rejected for that object. Sparse saved patches must never
be described as full voxel coverage. Failed patch reads are recorded, and an
object with partial success is flagged in `coverage.tsv`.

The reader streams individual patches and model batches, and accumulates vectors
without retaining all per-view features. Cache keys include the grouped array
files, preprocessing manifest, algorithm and sampling recipe. Shared-space checks
compare distributed-patch recipes; source paths/IDs retain dataset identity.
Prepared-object inspection displays one representative patch, explicitly labeled
as such; original-volume inspection and mesh/MoBIE export use the preserved source
volumes and native IDs, not the patch stack as a segmentation.

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

## Finish a workflow in MoBIE

Open a saved analysis/comparison in **Results → Export this analysis to MoBIE**,
or use **Change which stages run → Add MoBIE export after analysis**. The export
runs through the normal Review & run step, locally or on Slurm, and creates a
new MoBIE project with original instance IDs, label/cluster tables and saved
UMAP/PCA views. A comparison includes all its representations.

Choose a portable image region (local or supported streamed sources) or a
PlatyBrowser project that continues streaming the published images. The export
report lists matched and excluded IDs. Saved analysis files and explicitly
selected training/inspection artifacts can accompany the project.

See the [MoBIE guide for biologists](analysis_and_mobie.md) for input preparation,
ID mapping, region selection, outputs, Fiji instructions and sharing, and
[workspace_mobie.example.yaml](../configs/workspace_mobie.example.yaml) for a
complete streaming export configuration.

## Target a chosen set of cells or nuclei

**Target objects** is part of the existing preparation and DINO workflows. Use it
to embed annotated cells, a particular cell type, a manually selected population,
or any other list of objects. It does not require class labels and does not train
the DINO model.

What you need: the existing image/crop inputs and their exact object IDs, plus
either an ID list or a CSV/TSV with an ID column. The IDs must refer to the **same
specimen and ID namespace** as your input. Cell IDs and nucleus IDs are different,
even when the numbers overlap. An optional mapping translates between them.

1. Open **Prepare data → Target objects** for preprocessing or each dataset in a
   fine/coarse experiment. For **DINO features from prepared objects**, the same
   section appears with extraction settings on **Analyze**.
2. Choose **Enter an ID list** (for example `[123, 456]`) or **Use an ID /
   annotation table**. Supply a file accessible to the UI and local/SLURM worker,
   and select its ID-column header. No new pipeline or separate preprocessing run
   is required for an already prepared input.
3. Optionally set an annotation/category column. **Only rows with a nonempty
   annotation** drops blank entries. **Categories to include** accepts a list
   such as `[neuron, muscle]`; `[]` retains every category. IDs are read exactly,
   duplicates are removed and counted, and table order does not affect matching.
4. If necessary, use **Translate target IDs** to provide a CSV/TSV with a positive,
   one-to-one mapping from target-table IDs to input IDs. This changes selection,
   **not the output ID namespace**. Physical sources can match workflow output IDs
   (default) or native segmentation IDs. Prepared crops expose their saved IDs.
5. Click **Check target table / IDs** to inspect the number of targets and a sample.
   This reads metadata only; it does not yet verify availability in the input/ROI.
   Missing mappings are highlighted. Missing source IDs stop the worker by default
   with a remedy; **Skip unavailable IDs and report them** is an explicit alternative.
   An empty or entirely unmatched selection always fails, never falls back to all
   objects.
6. Check the existing object cap. It applies **after** targeting. For fine/coarse
   experiments, the default 100-object pilot still selects at most 100 from your
   targets; **Use every targeted object (set object cap to 0)** selects all of them.
   That cap is shared by all datasets in an experiment. Existing explicit ID lists
   and ROIs remain additional restrictions. Boundary, size and quality policies
   still apply. All fine/coarse variants use the same selected cohort.

### Use the bundled Platynereis ground-truth cells

Click **Use bundled Platynereis ground-truth IDs**. It loads
`analysis/data/class_labels.tsv`, selecting `label_id` rows with a nonempty
`cell_type`. The current file contains **390 unique cell IDs** across seven types.
It uses **cell IDs**, even when the images being embedded are nuclei.

- For a physical fine/coarse or distributed-patch **PlatyBrowser** source, choose
  `cell` under **Object IDs used by annotations** and keep **Match selected IDs
  against → Workflow output IDs**. The published mapping connects these cells to
  the nuclear masks.
- For **nucleus-indexed** saved crops or local nuclear masks, supply
  `cells_to_nuclei.tsv` under **Selection ID mapping table**, with `label_id` as
  the mapping's input column and `nucleus_id` as its output column. Single-crop
  preparation retains native mask IDs. If you later classify nucleus-indexed
  embeddings, your annotations must also be explicitly mapped to nucleus IDs;
  selection does not silently rewrite classification annotations.
- For prepared embeddings/crops already indexed by **cell IDs**, no translation
  is needed. Check the preparation metadata rather than guessing from numeric
  overlap.

To select all annotated cells rather than a pilot subset, set the cap to zero.
To try one class, set **Categories to include** to `[neuron]`, for example.
Using annotation availability to define a cohort does not feed the class labels
to DINO. It does change the population being studied: clustering and held-out
scores on these cells need not describe the entire animal. Configure classification
annotations separately on Analyze when you want a probe.

### YAML, reports and computation

The same additive block is accepted at the root of a `preprocess` or DINO `extract`
stage, or under `datasets[].source` in `dino_experiment`:

```yaml
object_selection:
  mode: table                 # all (default), ids, or table
  ids_file: /path/to/targets.tsv
  id_column: label_id
  filter_column: cell_type    # null: use every row, no annotation filter
  require_nonempty: true
  filter_values: []           # e.g. [neuron, muscle]
  id_space: output            # physical sources also accept segmentation
  missing: error             # skip records unavailable IDs and continues
  mapping_file: null          # optional explicit cell → nucleus translation
  mapping_from: label_id
  mapping_to: nucleus_id
# Alternatively: object_selection: {mode: ids, ids: [123, 456]}
```

Relative paths in imported YAML resolve beside that YAML; paths entered directly
in the UI resolve from the repository root. The reusable
[fine/coarse YAML](../configs/workspace_dino_fine_coarse.example.yaml) includes the
selection fields with `mode: all`, preserving its original behavior.

Selection happens before **per-object crop reads and encoding**. Existing crops
can be filtered from their saved ID array. A volume without an object index still
needs a bounded-chunk segmentation scan to locate objects; knowing an ID does not
give its bounding box. PlatyBrowser's published bounds or an existing compatible
object table avoid this scan. Remote sources retain their ROI/read safeguards.

When enabled, `selection.json` and `selection.tsv` record requested IDs, translated
IDs, output IDs, missing entries, and cap exclusions. Single-crop preparation also
records its preparation outcome/reason. Fine/coarse reports live in
`dataset-N/input/`; the experiment Results page shows counts and a downloadable
selection table. Physical extraction keeps its reports beside `objects.tsv` in
the materialized view output, and stored-crop extraction writes them in its stage
directory, including on cache hits. Later view/contrast/quality exclusions remain
in the usual `coverage.tsv`, `excluded.tsv`, or extraction metadata. The downstream
`matched-*` directories may contain a smaller intersection across representations.

Selection settings and the contents of ID/mapping files participate in extraction
cache identity, so editing a table cannot silently reuse embeddings for an old
selection. Target selection does not change the model/view recipe or the dataset's
identity; subsets remain comparable with other runs of the same representation.

## DINO fine/coarse experiments

**Workflow → Start → Compare DINO fine/coarse representations** compares several
ways of describing the same nuclei. It uses frozen DINO weights: there is no DINO
training step. Existing DINO workflows that read stored crops or grouped patches
keep their previous behavior.

Start with [the reusable experiment YAML](../configs/workspace_dino_fine_coarse.example.yaml),
or [the public Platynereis preset](../configs/sites/workspace_dino_platybrowser.example.yaml).
Load the YAML in Start, replace the checkpoint and official repository paths, and
work through Prepare data and Analyze. The commented Parhyale entry is a template
for when its **instance segmentation** is ready. Each nonzero segmentation ID must
identify one nucleus; a binary foreground mask cannot distinguish nuclei.

Creating or importing this workflow opens **Prepare data**. Configure sources,
objects, physical patch widths, coverage cutoffs, placement and slice/image
preparation there, then inspect **Preview DINO inputs** for each dataset. In
**Analyze**, select the backbones, model execution settings, feature-combination
weights, projections and classification/transfer checks. A read-only summary and
**Edit data and DINO views** button link back to the preparation settings. The
same options remain available, with each edited on one page.

### Prepare a small, physically meaningful pilot

1. In **Prepare data**, give each independent dataset a distinct name. Connect its
   local raw/segmentation containers, PlatyBrowser sources, or two OME-Zarr layers.
   OME-Zarr offers resolution discovery plus channel and timepoint selection.
   An existing inspection YAML can supply source settings.
2. Check the **voxel spacing, origin and unit** for the selected arrays, in Z/Y/X
   order. Local arrays require measured metadata. Remote sources use their
   published metadata. Coordinates refer to voxel centers. Different grids are
   supported only when they already share a registered physical coordinate frame;
   this workflow does not estimate registration. Raw images are interpolated and
   filtered when downsampled; instance IDs are sampled with nearest neighbors.
3. Use **Target objects** for an ID list or annotation table, and/or select a
   segmentation-grid ROI (see [targeting instructions](#target-a-chosen-set-of-cells-or-nuclei)). PlatyBrowser's
   published bounds avoid a whole-volume scan. Other sources may supply a TSV
   containing `label_id`, `bbox_min_z`, `bbox_min_y`, `bbox_min_x`, `bbox_max_z`,
   `bbox_max_y`, `bbox_max_x`. Coordinates are segmentation voxel indices and stop
   coordinates are exclusive. Otherwise the worker indexes the selected region in
   chunks. Remote indexing requires an explicit ROI. An interactive preview needs
   an index or an ROI no larger than 128³ voxels.
4. Keep **100 objects per dataset** initially. A seeded selection is made once,
   without consulting class labels, and reused for all backbones and scales. Use
   zero for the complete selection. If annotations use cell IDs while the mask
   contains nucleus IDs, supply a one-to-one `label_id, segmentation_id` mapping;
   PlatyBrowser also offers the published cell-to-nucleus mapping.
5. Select **Preview actual DINO inputs**. Check raw detail, mask correspondence,
   physical canvas size and the fine tile width in native pixels for both raw and
   segmentation. The adjacent overlay shows foreground and spatial token cells.
   A 1 µm tile is an experimental starting value. Upsampling a coarse raw image or
   a coarse segmentation does not restore missing detail.

### Check which objects can be processed

On **Prepare data → your dataset → Object eligibility, boundaries and memory**,
the following controls are expanded and have individual help explanations:

| Control / YAML key under `source` | Meaning |
| --- | --- |
| Maximum object-box voxels / `max_object_voxels` | Default **16,777,216 (256³)**. Limits the complete segmentation bounding box plus a one-voxel halo, including background. This is **not** the fine tile size, foreground voxel count or GPU batch size. |
| Boundary objects / `boundary_policy` | `skip` (default) excludes objects touching the volume or ROI edge. `error` stops if any selected object touches an edge. `allow_partial` explicitly uses the available portion and marks both fine and coarse coverage as partial. |
| Objects above the memory limit / `oversized_policy` | `skip` (default) excludes them and reports their required sizes. `error` stops the run. Objects are never clipped to fit the budget. |
| Other object-reading or encoding errors / `on_object_error` | `skip` (default) records object failures such as stale bounds, unusable contrast or no valid views. `error` stops at the first such failure. Configuration errors always need correcting. |

Choose **Check object eligibility (no DINO)** before submitting. With an existing
bounds table, this compares object boxes against the chosen volume, ROI and
memory limit without reading image chunks or loading a model. Without a bounds
table, the interactive check indexes a small ROI (at most 128³ segmentation
voxels); larger scans run in the worker. The table lists every selected ID,
volume/ROI edge contact, required voxel count and expected decision. Download it
to inspect exclusions. Changing settings invalidates the displayed check.

For a failed run, reuse its `objects.tsv` under **Object selection and source
bounds → Existing object bounds table**, provided it describes the exact same
segmentation grid. This avoids rescanning a large volume. Selection and the pilot
cap still apply. Skipped objects are not replaced with other objects.

The worker runs the same bounds check before loading DINO. If none can proceed,
or a selected stop policy is triggered, the error gives counts for all reasons,
the required size range and the relevant UI settings. `dataset-N/input/eligibility.tsv`
and `eligibility.json` preserve the check; a blocked run also writes `excluded.tsv`.
Passing this check does not guarantee successful encoding: actual mask contents,
contrast and fine-tile cutoffs are checked during extraction. Its view directory
also contains `eligibility.tsv`, `coverage.tsv`, `excluded.tsv` and `views.jsonl`.

For example, a **500 × 700 × 600** voxel object box needs **502 × 702 × 602 =
212,147,208** voxels including its halo. Use the precise required count in the check
table, then choose a limit that your worker can support. The binary mask is only
part of memory use: source labels, foreground coordinates and sampling buffers
also occupy RAM. Raising the voxel limit does **not** raise the SLURM memory
request. Choosing a coarser segmentation level reduces box size but also loses
mask detail; choosing a coarser raw level alone does not reduce this limit.

Keep **Skip objects touching a volume or ROI edge** for complete-nucleus
comparisons. Expanding an ROI can recover objects cut by the ROI, but cannot
recover tissue missing beyond the original volume. Use **Use the available part**
only when partial objects are acceptable for the experiment. Previews warn about
partial coverage and per-object/per-view records carry `coverage_scope: partial_object`;
coarse views then describe the visible portion, not a complete nucleus. An index
that cuts through foreground away from a real ROI/volume boundary remains an
error: refresh that index instead of using the partial-object option.

### What each representation measures

| Representation | Images given to DINO | Feature used |
| --- | --- | --- |
| Fine CLS | Fixed physical tiles distributed throughout the nucleus | Global CLS vector |
| Fine foreground | The same fine images | Foreground-weighted spatial tokens |
| Coarse CLS | Whole-label cross-sections resized with physical aspect preserved | Global CLS vector |
| Coarse foreground | The same coarse images | Foreground-weighted spatial tokens |
| Fine + coarse | Fine foreground plus coarse CLS | Balanced concatenation of both full vectors |

The coarse mode spans the complete indexed label. By default it takes five evenly
spaced positions across its foreground extent along each of three axes. Each
image is fitted into a square canvas without distorting physical aspect ratio.
The effective pixel size therefore varies with nucleus size and is recorded.

The fine mode places a non-overlapping physical tile lattice over the label.
Foreground volume fractions account for voxel/tile intersections, including when
a tile is smaller than a segmentation voxel. Tiles with at least 1% foreground
are eligible. Starting near the center, deterministic farthest-point selection
retains up to 64 spatially distributed tiles. Each has one foreground-centered
cross-section per selected axis. **Exhaustive** processes all eligible tiles;
it ignores **Maximum fine tiles per nucleus**, even when a cap is entered.
It still samples cross-sections, not every foreground voxel. The independent
candidate-lattice budget defaults to 250,000 tiles per object and fails explicitly
if exceeded. Bounded raw reads and model batches apply in either mode.

The expanded **Fine texture: patch size, nucleus content and placement** panel
is on **Prepare data**. There are two distinct cutoffs:

| Setting | Meaning | Example |
| --- | --- | --- |
| Minimum nucleus content per **3D tile** (`min_foreground_fraction`) | The selected nucleus must occupy this fraction of the entire tile volume. Other nuclei do not count. | `0.25` keeps tiles at least one quarter full; `0.50` requires half full. Default: `0.01`. |
| Minimum nucleus content per **2D image** (`min_slice_foreground_fraction`) | Each fine cross-section sent to DINO must also pass this image-area cutoff. Padding counts as background. | `0.50` requires half the image to contain the nucleus. Default: `0.0`, retaining every nonempty slice. Coarse images are unaffected. |

High cutoffs emphasize the interior and can exclude nuclear edges or small
nuclei. A 3D tile can pass its cutoff while a particular cross-section fails the
2D cutoff. Preview before a large run: eligible/selected tile counts and exclusion
reasons are shown beside the actual DINO inputs. Worker `coverage.tsv`,
`excluded.tsv` and `views.jsonl` retain those decisions. If no objects survive,
the error identifies the dataset, backbone, scale and exclusion report.

**Position of the fine tile grid** controls where the lattice begins:

- `bbox_min` starts at the lowest bounding-box corner, preserving the original behavior.
- `bbox_center` places one tile at the bounding-box midpoint and extends the grid outward.
- `centroid` places one tile at the mean position of the nucleus foreground voxels, then extends outward. For an irregular nucleus this may differ from the box midpoint; for a hollow/concave object it can even lie outside the foreground.

In the two centered modes, capped selection begins with the eligible tile nearest
the chosen center, then spreads across the object. This controls grid placement;
`fine_fractions` still controls the slice positions within each tile's foreground.
All modes cover the candidate lattice before filtering: choosing a center does
not discard one side of the nucleus. The view algorithm version is recorded in
cache identities, so the updated sampler cannot silently reuse older results.

Each object has one deterministic foreground intensity sample, capped at 65,536
values, for shared 1st–99th percentile normalization of its fine/coarse views.
The default masks surrounding tissue and pads outside the image. Turn off masking
for a separately identified context experiment. For foreground-weighted pooling,
the transformed mask's mean occupancy in each spatial token is its weight; CLS
and register tokens do not participate. CLS and foreground features come from the
same model pass. Valid view vectors are averaged equally; maximum aggregation is
an advanced alternative.

### Combine vectors and compare fairly

With weights `w_fine + w_coarse = 1`, the default combined vector is:

```text
concatenate(sqrt(w_fine) * L2(fine), sqrt(w_coarse) * L2(coarse))
```

The default weights are 0.5/0.5. Both full feature blocks are retained, with their
column boundaries recorded. The experiment uses whole-vector L2 normalization for
analysis; this preserves the relative block weights. Avoid per-feature
standardization when assessing those weights. PCA and UMAP operate on these
combined **feature vectors**, never on concatenated UMAP coordinates.

This follows the principle of joining complementary features in the
[original MorphoFeatures method](https://elifesciences.org/articles/80918). These
DINO encoders, dimensions and pooling choices are different from the original
trained shape/texture networks.

In **Prepare data**, add physical widths to compare. In **Analyze → Fine/coarse
feature combination and model execution**, optionally add fine/coarse pairings.
The corresponding stage YAML is, for example:

```yaml
tile_sizes_um: [0.5, 1.0, 2.0]
extra_pairings:
  - name: Both foreground
    fine: foreground_mean
    coarse: foreground_mean
    weights: [0.5, 0.5]
```

Coarse outputs are reused across fine scales. An official backbone is loaded once
per experiment backbone, and remote chunks and complete extractions can be cached.
Keep a checkpoint fixed when comparing views. Add DINOv2 and DINOv3 as separate
backbones to compare their results on the same objects.

**DINO execution, caching and recovery** on this page exposes preparation workers,
RAM/queue/cache budgets, scratch storage, inference precision and per-object
recovery. Existing YAMLs default to serial preparation and float32; the reusable
example starts with two preparation workers. Completed physical-view objects are
saved independently, so an interrupted extraction can resume from its shared
embedding cache. `progress.json` and `performance.json` report throughput, timing,
memory and cache activity. These settings leave the selected views unchanged;
mixed precision is an explicit numerical experiment. See the
[runtime and benchmark walkthrough](troubleshooting.md#using-the-new-controls)
before choosing settings for a multi-day run. Stored-crop/grouped-patch extraction
remains available with its existing settings.

### Let the job's RAM allocation set the DINO budgets

Keep **Analyze → DINO execution, caching and recovery → DINO memory planning**
on **Automatic from job RAM**. In **Review & run**, choose your SLURM memory
allocation, for example `100G`. The **DINO job memory plan** shows how this is
split between the model/main process, complete-object preparation, caches and
queued images. You do not need to divide up the RAM yourself. For a local run,
set **Job RAM budget (MiB)**; zero starts with 8192 MiB locally. For SLURM, zero
uses its allocation. A nonzero value is an additional ceiling, never permission
to exceed SLURM or container limits. 1024 MiB = 1 GiB.

The plan reserves at least 25% of the budget (and at least 2 GiB) for the main
process/model and other allocations. At the worker, current main-process RAM can
increase this reserve. Host image batches and the queue have separate allowances;
the remaining preparation budget includes **both raw and segmentation caches per
reader**, plus 512 MiB estimated overhead per extra reader process. Complete
object scratch is estimated at 128 bytes per bounding-box voxel, including its
halo. The scheduler admits only as many objects as fit that budget, even when
more CPU workers were requested. Serial preparation obeys the same object-size
check. GPU RAM is a separate limit.

For example, a `100G` job with two readers and 224-pixel images in batches of 16
initially receives about **74.7 GiB for preparation**, **25 GiB main/model reserve**,
**256 MiB queue**, and **49 MiB host image buffers**. The preparation allowance
already contains 512 MiB of decoded caches and 1 GiB of reader overhead. These
are planning estimates; worker limits and the actual indexed object boxes can
change them before extraction. Review does not scan your volume to find objects.

Choose **Manual preparation, queue and cache budgets** to use the existing RAM
controls directly. **Repair manual RAM conflicts and retry smaller GPU batches**
allows the worker to raise an undersized preparation allowance within the job
budget, reduce caches/queues, or use fewer workers. Disable it to reject conflicts
instead. All requested values remain saved; effective values and reasons appear
in `memory_plan.json`, `performance.json`, standard output and `metrics.jsonl`.
Automatic host planning always balances its sub-budgets. The retry checkbox also
controls recovery from GPU out-of-memory: it retries the **same images** in smaller
batches and retains that smaller batch size for later inference.

The worker checks its actual SLURM allocation, available host RAM and readable
container/cgroup limits, then checks the selected object's indexed bounds before
preparing pixels. If an entire object cannot fit even serially with caches off,
it reports the object ID, estimated need and available preparation budget. It
does not silently skip objects, crop them, change their resolution, reduce their
views or change precision. Likewise, one image that cannot fit on the GPU requires
a larger GPU or an explicit change of experiment settings. The existing
**Maximum object-box voxels** and exclusion policies remain independent controls;
raising the RAM allocation does not change your scientific selection.

These checks reduce avoidable configuration failures; they are estimates, not
an OS reservation or a guarantee against every allocation failure. They cover
physical DINO extraction; subsequent PCA/UMAP/classifier memory is separate.
A process already killed by the OS cannot repair itself, but completed object
checkpoints remain available for resumption. See the
[SLURM memory options](https://slurm.schedmd.com/sbatch.html) and
[Linux memory-controller reference](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html)
for the allocation limits used by the worker.

Each dataset can include named saved baselines. They participate in the same
comparison without another model run. Preserve their `.metadata.json` sidecars.
When an older artifact lacks original-volume identity, verify the IDs yourself and
set `assume_same_dataset: true` on that baseline. A known conflicting identity is
rejected. Baselines with matching names across datasets also receive joint
analyses, subject to the usual common-model provenance checks.

After extraction, the experiment intersects IDs across **all configured
backbones, scales, pairings and baselines within each dataset**. Missing methods
are recorded and never filled with zero vectors. Nonfinite and zero-norm fusion
components are excluded by object. Comparisons use these matched cohorts and
identical deterministic held-out folds. Supply a specimen/acquisition column to
hold out specimens; otherwise results are explicitly object-level validation.
Labels with insufficient examples cannot support a classification test. Disable
classification to inspect such labels without claiming held-out performance.

Each representation receives its own within-dataset analysis and, when multiple
datasets are present, a joint PCA/UMAP fit. Coordinates from different
representations are separate fitted spaces; compare their biological neighborhoods
and held-out metrics rather than their axis positions.

### Compare raw and normalized combinations in one run

Older saved features can be combined with newer views of the same specimen through
**Combine saved feature blocks**. Enter the individual NPZ paths, rather than their
folders. For a label-transfer reference export, `original_embeddings.npz` contains
the original input vectors; `embeddings.npz` may have already been normalized by
that transfer workflow. Choose the original file when comparing raw and balanced
fusion without inheriting that earlier scaling.

When old crop containers and new volume extractions have different source records,
the source identity check can block fusion even though both depict the same cells.
Open **Object identity across source files**, verify the specimen and actual ID
namespace, and record a shared dataset name and the evidence for correspondence.
The explicit confirmation allows these differing source records; it never translates
IDs or changes source metadata. Each output records the declaration, original source
identities, feature recipes and matching exclusions. Numeric overlap or a common
DINO checkpoint alone is insufficient evidence. The existing missing-metadata
confirmation remains available and does not override conflicting source identities.

For example, two separately prepared views of one verified specimen can use:

```yaml
identity_override:
  confirmed: true
  dataset: specimen-A
  id_namespace: cell
  reason: Both exports use the same cell table and verified cell-to-nucleus mapping.
```

Use the same dataset name and namespace for subsequent combinations of that specimen.
Different fine/coarse view recipes are expected and remain separately recorded.

In **Analyze → Fine/coarse feature combination and model execution**, there are
two separate decisions:

1. **Feature combination methods** chooses how vectors are built. **Raw
   concatenation** joins the original feature values, ignoring weights. **Balanced
   block L2** divides each component vector by its length, multiplies it by the
   square root of its weight, then joins all dimensions. Select both to test both.
2. **Analysis scaling methods to compare** chooses how the saved vectors enter
   PCA, UMAP, clustering and within-dataset classification. **No analysis scaling**
   leaves them unchanged; **L2 of the complete vector** gives each concatenated
   vector unit length; **Standardize each feature** centers/scales each column.
   Choose several to evaluate them in the same run. Classifier scaling is fitted
   only on training folds, and all conditions share object IDs and fold assignments.

For example, raw fusion plus no analysis scaling is `[fine, coarse]`; raw fusion
plus complete-vector L2 is `[fine, coarse] / ||[fine, coarse]||`. Neither is the
same as normalizing the two blocks independently. Balanced fusion already has
unit total length (weights sum to one), so an additional whole-vector L2 normally
has no further effect apart from numerical roundoff. It is retained as an
explicit selectable condition. Additional classifier standardization remains a
separate control; disable it when testing a classifier with no scaling at all.

**Additional feature combinations** adds other fine/coarse pairings, both pooling
methods at one scale, or **All four features**. The original four individual
representations and primary fine-foreground + coarse-CLS pair remain available.
For all four, the fine/coarse weights are divided equally between the two pooling
methods at each scale: `[0.5, 0.5]` gives four weights of `0.25`. Within-scale pairs
use equal weights. Raw fusion ignores these weights. Advanced named YAML pairings
remain supported; an explicit `combination` there overrides the selected methods.

```yaml
# Inside a dino_experiment stage:
fusion_weights: [0.5, 0.5]
combination_methods: [balanced, raw]
additional_combinations: [all_four]
analysis_normalizations: [none, l2, standardize]
```

This example generates eight representations (four individual and four fused)
and analyzes each with three scaling methods: 24 conditions per dataset/backbone/
fine scale. Extraction runs once for those settings; the extra work is vector
combination, projection/clustering and classifier fitting. **Review & run** lists
the fusion recipes, weights and scaling choices. Multiple datasets also receive
joint maps under each selected scaling. Label transfer retains its separate
reference preprocessing, including its initial L2; these scaling selections do
not create additional transfer-classifier conditions.

The saved `experiment.json` records `fusion_recipes` and `analysis_normalizations`.
Comparison entries record the effective settings for each condition, and metrics
remain separated by representation/scaling. Plot captions, classifier summaries
and feature-space neighbor inspection use that condition's actual scaling.
NPZ downloads contain the saved input/fused vectors; plotting or classifier
preprocessing does not overwrite them. `matched-R` indexes vector representations,
while `comparison/representation-R` indexes representation/scaling conditions;
their numbers need not correspond when multiple scalings are selected.

**Already have the vectors?** Open the DINO `experiment.json` in Results, expand
**Compare combinations from these saved vectors**, select a dataset/backbone/scale
and two or more blocks, then **Create saved-vector comparison workflow**. This
creates an editable draft with balanced and raw fusion plus both unchanged and
whole-vector-L2 analysis; it submits nothing until Review & run. Original matched
IDs, source metadata and available annotation settings are retained. No image
volume or DINO inference is needed. For other saved feature files, use **Combine
saved feature blocks → Add raw/balanced comparison** to construct the same kind
of pipeline while keeping your existing weights and classifier settings.

Ordinary Analyze/Compare stages also provide **Compare multiple analysis scaling
methods**. Their `analysis_normalizations` list controls the full analysis,
including newly evaluated classifiers. The separate **Additional analysis
combinations** and Results' clustering/UMAP exploration remain **plot/clustering
only** and reuse classifier predictions. These options answer different questions.

Existing YAML without the new fields keeps its original fusion recipe and
analysis scaling (whole-vector L2 for the DINO experiment); existing saved results
are not modified. The reusable fine/coarse example now demonstrates both modes.

### Reading a fine/coarse experiment's files

The stage folder is usually `workspace/00-dino_experiment/` (the number is its
position in the pipeline). Its main relationships are:

```text
experiment.json                         open this in Results
metrics.tsv, metrics_summary.tsv         held-out scores, when classification runs
export.zip                              packaged experiment outputs
dataset-D/
  input/{source.json,objects.tsv,eligibility.json,eligibility.tsv}
  matched_ids.tsv, excluded_from_comparison.tsv
MODEL-B/scale-S/dataset-D/
  extract/
    result.json                         extraction-report pointer and cache-hit flag
    views/
      fixed_tiles.{cls,foreground_mean}.npz + .metadata.json
      whole_label.{cls,foreground_mean}.npz + .metadata.json
      extraction.json, objects.tsv, cohort.json
      eligibility.json, eligibility.tsv, excluded.tsv, coverage.tsv
      views.jsonl, input_preview.{png,svg}, preview-*.npz
      progress.json, performance.json, resume.json
  fusion-F/
    embeddings.npz, embeddings.metadata.json, fusion.json
    component-*.npz + .metadata.json, excluded.tsv
  matched-R/
    embeddings.npz, embeddings.metadata.json
  comparison/
    comparison.json, report.md, coordinates.tsv, splits.tsv, export.zip
    representation-R/
      embeddings.npz, coordinates.tsv, object_labels.tsv
      projection.svg, projection.png
      predictions.tsv, splits.tsv       when classification runs
```

`D`, `B`, `S`, `F` and `R` are **zero-based list indices**, respectively for dataset,
backbone, fine scale, fusion recipe and representation. They are not cell IDs,
replicates or resolution levels. `dinov2-0` means the first configured backbone,
from the DINOv2 family; use `experiment.json` for its actual variant/checkpoint.
Coarse vectors can be reused across multiple fine scales. Optional baselines and
additional fusion recipes change representation numbering; use the manifest's
names and paths rather than assuming an index always means a particular method.

- **Selected objects and eligibility:** `dataset-D/input/objects.tsv` is the
  selected object index; `source.json` records resolved volumes, geometry and ID
  mappings. Eligibility files record boundary and bounding-box budget decisions
  before image preparation. Extraction retains copies for provenance. `cohort.json`
  identifies the selected cohort/cache; it is not a second feature array.
- **Original extraction:** fine and coarse NPZs contain equally aggregated valid
  view features per object, with either CLS or foreground-weighted spatial-token
  pooling. Fine/coarse modes can retain different objects. Sidecars record the
  source, checkpoint, physical views, pooling and coverage paths. `extraction.json`
  indexes these artifacts and records effective settings, exclusions and runtime.
- **What DINO saw:** `views.jsonl` has one record per used view, including object,
  mode, axis, physical bounds/location, pixel size, foreground fraction and
  normalization. Read it line by line: millions of views can make it several GB.
  `coverage.tsv` summarizes tiles/views and represented extent; complete object
  bounds do not mean all voxels were encoded. `excluded.tsv` gives preparation
  failures. `input_preview.png/.svg` shows a small sample; `preview-*.npz` contains
  the corresponding prepared image, mask and metadata, not the complete source.
- **Runtime records:** `progress.json` counts processed objects/images, including
  objects eventually excluded; it is not the retained embedding count.
  `performance.json` records timings, memory and cache use. `resume.json` is
  restart bookkeeping; actual per-object recovery artifacts can also live in the
  configured cache. Newer runs additionally save memory-planning reports. These
  files describe execution, not biological performance.
- **Fusion:** `fusion-F/component-*.npz` preserves the original input components,
  before block normalization/weighting. With the default recipe, the output is
  `[sqrt(0.5) × L2(fine foreground), sqrt(0.5) × L2(coarse CLS)]`.
  `fusion.json` and the embedding sidecar record each block's start/stop columns,
  source and weight. `excluded.tsv` reports invalid or missing components among
  the supplied component IDs; objects absent from every component are recorded
  in extraction/cohort exclusions instead.
- **Fair comparison:** `matched-R/` holds the common IDs across all requested
  methods/backbones/scales for that dataset, without re-encoding or changing the
  feature values. **Use these NPZs and sidecars for comparing or reusing methods.**
  `excluded_from_comparison.tsv` explains missing methods; one object may appear
  on several rows. `comparison/representation-R/embeddings.npz` is another copy of
  those matched vectors for a self-contained analysis, not a new model output.
- **Maps and labels:** each representation's `coordinates.tsv` holds its PCA/UMAP
  positions and clusters; `projection.svg/.png` is the saved graphical view.
  `object_labels.tsv` records known/predicted labels, confidence, prediction source
  and cluster. Evaluated objects use held-out predictions; other objects may use
  the classifier trained on all eligible annotated objects. These inferred labels
  are not new ground truth. The parent `comparison/coordinates.tsv` stacks the
  separate representation maps; it is **not one joint coordinate space**.
- **Classifier evidence:** `predictions.tsv` contains held-out predictions by
  classifier/fold; `splits.tsv` records train/test memberships. Confusion matrices,
  per-class results, projection settings, diagnostics and provenance are saved
  in `comparison.json`; `report.md` is a shorter human-readable report. Top-level
  `metrics.tsv` collects per-fold metrics, while `metrics_summary.tsv` reports
  their means and sample standard deviations. Historical `comparison.json`
  summaries use population standard deviations, so these deviations can differ
  slightly for the same folds. `linear_*` metrics refer to logistic regression.
  Retrieval precision/recall measures same-class neighbors among training objects;
  `test_kmeans_ari` compares held-out cluster assignments with known classes.
- **Archives and optional branches:** `comparison/export.zip` packages one
  comparison; the stage's `export.zip` packages the experiment, omitting nested
  ZIP files. Joint maps need multiple datasets and label transfer must be enabled;
  their exact paths are indexed in `experiment.json`. Optional TSV vector exports
  and interactive analysis variants add files alongside these outputs.

**Worked example: `run-b4607830`.** This completed run uses Platynereis nuclei
images, DINOv2 ViT-B/14 with four register tokens, one 1 µm fine scale, up to 32 fine
tiles per object and five slice positions on each of three axes. Both tile and
slice foreground cutoffs are 75%. Object IDs in this run are **cell IDs mapped to
nucleus segmentation IDs**, according to its saved PlatyBrowser settings. Do not
apply these IDs directly to a nucleus volume without that mapping.

Of 11,382 selected objects, IDs **22140** and **26223** exceeded the configured
500,000,000-voxel object-box budget (requiring 622,988,320 and 1,021,695,444 voxels).
ID **10141** had no fine tiles meeting the 75% cutoff. Consequently, original
coarse vectors have 11,380 rows, original fine vectors have 11,379, and all five
matched comparison representations have **11,379 rows**:

| Index in `matched-R` / `representation-R` | Representation | Feature dimensions |
| --- | --- | ---: |
| 0 | Fine CLS | 768 |
| 1 | Fine foreground | 768 |
| 2 | Coarse CLS | 768 |
| 3 | Coarse foreground | 768 |
| 4 | Fine + coarse (fine foreground + coarse CLS) | 1536 |

Fusion block 0 occupies columns `[0, 768)` and block 1 `[768, 1536)`, each with
weight 0.5. The plots use whole-vector L2 normalization, 10 k-means clusters and
direct UMAP (15 neighbors, minimum distance 0, 50 epochs); neither clustering nor
UMAP uses PCA input in this run. Separate PCA coordinates are also saved.
Classification uses 390 annotated objects from seven classes in five identical
object-level folds, with logistic regression and kNN; these scores do not measure
between-specimen or cross-species generalization. No joint map or transfer was
requested because there is only one dataset and transfer is disabled.

The extraction processed 5,184,521 images. Its `views.jsonl` is about 2.83 GiB,
and the complete experiment ZIP is 711.3 MiB (the comparison ZIP is 200.6 MiB).
Both ZIPs exceed the 200 MiB browser-download safeguard, while each matched NPZ
is about 31 MiB, or 62 MiB for the combined vectors. Those individual files and
their metadata remain downloadable from Results. This older run has no memory
planning report; the completed outputs do not need to be regenerated to open them.

### Transfer labels from Platynereis to a target dataset

Before training, the workflow compares the saved representation recipes. A
blocking error lists each differing parameter and both dataset values, and points
to **Analyze → Transfer labels → Reference / target embedding files**. Different
weights, feature pooling (for example CLS versus foreground mean), fine/coarse
sampling recipes, fusion weights or fitted adapter spaces cannot be made
compatible by changing the classifier or UMAP settings. Select matching saved
representations or regenerate the differing input. The confirmation for missing
provenance never overrides a known mismatch.

For frozen DINO, grouped patches versus single object crops are **input preparation
differences**, not different neural models. Transfer and shared projection can
run with matching DINO recipes, but display and save a comparison of patch/crop
sizes, voxel spacing, masks and sampling settings when recorded. A warning means
the calculation is possible, not that the biological comparison is controlled.
For example, averaging many small nuclear patches and encoding one clipped crop
can describe different texture and spatial coverage even with identical DINO
weights. Use matched physical views when testing whether representation choices
improve cross-dataset biology. MAE input-layout compatibility remains strict.

Saved analysis copies retain their extraction metadata. Older re-analysis copies
can recover it through their recorded input only while that file still matches
its saved size and modification time; missing or changed sources remain unverified.

In Analyze, enable **Train a reference classifier and predict other datasets**.
Choose the labeled reference, logistic regression, MLP or k-nearest neighbors, and
uncorrected, classical CORAL and/or neural adapter conditions. The target does not need annotations.

**Reference classifier settings** exposes MLP hidden-layer widths (default `[64]`,
as in the Classification Tool), iteration limits, KNN neighbor count, logistic
regularization, classifier PCA dimensions and validation folds. All three use
the shared Classification Tool estimators. Transfer first L2-normalizes each
object vector. Additional **Classifier feature scaling** defaults to `none` to
preserve fine/coarse weighting; `standardize` learns per-feature scaling from the
reference training fold and applies it to held-out and target objects. This
changes the relative block weighting, so keep it consistent when comparing runs.
**Classification input** now chooses full embedding vectors, PCA components or
UMAP coordinates. Set **Classifier dimensions** to 2 to test a two-dimensional
representation, or use more dimensions to retain additional information. PCA
dimensions cannot exceed the feature count or number of training objects in a
fold. **Standardize selected classifier dimensions** centers and scales the
selected coordinates *after* projection, independently of feature scaling before
projection. These controls are also available under Analyze's classification
check, within DINO experiments, and in the Classification Tool.

Logistic regression is displayed as **Linear classifier (logistic regression)**.
Choose it alone to run only a linear classifier. Its boundary is linear in the
selected input space; UMAP itself is a nonlinear transformation of the original
embedding vectors. A linear SVM is a different classifier and is not implied by
this option.

For example, this setting block fits linear logistic regression on standardized
PCA components in a Transfer labels stage (keep the reference and target inputs
from your existing configuration):

```yaml
classifier: logistic
classifier_normalization: standardize # Scale features before projection, after transfer's L2 step.
classifier_space: pca                 # embeddings, pca or umap
classifier_dimensions: 50            # Use 2 for a two-dimensional classification experiment.
classifier_standardize: true         # Scale the selected PCA/UMAP dimensions after projection.
```

To use UMAP, change `classifier_space` to `umap` and choose its dimensions; optional
settings are `classifier_umap_neighbors: 15`, `classifier_umap_min_dist: 0.1` and
`classifier_umap_epochs: 200`. UMAP classification needs at least four training
objects in every fold and is slower than full-vector or PCA classification.
For ordinary Analyze/Compare stages, use `normalization` for pre-projection
scaling and `classifier_models: [logistic]` for a linear-only evaluation.

All classifier projections and scalers are fitted on each training fold only.
UMAP is **unsupervised** here: known cell-type labels do not shape its projection.
Held-out objects are transformed into that training map. In transfer, a final
reference-only map transforms each target, after any CORAL correction. The joint
plot fitted to all reference/target objects is a separate exploratory view;
classifying its already-fitted coordinates would expose validation objects to
the projection. This follows the [UMAP transform workflow](https://umap-learn.readthedocs.io/en/latest/transform.html).
The distance-based unknown check continues to use the full L2/CORAL feature space,
independently of the dimensions selected for the classifier.

Existing YAML with `evaluation_pca` keeps its previous behavior when
`classifier_space` is absent. An explicit `classifier_space` takes precedence;
post-projection standardization is off unless requested, preserving existing
results by default.

All transfer classifiers learn cell-type assignments only from reference labels.
Reference cross-validation keeps learned preprocessing inside its training folds.
The original `coral` condition maps **unlabeled target features** into reference
coordinates, using Ledoit–Wolf shrinkage covariance estimates and bounded
eigenvalues. It keeps the reference classifier fixed. The additional
`coral_source_to_target` condition instead maps reference vectors toward the target
and refits the classifier. This follows the direction used in the
[original CORAL paper](https://arxiv.org/html/1511.05547v2), while retaining our
shrinkage regularization and explicit mean matching in both directions.

#### Compare all five transfer conditions

You need full embedding vectors from matching representation recipes for the
reference and target, plus reference cell-type annotations. Images and DINO
checkpoints are not needed again when starting from saved vectors. Optional target
annotations should be reserved for evaluation. If reusing `reference/reference.json`,
keep its `parameters.npz`, `annotations.tsv`, `splits.tsv` and
`original_embeddings.npz` alongside it; the saved splits define the common folds.
Retain `oof_predictions.tsv` as well so an edited rejection quantile can be
recomputed from the saved reference distances without retraining the baseline.

In **Analyze → Alignment conditions**, select the five entries below. The same
controls are available inside the DINO fine/coarse experiment. Each representation,
checkpoint, physical scale and target gets its own comparison and fitted spaces.

| YAML condition | What it does | Uses target features to fit? |
| --- | --- | --- |
| `uncorrected` | Apply the reference classifier to L2-normalized target vectors. | No |
| `coral` | Align target vectors to reference; keep the reference classifier fixed. | Yes |
| `coral_source_to_target` | Align reference vectors to target; fit a new classical classifier. | Yes |
| `adapter` | Learn a small neural adapter and linear classification head from labeled reference objects. | No |
| `deep_coral` | Train the same adapter/head with an additional covariance-alignment loss. | Yes |

The neural adapter uses the **saved full vectors**, independently of the classical
classifier's PCA/UMAP controls. It applies optional feature standardization fitted
on labeled reference training objects, followed by a hidden linear layer, ReLU,
a linear latent layer, and a linear softmax classification head. Its latent vectors
are the exported adapted representations. The DINO backbone remains frozen.
This is Deep CORAL on a small embedding adapter, not end-to-end DINO fine-tuning.

Training minimizes class-balanced source cross entropy plus
`coral_weight × ||C_reference − C_target||²_F / (4 × latent_dim²)`, using unbiased
mini-batch covariance estimates of latent activations. The neural control sets
the CORAL weight to zero. Both conditions share initialization, source batches,
optimizer and epoch count. Alignment uses separate random streams, so sampling
target objects cannot change the supervised batch schedule. The loss follows
[Deep CORAL](https://arxiv.org/abs/1607.01719); our adapter architecture and
class balancing are choices for this workflow.

Open **Neural adapter training** to adjust layer widths, epochs, batch size,
learning rate, weight decay, CORAL weight, input scaling and CPU/CUDA execution.
CPU is the default. Only bounded batches move to a GPU; complete vector tables
remain in host memory. The same configuration/device and seed support repeatable
runs, but identical floating-point results across hardware are not promised.
Classical CORAL needs at least three objects in each fitted domain, including each
reference training fold. Neural covariance alignment needs at least two per domain.
The reusable [fine/coarse experiment YAML](../configs/workspace_dino_fine_coarse.example.yaml)
includes all five conditions. A transfer stage using existing vectors can contain:

```yaml
action: transfer
reference:
  name: Platynereis
  embedding: /path/to/platy/combined.npz
  annotations: /path/to/platy/labels.tsv
  label_column: cell_type
  group_column: specimen # Omit if independent specimen IDs are unavailable.
targets:
  - name: Parhyale
    embedding: /path/to/parhyale/combined.npz
    annotations: null # Add reserved target labels only for evaluation.
conditions: [uncorrected, coral, coral_source_to_target, adapter, deep_coral]
classifier: logistic # Classical conditions only; neural conditions have their own head.
folds: 5
seed: 42
adapter:
  hidden_dim: 128
  latent_dim: 64
  epochs: 100
  batch_size: 128
  learning_rate: 0.001
  weight_decay: 0.0001
  coral_weight: 1.0
  standardize: true
  device: cpu
min_confidence: 0.6
distance_quantile: 0.99
```

All conditions use the same exact native object IDs and exported source folds.
Within an adapted source-validation fold, held-out source objects contribute
neither to the classifier nor to covariance/adapter fitting; the alignment source
pool is the labeled training fold. Final CORAL/adapted models can use the complete
reference feature cohort, including unlabeled reference objects. Classical CORAL
and Deep CORAL use the current target cohort without its labels. This is
**adaptation to that target dataset**, not validation on an entirely unseen dataset.
Neither target labels nor target accuracy control stopping; training uses the
configured fixed epoch count. When specimen groups exist, the shared splits keep
specimens separate. Without them, source validation remains object-level.

**Review & run** lists the targets and selected conditions. New classifiers and
adapters require one fit per source validation fold plus a final fit. Start with
the pilot before applying this across every representation and scale. Choosing a
small epoch count verifies the workflow but does not establish convergence.

**Results** separates the original reference score, the selected condition's source
validation, and target evaluation. Neural conditions also show their training-loss
curves. The CORAL curve is unweighted; compare it with its configured weight and
the total loss. A low covariance loss does not establish correct cell-type transfer.
The same targets are scored before and after the unknown-rejection rules:
`candidate_accuracy` and `candidate_balanced_accuracy` describe candidate labels
before rejection; the usual accuracy fields include rejections. Compare these with
assignment coverage and per-class confusion matrices, not just UMAP mixing.

Each target/condition folder saves target `embeddings.npz`, matched-space
`reference_embeddings.npz`, predictions, `condition.json`, source confusion matrices,
and its joint plots. Optional TSV vector exports apply to both datasets. Neural
conditions additionally save `adapter.json`, `adapter_parameters.npz`,
`training_history.tsv`, exact `splits.tsv`, and held-out source predictions.
Reference-to-target CORAL saves its refitted classifier under `model/` and a
`coral.npz` matrix mapping reference to target. In the original `coral` folder,
that matrix instead maps target to reference. Directions and native cohorts are
recorded in metadata. Both CORAL exports include `input_mean` and `output_mean`;
the moving dataset is transformed as `(vectors - input_mean) @ matrix + output_mean`
after L2 normalization. Learned representation IDs prevent accidentally combining
vectors from independently fitted adaptation spaces. Original vectors are retained.

Download `condition_comparison.tsv` for a table of source/target scores and coverage.
Keep the complete transfer output directory for model reuse; models contain numeric
arrays rather than executable pickle files. For a reproducibility check,
`predict_transfer_condition('/path/to/condition.json', raw_embedding_matrix)` from
`morphofeatures.analysis.transfer` returns transformed target vectors and class
scores without fitting. It reuses that condition's fitted transform, even if given
only some target rows. Run a new comparison to adapt to a different target cohort.
Use the original `reference/reference.json`, not a target-specific `model/reference.json`,
as the reusable reference in another workflow.

Matching whole-dataset distributions can also erase biological differences when
cell-type proportions or species-specific types differ. Evaluate on reserved
target labels where available. To select hyperparameters using target labels,
create a separate development set first and retain a final untouched evaluation
set; the current workflow does not automatically create that development split.

An object is provisionally marked **unknown** if its maximum class score is below
0.6 **or** its nearest-reference distance within the proposed class exceeds that
class's 99th-percentile reference out-of-fold distance. Both controls are editable.
For neural conditions, distance thresholds are re-estimated from that adapter's
source out-of-fold distances in L2-normalized latent space; classical conditions
use their full input/aligned feature space. These are heuristics, not calibrated
guarantees. The candidate label, class
scores, distance and rejection reason are retained even when a label is rejected.

Optional target annotations are joined only after adaptation and prediction, for
balanced accuracy, macro-F1, per-class results, confusion matrices and assignment
coverage. An explicit target-to-reference label-name mapping can define known
correspondences. Target classes absent from the reference remain named in the
export and are evaluated as potentially novel/unknown classes. Without target
labels, there is **no reported cross-species accuracy**. Improved mixing on a UMAP
alone does not establish biological correspondence; CORAL can also suppress real
biological differences.

In **Results**, reference cross-validation and each annotated target have a
confusion-matrix heatmap: rows are **known** cell types and columns are
**predicted** types. Switch between object counts and percentages within each
known class; percentages make rare types easier to assess. Download SVG, PNG or
the counts TSV. Newly run transfers also save `confusion_counts.svg/.png`,
`confusion_normalized.svg/.png` and `confusion_matrix.tsv` under `reference/` and
each annotated target condition. Existing saved reports can display and download
these figures without rerunning classification. Targets without matched known
labels show assignment coverage, not a confusion matrix or accuracy. Rejected
predictions and target types absent from the reference are represented as
`unknown`; their original target labels remain in the prediction table.

For ordinary analyses and fine/coarse comparisons, the **Classification check**
section lets you select the representation and classifier, inspect the same
heatmaps and download them. These are held-out classifier predictions; agreement
between an unsupervised cluster number and a cell-type name is a different question.

When comparing scores, **accuracy** is the fraction of correctly classified
objects; **balanced accuracy** gives each cell type equal weight by averaging
its recall. A dataset dominated by neurons can therefore have high accuracy
while performing poorly on rare types. Macro F1 gives each type equal weight
while considering precision as well as recall. Compare the same metric, objects,
folds, classifier and preprocessing. The original reference score is calculated **before**
adaptation and joint PCA/UMAP, so adding a target cannot change that baseline.
Adapted conditions have their own source-validation scores, which may depend on
the unlabeled target cohort. Combining
fine/coarse feature vectors is different: it changes classifier inputs and can
improve or worsen held-out performance, depending on useful versus distracting
information, block weights and the number of labeled examples. A more attractive
UMAP does not establish a better classifier.

The numeric `reference/reference.json`, `parameters.npz`, and `annotations.tsv`
files form a reusable reference artifact. Keep them together. A new Transfer
labels stage can use `reference_model: /path/to/reference/reference.json` with new
targets, without the original reference embedding or annotation files. CORAL is
fitted anew for each target cohort, and its matrix, means, direction and cohort IDs
are saved. Original input vectors are retained as `original_embeddings.npz`. The uncorrected
condition uses whole-vector L2 normalization for the reference classifier; CORAL
applies its additional alignment to that input. Both conditions remain separate.
MLP layer weights and biases are saved as numeric arrays and restored for
prediction without retraining; saved reference settings override the new form's
classifier choice when reusing `reference.json`.
PCA, post-projection scalers and fitted UMAP inputs/coordinates are also retained
as numeric arrays. Reopening a reference does not refit its PCA or UMAP. UMAP
transforms use bounded batches with exact reference-neighbor distances and require
the same `umap-learn` version that saved the reference. The required version is
recorded in `reference.json`; these artifacts do not contain executable pickle
objects.

### Review, run and find the actual outputs

**Review & run** shows datasets, object caps, physical scales, backbones, view-count
bounds, fusion weights and the normal local/Slurm controls. Source checks link back
to Prepare data; checkpoint and experiment settings link to Analyze. The object
cap is an estimate of work, not a promise that every sampled object passes coverage
checks. With defaults, there are at most 207 views per object per backbone/scale
(64 × 3 fine plus 5 × 3 coarse), before empty-view exclusions and coarse reuse.

In **Results**, open the experiment summary, select a representation or transfer
condition, and inspect metrics and projections. Download NPZ for the actual feature
matrix and exact integer IDs; optional TSV downloads are available. Enable
**Write TSV feature tables in worker outputs** (`export_tsv: true`) for larger
tables that should be formatted by the worker. A transfer
result also offers prediction and corrected-vector downloads. Existing graph
controls provide independent and combined figure downloads and image/mesh
inspection. Browser downloads retain the 200 MB safeguard; larger artifacts stay
in the worker output directory.

| Output | Use |
| --- | --- |
| `experiment.json`, `metrics_summary.tsv`, `metrics.tsv` | Experiment matrix, score means/deviations and individual fold results |
| `dataset-*/input/objects.tsv`, `matched_ids.tsv`, `excluded_from_comparison.tsv` | Chosen native IDs, common evaluation cohorts and missing methods |
| `extract/.../views.jsonl`, `coverage.tsv`, `excluded.tsv` | Physical locations, fields of view, normalization, tile counts and exclusion reasons |
| `extract/.../input_preview.png/.svg`, `preview-*.npz` | Representative real input images, masks and token grids |
| `matched-*/embeddings.npz` and `.metadata.json` | ID-preserving vectors used for fair comparisons |
| `fusion-*/embeddings.npz`, `fusion.json`, `component-*.npz` | Full combined vectors, original blocks, boundaries, weights and provenance |
| `comparison/`, `joint/` | Metrics, held-out splits, coordinates, SVG/PNG figures and native-ID reports |
| `transfer/target-*/uncorrected/` and `coral/` | Predictions, scores, rejection fields, vectors, metrics and adaptation parameters |
| `export.zip` | Collected reports and scientific artifacts; original large source volumes are not bundled |

Physical source metadata, source version, object cohort, view algorithm version,
sampling, pooling and checkpoint identities contribute to cache keys. Increment
`data_version` after modifying source arrays in place. Keep sidecars with vectors
so comparisons can distinguish object identity from the representation recipe.

### Follow the history of a result

Open **Data provenance · workflows, settings and failures** in **Workflow →
Review & run**, **Runs**, or **Results**, then enable **Show provenance graph**.
The graph connects saved input files, processing steps and outputs, including
earlier runs that produced imported embeddings. It starts with workflow steps;
enable **Include individual files and detailed processing records** for the full
history. Solid arrows represent recorded
data dependencies; dashed arrows represent execution order only. Failed stages
are red, completed stages green, and unexecuted draft stages are marked planned.

Use **Inspect a workflow step or data file** below the graph to reveal its exact
settings, file path and saved error. Selecting a transfer stage also checks its
inputs with the current software; this does not rewrite the historical failure
or resubmit the job. Download the graph as DOT or the settings and connections as
JSON to share the processing history with a collaborator.

This view reads local workflow snapshots and small metadata records. It does not
open image volumes or download remote data. Missing metadata is not reconstructed
by guessing, and a bounded node/depth limit keeps large histories manageable.
Open an upstream result separately if the graph reports that it was truncated.

For saved vectors alone, choose **Combine embeddings** or **Transfer labels** in
Start. They are also available as additional stages. A minimal fusion followed by
analysis is:

```yaml
stages:
  - action: fuse
    name: Combined
    embeddings:
      fine: /path/to/fine.npz
      coarse: /path/to/coarse.npz
    component_order: [fine, coarse]
    weights: [0.5, 0.5]
    combination: balanced # raw concatenation is an advanced alternative.
  - action: analyze
    from_embedding: Combined
    normalization: none
    umap: true
    clusters: 8
```

`from_embeddings: [Fine, Coarse]` can replace the saved paths when earlier Extract
stages carry those names. Standalone physical extraction uses `action: extract`,
`views.mode: fixed_tiles` or `whole_label`, and `config.data.source: physical_volume`
with the same source settings under `config.data.volume`. Choose
`views.feature: cls`, `patch_mean`, or `foreground_mean`. Stored-crop inputs remain
explicitly limited to the supplied crops/patches; this setting cannot reconstruct
missing parts of their original labels.

For volume overlays and MoBIE, select the appropriate native-ID dataset report,
following [the export guide](analysis_and_mobie.md#dino-experiments-and-cross-dataset-predictions).

## Bilateral retrieval and spatial context

These workflows address two different questions: **Do similar cells recover their
expected left/right counterparts?** and **Does the neighborhood around a cell add
useful information?** Start with saved embeddings; no retraining or GPU is required.
The method follows the [MorphoFeatures publication](https://elifesciences.org/articles/80918).

### Explore the original results

Choose **Workflow → Explore published MorphoFeatures / MorphoContextFeatures**.
The preset opens the original 200-dimensional context vectors and preserves their
saved UMAP and 23 cluster assignments. The reference selector also offers the
480-dimensional MorphoFeatures vectors and 140 manually defined features. After
changing the selector, click **Use this published feature file**. Manual features
have no bundled saved map. Disable **Preserve published UMAP coordinates and
clusters** to fit a new projection and clusters using the ordinary analysis controls.

Saved maps and recomputed maps are labeled differently in Results. Classification
uses the underlying feature vectors, never the saved UMAP. Optional cell-ID labels
work through the existing classification probe, confusion matrices and inspection.
The original MorphoFeatures vectors include cell/nucleus shape and texture, whereas
MAE/DINO experiments may describe only nuclear texture. Differences are informative
representation comparisons, not controlled tests of the encoder architecture alone.

The saved MorphoFeatures map contains 10,391 cells, while its feature file contains
11,382. The 991 cells without saved coordinates remain in the exported vectors and
optional feature-based probes; their IDs are listed in the report diagnostics.
No coordinates or cluster labels are invented for them. The published context
feature file and its saved map both contain 10,391 cells.

### Compare bilateral partners

1. Choose **Compare bilateral-partner retrieval**. The three published baselines
   are preloaded. Add named MAE/DINO paths in **Named saved embedding files**; NPZ,
   label-first NPY, TSV and CSV are supported. Earlier named extraction/context
   outputs can be selected directly. Confirm all files describe the same specimen.
2. Check the **IDs stored in these embeddings**. The Platy candidate mapping uses
   **cell IDs**. Nuclear images can still carry cell IDs. If a file actually uses
   nucleus IDs, provide a one-to-one cell/nucleus table; per-representation overrides
   allow published cell-ID arrays and nucleus-ID arrays to be compared together.
3. Keep **Standardize → Euclidean → All** for the historical all-cell protocol.
   Use **Add bilateral settings variant** to evaluate cosine distance, opposite-side
   search, L2/no scaling or PCA dimensions in the same worker run. Null PCA keeps
   every feature dimension. Opposite-side search needs the supplied body-side table.
4. Review the settings and submit locally or to SLURM. Larger cohorts require more
   time: exact retrieval evaluates all query-to-population distances in bounded
   blocks. **Query cells per distance block** controls RAM; CPU threads must fit
   within the requested worker CPUs. No full sorted distance matrix is saved.
5. Results shows cumulative retrieval curves and a per-cell rank table. Select a
   cell and click **Inspect this cell and its best candidate** to compare images or
   meshes with the existing inspection controls. Supply the matching inspection
   YAML; cell IDs are translated back to that representation's native embedding IDs.

**Adding this probe to a fusion or representation comparison:** open **Analyze →
Bilateral-partner retrieval (optional)**. When comparing multiple representations,
select **These representations describe the same specimen and cells** after
checking their origin and ID namespace. Raw and balanced combinations of the same
input cells qualify: different feature values do not imply different cells. This
confirmation is separate from the fusion-stage identity setting and is required
even when fusion has already succeeded. Numeric IDs alone cannot establish identity
across animals. The candidate table must also belong to the specimen being tested.

If you only want embedding combinations, PCA/UMAP and classification, leave
**Evaluate bilateral-partner retrieval** off. Missing confirmation and invalid
probe settings are checked before submission and before analysis reads the vectors;
errors point to the specific field in the bilateral settings. Multiple **analysis
scaling** choices do not create additional bilateral conditions: the probe evaluates
the original representations using its own **Scale features before retrieval**
variants. Add bilateral variants explicitly to compare its scaling methods.

The closest listed candidate is ranked among all eligible **other** cells. Rank 1
means it is a nearest neighbor. Recall@10 is the fraction of evaluable query cells
whose best candidate is among the ten nearest others. Equal distances receive the
best possible rank, with tie counts reported. Both modes use one-based ranks; the
historical opposite-side script instead used zero-based indices. Opposite-side mode
also follows its exclusion of ambiguous candidate sets containing same-side cells.

These candidates are anatomical hypotheses, not verified one-to-one partners;
recall is not classification accuracy. Every representation is recomputed on the
**intersection of cell IDs**, using the same search population. Missing query
mappings/candidates and exclusions are reported. Smaller search populations make
retrieval easier, so compare methods within the same report rather than comparing
raw ranks across runs with different cohorts. The chance expectation accounts for
the number of available candidates and eligible searched cells. Use full cohorts
for scientific conclusions; small pilots are usability checks.

The probe is also available in the **Bilateral-partner retrieval (optional)**
expander of Analyze/Compare, or as an added stage. Independently derived candidates
are required for another animal; numeric ID overlap with Platy is not correspondence.

### Compare with any previous bilateral run

In the bilateral settings, enable **Compare with previous bilateral runs**. Choose
completed runs from this workspace and click **Add selected runs to this comparison**,
or enter named `bilateral.json` paths in **Named saved bilateral reports (YAML)**.
The saved runs can contain your own MAE/DINO/context features; they need not use a
published baseline. Keep their whole result directories, including `embeddings.npz`
and `id_mapping.tsv`. Historical rank tables alone are insufficient.

Add the new embeddings as usual and confirm that all inputs describe the **same
animal and cell identities**. The worker reads saved vectors, intersects the cells
across every new and saved representation, and computes all curves again using the
**current** candidate table, distance/normalization/PCA variants and search options.
No encoder runs again. A previous curve can therefore change when the common search
population shrinks. Its original cohort size and settings remain in provenance;
the previous files are not modified. Compare the curves in the new report, not an
old score measured on a different population. To change candidates for a different
animal, also supply embeddings and correspondences for that animal.

```yaml
- action: bilateral
  embeddings:
    New DINO: /path/to/new_embeddings.npz
  same_specimen: true
  id_namespace: cell
  candidates: /path/to/candidate_pairs.tsv
  compare_saved_runs: true
  comparison_reports:
    Earlier MAE: /path/to/earlier_run/workspace/01-bilateral/bilateral.json
  variants:
    - name: Same settings for every representation
      normalization: standardize
      metric: euclidean
      search: all
      pca: null
```

### Inspect retrieved pairs on the animal

Use **Export bilateral pairs in 3D** in Results, or **Add bilateral 3D export**
under the Analyze page's stage controls. Select representations, probe variants,
specific query **cell IDs**, or an automatic best/worst-rank subset. **Maximum pairs
per representation / variant** limits visualization size without changing probe
metrics. The partner is the query's closest **supplied candidate**, which is not
necessarily its nearest unrestricted feature neighbor.

Choose a streaming PlatyBrowser MoBIE project or a portable, bounded segmentation
ROI. MoBIE gets rank-colored objects, per-cell result tables, and one selectable
3D view per pair. TSV/NPZ/VTK links also support other viewers. Without MoBIE, supply
a physical position table such as Platy's cell `default.tsv` (`anchor_z`, `anchor_y`,
`anchor_x`, in µm). For nucleus-volume views use the cell-to-nucleus mapping;
numeric equality between the two ID namespaces does not establish correspondence.
Optional HDF5 masks hold each pair in its own layer, so shared cells are preserved.
See the [3D export walkthrough](analysis_and_mobie.md#export-bilateral-pairs-in-3d)
for file meanings, napari code and ROI limitations.

### Build new context vectors from MAE or DINO

Choose **Build spatial context from embeddings**. Supply the object's original
feature file and one of these physical neighborhood definitions:

| Method | Input | Interpretation |
| --- | --- | --- |
| `adjacency_table` | TSV/CSV with `source_id`, `target_id` | Undirected physical neighbor/contact pairs. Reuse an exported graph to compare different encoders. |
| `segmentation` | Integer instance volume and dataset key | Touching labels, read in bounded blocks with halos. Connectivity 1/2/3 includes faces / faces+edges / faces+edges+corners. |
| `radius` | `label_id`, physical `z,y,x` coordinates | Every centroid within a chosen distance in µm. Coordinate columns and µm/nm units are configurable. |
| `knn` | The same centroid table | Each centroid's k nearest physical neighbors, symmetrized by union. A cell can consequently have more than k neighbors. |

Use cell segment contacts for the paper-style biological neighborhood. Nuclei
usually do not touch; radius/KNN neighborhoods are useful alternatives but are not
the same construction. Do not use embedding-space neighbors or bilateral evaluation
candidates as physical contacts. Centroid tables must contain physical coordinates,
not voxel indices; convert using measured spacing before import. Radius distances
are independent of coordinate origin when all rows share the same physical frame.

For segmentation scans, select the exact resolution, dimension order, channel/time
point and ROI. Local NPY/H5/N5/Zarr and remote N5/OME-Zarr readers work.
**Maximum segmentation
voxels to scan** limits total work; **block shape** limits per-read memory. Raise
budgets deliberately for full volumes. A completed contact graph can be cached and
reused across encoders. Set **Source revision** and change it when remote/container
chunks are edited in place; a URL or directory timestamp cannot detect every edit.
Cached graphs reuse scientific topology even if a later scan budget is smaller;
they still respect the configured edge limit. No partial graph is cached.

**Stream Platy data directly:** set the neighbor method to `segmentation`, then
choose **Spatial context segmentation source**:

- **Choose a published PlatyBrowser segmentation:** select project, pinned revision
  and dataset, click **List published context segmentations**, choose cells and a
  resolution level, then **Connect context segmentation**. The catalogue may use
  N5 rather than OME-Zarr; the connection reports the actual format. Only published
  segmentations with an image source appear in the list.
- **Stream an OME-Zarr segmentation (including Platy):** paste the root URL of the
  instance-label layer, select image/channel/time, click **Discover context
  segmentation resolutions**, choose a level and connect. Physical voxel sizes,
  origins and axis order are read from NGFF metadata. This also accepts a Platy
  OME-Zarr mirror when it is not listed in the published catalogue.
- **Enter a local container or remote array:** retains the manual controls. For a
  manually entered remote OME-Zarr array set `remote_options: {format: ome_zarr}`.

Connecting does not scan the volume. It displays dimensions and physical spacing;
**Preview context segmentation** reads a bounded XY window. Set **Graph scan ROI**
and the scan budget before submission. The worker only reads segmentation data,
not raw EM images; it builds physical neighbors around your **existing** vectors.
Coarser segmentation levels can remove small cells or change contacts. Record the
level and use the same graph when comparing encoders. Disconnected nuclei may yield
self-only neighborhoods: use cell segments for contact graphs, or a physical
position table with radius/KNN if nucleus proximity is the intended question.

```yaml
- action: context
  name: DINO contact context
  embedding: /path/to/cell_id_embeddings.npz
  id_namespace: cell
  graph_id_namespace: cell
  graph_mode: segmentation
  spatial_source: ome_zarr
  ome_zarr:
    url: https://your-server.example/platy-cell-labels.ome.zarr
    image_index: 0
    level: 0
    channel: 0
    timepoint: 0
  remote_options: {cache_size_mb: 512}
  roi: [[0, 0, 0], [256, 256, 256]]  # Example voxel region; choose yours.
  max_scan_voxels: 16777216
  block_shape: [64, 64, 64]
  connectivity: 1
  include_self: true
  combination: raw
  reduction: none
  missing_neighbors: error
  roi_boundary: skip
```

For object vector `x_i` and its physical neighbors `N(i)`, the default is:

```text
neighbor_mean_i = mean(x_i and all x_j for j in N(i))
context_i = concatenate(x_i, neighbor_mean_i)
```

A D-dimensional encoder therefore produces **2D dimensions**. Both the individual
and mean blocks are exported separately. The focal cell is included in the mean
by default, as in the paper. The original 480-dimensional vectors would give 960
unreduced dimensions. The bundled context artifact has 200 dimensions; its original
contact graph and fitted reducer are unavailable, so generating new vectors does
not reproduce that historical artifact exactly.

Advanced options remain explicit: omit the focal cell from the mean, balance L2
normalized focal/context blocks with a weight, or reduce feature columns using Ward
feature agglomeration. Raw concatenation and no reduction are the defaults.
Agglomeration saves its feature-group assignments so its transform is reproducible.
It fits on the available cohort for exploration; use unreduced context vectors when
performing leakage-aware classifier evaluation, and fit any learned reduction within
training folds. Context also introduces shared information across neighboring cells:
use spatially blocked or independent-specimen validation for strong predictive claims.
Random cell splits do not establish generalization to another tissue or animal.

**Missing data is a scientific choice.** Default behavior stops if any graph neighbor
lacks an embedding. Choose `available` and a minimum represented-neighbor fraction
only deliberately; the coverage table records the effect. Isolated objects can use
a self-only mean, be skipped, or cause an error. ROI-boundary objects are skipped by
default because neighbors outside the region may be absent. Imported edge tables
cannot diagnose boundaries unless a sibling `graph_nodes.tsv` supplies node IDs and
boundary flags; keeping that file also preserves isolated nodes. The embedding IDs
and graph IDs each have a cell/nucleus selector, and nucleus IDs are mapped explicitly.

### Files, handoffs and reproducibility

The reusable [workspace example](../configs/workspace_bilateral_context.example.yaml)
runs published context display and both bilateral search modes; commented stages
show where to supply a real physical graph and MAE/DINO vectors. The original graph
is not substituted with the bundled bilateral candidates.

- **Bilateral:** `bilateral.json`, `summary.tsv`, `bilateral.svg/png`, and per-method
  matched vectors, native-ID bridges and `ranks-*.tsv` with reasons for exclusions.
- **Context:** `context.json`, `embeddings.npz`, `full_context.npz`, `self.npz`,
  `neighborhood_mean.npz`, `coverage.tsv`, `adjacency.tsv`, `graph_nodes.tsv`,
  `id_mapping.tsv` and, when requested, `agglomeration.npz`. Output vectors use cell
  IDs. `graph_id_mapping.tsv` additionally records conversions of graph nucleus IDs.
- Both produce `export.zip`. Browser downloads retain the 200 MiB safeguard; larger
  artifacts remain accessible in the worker output directory.

Use the context stage's **Name** as `from_embedding` in Analyze or as an entry in
`from_embeddings` in Compare/Bilateral. Results offers **Analyze these embeddings in
a new workflow**. Provenance shows inputs, graph sources, stages and settings; jobs
are labeled **Spatial context features** or **Bilateral-partner retrieval** in Runs.
Existing generic mean/max aggregation remains in Tools/CLI and saves only the
aggregate, so it is distinguished from the complete concatenated context recipe.
