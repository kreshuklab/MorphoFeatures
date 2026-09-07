# MorphoFeatures workspace redesign

Design and implementation notes, 7 September 2026. The design below guided the
Streamlit redesign in this change. Launch `python -m morphofeatures ui` to use it.

## Recommended direction

Replace the eleven peer-level pages with three primary destinations:
**Workflow**, **Runs**, and **Results**. A workflow is a retained draft that
connects **Prepare data → Train → Analyze → Review & run**. Users enter at the
stage matching the artifacts they already have. A single submission surface
explains exactly what will run, where it will run, and which settings it uses.

Put workspace settings, specialist tools, and help below the main navigation.
Use a quiet, theme-compatible layout, one open stage at a time, and explicit
input/output summaries. Keep advanced controls beside the settings they extend.

The implemented UI includes the starting points, connected stage forms,
advanced settings, configuration loading, draft save/reopen, artifact selectors,
shared workspace root, review invalidation and exact submission preview. The
original CLI and worker entry points remain available. Slurm scripts are tested
with a fake scheduler; live cluster submission is outside this validation.

Specialist shape/texture workflows, feature assembly, published classification,
mesh inspection and the existing MAE sweep manager remain in **Tools**. They
retain their existing execution adapters. Dedicated sweep editing within Train,
separate CPU/GPU stage allocations, and field-linked validation messages are
follow-up refinements, not features claimed by this implementation.

## Findings before the redesign

| Observation | Consequence | Proposed change |
| --- | --- | --- |
| `dashboard.PAGES` has eleven entries; training and analysis have additional overlapping tabs. | Users must learn which implementation owns each task. | One workflow editor, one run browser, one result browser. |
| The sidebar calls `load_config()` on a field labelled **Pipeline config**. Training, extraction, and pipeline pages load separate documents. | “Config ready” can describe workspace paths while the training draft still uses another file. | Call this **Workspace settings**; load scientific settings only into the active workflow or stage. |
| The generic editor renders dotted field names and serializes `stages` as a YAML value. | Common tasks require understanding the internal schema. | Stage forms with ordinary labels, contextual help, and linked artifact selectors. |
| `launch_controls()` repeats run ID, output root, execution mode, and dependency on each page. | Run identity and execution choices drift as users navigate. | One run identity and one execution panel per draft. |
| **Resolved submission** displays the incoming document before `submit_job()` resolves paths and scopes training outputs. | The preview does not show the final worker configuration. | Review a plan produced by the same resolver and renderer that submission consumes. |
| `dry-run` creates an experiment directory and registry entry. | Using it merely to inspect a script consumes the run ID. | Separate read-only **Review run** from **Save dry-run bundle**. Explain that saving a bundle reserves its ID. |
| Analysis and comparison reuse `resources` created inside the extraction tab. | Scheduler settings depend on an unrelated form. | Resources belong to the whole workflow or explicitly to a supported separate job. |
| Launch controls allow an output root different from the root used by Runs and Results. | A successfully submitted job may be absent from those browsers. | Use one active workspace root throughout; offer explicit workspace switching. |
| `submit_job()` updates the registry with the Slurm ID but returns the earlier frozen record. | Immediate feedback cannot reliably show the scheduler ID from that return value. | Return the updated record and link directly to its run details. |
| Structured edits update the document but do not advance the revision used by an already-instantiated YAML text area. | Applying stale YAML can overwrite newer form edits. | One canonical draft and revision-aware YAML buffers. |
| Default training uses `smoke.yaml`; `_load_crops()` generates synthetic crops when a crop path is absent. | A newcomer can run a demonstration without realizing it. | An explicitly named **Synthetic demonstration** starting point; real-data routes require explicit inputs. |
| Global CSS forces bright text and dark metric surfaces independently of the theme. | The visual hierarchy depends on the surrounding theme and mixes competing colors. | Use native theme colors, restrained accents, consistent spacing, and text status labels. |

Source locations: [navigation and workspace loading](../morphofeatures/dashboard.py),
[stage views and launch controls](../morphofeatures/workspace_ui.py),
[document loading](../morphofeatures/configuration_editor.py),
[job resolution and execution](../morphofeatures/workspace_jobs.py), and
[crop fallback](../morphofeatures/mae3d.py).

## Navigation and entry points

```text
MorphoFeatures                         Active workspace: outputs/my-project
  Workflow                            Draft name · Saved / Unsaved changes
  Runs
  Results                             Prepare data → Train → Analyze → Review & run

  Tools                               Current stage: inputs, essential settings,
  Workspace settings                  advanced settings, expected outputs
  Help
                                      Back                     Continue to …
```

The stage sequence is navigation, not a progress bar claiming work has run.
Use **Needs input**, **Configured**, **Using existing …**, and **Excluded** for
drafts. Reserve **Queued**, **Running**, **Completed**, and **Failed** for jobs.
Allow direct stage selection without losing edits. A missing prerequisite blocks
submission and links back to the relevant field; it does not block navigation.

On a new workflow, ask **What are you starting with?**:

| Starting point | First task | Default connected stages |
| --- | --- | --- |
| Raw image and instance segmentation | Prepare aligned volumes and choose objects | Prepare → Train MAE → Extract embeddings → Analyze |
| Prepared crops or grouped N5 patches | Select a data configuration and train | Train MAE → Extract embeddings → Analyze |
| An existing model checkpoint | Select matching model/data settings | Extract embeddings → Analyze |
| Saved embeddings | Select one representation or several to compare | Analyze, or Compare |
| Saved workflow | Import an existing pipeline YAML | Preserve its stage sequence and settings |

Offer **Try the synthetic demonstration** as a secondary, clearly labelled
action. Offer **Open saved results** beside the starting points. Users can stop
after preparation, train without extracting, or extract without analysis.
Collapsed optional stages stay in the outline with their disposition visible.

**Analyze** is a user-facing stage containing two decisions: obtain embeddings
(use saved files or extract with MAE/DINO), then inspect one representation or
compare several. Extraction remains an explicit execution stage in the review.
Selecting DINO exposes its local repository, backbone variant, weights and view
settings. A raw-data workflow can prepare crops and use an existing model
without training a new MAE.

### Where the current features go

| Current page or tab | New home |
| --- | --- |
| Overview | Workflow starting points; runtime and bundled-data checks in Tools |
| Preprocessing | Workflow → Prepare data |
| Train and encode → Persistent training configuration | Workflow → Train |
| Scientific pipeline | The workflow itself; full pipeline YAML under advanced editing |
| Embeddings and comparison → Extract / Analyze / Compare | Workflow → Analyze |
| Embeddings and comparison → Reopen results | Results |
| Analyze → Classification / Projection | Workflow → Analyze for configuring new work; Results for saved outputs |
| Experiments, including its separate workspace job selector | Runs with one selection and one run detail view |
| Build features → combination / neighbor context | Tools → Feature assembly, with **Use output in a workflow** |
| Legacy shape/texture training and encoding | Tools → Published shape and texture workflows, using the shared execution review |
| MAE soft grid | Workflow → Train → Parameter sweep; sweep manifests and summaries in Runs |
| Meshes | Tools → Mesh inspection, linked from data and result inspection |
| Data and config → Paths | Workspace settings |
| Data and config → Synthetic fixture / Inspect embedding | Tools, with contextual links from Workflow and Results |
| Documentation | Help and links beside relevant settings |

Do not delete distinct scientific functionality to simplify navigation. Some
legacy tools run synchronously today; their new location must retain that
behavior explicitly until there is a tested worker adapter. Do not describe
unsupported legacy operations as integrated workspace stages.

## Stage forms and connected artifacts

Each stage begins with **Input** and ends with **Produces**. Linked inputs show
their source in words, for example **Crops from Prepare data in this workflow**
or **Checkpoint from run nucleus-014**. A **Change source** action opens the
selector; users should not copy generated paths between pages.

| Stage | Essential controls | Advanced controls |
| --- | --- | --- |
| Prepare data | Raw and segmentation paths; relevant keys, axes and channels; explicit ROI; crop size; object limit; spacing/unit summary | Origin, block shape, ID filter, minimum volume, centering, padding, truncation and normalization |
| Train | Data source/contract, model preset, epochs, batch size, learning rate, compute device | MAE architecture, masking, regularization, validation/split policy, loaders, checkpoint resume, metrics, parameter sweeps |
| Analyze: extraction | Target data, MAE/DINO family, checkpoint or linked training output | Architecture/data compatibility, DINO views and masks, inference batches, cache and data version |
| Analyze: exploration | Embedding source, PCA/UMAP choice, cluster method/count | Normalization, neighbors, minimum distance, epochs, seed, optional Leiden settings |
| Analyze: comparison | Named representations; optional annotation table and label/group columns | Evaluation folds, KNN, linear probe, evaluation PCA and extraction cost provenance |

Use explicit numeric/list controls for common shapes and ROIs; retain full YAML
for unmodeled settings. Reveal source-dependent fields only when relevant.
Physical units and data contracts remain visible because they affect scientific
interpretation. Show effective advanced settings in the review even if their
controls were never opened.

Within one job, use existing `from_preprocessing`, `from_training`,
`from_extraction`, and named `from_embeddings` links. Make the compiler choose
the correct links rather than asking users to type them. Derive training input
shape from preparation unless explicitly overridden; flag conflicting shapes.
Whole-object crop MAE and grouped N5 MAE remain distinct contracts.

Across completed runs, select recorded artifacts and attach their provenance to
the new draft. Missing files produce a repair action. Selecting an unfinished
run must not pretend its outputs exist; the first implementation supports
completed artifacts and existing Slurm `afterok` dependencies, while assembling
new cross-job data dependencies requires a separate adapter and validation.

Changing an upstream input invalidates downstream validation and any reviewed
plan, while retaining editable settings. It does not alter completed runs.
Unknown or repeated stages in imported pipelines must remain lossless: use an
ordered advanced stage list when the simple forms cannot represent them.

## Configuration vocabulary and state

Use these labels consistently:

| Action | Effect |
| --- | --- |
| **Load settings into draft** | Read a selected YAML file/profile into the editor. Display source path, resolved profile and what will be replaced. No job is launched. |
| **Save draft** | Persist editable workflow state under the active workspace. Preserve source provenance and unknown fields. |
| **Download pipeline YAML** | Export the current runnable pipeline document. It is independent of the original source file. |
| **Review run** | Validate inputs and build the effective configuration, destination paths, command and script. No persistent run is created. |
| **Save dry-run bundle** | Save the reviewed snapshot/script and a dry-run record, reserving its run ID. No worker runs. |
| **Run locally** | Start the reviewed workflow in a detached process on the host running the app. |
| **Submit to Slurm** | Save the reviewed snapshot and call `sbatch` once, then display the returned job ID. |
| **Use settings for a new run** | Copy a previous run into a new editable draft with a fresh run ID. Does not resume a model. |

Keep a small provenance line near the workflow title, such as
**Loaded: mae_platynereis_nuclei_embl.yaml · Profile: full · Edited · Draft saved**.
Changing a path field alone does not load anything. Preview a replacement if a
draft has edits, then apply it with an explicitly labelled replacement action.
A failed load keeps the existing draft intact. Distinguish training profiles
from cluster resource presets in labels and state.

Use one canonical document separate from widget keys, building on `remember()`.
Track document revision, saved revision, source provenance, execution settings,
artifact bindings, and reviewed-plan fingerprint. Keep transient UI metadata
outside scientific YAML. Store drafts atomically under
`<workspace>/.morphofeatures/drafts/<draft_id>.json`, with schema version and
revision checks to avoid silently overwriting edits from another session.
Save explicitly and retain session changes on navigation; state clearly when
changes have not been saved for reopening after a restart.

The full YAML editor is an explicit edit/apply surface. Opening it uses the
current draft revision. Applying an older buffer after form changes requires
reconciliation rather than silently replacing the newer document. Import and
export retain unfamiliar fields and nonselected training profiles.

## Review and submission

Show a single page with:

1. **Run name and destination**: the active workspace root and complete run
   directory. Explain output overrides by listing the effective checkpoint,
   embedding, report and log paths.
2. **What will run**: ordered stage summaries, their inputs/outputs, and settings
   changed from the loaded source. Show the actual extraction stage separately.
3. **Where it runs**: Local or Slurm. Local shows the app host, interpreter and
   CPU threads. Slurm shows partition/account, GPUs, CPUs, memory, time and
   interpreter. Put QoS, topology, trusted setup and dependencies in advanced
   controls. Show whether `sbatch` is available on the app host.
4. **Validation**: actionable errors linked to their source stage. Distinguish
   validated local paths from compute-node access assumptions; avoid a blanket
   “ready” claim about untested cluster access.
5. **Exact submission details**: expandable resolved YAML, script and worker
   command, generated from the reviewed plan.
6. **Actions**: one primary **Run locally** or **Submit to Slurm**, with secondary
   **Save dry-run bundle**. Editing any relevant setting disables launch until
   the revised plan has been reviewed.

The initial linked pipeline remains one allocation with sequential stages, as
the worker supports today. Say **One Slurm job; stages run in order; these
resources apply to every stage**. GPU resources can remain reserved during CPU
stages. Separate CPU/GPU allocations are a later option, not an implied feature.

After submission, navigate to the new run and show its Slurm ID or local launch
state, stages, logs and output links. Failed scheduler submission is a failed
submission, not a queued job. A saved dry run can be cloned into a fresh run for
submission; do not silently repurpose its immutable directory.

## Runs and Results

**Runs** lists all records in the active workspace with one selection model.
The detail view starts with stage status and the useful next action. Logs,
metrics, configuration and scheduler details are expandable sections. Show
scheduler state separately when it differs from application state. Retain
existing cancellation support for Slurm and do not imply local cancellation is
implemented. Retain the last known status when refresh is unavailable.

**Results** reopens registered analysis/comparison artifacts and manually entered
paths, including the existing `outputs/workspace-validation-20260907/comparison/
comparison.json`. Opening a result does not trigger extraction or analysis.
Keep linked ID selection, nearest neighbors, object previews, scientific
interpretation, exclusions and exports. Offer **Analyze these embeddings**,
**Compare with another representation**, and **Use this checkpoint** only where
the selected artifact supports the action. Each creates a new draft with
explicit artifact bindings.

## Implementation sequence

| Increment | Concrete work | Completion check |
| --- | --- | --- |
| 1. Workspace and navigation | Introduce the three primary destinations, secondary tools/help/settings, one workspace root, stage navigation and explicit starting points. Rehome legacy functionality without dropping it. Remove forced theme colors. | Every current capability has a reachable home; navigation retains drafts and workspace selection. |
| 2. Draft and forms | Add a versioned draft model/store and stage compiler; replace generic primary forms with essentials/advanced sections and artifact bindings. Add configuration provenance and lossless import/export. | Real-data routes cannot silently use synthetic crops; forms/YAML round-trip unknown settings; invalid links and input contracts are explained. |
| 3. Shared planning and submission | Factor `submit_job()` into planning and execution with a single resolved plan; retain the CLI wrapper. Reuse the Slurm renderer and existing legacy plan adapters. | Preview creates no experiment/registry entry; submitted paths/script match the plan; a changed revision invalidates review; double clicks cannot launch twice; returned records contain scheduler IDs. |
| 4. Connected runs and results | Merge run selectors; share the active registry/root; add artifact-driven continuation, draft reopening, and saved-result navigation. | Completed preparation/training can feed a new run without path copying; reopening results performs no computation. |
| 5. Regression and documentation | Update UI docs/CLI navigation hints, add interaction tests, retain scientific smoke tests, and inspect the UI at narrow and wide widths in light/dark themes. | Existing workflows and artifacts remain compatible; task checks below pass. |

The implementation uses `workspace_state.py` for drafts and persistence,
`WorkspacePlan` in `workspace_jobs.py` for the plan/submit contract, and
`workflow_ui.py` for the guided stage views. Reuse `configuration_editor.py`,
`workspace_jobs.py`, `registry.py`, `slurm.py`, and scientific worker functions;
do not duplicate their scientific logic in forms. Keep existing CLI YAML valid.
Check the repository's conditional Streamlit 1.12/Python 3.9.7 support before
using newer navigation APIs; the design can use ordinary radio/buttons and
conditional rendering without requiring those APIs.

## Acceptance scenarios

- A newcomer starts with aligned volumes, configures a bounded ROI, trains and
  analyzes through explicit stage links, then names the run, reviews its exact
  Slurm script and submits once.
- A researcher loads a grouped N5 config/profile, changes epochs, visits Results,
  returns, and sees the same values. Exported and submitted settings agree.
- A user changes a file path but has not clicked **Load settings into draft**;
  the active source indicator and effective settings remain unchanged.
- A user loads another config over edits, sees the replacement summary, and can
  keep the old draft. Invalid YAML cannot destroy it.
- A user starts from an existing checkpoint or embeddings and skips irrelevant
  stages without inventing preprocessing or training dependencies.
- A user chooses saved MAE and DINO embeddings, sees the ID/provenance matching
  requirements, configures comparison, and later reopens the saved report.
- A user previews repeatedly without reserving a run ID; saving a dry-run bundle
  explicitly reserves it and points to the saved files. A later real submission
  uses a fresh run ID.
- A user edits a setting after review or double-clicks submit; stale plans and
  duplicate launches are prevented at the backend boundary.
- Changing the active workspace changes draft storage, run lookup and result
  lookup together; submission cannot disappear into a different default root.
- A missing checkpoint, incompatible crop/grouped contract, failed stage,
  unavailable `sbatch`, or failed Slurm submission yields a specific next action.
- Keyboard navigation, visible focus, descriptive labels, theme contrast,
  narrow layouts and reduced motion work without relying on color alone.

Use Streamlit AppTest for retained state and interaction paths, fake schedulers
for submission/duplicate behavior, temporary workspaces for draft/plan tests,
and the existing bounded train→extract→analyze and preprocessing smoke tests.
No live cluster allocation is required to verify the UI redesign.
