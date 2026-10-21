# Bring MorphoFeatures results into MoBIE

MorphoFeatures can now create a **MoBIE project** from a saved analysis or comparison. In Fiji, collaborators can explore the original images, select a cell or nucleus, see its cluster or cell-type label, and locate it in the saved UMAP/PCA plot. They do not need to install MorphoFeatures or rerun the model.

The export creates a new project. It does not modify the original images, your analysis, or the published PlatyBrowser project. Nothing is uploaded or published automatically.

For original/PCA/neural-adapter comparisons, open the distillation report, select
**Maps & inspection**, then **Exports** on the selected comparison to prepare a
MoBIE export. The vectors retain the original object IDs and inspection source
metadata; the reduced dimensions never replace segmentation IDs. The maps include
training objects and are exploratory. Held-out classifier predictions are saved
separately in `predictions.tsv`; they are not silently assigned to all objects in
the volume. See [embedding distillation and its output files](workspace_workflows.md#distill-embeddings-using-known-cell-types).

For independent specimens projected into a shared PCA/UMAP, open the joint result
in **Results**, choose **Dataset to inspect / export**, and create the MoBIE export
from that dataset's report. Its original object IDs, shared coordinates and shared
cluster IDs are preserved. Export each specimen against its own raw image and
segmentation; the combined report's internal IDs must not be used as segmentation
IDs. This creates a separate MoBIE project per specimen. See the
[shared-dataset workflow](workspace_workflows.md#independent-datasets-in-one-shared-pcaumap)
for preparation and output details.

## Explore binary expression with morphology

The standalone [binary expression methods notebook](../binary_expression_morphology_methods.ipynb)
extends [the joint morphology/expression tutorial](../joint_morphology_expression_tutorial.ipynb).
It compares binary PCA, positive-call cosine/SVD, TF–IDF/LSI, Jaccard and Hamming
neighborhoods, weighted vector/distance fusion, and a fixed graph blend. The guide
inside distinguishes registered binary spatial calls from raw scRNA-seq counts
and explains when Bernoulli factors, MOFA2 or WNN would be useful next steps.

Run top to bottom in `MorphoFeats_dev`. The default uses 1,500 uniformly sampled
cells for distance matrices and layouts, full matched cells for descriptive
vectors, and all eligible annotations for classification with preprocessing
fitted inside training folds. Adjust `LATENT_DIM`, `EXPRESSION_WEIGHTS`, and
`MORPHOLOGY_FOR_FUSION` in the configuration cell; compare separate runs before
selecting a recipe. `RUN_UMAP=False` skips UMAP without disabling the other
analyses. Outputs go to a new directory under
`outputs/exploration/binary_expression_methods/`.

Each run saves cohort audits, source hashes, gene loadings, diagnostic tables,
PNG/SVG figures, vectors and sparse affinity graphs. Vector `label_ids` are cell
IDs; separate `nucleus_ids` preserve the mapping. Layout tables contain both
namespaces. Match the appropriate column to the displayed segmentation before
MoBIE import. Graph rows follow `tables/geometry_cohort.tsv`; graphs and 2D
coordinates are not embedding feature vectors. The fixed graph blend is not
Seurat WNN, and the notebook does not fit or export a deployment classifier.

For **learned correspondence between the modalities**, continue with
[shared_space_morphology_expression.ipynb](../shared_space_morphology_expression.ipynb).
It opens a completed CPU pilot of regularized CCA, PLS, MOFA2 with separate ridge
projections, and paired neural encoders. Matched-cell retrieval, gene prediction,
random/spatial holdouts and anatomical controls distinguish a useful representation
from an attractive plot. The [method guide](shared_space_analysis.md) explains the
dedicated environment, reusable scripts, saved models and cell IDs for inspection.

## What will I be able to explore?

| Result | In MoBIE |
|---|---|
| Original raw image | Image background, when supplied |
| Original instance segmentation | Individually selectable cells/nuclei |
| Embedding clusters | Segmentation and scatter plots colored by cluster |
| Known class labels | Separate views colored by supplied annotations |
| Predicted class labels | Separate views colored by classifier output |
| UMAP and PCA coordinates | Saved scatter plots linked to segmentation objects |
| Prediction confidence and prediction source | Columns in the analysis table, when present |
| Several representations, such as MAE and DINO | Separate tables and views in one project when exporting a comparison |
| Full embedding vectors | NPZ files among the accompanying evidence; optionally also wide MoBIE feature tables |
| Figures, classifier reports and evaluation splits | Accompanying files, available outside the MoBIE viewer |

MoBIE can create 3D displays from selected segmentation objects. Previously exported mesh files and image sheets can be attached as evidence, but they are not automatically imported as MoBIE image sources.

**Keep three meanings separate:** known labels are annotations supplied by the user; predicted labels are estimates from a classifier; clusters are groups in feature space and are not automatically cell types. The export preserves these distinctions. It does not run another classifier, recompute UMAP, or improve the validation evidence. When available, `prediction_source` distinguishes held-out predictions from predictions fitted on labeled objects.

If you enable **PCA components before UMAP** during analysis, MoBIE receives that
PCA-preprocessed UMAP's saved coordinates. This option is off by default; 50 is
one starting count to compare with the full-vector map. Results records the
actual count and variance retained. It does not change the saved feature vectors,
clusters or classifier scores. See the [workflow guide](workspace_workflows.md)
for the normalization → PCA → UMAP sequence and shared-dataset fitting rules.

You can also choose PCA before clustering separately, or save several named
parameter combinations from **Results → Explore clustering and UMAP on these
embeddings**. Select the desired **Saved analysis combination** before inspecting
objects or preparing MoBIE export: its coordinates and cluster IDs will be used.
Known and predicted cell types retain their original meaning; changing clusters
does not retrain a classifier. Shared maps preserve each specimen's original
segmentation IDs. Display names can be edited for completed runs and results;
file paths and object correspondence stay stable.

## 1. Prepare the ingredients

You need a completed **Analyze** or **Compare** result: `analysis.json` or `comparison.json`, together with its result folder. Keep `coordinates.tsv`, `object_labels.tsv` when present, and the representation subfolders together. The usual Workflow produces these files for you.

For a spatial project, also identify:

1. **The original instance segmentation.** Every cell/nucleus has its own positive integer ID; zero is background. A binary foreground mask or a foreground-probability image cannot identify individual objects. An array of isolated prepared crops also does not preserve the animal's spatial arrangement.
2. **The raw image**, if desired. For a portable export, raw and segmentation must use the same spatial grid. Selecting a raw channel or timepoint is supported. Different resolutions must first be selected or resampled onto a matching grid.
3. **Voxel size, axis order and origin.** MorphoFeatures settings use Z, Y, X order. For example, `[0.100, 0.080, 0.080]` with unit `um` means 100 nm in Z and 80 nm in Y and X. Use the actual source metadata; these are not universal values for all PlatyBrowser levels.
4. **The relation between embedding IDs and segmentation IDs.** New PlatyBrowser preprocessing uses the segmentation's own IDs. Some older patch datasets instead use cell IDs while the mask contains nucleus IDs.

For different ID systems, supply a TSV or CSV with a unique `label_id` for each embedding object and either `segmentation_id` or `nucleus_id` for its segmentation object:

```tsv
label_id	nucleus_id
8578	11486
```

The example shows a cell-to-nucleus mapping, not two interchangeable IDs. Positive assignments must be one-to-one. A target ID of zero means no assigned object. PlatyBrowser's **cell** ID option can retrieve the published cell-to-nucleus table automatically for its nuclei source.

If you want known/predicted cell-type views, run analysis with the annotation table first. Export cannot invent absent annotations or classification results. Review the classification report before sharing predictions as biological evidence.

## 2. Choose how the images will be supplied

| Option | Best suited to | What is copied | What collaborators need |
|---|---|---|---|
| **Copy a region — portable project** | Local volumes, generic remote OME-Zarr, offline sharing, or a small representative region | The chosen raw/segmentation region as OME-Zarr v0.4 using Zarr v2, plus tables and evidence | The whole exported project folder and Fiji/MoBIE |
| **Stream published PlatyBrowser images** | PlatyBrowser analyses and whole-animal context | Published image-reference XML files, object table, analysis tables and evidence; no image chunks | The project folder, Fiji/MoBIE and internet access to the published image storage |

Portable export accepts the volume formats already supported by MorphoFeatures: local HDF5, N5, Zarr and NumPy, plus supported remote N5/OME-Zarr sources. A generic remote OME-Zarr URL is **copied only over the selected region**, rather than made into a new live remote reference. The source's chosen array key, channel and timepoint determine what is exported. Zarr v3 input requires a compatible reader; the output deliberately uses Zarr v2 for MoBIE compatibility.

A portable export contains **one resolution**, at the selected source resolution. It does not create a new image pyramid. Start with a modest region; this is especially useful for checking an export before copying a larger volume. PlatyBrowser streaming preserves the published multiresolution image pyramid, so MoBIE can load appropriate resolutions while browsing. The MorphoFeatures resolution used to compute the embeddings is still recorded in the evidence.

The streaming project pins the PlatyBrowser metadata to a Git commit. Its images remain at the published storage locations and are not an offline archive of those images. Published storage must remain accessible. It is a new local project, not a contribution back to PlatyBrowser.

## 3. Export from the UI

1. Open **Results** and select your `analysis.json` or `comparison.json`.
2. Expand **Export this analysis to MoBIE** and click **Create MoBIE export workflow**. A comparison includes all its representations.
3. In **Workflow → Analyze**, choose the image option above. Source settings are recovered from the saved preprocessing/extraction metadata when available. If needed, enter the original data/inspection YAML and click **Check export settings and recover source metadata**. This deliberately reloads the source fields from that YAML; check them again after recovery.
4. Give the dataset a short name and a biological description. For a portable project, choose the original raw/segmentation sources and ROI. For streaming, check the PlatyBrowser revision, dataset, source names and ID system.
5. Check the mapping and coverage policy. Leave **Stop export if any embedding objects are missing** off when intentionally exporting only a region. Enable it when every embedded object should be present, for example a whole-sample export.
6. Under **Evidence files, display and download**, choose whether to include the saved analysis files, a wide feature table and a ZIP. Add any extra files explicitly: a training checkpoint, training metrics, model configuration, mesh archive, or inspection figure. They count toward the evidence size limit.
7. Continue to **Review & run**, choose local execution or Slurm, and review the generated configuration and destination. Export needs no GPU. Four CPUs and 8 GB RAM are a reasonable starting allocation for a small region; larger tables or evidence bundles can need more resources. The worker must be able to access the files and, for remote data, the network.
8. Run the export. In **Results**, select the registered `mobie_export.json`. Check the matched/excluded counts. Download the ZIP if offered, or copy the project folder/ZIP from the displayed disk location. Browser download is offered only for archives up to 200 MiB.

To finish an ongoing pipeline with MoBIE export, use **Change which stages run → Add MoBIE export after analysis**. This exports the **last** Analyze or Compare result in that workflow. To include MAE and DINO together, put a Compare stage before the export. Exporting a single Analyze result includes that representation only.

### Choosing a portable region

ROI start is inclusive and stop is exclusive, in **voxels of the selected source resolution**, with both triples in Z, Y, X order. For example, start `[100, 200, 300]`, stop `[164, 264, 364]` copies a 64 × 64 × 64 region. It does not mean physical micrometres or a region relative to a prepared crop.

The default copy safeguard is 256³ voxels. Raise it only after estimating the output: raw float32 plus segmentation int64 require roughly **12 bytes per voxel before compression**, plus tables, evidence, and an additional ZIP copy if enabled. Copying uses bounded blocks; the limit governs total work, not just RAM. Compression depends on the data. Disable ZIP for large projects if you will transfer the folder directly.

Use the existing Prepare data preview or Results object inspection to check alignment before export. Matching IDs and dimensions are necessary checks, but cannot prove that a file belongs to the correct specimen.

## 4. Open the project in Fiji

1. Install Fiji and enable the **MoBIE** update site through **Help → Update… → Manage Update Sites**. Apply updates and restart Fiji. See the [official installation instructions](https://mobie.github.io/tutorials/installation.html).
2. Extract the ZIP first, if using one. Keep the entire `project/` folder intact.
3. In Fiji's search bar, find **Open MoBIE Project Expert Mode…**. Set **Project Location** to the local folder containing `project.json`—not to a TSV, `dataset.json`, or the ZIP itself. Use local image data for a portable project, or remote/S3 image data for a PlatyBrowser streaming project. Leave separate image/table root overrides unset. Wording can differ between MoBIE versions; the [expert-mode guide](https://mobie.github.io/tutorials/expert_mode.html) describes these location settings.
4. Choose a view from the **MorphoFeatures** menu. `default` shows the image and instance identities. Other views identify the representation (`mf0`, `mf1`, …), coloring column and plot type. The included `README.txt` and manifest map those prefixes to MAE, DINO, or your representation names.
5. Choose a `umap` or `pca` view to show a saved scatter plot. Select an object in the table or scatter plot to locate it in the spatial image. Use MoBIE's segmentation display controls to show selected objects in 3D. Adjust image contrast if needed.

Analysis columns are prefixed to avoid collisions between models. For example, `mf0_known_label`, `mf0_predicted_label`, `mf0_cluster`, `mf0_prediction_source`, `mf0_umap_1`, and `mf0_pca_1`. Missing category values are displayed as `unlabeled`. Spatial objects outside the analyzed subset have no embedding coordinates. Viewer palettes may differ from the MorphoFeatures plot palette; categories and coordinates are preserved.

If you enabled full feature tables, load `mf0-features.tsv` (or the corresponding representation file) as an additional table for the segmentation using MoBIE's table controls. They are not automatically opened because hundreds of feature columns can make the table slow.

## 5. Read the coverage report before sharing

The report compares **actual IDs**, not only object counts. An equal count does not establish a valid correspondence.

- Portable mode checks against the actual copied segmentation pixels. It computes object bounds and centroids from those pixels. Objects touching the ROI boundary are marked `touches_export_boundary`; their displayed geometry may be cut and their exported centroid/bounds describe only the copied portion.
- PlatyBrowser streaming checks against the published object catalogue and retains its physical anchors/bounds. It does not scan the entire remote segmentation or guarantee that every object survives at every coarser level. A very small nucleus may only be visible at a finer resolution.
- Excluded IDs are recorded separately for each representation. They may be outside the ROI, missing from the catalogue, or lack a mapping. Their analysis evidence can still be included, but they have no corresponding spatial overlay.
- An export with **zero matched IDs** for any representation stops. With strict coverage enabled, any missing embedding object also stops the export. A failed run may leave partial files for diagnosis; do not share them as a complete project. Use a new run ID for the corrected export.

## What files are produced?

A successful export stage has this layout. The dataset folder uses your chosen name.

```text
00-export_mobie/
├── mobie_export.json             # Completion report shown in MorphoFeatures Results
├── project.zip                   # Optional transfer archive
└── project/                      # Open this folder in MoBIE
    ├── project.json              # Lists the dataset
    ├── README.txt                # Opening instructions and coverage summary
    └── morphofeatures/
        ├── dataset.json          # Image sources and saved viewer states
        ├── images/
        │   ├── raw.ome.zarr/     # Portable mode; omitted if no raw image supplied
        │   └── objects.ome.zarr/ # Original instance IDs within the ROI
        │                        # Streaming mode has raw.xml and objects.xml instead
        ├── tables/objects/
        │   ├── default.tsv      # Segmentation IDs, physical anchors and bounds
        │   ├── mf0.tsv          # First representation's labels and coordinates
        │   ├── mf1.tsv          # Further representations, if present
        │   └── mf0-features.tsv # Optional full feature vectors
        └── misc/
            ├── export-manifest.json  # Coverage, representation/column mapping, evidence checksums
            ├── export-settings.json  # Spatial sources, export settings and recovered input config
            ├── analysis-report.json  # Original analysis/comparison metadata
            ├── id-mapping.tsv        # Supplied mapping, if present (CSV remains CSV)
            ├── mf0-excluded-ids.tsv  # IDs with no spatial overlay; one file per representation
            ├── analysis/            # Optional copy of the saved analysis directory
            └── attachments/         # Extra files explicitly selected by the user
```

The table `label_id` always refers to the **segmentation ID**. `mf0_embedding_label_id` preserves the corresponding original embedding ID. Tables are joined by ID, never by row position. Background zero is omitted from object tables. Integer IDs are written without passing through floating-point representations.

Physical coordinates use the same transformation as the exported images: `physical = original_origin + original_voxel_index × voxel_spacing`. Portable image metadata includes the ROI offset, so cropping does not reset the object's location to the origin. Table bounds are extrema of voxel centres; the ROI stop remains exclusive in the configuration.

The portable segmentation retains its IDs. The raw image is stored as float32 and is not normalized; converting higher-precision input to float32 can round intensities. Selected channel/timepoint axes are removed from the exported 3D image, but their selection is retained in provenance. These exported images are for sharing/exploration; retain the original acquisition as the archival source.

By default the evidence copy includes files in the saved analysis directory, excluding existing ZIPs, temporary files, symbolic links and model checkpoints. It does **not** scan unrelated training folders or copy all training inputs. Add checkpoints, training logs/configs or other files through **Additional files to include**. The size limit is checked before copying evidence, and copied files have SHA-256 checksums in the manifest. The recorded exporter version includes the repository commit and whether the working tree contained changes.

## Configuration and command-line use

[The example workflow](../configs/workspace_mobie.example.yaml) starts from a saved result and uses PlatyBrowser streaming. Edit its analysis path and ID system, then load it from **Workflow → Start**, just like any other pipeline YAML.

A portable stage can instead use:

```yaml
stages:
  - action: export_mobie
    analysis: /path/to/analysis/analysis.json
    mode: portable
    dataset_name: specimen_01
    description: "Specimen 01: MAE embeddings and cell-type predictions"
    raw: /path/to/aligned_volumes.h5
    raw_key: raw
    raw_axes: zyx
    raw_channel: 0
    raw_timepoint: 0
    segmentation: /path/to/aligned_volumes.h5
    segmentation_key: instances
    segmentation_axes: zyx
    segmentation_channel: 0
    segmentation_timepoint: 0
    spacing_zyx: [0.100, 0.080, 0.080]  # Replace with this volume's voxel size
    origin_zyx: [0.0, 0.0, 0.0]
    unit: um
    roi: [[100, 200, 300], [164, 264, 364]]
    max_voxels: 16777216
    block_shape: [32, 32, 32]
    id_mapping: null
    require_all_ids: false
    include_artifacts: true
    include_feature_table: false
    attachments: []
    max_artifact_gb: 4.0
    contrast_limits: [0, 255]
    make_zip: true
slurm:
  cpus: 4
  gpus: 0
  memory: 8G
  time: "01:00:00"
```

For generic remote OME-Zarr, replace raw/segmentation paths with the source URLs, set their array keys and axis orders to the selected resolution, and add:

```yaml
    remote_options:
      format: ome_zarr
      cache_size_mb: 1024
      timeout_seconds: 20
```

The normal Workflow can fill these values from the sources connected in Prepare data. Raw-only data can be previewed there, but a spatial MoBIE segmentation project needs the instance segmentation to be ready.

To link an export stage directly to an earlier Analyze or Compare stage, replace `analysis: ...` with `from_analysis: true`. A preceding Prepare data stage can supply the spatial source and ROI; otherwise configure them explicitly. Keep the export after the comparison if you want every compared representation.

For local command-line execution, save the workflow YAML in a **new run directory** and run:

```bash
python -m morphofeatures workspace-run --config /path/to/new-run/export.yaml
```

Without a generated worker block, results are written beneath `results/` next to that YAML. Existing worker runs are not overwritten. Use the UI's Review & run step for Slurm submission and resource settings; the command above runs locally.

## Method and maintenance notes

The exporter writes native MoBIE project specification **0.3.0** metadata and TSV tables directly, using the [official MoBIE specification](https://mobie.github.io/specs/mobie_spec.html). Portable images use [OME-NGFF v0.4](https://ngff.openmicroscopy.org/0.4/) axes, scale and translation metadata over gzip-compressed Zarr v2 arrays. There is no extra MoBIE Python SDK dependency. The normal workspace environment supplies the volume readers; Fiji/MoBIE is the viewer.

For portable data, bounded copies precede an on-disk object-index scan of the exported segmentation. This establishes ID coverage and physical bounds without loading the full volume into memory. PlatyBrowser mode instead reuses the existing metadata resolver, copies the published BDV XML references, and changes only their source names to match the new project. The published raw and segmentation images remain remote.

Saved analysis columns are preserved and namespaced per representation. No new clusters, projections, classifier predictions or category masks are calculated. MoBIE applies table-based coloring to the original segmentation, preserving selection of individual objects.

Regression checks cover exact IDs (including large integers), shuffled joins, cell-to-nucleus mapping, ROI offsets, anisotropic spacing, channel/timepoint selection, unmatched-object policies, comparison views, ZIP contents, metadata-only streaming and worker handoffs. Native metadata is also checked against MoBIE's published project/dataset/source/view JSON schemas during development. This checks the file contract; it does not replace opening a representative project in your installed Fiji/MoBIE version.

### Existing table-only analyses

`morphofeatures.data.io.export_embeddings` still writes ID-first TSVs that can be joined to existing MoBIE tables. The `project` CLI still exports coordinates and clusters without creating a spatial project:

```bash
python -m morphofeatures project \
  --embedding path/to/embeddings.tsv \
  --cluster-method leiden \
  --output path/to/clusters_umap.tsv
```

For an existing MoBIE segmentation, additional TSVs must have one row per unique `label_id`, with IDs drawn from that source's default table. Keep published column names stable when updating tables already referenced by saved views. The new project export automates this contract and supplies the spatial images, default table and views as well.

## DINO experiments and cross-dataset predictions

The [DINO fine/coarse workflow](workspace_workflows.md#dino-finecoarse-experiments)
retains a separate feature matrix for fine texture, coarse texture and their
combined representation. Choose the representation in the experiment's Results
summary, then the dataset you want to inspect or export. Its per-dataset report
contains that dataset's original object IDs and original volume configuration.

For label transfer, choose the target dataset and one of the five conditions:
**uncorrected**, **classical CORAL in either direction**, or the **neural adapter
with/without Deep CORAL**. The [transfer comparison walkthrough](workspace_workflows.md#compare-all-five-transfer-conditions)
explains how to configure them and compare their scores. Select each condition's
own reference/target map; independently fitted adapted spaces cannot be merged.
The native target report is under
`target-*/<condition>/joint/dataset-1/analysis.json`. It carries the same predicted
labels shown in the saved plots and prediction table, together with known labels,
cluster IDs, class scores, nearest-reference distances and rejection reasons.
An `unknown` prediction is a provisional rejection, not a newly discovered cell
type. Classical CORAL and Deep CORAL results identify that the target's unlabeled
features participated in adaptation. Target annotations contribute only to
evaluation. Neural conditions export new latent vectors, while preserving native
object IDs and the original image/segmentation provenance for inspection and MoBIE.

Use **Export to MoBIE** on that native report. Do not pass `experiment.json`,
`transfer.json`, or the pooled joint report as a segmentation analysis: pooled
internal IDs are not IDs in either animal. For a within-dataset fine/coarse
comparison, its `comparison.json` can export all compared representations together.

Original sources are recovered from embedding provenance, including any
cell-to-nucleus mapping. Published PlatyBrowser sources can remain streamed.
For a portable local/OME-Zarr export with differently sampled raw and segmentation
grids, the physical-view configuration enables `resample_raw: true`: raw images
are interpolated and filtered onto the segmentation grid for the chosen export
ROI. **Instance IDs remain unchanged.** Check the exported resolution; this
portable image may have less raw detail than the source used for fine texture.
Choose a finer segmentation grid if that detail is needed in the portable viewer.

Follow the same preview, bounded ROI, ID-matching and reviewed export steps earlier
in this guide. The project tables preserve prediction evidence and coordinates;
optional feature tables include the real combined feature vectors, not just the
UMAP positions. The NPZ files and their metadata sidecars remain available in the
worker outputs for further quantitative analysis.

### Distributed fine-patch inputs

DINO embeddings made from **Prepare data → distributed_patches** retain native
object IDs and their original raw/instance volumes in `data_config.yaml`. Keep
that file and `preprocessing.json` with the results. MoBIE export uses those
original spatial volumes; `crops.h5`/`crops.n5` is a stack of isolated patches and
must not be used as the whole-animal segmentation. The prepared-image inspector
shows one representative patch, while original-image/mesh inspection returns to
the source object. See the [patch preparation walkthrough](workspace_workflows.md#distributed-fine-patches-for-dino)
for the sampling controls and coverage/exclusion reports.

## Bilateral probes and spatial context results

Use the new Workflow starting points to explore the published MorphoContextFeatures
map, compare bilateral-partner retrieval, or build context vectors from MAE/DINO.
The [workflow walkthrough](workspace_workflows.md#bilateral-retrieval-and-spatial-context)
explains the required data, graph choices, rank conventions and limitations.

Context output is an ordinary ID-preserving embedding. Analyze it to produce
coordinates, clusters and optional classifier predictions, then use the existing
**Export this analysis to MoBIE** action. Saved published UMAP/clusters can also go
through Analyze and the same MoBIE export without fitting a new map.

Context vectors and published bilateral candidates use **cell IDs**. A nucleus-ID
input is explicitly mapped, and `id_mapping.tsv` preserves its native embedding ID.
For nucleus-segmentation export, provide the corresponding cell-to-nucleus mapping
through the existing label-export/inspection settings; do not relabel a nucleus
volume by numeric overlap with cell IDs. Bilateral Results performs this conversion
when opening a selected query/candidate pair using the original embedding file.
Its ranks measure partner retrieval, not predicted cell types, and are exported as
separate per-cell tables rather than silently converted into segmentation classes.

Keep `context.json`, graph/coverage tables, ID bridges and embedding metadata with
exports. They explain which neighborhood definition and missing-neighbor policy
produced the vectors, and which original volume IDs an inspected object refers to.

### Export bilateral pairs in 3D

**What you need:** a completed `bilateral.json` with its result folders; a matching
cell/instance segmentation for MoBIE, or a physical position table for link geometry;
and an explicit cell-to-nucleus mapping if visualizing cell-ID results on nuclei.
Platy cell `default.tsv` supplies usable `anchor_z`, `anchor_y`, `anchor_x` columns
in micrometers. Anchors locate objects but are not necessarily their centroids.

1. In bilateral Results click **Export bilateral pairs in 3D**. You can instead add
   the export as a stage after a bilateral probe, using `from_bilateral: true`.
2. Choose a representation and variant, or leave them empty to export all groups.
   Enter query cell IDs for particular examples, or choose best/worst ranks. The
   default limit is 20 pairs **per representation and variant**. Each query is
   connected to its closest candidate in that saved result, not a newly predicted
   anatomical connection. The full probe table remains available independently.
3. Enable **Create a MoBIE project with 3D pair views** and choose published streaming
   or portable ROI export. The Platy default uses cell segments. Nucleus segments
   require the cell/nucleus mapping. Alternatively disable MoBIE and enter a position
   table with its Z/Y/X columns, unit (`um` or `nm`) and ID namespace.
4. Optionally enable **Also export query/partner masks as HDF5**. This requires
   portable mode and an explicit ROI. Pairs × ROI voxels must fit the total mask
   budget; the export checks this before copying image data. For large animals,
   streamed MoBIE pair views or link geometry are usually more practical than
   many full-volume masks.
5. Review and run locally or on SLURM. Results provides the archive and individual
   files. Browser downloads keep their size safeguards; copy large worker outputs
   directly. Read exclusion/coverage tables before interpreting a missing object.

| Output | Meaning and use |
| --- | --- |
| `bilateral_export.json` | Source result, export settings, position provenance, groups and exclusion counts. |
| `pairs-N/pairs.tsv` | Query/partner cell IDs, mapped segmentation IDs, rank/distance and physical endpoint coordinates. |
| `pairs-N/pairs.npz` | Exact integer IDs and `endpoints_zyx_um` (`pairs × 2 × 3`). `vectors_zyx_um` stores start position plus displacement for napari. All coordinates are µm. |
| `pairs-N/pairs.vtk` | Line geometry in physical **X/Y/Z µm** with retrieval rank per line, for ParaView or another VTK reader. |
| `pair_masks.h5` (optional) | `roles` array in **P/Z/Y/X** order; each P layer is a separate pair with 0 background, 1 query, 2 candidate. Attributes record volume unit, spacing and ROI-adjusted origin. Cell IDs are separate integer arrays. |
| `pair_mask_layers.tsv` | Layer order, pair/group identity and whether each endpoint actually appears in the ROI. |
| `excluded_pairs.tsv` | Selected pairs lacking positions or required ID mappings. Strict coverage can turn these omissions into an error. |
| `mobie/project/` (optional) | Whole MoBIE project, spatial sources, tables, pair views and provenance; retain the directory structure. |

**In MoBIE:** open the exported project folder containing `project.json` with Fiji's
MoBIE expert-mode project opener. Choose `mfN-bilateral-rank` to color spatial objects
by rank (lower is better), or `mfN-pair-CELL_ID` to select the two objects and enable
their 3D display. Each `mfN` corresponds to a representation/variant listed in the
export README. Pair views use MoBIE's
[selected-segment and 3D-display settings](https://mobie.github.io/specs/mobie_spec.html).
Missing or outside-ROI partners cannot be rendered. Connecting lines are exported
separately as VTK/NPZ; MoBIE pair views display the **segmented objects**.

**In napari:** add links in the same physical frame as the original volume:

```python
import numpy as np
import napari

viewer = napari.Viewer(ndisplay=3)
with np.load("pairs-0/pairs.npz", allow_pickle=False) as pairs:
    viewer.add_vectors(pairs["vectors_zyx_um"], name="Bilateral candidate links")
    viewer.add_points(pairs["endpoints_zyx_um"].reshape(-1, 3), size=0.3,
                      name="Query and candidate anchors")
# Add your raw/label array lazily, with scale=spacing_zyx_um and
# translate=origin_zyx_um. Cropped volumes need the ROI-adjusted origin.
napari.run()
```

To display a bounded pair-mask layer in napari (or load it as a label image through
an HDF5 reader in Fiji/ilastik):

```python
import h5py

with h5py.File("pair_masks.h5", "r") as masks:
    factor = 0.001 if masks.attrs["unit"] == "nm" else 1.0
    viewer.add_labels(masks["roles"][0], name="Query 1 / candidate 2",
                      scale=np.asarray(masks.attrs["spacing_zyx"]) * factor,
                      translate=np.asarray(masks.attrs["origin_zyx"]) * factor)
```

A cell may occur in several pairs, so masks intentionally do not collapse all pairs
into one competing label image. An endpoint absent from the selected ROI produces
an incomplete layer and is reported; enlarge the ROI or use streamed views. These
links represent a retrieval hypothesis, not a synapse, neurite or proven bilateral
correspondence. Do not interpret retrieval rank as a cell-type confidence score.
