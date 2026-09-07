# Local Platynereis data audit

> This document covers the earlier whole-volume ROI route. The primary real nucleus-texture
> example now uses the much larger, ID-indexed masked-patch store documented in
> [Legacy MAE workspaces and Platynereis patch inventory](legacy_workspaces_inventory.md).
> The ROI route remains useful when explaining how crops are derived from raw and segmented
> volumes, but it is not the selected training cohort.

This is a read-only audit of the cluster data made available for the real-data tutorial on
2026-08-31. The audited location was
`/g/kreshuk/data/arendt/platyneris_v1` (the folder name omits the second "e" in
*Platynereis*). Repository code does not hard-code this location; set
`MORPHOFEATURES_PLATYNERIS_ROOT` to the corresponding root on a given system.

No source data were modified. Generated crops, manifests, checkpoints, metrics, embeddings, and
the executed notebook were written below `outputs/notebooks/04_platynereis_mae/`, which is
ignored by Git.

## Provenance found in the local README

The folder README describes data derived from a full six-day *Platynereis dumerilii* larva. It
identifies ssTEM resolution as 25 × 10 × 10 nm in `(z, y, x)` and associates the data with
"Whole-body integration of gene expression and single-cell morphology" (Cell, 2021,
DOI 10.1016/j.cell.2021.07.017). It also points to a separate complete MoBIE project and raw-data
location on the Arendt share. These external locations and the exact historical MorphoFeatures
configuration were not assumed to be part of the permitted audit scope.

## Top-level inventory and relevance

| Entry | What was inspected | Decision for this tutorial |
|---|---|---|
| `README.md` | provenance, modalities, native resolution | authoritative local starting point |
| `data.n5` | raw pyramid, cell/nucleus labels, mappings, morphology, overlaps | selected source |
| `trained_networks` | segmentation-network configs, weights, and logs | not MorphoFeatures embedding checkpoints |
| `training_data` | segmentation training data for membrane, nuclei, tissue, cilia, and cuticle | not a whole-cell embedding cohort |
| `exp_data` | lifted-multicut/multicut graph and blockwise intermediate stores | segmentation internals, not direct MAE inputs |
| `fib_registration` | HDF5/N5 registration landmarks for FIB-SEM/SBEM | different registration task |
| `validation` | segmentation validation projects and label volumes | useful for segmentation evaluation, not selected here |
| `tracings` | neuron tracing/NMX material | not a cell-volume input contract |

The audit intentionally inspected metadata and bounded blocks rather than recursively reading the
multi-terabyte N5 chunks.

`data.n5/volumes/raw` is a symbolic link to the Arendt-share raw store
`/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/rawdata/sbem-6dpf-1-whole-raw.n5/setup0/timepoint0/`.
The tutorial reads it through the permitted `data.n5` path. A copied `data.n5` directory without
an equivalent raw link will therefore not be self-contained; configure or recreate the link
explicitly rather than assuming the current cluster layout.

## N5 datasets inspected

N5 stores dimension/downsampling attributes in `(x, y, z)`, while `z5py` exposes NumPy arrays in
`(z, y, x)`. The table below reports the exposed array order.

| Dataset | Shape `(z,y,x)` or table shape | dtype | Interpretation and decision |
|---|---:|---:|---|
| `volumes/raw/s1` | `(11416,12958,13750)` | `uint8` | raw intensity; factors `[2,2,1]` in N5 `(x,y,z)` |
| `volumes/raw/s3` | `(2854,3240,3438)` | `uint8` | selected raw scale; effective `(0.10,0.08,0.08)` µm `(z,y,x)` |
| `volumes/paintera/proofread_cells/data/s2` | `(2854,3240,3438)` | `uint64` | selected multiscale **fragment** labels, aligned to raw `s3` |
| `node_labels/curated_lmc/filtered_unmerge` | `(18198043,)` | `uint64` | selected fragment-to-final-cell mapping; max final ID 31731 |
| `volumes/segmentation/curated_lmc/result` | `(11416,12958,13750)` | `uint64` | unfiltered agglomeration, max ID 1,641,758 |
| `volumes/segmentation/curated_lmc/filtered_size` | same | `uint64` | filtered cells, max ID 30,001 |
| `volumes/segmentation/curated_lmc/filtered_unmerge_tmp` | same | `uint64` | temporary unmerge result, max ID 31,807 |
| `volumes/segmentation/curated_lmc/filtered_unmerge` | same | `uint64` | final full-resolution cell labels, max ID 31,731 |
| `volumes/paintera/nuclei/data/s0` | `(2854,3240,3438)` | `uint64` | nucleus fragments/labels at the selected raw scale, max ID 11,498 |
| `volumes/nuclei/mws_mc_biased_filtered` | `(2854,3240,3438)` | `uint64` | filtered nucleus segmentation candidate |
| `morphology/curated_lmc/result` | `(1641759,11)` | `float64` | rows encode label, size, center, min, and max for the unfiltered result |
| `morphology/nuclei_filtered` | `(11499,11)` | `float64` | corresponding nucleus morphology rows |
| `morphology_curated_lmc_filtered_unmerge` | nominally full-volume shape | `float64` | variable-length derived chunks; not readable as a normal array with installed `z5py`; rejected |
| `label_overlaps_lmc_filtered_size` | nominally full-volume shape | `uint64` | variable-length derived chunks; not used as segmentation input |
| `nuclei_overlaps/curated_lmc_filtered_size` | nominally `(30002,)` | `uint64` | variable-length chunk; installed `z5py` cannot read it as an ordinary vector |

The raw pyramid was inspected from `s1` through `s9`; the cell-fragment pyramid from `s0` through
`s8`; and the nucleus pyramid from `s0` through `s6`. Raw `s3`, cell fragments `s2`, and nuclei
`s0` share the same array shape. The real tutorial uses only cell texture, so it does not infer a
cell–nucleus association from spatial overlap.

## Why the selected mapping route is suitable

The selected route reads a bounded block from raw `s3` and cell-fragment `s2`, then indexes the
one-dimensional `filtered_unmerge` node-label mapping with each fragment ID. The mapping has
18,198,043 entries, 31,732 unique outputs including background, and maximum final cell ID 31,731.
Those final IDs occupy the same numeric ID space as the bundled published feature tables.

A small comparison between the mapped low-resolution pyramid and every-fourth-voxel sampling of
the full-resolution final segmentation agreed at 89.5% of voxels. This is not evidence of an
alignment error: stored label pyramids use label-aware downsampling and are not equivalent to
naive striding. A workflow must choose and document one pyramid representation consistently.

## Bounded ROI used for the executable run

| Property | Value |
|---|---|
| raw/fragment origin `(z,y,x)` | `(1584,1492,1591)` at the selected scale |
| ROI shape | `(256,256,256)` voxels |
| resolution | `(0.10,0.08,0.08)` µm |
| physical extent | `(25.6,20.48,20.48)` µm |
| raw range observed | 14–232 (`uint8`) |
| distinct non-background final IDs | 136 |
| complete objects with at least 1,000 ROI voxels | 41 |
| selected fixed crops | 31 × `32³` after ≥90% crop coverage |
| median crop coverage | 0.994 |
| median foreground fraction | 0.065 |
| labels also present in published coarse-cell features | 7 of 31 |

The crop window spans `(3.2,2.56,2.56)` µm. Its `8³` MAE patch spans
`(0.8,0.64,0.64)` µm. Intensities are converted from the `uint8` dtype range to `[0,1]`, then
zeroed outside the target cell. The separately saved binary mask restricts MAE reconstruction
loss to cell-interior voxels so sparse background cannot dominate the objective.

## Actual execution record

The `MorphoFeats_dev` environment contained PyTorch 2.13.0+cu130 and the required N5/scientific
libraries. Its CUDA build was present, but no GPU/NVML device was exposed to this session, so the
run used CPU. The 10-epoch, 31-cell tutorial produced:

- a portable checkpoint at `outputs/notebooks/04_platynereis_mae/training_foreground/checkpoint.pt`;
- train loss decreasing from approximately 0.400 to 0.198;
- validation loss decreasing from approximately 0.439 to 0.215;
- a finite label-first `(31,33)` array (31 IDs plus 32 learned features);
- exact equality between preprocessing IDs and encoded IDs;
- an executed notebook with rendered QC figures in the same output directory.

The run proves that the bounded real-data path executes. It does not establish convergence or
biological generalization. Only seven cells overlap the historical feature cohort, and their
descriptive new-versus-published pairwise-distance correlation was about 0.20 (`p≈0.39`). The
notebook labels this comparison as underpowered rather than treating it as evidence.

## Information still missing for a full migration

- the authoritative complete cell-to-nucleus mapping for final unmerged IDs;
- the published MoBIE cell/nucleus default tables with physical bounds and anchors;
- exact historical crop transforms, selection exclusions, train/validation membership, and
  checkpoints;
- independent animals or batches for biological generalization;
- a GPU visible to the current execution session.

Do not claim exact historical retraining from this audited folder alone. For a full experiment,
recover the authoritative tables, document segmentation and inclusion versions, select cells
across the whole animal without ROI-completeness bias, and compare legacy and MAE objectives on
identical label cohorts and downstream tasks.
