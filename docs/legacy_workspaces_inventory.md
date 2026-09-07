# Legacy MAE workspaces and Platynereis patch inventory

This document records a read-only audit of four legacy workspaces made on
2026-09-01:

- `/g/kreshuk/zinchenk/mae`
- `/g/kreshuk/zinchenk/mae_nuclei`
- `/g/kreshuk/zinchenk/nuclei`
- `/g/kreshuk/mansaray`

The audit was intended to answer two practical questions: what the old experiments actually
trained, and which real data can support a maintained MorphoFeatures example. Paths in this
document are evidence from one cluster, not portable defaults. No file in an external workspace
was changed and no legacy script was launched. Metadata and bounded samples were read; large N5
containers and checkpoint series were not copied or exhaustively hashed.

## Conclusions first

The most defensible maintained input is the paired dataset
`/g/kreshuk/mansaray/raw_patches_masked.n5` and
`/g/kreshuk/mansaray/abs_crop_centers_radius4_only_nucl.n5`. It contains 2,506,460 masked
`32³` EM patches associated with 11,382 unique parent cell IDs. The positions table preserves the
parent ID in column zero and absolute low-resolution `(z, y, x)` crop centers in columns one to
three. Its ID list maps one-to-one onto the nonzero cell-to-nucleus associations in the bundled
Platynereis tables.

Each model output from these inputs is a **cell-associated, nucleus-derived texture embedding**.
It is not a complete cell embedding and not the historical 480-dimensional representation. The
latter was formed from six independently trained 80-dimensional groups: cell shape, nucleus
shape, coarse cell texture, coarse nucleus texture, fine cell texture, and fine nucleus texture.

The legacy nucleus work used a transformer across as many as 200 local patches per nucleus. Each
patch was first encoded by a 3D ResNet or linear projection; hidden **whole-patch tokens** were
decoded to `8³` downsampled patch targets. An initial maintained implementation incorrectly masked
internal `4³` blocks inside each stored `32³` patch. Real reconstructions exposed that sampling-unit
error even though its low MSE beat a scalar-mean baseline. The corrected
`grouped-nucleus-patches-v3` route now groups patches by parent, retains their relative `(z,y,x)`
positions, masks complete stored patches, and exports one sequence representation per `label_id`.
The rejected v1/v2 real-data checkpoints are not compatible with v3.

## Audit method and limits

Git remotes, commits, diffs, tracked files, README/license files, Python source, YAML/configuration
snapshots, SLURM logs, small checkpoint metadata, N5 `attributes.json`, z5py metadata, tables, and
bounded patch samples were inspected. Directory sizes were summarized without walking every N5
chunk when doing so would be prohibitively expensive. A small checkpoint was opened with
PyTorch's restricted `weights_only=True` loader after explicitly allow-listing
`argparse.Namespace` and `pathlib.PosixPath`; large checkpoints were not deserialized.

The Mansaray workspace is not a Git repository and its recorded source path under
`/home/mansaray` is inaccessible. Consequently, its producer code, license, and local source
modifications cannot be completely audited. Configuration files, logs, checkpoints, and data are
useful evidence, but filenames alone were not treated as proof. Confidence labels below mean:

- **high**: directly established from source plus data metadata/content;
- **medium**: established from multiple artifacts but a producer step or author record is missing;
- **low**: suggested by filenames or incomplete artifacts only.

## Code and data inventory

| Path or group | Owner workspace | Category | LM/EM | Biological content | Format/dataset keys | Shape/dtype/chunks | IDs and coordinates | Producer/consumer code | Evidence | Confidence | Recommended use |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `mae/` | Zinchenk | Upstream model code | natural-image, not biological | ImageNet-style RGB images | Python/Git | 2D patch embedding and 2D augmentations | class folders; no biological IDs | Meta MAE pretraining/fine-tuning scripts | origin `facebookresearch/mae`, commit `efb2a80`, Apache-2.0 license, Git diff | high | Cite as architectural history only; use maintained 3D implementation |
| `mae/models_mae.py`, `models_vit.py` | Zinchenk | Locally adjusted upstream code | not LM/EM-specific | 2D images | PyTorch | small model factory added; obsolete `qk_scale` removed | none | upstream MAE/ViT | direct diff from Git commit | high | Concepts only; not a 3D biological loader |
| `mae/main_*`, `util/*` | Zinchenk | Compatibility changes | not LM/EM-specific | 2D images | Python | timm assertion disabled; distributed/sampler changes | none | ImageFolder training | direct diff | high | Do not use for the real pipeline |
| `mae/output_dir` | Zinchenk | Generated output | unknown | no useful cohort discovered | small output folder | negligible | none established | upstream scripts | bounded directory inspection | high | Exclude |
| `mae_nuclei/` | Zinchenk | Incomplete upstream branch | not demonstrated as biological | nominally “nuclei” by folder name only | Python/Git | same 2D Meta MAE core; no N5/nucleus loader | no biological IDs | reduced Meta MAE clone | same origin/commit/license and local diff as `mae`; fine-tune/demo files deleted | high | Do not infer nucleus training from name; exclude from maintained path |
| `nuclei/nuclei_utils/nuclei_loader.py` | Zinchenk | Biological data loader | EM, with separate Planaria code elsewhere | complete nuclei represented by sequences of local texture patches | Python; N5 and TSV consumers | up to 200 patches/nucleus; raw patches `32³` | parent cell ID, nucleus ID mapping, low-resolution `(z,y,x)` centers | training/prediction scripts | direct source inspection | high | Provenance and sampling reference; adapt concepts, not code |
| `nuclei/nuclei_utils/nuclei_models.py` | Zinchenk | Sequence MAE/ViT model | EM | one nucleus as a patch sequence | PyTorch | transformer usually 768-D; 3D patch encoder or feature-token input | sequence mask preserves non-empty patch positions | `scripts/train.py`, prediction scripts | direct source; header says modified from Meta MAE | high | Architectural comparison only; repo has no license |
| `nuclei/nuclei_utils/resnet_encoder.py` | Zinchenk | Patch encoder | EM | local raw nucleus texture | PyTorch 3D ResNet | `32³` input, optional reconstruction downsampling | no ID logic inside model | sequence model | direct source; header says modified from torchvision | high | Conceptual baseline only; do not copy without license clarification |
| `nuclei/scripts/train.py` and cluster launchers | Zinchenk | Training orchestration | EM | nucleus texture | Python/SLURM | common runs: 200 epochs, batch 32, mask ratio 0.9, seed 0 | samples all 11,382 nuclei with random sampler | loader/model/checkpoints/WandB | source, args, and logs | high | Historical hyperparameter evidence; not a validated maintained default |
| `nuclei/scripts/predict*.py` | Zinchenk | Encoding/evaluation | EM | nucleus texture | Python/NumPy | averages non-empty patch-token features | exports `np.c_[cell_labels, features]` | legacy checkpoints | source inspection | high | Preserve label-first and per-parent aggregation contracts |
| `nuclei/scripts/save_patches_separately.py` | Zinchenk | Patch producer | EM | masked local nucleus texture | N5 producer | `patches`: `(2506460,32,32,32)`, `uint8`, chunks `(1,32,32,32)` | row order matches position table | reads raw s1 and low-resolution nucleus mask | source plus matching current metadata | high | Primary producer evidence for selected patch store |
| `nuclei/data/hr_nucleus_cell_id_3.npy`, `lr_nucleus_cell_id_3.npy` | Zinchenk | Test/development fixtures | EM | single nucleus/cell examples | NumPy | approximately 163 MB and 290 KB | single example, not cohort IDs | loader tests | file metadata and tests | high | Tests/QC only, not model training cohort |
| `nuclei` Planaria scripts/data references | Zinchenk | Cross-dataset experiments | EM, different species | Planaria nuclei | Python/N5 references | distinct layout | species-specific | `planaria_loader.py`, `other/convert_planaria_data.py` | direct source | high | Exclude from Platynereis example |
| `nuclei/old_trainings`, `trainings_*`, `delete`, `contr*`, `regular` | Zinchenk | Generated training series | mostly EM | nucleus texture experiments | checkpoints/logs/images | grouped size about 353 GB outside raw-patch series | IDs/config embedded inconsistently | training scripts | directory sizes, args, logs, checkpoints | medium | Summarize as historical outputs; do not copy or enumerate each epoch |
| `nuclei/trainings_raw_patches` | Zinchenk | Generated training series | EM | raw masked nucleus patches | checkpoints/logs/reconstructions | about 3.5 TB | parent IDs implied by loader | raw-patch training | directory inspection | medium | Retain externally; inspect selected metadata only |
| representative `nuclei/other/slurm.41563064.out` | Zinchenk | Training log | EM | raw nucleus patches | text | 200 epochs; ~432M parameters; seed 2; mask 0.9; normalized-pixel loss | bilateral-neighbor evaluation uses parent IDs | legacy train/predict | direct log inspection | high | Historical evidence only; proxy metric is not biological validation |
| `raw_patches_masked.n5/patches` | Mansaray | Authoritative candidate model input | EM | nucleus-masked raw texture patches | N5 key `patches` | `(2,506,460,32,32,32)`, `uint8`, chunks `(1,32,32,32)`, gzip level 5 | row aligns with positions table | producer lineage matches Zinchenk save script; consumed by configs | N5 metadata, bounded content, source comparison | high | **Selected maintained real-data input** |
| `abs_crop_centers_radius4_only_nucl.n5` | Mansaray | Authoritative candidate index | EM | cell-associated nucleus patch locations | N5 keys `positions`, `ids` | positions `(2,506,460,4)` `int64`, chunks `(512,4)`; ids `(11,382,)` `int64` | `(cell_id,z,y,x)` absolute low-resolution grid; IDs unique | generated after candidate-center selection; consumed by legacy loader | full positions/ID validation and producer lineage | high for contents; medium for missing absolute-conversion step | **Selected maintained index** |
| `copied/raw_patches_masked.n5`, `copied/abs_*`, related `platy_data` copies | Mansaray | Candidate duplicate data | EM | same apparent patch/index content | N5 | sampled chunks share size/mtime but have different inodes | apparent same ID space | copied training environment | sampled metadata/content comparison | medium | Do not prefer; exact whole-store identity not proven |
| `platy_data/1.0.1/tables/...cells/cells_to_nuclei.tsv` | Mansaray | Biological mapping table | EM | cell-to-nucleus association | TSV | 32,699 cell rows plus header; 11,382 nonzero mappings | cell and nucleus label IDs | legacy loader and analysis | full table read and ID set comparison | high | Join annotations by `label_id`; preserve externally configured path |
| `platy_data/1.0.1/tables/...cells/default.tsv` | Mansaray | Biological annotation table | EM | cells, positions, types, metadata | TSV | 32,699 rows | cell label ID and physical/table coordinates | downstream analysis | headers/content inspection | high | Optional label-first downstream annotation join |
| `platy_data/1.0.1/tables/...nuclei/default.tsv` | Mansaray | Biological annotation table | EM | nuclei | TSV | 11,497 rows | nucleus label IDs | mapping/analysis | headers/content inspection | high | Optional provenance and nucleus lookup |
| `platy_data/1.0.1/tables/...cells/symmetric_cells.tsv` | Mansaray | Biological relation table | EM | bilateral/symmetric cells | TSV | 32,570 rows | cell IDs/pairs | bilateral evaluation | table and analysis inspection | high | Optional evaluation; not a train/test target by default |
| `raw_patches/` and cell-specific visualization arrays | Mansaray | Generated examples/QC | EM | individual cell/nucleus patches | N5/NumPy/PNG | variable; examples such as cell 7685 | often one cell ID | training/reconstruction inspection | bounded file and image inspection | medium | Representative QC only, not cohort input |
| `lm_platy`, LM configs and LM data references | Mansaray | Light-microscopy experiments | LM | light-microscopy nuclei/cells | N5/configs/checkpoints | separate patch sizes/resolutions | LM-specific identifiers | configs have `LightMicroscope: true` | direct config inspection | high | Exclude from selected EM pipeline; document separately if revived |
| EM training configs under `trainings` and `copied/trainings` | Mansaray | Experiment configuration/output | EM when `LightMicroscope: false` | nucleus texture | YAML/checkpoints/logs | ViT Tiny/Small/Base; 3D ResNet/linear tokenizers; 200/400/1000 epochs | position and mapping paths in config | missing external source package | direct configs/logs | medium-high | Hyperparameter and ablation evidence; not canonical maintained configs |
| LM training configs under the same groups | Mansaray | Experiment configuration/output | LM when `LightMicroscope: true` | LM samples | YAML/checkpoints/logs | separate model/data choices | LM paths in config | missing source package | direct configs/logs | medium-high | Exclude from EM example |
| `trainings` and `copied/trainings` checkpoint series | Mansaray | Generated model outputs/candidate duplicates | mixed LM/EM | MAE/ViT/ResNet nucleus experiments | PyTorch checkpoint dictionaries | about 1.9 TB and 1.4 TB respectively; individual files ~0.7–1.9 GB common | args may contain input paths and IDs indirectly | missing local source | directory metadata, configs, sampled safe checkpoint | medium | Group by run; do not load wholesale or claim compatibility |
| small sampled legacy checkpoint | Mansaray | Checkpoint-format evidence | unknown/toy | model experiment | PyTorch dict | 252 model tensors in inspected sample | no label-first table inside | expected keys `args`, `epoch`, `model`, `optimizer`, `scaler`, WandB fields | restricted safe load | high for format | Metadata audit only; maintained MAE architecture is different |
| `analysis/` and biological analysis subtrees | Mansaray | Downstream analysis | mostly EM | morphology, gene expression, bilateral pairs, nephridia | Python/tables/plots | heterogeneous | joins appear ID-based but require script-specific review | post-training consumers | bounded source/file inspection | medium | Potential annotation/QC sources; not MAE producers |
| `submit_gpu.py` | Mansaray | Legacy scheduler helper | any | none | Python/SLURM | hard-coded account/nodes/email and `sbatch` subprocess | none | job launcher | direct source | high | Do not reuse; maintained safe scheduler supersedes it |
| environment/package/cache directories | both | Reproducibility residue/cache | mixed | none directly | conda/pip caches and packages | very large/repetitive | none | historical environments | directory inspection; some permissions denied | low-medium | Intentionally summarized; never treat as scientific input |

## Dataset relationships and patch provenance

The best-supported lineage is:

1. A low-resolution nucleus mask and cell-to-nucleus mapping identify the nucleus belonging to a
   cell.
2. Candidate centers are selected inside the nucleus. The inspected producer applies an
   occupancy threshold of 0.4 and caps candidate positions in an earlier stage.
3. Relative candidate centers are converted to absolute low-resolution coordinates. The output
   script for this conversion was not present. A checked example differed from its relative row by
   a constant cell bounding-box offset, which supports—but does not fully prove—the interpretation.
4. Each low-resolution coordinate is multiplied by four on every axis to index raw pyramid `s1`.
5. A `32³` raw patch is extracted and multiplied by an upsampled nucleus mask.
6. The row is written to `patches`; `(cell_id,z,y,x)` is written at the same row in `positions`.
7. Training samples local patches. Encoding aggregates local patch representations back to one
   parent cell ID and writes a label-first feature table.

The legacy source declares native EM resolution `(0.025,0.010,0.010)` µm in `(z,y,x)`. Raw `s1`
is downsampled in-plane, yielding patch resolution `(0.025,0.020,0.020)` µm. The center table is on
the grid before the all-axis factor-four conversion, so its derived resolution is
`(0.100,0.080,0.080)` µm. The stored position coordinates and patch array axes are explicitly
handled as `(z,y,x)` by the Python loaders. This resolution chain is supported by source and
metadata but should still be confirmed with the original author because the absolute-center
conversion program is missing.

## Integrity and bounded quality control

Full metadata/index checks, which did not read the patch payload, established:

- 2,506,460 position rows and exactly the same number of patches;
- 11,382 unique `ids`, all finite, positive, integer-valued, and unique;
- exact equality between the unique position parent IDs and the `ids` dataset;
- monotonically grouped parent IDs and no duplicate full `(id,z,y,x)` rows;
- patch counts per parent: minimum 35, median 212, 95th percentile 324, maximum 1,140;
- exact set agreement between selected parent IDs and nonzero entries of the cell-to-nucleus table;
- coordinate range `(20,344,518)` through `(2812,2960,3046)` in `(z,y,x)`.

A deterministic seed-42, 80/10/10 split by **parent ID**, not patch row, produces 9,106/1,138/1,138
IDs and 2,005,268/252,212/248,980 patch rows. The three ID sets have no overlap. This grouped split
is essential: a random patch-level split would put texture from the same nucleus on both sides of
the evaluation boundary.

A bounded stratified read of 256 patches (four patches from each of 64 deterministic parent IDs)
found 56 all-zero patches (21.9%). Nonzero fractions ranged from zero to one, with median about
0.555. Content-duplicate detection in this sample found 55 duplicates, all attributable to the
all-zero payload. This is a sampling estimate, not a global percentage. Parent-level spot checks
also varied widely, from no empty patches to 80 empty patches among 128 reads. The maintained
loader records foreground fraction and, when training/encoding a bounded number per parent,
deterministically replaces an empty/near-empty candidate with another patch from the same parent.
It never silently changes the biological ID.

These patches are already nucleus-masked: zero is both outside-mask background and a possible raw
intensity. The nonzero mask is therefore useful but not an exact independent segmentation mask.
The notebook displays this limitation and does not call a nonzero-derived outline ground truth.

## Model and objective comparison

| Aspect | Verified legacy nucleus model | Maintained grouped-patch MAE v3 |
|---|---|---|
| biological sample | one nucleus as up to 200 local patches | one nucleus as up to configured `group_size` local patches |
| local encoder | 3D ResNet or linear `32³ → embedding_dim` projection | maintained linear `32³ → embedding_dim` projection |
| masking unit | whole local patch token in a nucleus sequence | whole local patch token in a nucleus sequence |
| position | fixed 11³ grid centered using nucleus-table COM | relative indexed coordinates centered by within-parent median |
| target | each hidden `32³` patch downsampled to `8³` | configurable downsample, audited default `8³` |
| output | nucleus-level encoder/decoded-token feature variants | one nucleus-level encoder CLS representation keyed by `label_id` |
| validation | many legacy runs sampled all nuclei and logged training loss only | deterministic parent-ID train/validation/test groups |
| checkpoint | legacy model/optimizer/scaler/args dict | portable maintained checkpoint plus resolved YAML/metrics/artifact metadata |
| output meaning | fine nucleus texture candidate | nucleus-derived texture embedding; still one feature group |

The maintained path uses `uint8` dtype normalization by default (`value / 255`) because that
matches the legacy loader, followed by optional per-target-patch normalized-pixel loss as in the
best-documented Mansaray configurations. Spatial augmentation is disabled: patch arrays and their
physical coordinates must undergo one coherent transform, and anisotropic axes cannot be swapped
casually.

## Checkpoints, duplicates, and storage implications

The Zinchenk nucleus repository occupies roughly 4.0 TB, dominated by approximately 3.5 TB of
raw-patch training outputs. Other training families range from a few GB to roughly 172 GB each.
Mansaray `trainings` and `copied/trainings` account for roughly 1.9 TB and 1.4 TB. Listing or
hashing every checkpoint would be expensive and scientifically unhelpful, so epoch series,
reconstruction images, logs, and caches are grouped in the inventory.

The selected patch store is about 82.1 GB uncompressed (`2,506,460 × 32³` bytes) and contains one
compressed chunk per patch. Millions of tiny filesystem objects make recursive traversal and
random I/O costly even when the byte count is manageable. Training reads a bounded central group
per parent and uses worker-local lazy N5 handles. Inventory discovery stops at
dataset metadata and never descends into chunk directories.

Root, `copied`, and `platy_data` variants have matching sampled metadata, sizes, and modification
times, but are distinct physical files. They are documented as **candidate duplicates**, not
declared bit-identical. The root paired inputs are selected because their relationship is clear
and both paths are accessible.

Legacy checkpoints are not loaded into the maintained MAE: tensor names and some architecture
details still differ despite the aligned sampling unit. They may support a future frozen-baseline comparison after the exact
source revision and license are recovered. Loading any untrusted PyTorch checkpoint should use
restricted loading and explicit allow-lists; arbitrary pickle loading is not part of this
pipeline.

## Why this dataset was selected

Compared with re-extracting a small spatial ROI from the whole-animal segmentation, the selected
store provides a much larger, author-derived sampling of 11,382 nuclei, complete stable parent
IDs, a verified row-level index, consistent patch dimensions, practical bounded reads, and a
clear relationship to the original nucleus-texture experiments. It is compatible with the
maintained MAE without importing legacy code. The earlier ROI workflow remains useful for showing
raw-volume-to-cell-crop preparation, but it is not the primary real nucleus-texture example.

The full maintained profile uses all 11,382 IDs, up to 200 central patches per parent, 80 output
dimensions, and 200 epochs. This matches the documented sample width and common legacy run length;
it is a starting budget, not evidence that the maintained model will converge. A real run must be
judged from held-out loss, reconstructions, embedding dispersion,
stability across checkpoints/seeds, and appropriate ID-joined downstream analyses.

## Superseded and current execution records

The earlier single-patch v2 notebook was executed against the selected external stores on 2026-09-01
with the `MorphoFeats_dev` environment. PyTorch was built with CUDA 13.0 support, but this session
exposed zero GPU devices and NVML could not initialize, so the run used CPU. No SLURM job was
submitted.

The deterministic superseded profile selected 24 parent IDs: 19 training, 2 validation, and 3 test.
It used two patches per ID for the split datasets, four patches per ID for encoding, a 32-D,
two-layer MAE, and two epochs. Six representative masked patches agreed exactly with the optional
aligned raw source at every nonzero voxel. Training loss changed from about 0.366 to 0.186 and
validation loss from about 0.251 to 0.114. Those numbers validate its software loop only. Because
it masked internal blocks inside already-local texture tokens, they are not evidence for the
correct nucleus-group objective and the artifacts must not be resumed as v3.

Artifacts under `outputs/notebooks/04_real_nucleus_mae/quick_grouped_v3/` include the resolved YAML, split
manifest, append-only metrics, epoch/final checkpoints, run metadata, a finite label-first
`(24,33)` array, embedding metadata, an ID-joined analysis table, and a MoBIE-compatible TSV. The
SLURM cell successfully rendered the argument list and script with placeholders and printed
“Not submitted. Preview only.” A descriptive distance-rank comparison with the bundled published
fine-nucleus group used 24 common IDs and gave approximately -0.08; it is deliberately presented
as evidence that two-epoch features are not a biological result.

After correction, a bounded v3 CPU run read 12 real parent IDs with four central patches each,
trained an 8-dimensional one-layer encoder/decoder for two epochs, saved/reloaded a portable
checkpoint, reconstructed complete hidden patches at `8³`, and exported a finite label-first
`(12,9)` array. Its sampled hidden-patch normalized MSE was approximately 1.30 versus 1.85 for the
visible-patch-mean baseline. Artifacts were written only to
`/tmp/morphofeatures_grouped_v3_final_smoke`. This validates the corrected I/O and objective
mechanics, not convergence or biological utility. CUDA was unavailable in the current process and
no SLURM job was submitted; the configured quick/full v3 runs remain unexecuted.

## Broken, ambiguous, or incomplete items

- The absolute-center conversion producer is missing from the inspected source history.
- The Zinchenk nucleus repository has no repository-level license; only the separate Meta clones
  carry Apache-2.0. Large legacy implementation blocks were therefore not copied.
- Mansaray model source and license could not be recovered from the accessible workspace.
- Some output families contain interrupted logs, incomplete checkpoint sequences, or only
  training loss. A directory's existence is not a successful-run certificate.
- Some N5 names and duplicated folders are historical aliases; complete equality has not been
  established.
- The apparent “VAE” lead was not verified in accessible project source or authoritative configs.
  The visible evidence is primarily MAE/ViT/3D-ResNet experimentation.
- A Meta-style 2D `mae_nuclei` folder exists, but it contains no verified nucleus loader or 3D
  modification.
- No independent animal/batch split is available in the selected data. Held-out nuclei from one
  animal test interpolation, not cross-animal biological generalization.

## Questions for the original authors

1. Which program converted relative nucleus candidate positions to `abs_crop_centers_*`, and
   which segmentation/table versions did it read?
2. Does every zero-valued patch mean failed/empty extraction, or can valid masked nucleus
   interiors be entirely zero after raw preprocessing?
3. Which of the root, `copied`, and `platy_data` stores is the designated immutable release?
4. Which checkpoint/config pair was considered the final nucleus result, and was it used in any
   published analysis?
5. Was the intended contribution specifically fine nucleus texture, or a prospective replacement
   for other MorphoFeatures groups? Source structure supports the former, but an author statement
   would settle intent.
6. Which resolution and orientation augmentations were scientifically approved, and should
   absolute body-axis orientation be retained?
7. Are there animal/batch identifiers or exclusion lists that could support a stronger split?
8. Under what license may the `platy_nuclei_texture` and inaccessible Mansaray model source be
   reused?

## Reproducing the metadata audit

The maintained command below writes JSON metadata without reading patch chunks:

```bash
morphofeatures n5-inventory \
  /g/kreshuk/mansaray/raw_patches_masked.n5 \
  /g/kreshuk/mansaray/abs_crop_centers_radius4_only_nucl.n5 \
  --output outputs/inventory/platynereis_n5.json
```

Use the site configuration
`configs/sites/mae_platynereis_nuclei_embl.yaml` only on a system where those external paths are
mounted. Use `configs/mae_nucleus_patches_template.yaml` elsewhere and supply paths through
environment variables or a private site YAML. Large external data remain external and all
generated artifacts belong under the configured output root.
