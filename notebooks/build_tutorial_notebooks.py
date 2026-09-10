"""Build the narrative tutorial notebooks from readable cell sources.

Run this file from any directory after installing the ``notebooks`` extra.  Keeping
the cell sources here makes review and maintenance substantially easier than editing
large notebook JSON documents by hand.
"""

from pathlib import Path

import nbformat as nbf
from build_real_mae_notebook import build as build_real_mae_notebook

HERE = Path(__file__).resolve().parent


def md(source):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source):
    return nbf.v4.new_code_cell(source.strip())


def write(name, cells):
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9"},
        },
    )
    nbf.write(notebook, HERE / name)


write(
    "01_data_preparation_and_contracts.ipynb",
    [
        md(
            """
# 1. See the data before training anything

MorphoFeatures starts *after* a 3D EM volume has been segmented. This tutorial slows
that hand-off down: we will inspect every input role, view the arrays in `(z, y, x)`,
check label alignment, and only then make a train/validation split.

**Questions answered here**

1. What do raw, cell, and nucleus arrays contain?
2. How do voxel coordinates become physical distances?
3. Why is `label_id` a join key rather than a row number?
4. Which checks should fail before a long cluster job is submitted?

The data are synthetic, so this notebook is safe on a laptop. Notebook 04 repeats the
same inspection on a bounded real Platynereis ROI.
"""
        ),
        md(
            """
## Step 0 — Reproducible setup

Run Jupyter from the repository root. Package helpers resolve the repository and output
root without relying on the notebook's current directory. The fixture is generated
beneath the configured output root; source data are never overwritten.
"""
        ),
        code(
            """
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from morphofeatures.config import load_config, repository_root
from morphofeatures.data.crops import describe_labels
from morphofeatures.data.io import load_embeddings
from morphofeatures.data.synthetic import save_synthetic_dataset

SEED = 7
random.seed(SEED)
np.random.seed(SEED)

REPO_ROOT = repository_root()
CONFIG = load_config(REPO_ROOT / "configs" / "default.yaml")
OUTPUT_DIR = CONFIG.paths.output_root / "notebooks" / "01_data_contracts"
FIXTURE_DIR = save_synthetic_dataset(OUTPUT_DIR / "synthetic", seed=SEED)
print("Repository:", REPO_ROOT)
print("Generated fixture:", FIXTURE_DIR)
"""
        ),
        md(
            """
## Step 1 — Identify the five input roles

| Input | What a voxel/row means | Essential rule |
|---|---|---|
| raw EM | image intensity | numeric and finite |
| cell segmentation | cell instance ID | integer; `0` is background |
| nucleus segmentation | nucleus instance ID | integer; `0` is background |
| cell–nucleus mapping | one biological association | join explicit IDs |
| metadata | annotation for one label | must contain `label_id` |

A real project may store the three volumes at different pyramid levels. Equal array
indices then do **not** imply equal physical positions: dataset key and resolution must
travel together.
"""
        ),
        code(
            """
raw = np.load(FIXTURE_DIR / "raw.npy")
cells = np.load(FIXTURE_DIR / "cells.npy")
nuclei = np.load(FIXTURE_DIR / "nuclei.npy")
mapping = pd.read_csv(FIXTURE_DIR / "cell_to_nucleus.tsv", sep="\t")
metadata = pd.read_csv(FIXTURE_DIR / "metadata.tsv", sep="\t")

summary = pd.DataFrame(
    [
        ("raw", raw.shape, str(raw.dtype), float(raw.min()), float(raw.max()), len(np.unique(raw))),
        ("cells", cells.shape, str(cells.dtype), int(cells.min()), int(cells.max()), len(np.unique(cells))),
        ("nuclei", nuclei.shape, str(nuclei.dtype), int(nuclei.min()), int(nuclei.max()), len(np.unique(nuclei))),
    ],
    columns=["array", "shape_zyx", "dtype", "minimum", "maximum", "unique_values"],
)
display(summary)
display(mapping)
display(metadata)
"""
        ),
        md(
            """
## Step 2 — Look at aligned slices

The first array axis is **z** (section/depth), followed by **y** (row) and **x**
(column). The panels below use the same z index. Cell outlines should coincide with
image structures and nuclei should fall inside their mapped cells. A displaced overlay
is an alignment error, not something a model should be expected to repair.
"""
        ),
        code(
            """
z = raw.shape[0] // 3
cell_boundaries = cells[z] != np.roll(cells[z], 1, axis=0)
cell_boundaries |= cells[z] != np.roll(cells[z], 1, axis=1)

fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
axes[0].imshow(raw[z], cmap="gray")
axes[0].set_title(f"raw intensity, z={z}")
axes[1].imshow(cells[z], cmap="tab20", interpolation="nearest")
axes[1].set_title("cell label IDs")
axes[2].imshow(nuclei[z], cmap="tab20", interpolation="nearest")
axes[2].set_title("nucleus label IDs")
axes[3].imshow(raw[z], cmap="gray")
axes[3].contour(cell_boundaries, levels=[0.5], colors="cyan", linewidths=0.8)
axes[3].contour(nuclei[z] > 0, levels=[0.5], colors="magenta", linewidths=0.8)
axes[3].set_title("raw + cell/nucleus outlines")
for axis in axes:
    axis.set_axis_off()
plt.show()
"""
        ),
        md(
            """
### What this view can and cannot establish

It can reveal gross axis swaps, offsets, background mistakes, and obviously misplaced
nuclei. One slice cannot establish 3D segmentation quality. Inspect orthogonal planes,
small/large objects, disconnected fragments, and suspected merges in a volume viewer
before training on real data.
"""
        ),
        code(
            """
label_table = describe_labels(cells)
resolution_zyx_um = np.array([0.10, 0.05, 0.05])  # illustrative fixture resolution
for axis, resolution in zip("zyx", resolution_zyx_um):
    label_table[f"extent_{axis}_um"] = (
        label_table[f"bb_max_{axis}"] - label_table[f"bb_min_{axis}"]
    ) * resolution

display(label_table[[
    "label_id", "voxel_count_roi", "touches_roi_border",
    "bb_min_z", "bb_min_y", "bb_min_x", "bb_max_z", "bb_max_y", "bb_max_x",
    "extent_z_um", "extent_y_um", "extent_x_um",
]])
"""
        ),
        md(
            """
## Step 3 — Inspect one biological unit in 3D

The next view isolates cell 1. The masked raw crop is the conceptual input to a coarse
cell-texture model: intensity inside one segmentation object, zero outside. Shape models
would instead derive a surface/point cloud from the binary mask. Fine-texture models
would sample multiple smaller high-resolution patches and aggregate them back to this
same `label_id`.
"""
        ),
        code(
            """
label_id = 1
mask = cells == label_id
masked_raw = raw * mask
centers = np.rint(np.argwhere(mask).mean(axis=0)).astype(int)

fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
views = [
    (raw[centers[0]], mask[centers[0]], masked_raw[centers[0]], "z"),
    (raw[:, centers[1]], mask[:, centers[1]], masked_raw[:, centers[1]], "y"),
]
for row, (raw_view, mask_view, masked_view, plane) in enumerate(views):
    axes[row, 0].imshow(raw_view, cmap="gray")
    axes[row, 0].set_title(f"raw ({plane} plane)")
    axes[row, 1].imshow(mask_view, cmap="gray")
    axes[row, 1].set_title("binary cell mask")
    axes[row, 2].imshow(masked_view, cmap="gray", vmin=raw.min(), vmax=raw.max())
    axes[row, 2].set_title("masked intensity")
for axis in axes.ravel():
    axis.set_axis_off()
plt.show()
"""
        ),
        md(
            """
## Step 4 — Audit IDs before relying on row order

IDs are semantic keys. Row 0 in one table is not automatically row 0 in another table.
The checks below expose missing segmentations, missing mappings, duplicate IDs, and
nuclei mapped to the wrong ID space.
"""
        ),
        code(
            """
cell_ids = set(np.unique(cells)) - {0}
nucleus_ids = set(np.unique(nuclei)) - {0}
mapped_cells = set(mapping["cell_id"].astype(int))
mapped_nuclei = set(mapping["nucleus_id"].astype(int))
metadata_ids = set(metadata["label_id"].astype(int))

alignment_qc = pd.Series({
    "segmented cells missing from mapping": sorted(cell_ids - mapped_cells),
    "mapping cells missing from segmentation": sorted(mapped_cells - cell_ids),
    "mapped nuclei missing from segmentation": sorted(mapped_nuclei - nucleus_ids),
    "segmented cells missing metadata": sorted(cell_ids - metadata_ids),
    "duplicate mapping cell IDs": int(mapping["cell_id"].duplicated().sum()),
})
display(alignment_qc.to_frame("result"))
assert all(len(value) == 0 for value in alignment_qc.iloc[:4])
assert alignment_qc.iloc[4] == 0
"""
        ),
        md(
            """
## Step 5 — Read the label-first embedding contract literally

A NumPy embedding has shape `(n_cells, 1 + n_features)`. Column zero contains the
segmentation `label_id`; every remaining column is a learned feature. `label_id` must be
finite, integer-valued, unique, and joined explicitly to metadata.
"""
        ),
        code(
            """
embedding = load_embeddings(FIXTURE_DIR / "embeddings.npy")
matrix = embedding.as_array()
print("matrix shape:", matrix.shape)
print("column 0 label IDs:", matrix[:, 0].astype(int).tolist())
print("feature matrix shape:", embedding.features.shape)
print("all features finite:", bool(np.isfinite(embedding.features).all()))

joined = pd.DataFrame({"label_id": embedding.label_ids}).merge(
    metadata, on="label_id", how="left", validate="one_to_one"
)
display(joined)

fig, axis = plt.subplots(figsize=(8, 3))
image = axis.imshow(embedding.features, aspect="auto", cmap="coolwarm")
axis.set_yticks(range(len(embedding.label_ids)), embedding.label_ids)
axis.set_xlabel("feature dimension")
axis.set_ylabel("label_id")
axis.set_title("Synthetic feature matrix (visual QC, not biology)")
fig.colorbar(image, ax=axis, label="feature value")
plt.show()
"""
        ),
        md(
            """
## Step 6 — Split IDs, then select rows

Splitting label IDs first makes the assignment reproducible and prevents a later table
sort from changing membership. For real biological data, also consider animal/batch,
spatial leakage, class balance, and related cells; a random cell split is not always an
independent biological test.
"""
        ),
        code(
            """
train_ids, validation_ids = train_test_split(
    embedding.label_ids, test_size=0.5, random_state=SEED, shuffle=True
)
train_rows = np.isin(embedding.label_ids, train_ids)
validation_rows = np.isin(embedding.label_ids, validation_ids)
print("train label IDs:", sorted(train_ids.tolist()))
print("validation label IDs:", sorted(validation_ids.tolist()))
assert not np.any(train_rows & validation_rows)
assert np.all(train_rows | validation_rows)
"""
        ),
        md(
            """
## Replacing the fixture with cluster data

Set `MORPHOFEATURES_DATA_ROOT` instead of pasting a private path into source code. Record
the container path, internal dataset key, `(z, y, x)` resolution, segmentation version,
mapping/table version, crop policy, normalization, and selection exclusions in the run
directory. Then repeat the visual and ID audits above.

Notebook 04 demonstrates this exact transition for the locally available Platynereis
N5 data. It also shows why a promising dataset name is not enough: its shape, encoding,
ID space, and alignment must be inspected before use.
"""
        ),
    ],
)

write(
    "02_cpu_mae_training_and_encoding.ipynb",
    [
        md(
            """
# 2. A masked autoencoder, one visible transformation at a time

This notebook trains a small 3D masked autoencoder (MAE) end to end on **segmented
synthetic cells**, not random noise. It is intentionally small enough for CPU execution,
but it uses the same package APIs, checkpoint format, metric events, and label-first
export as a larger experiment.

An MAE here is a **representation learner**, not a segmentation network: the cell mask
has already been produced upstream. The model hides 3D patches from a masked intensity
crop, predicts their cell-interior voxels, and uses its encoder output as an embedding.
"""
        ),
        md(
            """
## Step 0 — Central configuration and fixed seeds

The generated config is saved with the run. This cell also reports the actual compute
device; CUDA is optional and this tutorial deliberately requests CPU.
"""
        ),
        code(
            """
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
import yaml

from morphofeatures.artifacts import inspect_checkpoint, inspect_embedding
from morphofeatures.config import load_config, repository_root
from morphofeatures.data.crops import extract_masked_cell_crops, save_crop_batch
from morphofeatures.data.io import load_embeddings
from morphofeatures.data.synthetic import save_synthetic_dataset
from morphofeatures.mae3d import build_mae_model, encode_from_config, train_from_config
from morphofeatures.metrics import metric_series, read_metric_events
from morphofeatures.training_runtime import load_checkpoint

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

REPO_ROOT = repository_root()
BASE_CONFIG = load_config(REPO_ROOT / "configs" / "default.yaml")
OUTPUT_DIR = BASE_CONFIG.paths.output_root / "notebooks" / "02_mae_smoke"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("PyTorch:", torch.__version__)
print("CUDA visible:", torch.cuda.is_available())
print("Tutorial device: cpu")
print("Output:", OUTPUT_DIR)
"""
        ),
        md(
            """
## Step 1 — Start from raw intensity plus a cell segmentation

The preprocessing API centers a fixed window on each cell, rescales intensity according
to an explicit policy, and zeros voxels outside that cell. It also retains the unmasked
raw crop and binary mask for QC. Those companions are not hidden model state.
"""
        ),
        code(
            """
fixture_dir = save_synthetic_dataset(OUTPUT_DIR / "synthetic", seed=SEED)
raw = np.load(fixture_dir / "raw.npy")
cells = np.load(fixture_dir / "cells.npy")

batch = extract_masked_cell_crops(
    raw,
    cells,
    crop_shape_zyx=(12, 12, 12),
    min_cell_voxels=100,
    min_crop_coverage=1.0,
    normalization="dtype",  # the synthetic float data are already in [0, 1]
    seed=SEED,
)
paths = save_crop_batch(
    batch,
    OUTPUT_DIR / "preprocessed",
    {"source": "deterministic synthetic fixture", "seed": SEED, "coordinate_order": "zyx"},
)
display(batch.manifest[["label_id", "voxel_count_roi", "crop_coverage", "mask_fraction"]])
print("model crop array:", batch.crops.shape, batch.crops.dtype)
"""
        ),
        md(
            """
## Step 2 — Inspect raw, mask, and model input together

The three columns below are different objects:

- **raw window** retains neighboring structures;
- **cell mask** says which voxels belong to this label;
- **model input** retains intensity only inside the target cell.

For real data, inspect several z slices and orthogonal planes. A centered crop can still
truncate a long process, and a good coverage number can still hide a segmentation merge.
"""
        ),
        code(
            """
fig, axes = plt.subplots(len(batch.label_ids), 3, figsize=(9, 2.4 * len(batch.label_ids)), constrained_layout=True)
for row, label_id in enumerate(batch.label_ids):
    z = batch.crops.shape[1] // 2
    axes[row, 0].imshow(batch.raw_crops[row, z], cmap="gray", vmin=0, vmax=1)
    axes[row, 1].imshow(batch.masks[row, z], cmap="gray")
    axes[row, 2].imshow(batch.crops[row, z], cmap="gray", vmin=0, vmax=1)
    axes[row, 0].set_ylabel(f"label {label_id}")
    for column, title in enumerate(["raw window", "cell mask", "masked input"]):
        if row == 0:
            axes[row, column].set_title(title)
        axes[row, column].set_xticks([])
        axes[row, column].set_yticks([])
plt.show()
"""
        ),
        md(
            """
## Step 3 — Why save a separate loss mask?

Most voxels in a cell-centered cube may be outside the cell. If ordinary MSE includes
those zeros, a model can lower loss by predicting background. MorphoFeatures therefore
accepts `data.loss_masks`: patches are still masked randomly, but reconstruction error
is accumulated only at foreground voxels inside masked patches. The mask is **not** an
extra encoder channel, so the input remains the historical masked-intensity modality.
"""
        ),
        code(
            """
configuration = {
    "seed": SEED,
    "device": "cpu",
    "paths": {"repo_root": str(REPO_ROOT), "output_root": str(OUTPUT_DIR)},
    "data": {
        "crops": str(paths["crops"]),
        "label_ids": str(paths["label_ids"]),
        "loss_masks": str(paths["masks"]),
    },
    "mae": {
        "input_shape": [12, 12, 12],
        "patch_size": [4, 4, 4],
        "input_channels": 1,
        "embedding_dim": 16,
        "encoder_depth": 1,
        "encoder_heads": 4,
        "mask_ratio": 0.50,
    },
    "training": {
        "epochs": 3,
        "batch_size": 2,
        "learning_rate": 0.001,
        "validation_fraction": 0.25,
    },
    "inference": {"batch_size": 4},
}
config_path = OUTPUT_DIR / "mae_config.yaml"
config_path.write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")
display(pd.Series(configuration["mae"], name="value").to_frame())
"""
        ),
        md(
            """
## Step 4 — See the MAE patch grid and random mask before training

A `12³` crop with `4³` patches becomes a `3 × 3 × 3 = 27` token grid. At a mask ratio
of 0.5, 14 tokens are hidden. Patch size is physical: it must be chosen together with
voxel resolution, not copied blindly between datasets.
"""
        ),
        code(
            """
preview_model = build_mae_model(configuration)
preview_input = torch.from_numpy(batch.crops[:1, None])
preview_loss_mask = torch.from_numpy(batch.masks[:1, None].astype(np.float32))
torch.manual_seed(SEED)
with torch.no_grad():
    preview_output = preview_model(preview_input, mask_ratio=0.50, loss_mask=preview_loss_mask)
preview_mask = preview_model.mask_volume(preview_output.mask, preview_input.shape)[0].numpy()

z = preview_input.shape[-3] // 2
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
axes[0].imshow(preview_input[0, 0, z], cmap="gray", vmin=0, vmax=1)
axes[0].set_title("model input")
axes[1].imshow(preview_mask[z], cmap="Reds", vmin=0, vmax=1)
axes[1].set_title("masked MAE patches")
axes[2].imshow(preview_input[0, 0, z], cmap="gray", vmin=0, vmax=1)
axes[2].imshow(preview_mask[z], cmap="Reds", alpha=0.35, vmin=0, vmax=1)
axes[2].set_title("mask over input")
for axis in axes:
    axis.set_axis_off()
plt.show()
"""
        ),
        md(
            """
## Step 5 — Train through the package API

This cell intentionally resets only this tutorial's metrics file so rerunning it yields
one clean event series. Training writes the same portable checkpoint and append-only
JSONL events used by the CLI and Streamlit/SLURM monitor.
"""
        ),
        code(
            """
checkpoint_path = OUTPUT_DIR / "checkpoint.pt"
metrics_path = OUTPUT_DIR / "metrics.jsonl"
if metrics_path.exists():
    metrics_path.unlink()

saved_checkpoint = train_from_config(config_path, output=checkpoint_path)
assert saved_checkpoint == checkpoint_path
print("checkpoint:", checkpoint_path)
"""
        ),
        code(
            """
events = read_metric_events(metrics_path)
loss_frame = pd.DataFrame(metric_series(events))
display(loss_frame[["epoch", "train_loss", "validation_loss", "learning_rate"]])

axis = loss_frame.plot(
    x="epoch", y=["train_loss", "validation_loss"], marker="o", figsize=(7, 4)
)
axis.set_ylabel("foreground masked-patch MSE")
axis.set_title("Bookkeeping check on four synthetic cells")
axis.grid(alpha=0.3)
plt.show()
"""
        ),
        md(
            """
The curve verifies that optimization and validation bookkeeping work. Three epochs on
four synthetic cells do not establish convergence, generalization, or biological value.
Foreground-aware MSE is also not numerically comparable to the legacy contrastive loss.
"""
        ),
        md(
            """
## Step 6 — Inspect and reload the checkpoint

Portable checkpoints store unwrapped model weights plus epoch, step, config, and final
metrics. Reloading into a freshly constructed model is the test that matters; the Python
object left in memory after training is not an artifact.
"""
        ),
        code(
            """
checkpoint_summary = inspect_checkpoint(checkpoint_path)
display(pd.Series({
    "epoch": checkpoint_summary["epoch"],
    "step": checkpoint_summary["step"],
    "parameter tensors": checkpoint_summary["parameter_tensors"],
    **checkpoint_summary["metrics"],
}).to_frame("value"))

reloaded_model = build_mae_model(configuration)
payload = load_checkpoint(checkpoint_path, reloaded_model, device="cpu")
reloaded_model.eval()
print("reloaded epoch:", payload["epoch"])
"""
        ),
        md(
            """
## Step 7 — View a masked prediction honestly

`reconstruction` contains predictions for every token, although the loss trains only
masked tokens. For visualization we keep visible input patches and insert predictions
only where the mask is true. Reconstruction quality is a diagnostic of the pretext task,
not the biological interpretation of the embedding.
"""
        ),
        code(
            """
inputs = torch.from_numpy(batch.crops[:2, None])
loss_masks = torch.from_numpy(batch.masks[:2, None].astype(np.float32))
torch.manual_seed(SEED)
with torch.no_grad():
    output = reloaded_model(inputs, mask_ratio=0.50, loss_mask=loss_masks)
    composite = reloaded_model.composite_reconstruction(inputs, output)
    voxel_mask = reloaded_model.mask_volume(output.mask, inputs.shape)

sample, z = 0, inputs.shape[-3] // 2
hidden_input = inputs[sample, 0].clone()
hidden_input[voxel_mask[sample]] = 0
error = (composite[sample, 0] - inputs[sample, 0]).abs()

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), constrained_layout=True)
for axis, image, title in zip(
    axes,
    [inputs[sample, 0, z], hidden_input[z], composite[sample, 0, z], error[z]],
    ["target", "visible patches", "composite prediction", "absolute error"],
):
    axis.imshow(image, cmap="magma" if title == "absolute error" else "gray", vmin=0)
    axis.set_title(title)
    axis.set_axis_off()
plt.show()
"""
        ),
        md(
            """
## Step 8 — Encode and validate the label-first result

Encoding uses the complete, unmasked crop. The output is sorted by `label_id`; no row
position from a DataLoader is treated as biological identity.
"""
        ),
        code(
            """
embedding_path = OUTPUT_DIR / "embeddings.npy"
encode_from_config(config_path, checkpoint_path, embedding_path)
embedding = load_embeddings(embedding_path)
summary = inspect_embedding(embedding_path)
display(pd.Series(summary.__dict__).to_frame("value"))
print("IDs preserved:", np.array_equal(embedding.label_ids, batch.label_ids))
print("label-first matrix shape:", embedding.as_array().shape)
assert np.isfinite(embedding.features).all()
"""
        ),
        code(
            """
coordinates = PCA(n_components=2).fit_transform(embedding.features)
fig, axis = plt.subplots(figsize=(6, 5))
axis.scatter(coordinates[:, 0], coordinates[:, 1], s=80)
for label_id, (x, y) in zip(embedding.label_ids, coordinates):
    axis.text(x, y, str(label_id), fontsize=9, ha="left", va="bottom")
axis.set_title("PCA of four synthetic MAE embeddings")
axis.set_xlabel("PC1")
axis.set_ylabel("PC2")
axis.grid(alpha=0.2)
plt.show()
"""
        ),
        md(
            """
## Legacy texture autoencoder versus proposed MAE

| Decision | Maintained legacy texture route | Proposed MAE route |
|---|---|---|
| model | 3D convolutional autoencoder | 3D patch tokenizer + Transformer |
| views | paired spatial/intensity augmentations | one crop with random token masks |
| objective | NT-Xent + full reconstruction + bottleneck | reconstruction on masked, cell-interior voxels |
| coarse input | cell/nucleus-centered masked volume | same biological unit, fixed patch-divisible window |
| fine input | many high-resolution patches, average by cell | train/encode patches, then aggregate by `label_id` |
| output | normally 80 dimensions | configurable; use 80 for six-group comparability |

The MAE does **not** remove the need for cell/nucleus mappings, physical-resolution
choices, coarse/fine sampling, or QC. It changes the self-supervised pretext task. A fair
comparison holds label cohort, resolution, crop policy, split, and downstream evaluation
constant.
"""
        ),
        md(
            """
## Scaling to the historical six groups

Train independent 80-dimensional models for cell shape, nucleus shape, coarse cell
texture, coarse nucleus texture, fine cell texture, and fine nucleus texture. Aggregate
fine patch embeddings per cell, join all six tables by `label_id`, require exact ID
alignment, then concatenate to 480 dimensions. Notebook 03 demonstrates the aligned
combination and downstream biological analysis; notebook 04 repeats this MAE workflow on
real EM and segmentation data while keeping its conclusions deliberately limited.
"""
        ),
    ],
)


# Notebook 04 is canonical in notebook form; preserve its cells and clear execution state.
build_real_mae_notebook()
