# Data preparation

## Coordinate and unit contract

Volumes, bounding boxes, crop sizes, and resolution triples are ordered `z, y, x`. Physical resolutions are positive voxel sizes in micrometers by default. Segmentation arrays must align with the raw volume in physical space; arrays at different pyramid levels must provide their dataset keys and matching resolution.

## Volume inputs

Each raw or segmentation input has:

- `path`: `.npy`, Zarr/N5 container, or BDV XML path.
- `dataset`: internal array key such as `setup0/timepoint0/s3`.
- `resolution`: three physical voxel sizes in `z, y, x`.
- `kind`: `raw`, `cell_segmentation`, or `nucleus_segmentation`.

Raw volumes are numeric intensity arrays. Segmentations contain integer label IDs with `0` reserved for background.

## Cell-to-nucleus mapping

The normalized schema is:

```text
cell_id	nucleus_id
1	101
2	102
```

Legacy headers can be selected with `data_config.mapping_columns`. Rows with nucleus ID `0` are excluded from paired cell/nucleus training.

## Bounding-box and metadata tables

Bounding-box tables require:

```text
label_id bb_min_z bb_min_y bb_min_x bb_max_z bb_max_y bb_max_x
```

The texture loader also expects `anchor_z`, `anchor_y`, and `anchor_x` for center-of-mass crops. Coordinates are physical micrometers in the published PlatyBrowser tables. Metadata and MoBIE tables require `label_id`; additional columns are workflow-specific.

## Embeddings

NumPy outputs have shape `(n_cells, 1 + n_features)`. Column 0 is finite, unique, integer-valued `label_id`. TSV/CSV outputs name the first column `label_id`. Published group files have 80 feature columns; the combined representation has 480.

```bash
python -m morphofeatures combine \
  data_mobie/features_shape_cell.tsv data_mobie/features_shape_nucl.tsv \
  data_mobie/features_coarse_ultr_cell.tsv data_mobie/features_coarse_ultr_nucl.tsv \
  data_mobie/features_fine_ultr_cell.tsv data_mobie/features_fine_ultr_nucl.tsv \
  --output outputs/morphofeatures.npy
```

## Synthetic fixture

```bash
python -m morphofeatures synthetic outputs/synthetic --seed 7
```

This writes aligned `raw.npy`, `cells.npy`, `nuclei.npy`, `cell_to_nucleus.tsv`, `metadata.tsv`, and label-first `embeddings.npy`. The committed test fixture under `tests/fixtures/synthetic` uses the same generator.

## Shape input manifest

Shape training accepts `.npy`/`.npz` point clouds or OFF/PLY/OBJ meshes. A TSV manifest is recommended:

```text
label_id	path
1	point_clouds/1.npz
2	point_clouds/2.npz
```

NPZ files contain `points` and optional `normals`, each `(n_points, 3)`. NPY files contain `(n_points, 3)` or `(n_points, 6)` arrays. Mesh inputs are sampled with trimesh.
