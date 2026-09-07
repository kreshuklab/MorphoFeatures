import json

import numpy as np
import pytest

from morphofeatures.data.crops import (
    describe_labels,
    extract_masked_cell_crops,
    map_fragment_labels,
    save_crop_batch,
)
from morphofeatures.mae3d import _load_crops, _load_loss_masks


def test_fragment_mapping_and_label_description():
    fragments = np.zeros((8, 8, 8), dtype=np.uint8)
    fragments[2:6, 2:6, 2:6] = 2
    mapping = np.array([0, 7, 11], dtype=np.int64)
    labels = map_fragment_labels(fragments, mapping)
    assert set(np.unique(labels)) == {0, 11}
    table = describe_labels(labels)
    assert table.loc[0, "label_id"] == 11
    assert table.loc[0, "voxel_count_roi"] == 64
    assert not bool(table.loc[0, "touches_roi_border"])


def test_masked_crop_preparation_and_round_trip(tmp_path):
    raw = np.arange(20**3, dtype=np.uint16).reshape(20, 20, 20)
    labels = np.zeros_like(raw, dtype=np.uint16)
    labels[6:14, 6:14, 6:14] = 5
    labels[:3, :3, :3] = 9  # excluded because it touches the ROI border
    batch = extract_masked_cell_crops(
        raw,
        labels,
        (12, 12, 12),
        min_cell_voxels=10,
        min_crop_coverage=1.0,
        normalization="dtype",
    )
    assert np.array_equal(batch.label_ids, [5])
    assert batch.crops.shape == (1, 12, 12, 12)
    assert np.all(batch.crops[~batch.masks] == 0)
    assert 0 <= batch.crops.min() <= batch.crops.max() <= 1
    paths = save_crop_batch(batch, tmp_path, {"coordinate_order": "zyx"})
    assert np.array_equal(np.load(paths["label_ids"]), [5])
    assert json.loads(paths["provenance"].read_text())["coordinate_order"] == "zyx"


def test_crop_preparation_rejects_non_integral_labels():
    labels = np.zeros((8, 8, 8), dtype=float)
    labels[2, 2, 2] = 1.5
    with pytest.raises(ValueError, match="integer"):
        describe_labels(labels)


def test_empty_label_description_has_stable_columns():
    table = describe_labels(np.zeros((4, 4, 4), dtype=np.uint8))
    assert table.empty
    assert "label_id" in table.columns


def test_mae_integer_inputs_are_scaled(tmp_path):
    path = tmp_path / "crops.npy"
    np.save(path, np.array([np.zeros((8, 8, 8)), np.full((8, 8, 8), 255)], dtype=np.uint8))
    crops = _load_crops({"data": {"crops": str(path)}}, seed=1)
    assert crops.shape == (2, 1, 8, 8, 8)
    assert crops.dtype == np.float32
    assert crops.min() == 0
    assert crops.max() == 1


def test_mae_loss_masks_are_checked(tmp_path):
    path = tmp_path / "masks.npy"
    masks = np.zeros((2, 8, 8, 8), dtype=bool)
    masks[:, 2:6, 2:6, 2:6] = True
    np.save(path, masks)
    loaded = _load_loss_masks({"data": {"loss_masks": str(path)}}, 2, (8, 8, 8))
    assert loaded.shape == (2, 1, 8, 8, 8)
    masks[0] = False
    np.save(path, masks)
    with pytest.raises(ValueError, match="foreground"):
        _load_loss_masks({"data": {"loss_masks": str(path)}}, 2, (8, 8, 8))


@pytest.mark.optional
def test_mae_qc_reconstruction_helpers():
    torch = pytest.importorskip("torch")
    from morphofeatures.mae3d import MaskedAutoencoder3D

    model = MaskedAutoencoder3D(patch_size=(4, 4, 4), embedding_dim=8, encoder_depth=1, encoder_heads=2)
    inputs = torch.rand(2, 1, 8, 8, 8)
    output = model(inputs, mask_ratio=0.5)
    mask = model.mask_volume(output.mask, inputs.shape)
    composite = model.composite_reconstruction(inputs, output)
    assert mask.shape == (2, 8, 8, 8)
    assert composite.shape == inputs.shape
    assert torch.equal(composite[:, 0][~mask], inputs[:, 0][~mask])
    assert torch.equal(model.unpatchify(model.patchify(inputs), inputs.shape), inputs)


@pytest.mark.optional
def test_mae_foreground_loss_ignores_background_error():
    torch = pytest.importorskip("torch")
    from morphofeatures.mae3d import MaskedAutoencoder3D

    torch.manual_seed(3)
    model = MaskedAutoencoder3D(patch_size=(4, 4, 4), embedding_dim=8, encoder_depth=1, encoder_heads=2)
    inputs = torch.rand(2, 1, 8, 8, 8)
    foreground = torch.zeros(2, 1, 8, 8, 8)
    foreground[:, :, :4] = 1
    output = model(inputs, mask_ratio=0.75, loss_mask=foreground)
    assert torch.isfinite(output.loss)
    output.loss.backward()
