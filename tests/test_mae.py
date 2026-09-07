import importlib.util
from pathlib import Path

import pytest


@pytest.mark.optional
def test_small_mae_forward_and_backward():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("Torch is not installed")
    import torch

    from morphofeatures.mae3d import MaskedAutoencoder3D

    model = MaskedAutoencoder3D(
        patch_size=(4, 4, 4), embedding_dim=16, encoder_depth=1, encoder_heads=4
    )
    inputs = torch.randn(2, 1, 8, 8, 8)
    output = model(inputs, mask_ratio=0.5)
    output.loss.backward()
    assert output.reconstruction.shape == inputs.shape
    assert output.embedding.shape == (2, 16)
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.visible_mean_baseline_loss)


@pytest.mark.optional
def test_masked_tokens_keep_distinct_3d_positions():
    torch = pytest.importorskip("torch")
    from morphofeatures.mae3d import MaskedAutoencoder3D

    torch.manual_seed(5)
    model = MaskedAutoencoder3D(
        patch_size=(2, 2, 2), embedding_dim=12, encoder_depth=1, encoder_heads=3
    )
    inputs = torch.zeros(1, 1, 4, 4, 4)
    torch.manual_seed(17)
    output = model(inputs, mask_ratio=0.5)
    predicted = model.patchify(output.reconstruction)[0, output.mask[0]]

    # The legacy implementation erased position and made these rows equal to
    # floating-point precision. Position-aware masked tokens must differ.
    assert predicted.std(dim=0).mean() > 1e-4
    assert (predicted - predicted[:1]).abs().max() > 1e-4


@pytest.mark.optional
def test_position_aware_mae_can_overfit_a_spatial_toy_volume():
    torch = pytest.importorskip("torch")
    from morphofeatures.mae3d import MaskedAutoencoder3D

    torch.manual_seed(4)
    model = MaskedAutoencoder3D(
        patch_size=(2, 2, 2),
        embedding_dim=24,
        encoder_depth=1,
        encoder_heads=4,
        decoder_dim=32,
    )
    volume = torch.arange(64, dtype=torch.float32).reshape(1, 1, 4, 4, 4) / 63
    inputs = torch.cat((volume, volume), dim=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    torch.manual_seed(91)
    initial = float(model(inputs, mask_ratio=0.5).loss.detach())
    for _ in range(30):
        # Hold the mask fixed so this specifically tests spatial reconstruction.
        torch.manual_seed(91)
        output = model(inputs, mask_ratio=0.5)
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
    torch.manual_seed(91)
    final = float(model(inputs, mask_ratio=0.5).loss.detach())

    assert final < initial * 0.02


@pytest.mark.optional
def test_legacy_position_blind_checkpoint_is_rejected(tmp_path: Path):
    pytest.importorskip("torch")
    from morphofeatures.mae3d import MaskedAutoencoder3D, load_mae_checkpoint
    from morphofeatures.training_runtime import save_checkpoint

    model = MaskedAutoencoder3D(
        patch_size=(2, 2, 2), embedding_dim=8, encoder_depth=1, encoder_heads=2
    )
    checkpoint = save_checkpoint(
        tmp_path / "legacy.pt", model, config={"mae": {"embedding_dim": 8}}
    )

    with pytest.raises(ValueError, match="sampling unit or masking objective is incompatible"):
        load_mae_checkpoint(checkpoint, model)


@pytest.mark.optional
def test_grouped_patch_mae_masks_whole_patches_and_propagates_position():
    torch = pytest.importorskip("torch")
    from morphofeatures.mae3d import GroupedPatchMaskedAutoencoder3D

    torch.manual_seed(9)
    model = GroupedPatchMaskedAutoencoder3D(
        input_shape=(4, 4, 4),
        reconstruction_shape=(2, 2, 2),
        embedding_dim=12,
        encoder_depth=1,
        encoder_heads=3,
        decoder_dim=12,
        decoder_depth=1,
        decoder_heads=3,
    )
    patches = torch.rand(2, 5, 1, 4, 4, 4)
    positions = torch.tensor(
        [[[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]] * 2,
        dtype=torch.float32,
    )
    valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    output = model(patches, positions, valid, mask_ratio=0.5)
    output.loss.backward()

    assert output.mask.shape == valid.shape
    assert torch.all(output.mask <= valid)
    assert output.mask.sum(dim=1).tolist() == [2, 2]
    assert output.target.shape == (2, 5, 1, 2, 2, 2)
    assert output.embedding.shape == (2, 12)
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.visible_mean_baseline_loss)


@pytest.mark.optional
def test_grouped_patch_mae_supports_a_3d_resnet_patch_encoder():
    torch = pytest.importorskip("torch")
    from morphofeatures.mae3d import GroupedPatchMaskedAutoencoder3D

    model = GroupedPatchMaskedAutoencoder3D(
        input_shape=(8, 8, 8),
        reconstruction_shape=(4, 4, 4),
        embedding_dim=8,
        encoder_depth=1,
        encoder_heads=2,
        decoder_dim=8,
        decoder_depth=1,
        decoder_heads=2,
        patch_encoder="resnet3d",
        resnet_channels=(4, 8),
        resnet_blocks=(1, 1),
    )
    patches = torch.randn(2, 4, 1, 8, 8, 8)
    positions = torch.randn(2, 4, 3)
    valid = torch.ones(2, 4, dtype=torch.bool)
    output = model(patches, positions, valid, mask_ratio=0.5)
    output.loss.backward()
    assert model.patch_encoder_name == "resnet3d"
    assert output.embedding.shape == (2, 8)
    assert any(parameter.grad is not None for parameter in model.patch_encoder.parameters())


@pytest.mark.optional
def test_grouped_patch_checkpoint_rejects_a_different_patch_encoder(tmp_path: Path):
    pytest.importorskip("torch")
    from morphofeatures.mae3d import GroupedPatchMaskedAutoencoder3D, load_mae_checkpoint
    from morphofeatures.training_runtime import save_checkpoint

    arguments = {
        "input_shape": (4, 4, 4),
        "reconstruction_shape": (2, 2, 2),
        "embedding_dim": 8,
        "encoder_depth": 1,
        "encoder_heads": 2,
        "decoder_dim": 8,
        "decoder_depth": 1,
        "decoder_heads": 2,
    }
    linear = GroupedPatchMaskedAutoencoder3D(**arguments)
    checkpoint = save_checkpoint(
        tmp_path / "linear.pt",
        linear,
        config={
            "mae": {
                "architecture_version": "grouped-nucleus-patches-v3",
                "patch_encoder": "linear",
            }
        },
    )
    resnet = GroupedPatchMaskedAutoencoder3D(
        **arguments,
        patch_encoder="resnet3d",
        resnet_channels=(4,),
        resnet_blocks=(1,),
    )
    with pytest.raises(ValueError, match="different state and inductive biases"):
        load_mae_checkpoint(checkpoint, resnet)


@pytest.mark.optional
def test_grouped_patch_mae_mixed_precision_has_consistent_decoder_token_dtype():
    torch = pytest.importorskip("torch")
    from morphofeatures.mae3d import GroupedPatchMaskedAutoencoder3D

    model = GroupedPatchMaskedAutoencoder3D(
        input_shape=(4, 4, 4),
        reconstruction_shape=(2, 2, 2),
        embedding_dim=8,
        encoder_depth=1,
        encoder_heads=2,
        decoder_dim=8,
        decoder_depth=1,
        decoder_heads=2,
    )
    patches = torch.rand(2, 4, 1, 4, 4, 4)
    positions = torch.rand(2, 4, 3)
    valid = torch.ones(2, 4, dtype=torch.bool)

    # CPU float16 exercises the same autocast dtype transition that previously
    # failed when CUDA float16 projections were assigned to a float32 buffer.
    with torch.autocast("cpu", dtype=torch.float16):
        output = model(patches, positions, valid, mask_ratio=0.5)
    output.loss.backward()

    assert torch.isfinite(output.loss)


@pytest.mark.optional
def test_grouped_patch_mae_rejects_single_patch_sampling_unit():
    torch = pytest.importorskip("torch")
    from morphofeatures.mae3d import GroupedPatchMaskedAutoencoder3D

    model = GroupedPatchMaskedAutoencoder3D(
        input_shape=(4, 4, 4),
        reconstruction_shape=(2, 2, 2),
        embedding_dim=8,
        encoder_depth=1,
        encoder_heads=2,
        decoder_dim=8,
        decoder_depth=1,
        decoder_heads=2,
    )
    patches = torch.rand(1, 1, 1, 4, 4, 4)
    positions = torch.zeros(1, 1, 3)
    valid = torch.ones(1, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="at least two"):
        model(patches, positions, valid, mask_ratio=0.5)


@pytest.mark.optional
def test_grouped_patch_mae_can_overfit_spatial_patch_group():
    torch = pytest.importorskip("torch")
    from morphofeatures.mae3d import GroupedPatchMaskedAutoencoder3D

    torch.manual_seed(3)
    model = GroupedPatchMaskedAutoencoder3D(
        input_shape=(4, 4, 4),
        reconstruction_shape=(2, 2, 2),
        embedding_dim=24,
        encoder_depth=1,
        encoder_heads=4,
        decoder_dim=24,
        decoder_depth=1,
        decoder_heads=4,
        norm_pix_loss=False,
    )
    group = torch.arange(6 * 64, dtype=torch.float32).reshape(1, 6, 1, 4, 4, 4) / 383
    patches = torch.cat((group, group), dim=0)
    positions = torch.tensor(
        [[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0]]] * 2,
        dtype=torch.float32,
    )
    valid = torch.ones(2, 6, dtype=torch.bool)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    torch.manual_seed(55)
    initial = float(model(patches, positions, valid, mask_ratio=0.5).loss.detach())
    for _ in range(40):
        torch.manual_seed(55)
        output = model(patches, positions, valid, mask_ratio=0.5)
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
    torch.manual_seed(55)
    final = float(model(patches, positions, valid, mask_ratio=0.5).loss.detach())

    assert final < initial * 0.01


@pytest.mark.optional
def test_single_patch_v2_checkpoint_is_rejected_by_grouped_v3(tmp_path: Path):
    pytest.importorskip("torch")
    from morphofeatures.mae3d import (
        GroupedPatchMaskedAutoencoder3D,
        MaskedAutoencoder3D,
        load_mae_checkpoint,
    )
    from morphofeatures.training_runtime import save_checkpoint

    old_model = MaskedAutoencoder3D(
        patch_size=(2, 2, 2), embedding_dim=8, encoder_depth=1, encoder_heads=2
    )
    checkpoint = save_checkpoint(
        tmp_path / "single-patch-v2.pt",
        old_model,
        config={"mae": {"architecture_version": "position-aware-3d-v2"}},
    )
    grouped_model = GroupedPatchMaskedAutoencoder3D(
        input_shape=(4, 4, 4),
        reconstruction_shape=(2, 2, 2),
        embedding_dim=8,
        encoder_depth=1,
        encoder_heads=2,
        decoder_dim=8,
        decoder_depth=1,
        decoder_heads=2,
    )

    with pytest.raises(ValueError, match="expects 'grouped-nucleus-patches-v3'"):
        load_mae_checkpoint(checkpoint, grouped_model)
