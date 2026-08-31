import importlib.util

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
