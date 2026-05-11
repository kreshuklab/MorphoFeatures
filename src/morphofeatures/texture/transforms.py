"""Texture transform construction."""

from __future__ import annotations

from typing import Any


def get_transforms(transform_config: dict[str, Any] | None) -> Any:
    """Build an Inferno transform composition from configuration.

    Args:
        transform_config: Transform configuration dictionary.

    Returns:
        Inferno ``Compose`` transform or ``None`` when no config is supplied.

    Raises:
        ImportError: If Inferno is not installed.
    """

    if not transform_config:
        return None
    try:
        from inferno.io.transform import Compose
        from inferno.io.transform.generic import AsTorchBatch, Cast, NormalizeRange
        from inferno.io.transform.image import ElasticTransform
        from inferno.io.transform.volume import CropPad2Size, RandomRot903D, VolumeRandomCrop
    except ImportError as exc:
        raise ImportError("Texture transforms require inferno.") from exc

    transforms = Compose()
    if transform_config.get("crop_pad_to_size"):
        transforms.add(CropPad2Size(**transform_config["crop_pad_to_size"]))
    if transform_config.get("random_crop"):
        transforms.add(VolumeRandomCrop(**transform_config["random_crop"]))
    if transform_config.get("cast"):
        transforms.add(Cast("float32"))
    if transform_config.get("normalize_range"):
        transforms.add(NormalizeRange(**transform_config["normalize_range"]))
    if transform_config.get("rotate90"):
        transforms.add(RandomRot903D())
    if transform_config.get("elastic_transform"):
        transforms.add(ElasticTransform(order=3, **transform_config["elastic_transform"]))
    if transform_config.get("torch_batch"):
        transforms.add(AsTorchBatch(3))
    return transforms
