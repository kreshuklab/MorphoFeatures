"""Frozen official DINO ViT backbones applied to explicit microscopy views.

The adapter uses a local checkout and a local backbone state dictionary; it
never downloads code or weights implicitly. See docs/workspace_workflows.md.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import yaml

VARIANTS = {
    "dinov2": {"dinov2_vits14", "dinov2_vitb14", "dinov2_vits14_reg", "dinov2_vitb14_reg"},
    "dinov3": {"dinov3_vits16", "dinov3_vitb16"},
}


def load_backbone(settings):
    import torch

    family = settings["model"]
    variant = settings.get("variant", "dinov2_vits14" if family == "dinov2" else "dinov3_vits16")
    if variant not in VARIANTS.get(family, set()):
        raise ValueError(f"Supported {family} variants: {sorted(VARIANTS.get(family, set()))}")
    repository = Path(settings.get("model_repository") or "")
    if not (repository / "hubconf.py").is_file():
        raise ValueError(
            "model_repository must point to a local official DINO checkout with hubconf.py"
        )
    model = torch.hub.load(str(repository.resolve()), variant, source="local", pretrained=False)
    state = torch.load(settings["checkpoint"], map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.eval()


def microscopy_views(raw, mask, settings):
    """Yield RGB tensors: selected axial slices, normalized, resized and ImageNet scaled."""
    import torch
    import torch.nn.functional as functional

    original = np.asarray(raw)
    raw = original.astype(np.float32)
    mask = np.asarray(mask, dtype=bool)
    if raw.ndim != 3 or mask.shape != raw.shape or not np.isfinite(raw).all():
        raise ValueError("DINO input requires a finite 3D grayscale crop and aligned mask")
    if not mask.any():
        raise ValueError("Object has an empty mask")
    mode = settings.get("normalization", "foreground_percentile")
    if mode == "foreground_percentile":
        percentiles = settings.get("percentiles", (1, 99))
        if len(percentiles) != 2 or not 0 <= percentiles[0] < percentiles[1] <= 100:
            raise ValueError("percentiles must be two increasing values in [0, 100]")
        low, high = np.percentile(raw[mask], percentiles)
        if high <= low:
            # Uniform objects retain a foreground/background distinction.
            raw = mask.astype(np.float32)
        else:
            raw = np.clip((raw - low) / (high - low), 0, 1)
    elif mode == "dtype":
        if not np.issubdtype(original.dtype, np.integer):
            raise ValueError(
                "DINO dtype normalization requires integer raw crops; use foreground_percentile or unit"
            )
        limits = np.iinfo(original.dtype)
        raw = (raw - limits.min) / float(limits.max - limits.min)
    elif mode == "unit":
        if raw.min() < 0 or raw.max() > 1:
            raise ValueError("DINO unit input must already lie in [0, 1]")
    else:
        raise ValueError("DINO normalization must be foreground_percentile, dtype, or unit")
    if settings.get("mask", True):
        raw[~mask] = 0
    # NumPy percentile scalars can promote float32 arithmetic to float64.
    # Official backbones use float32 weights outside autocast.
    raw = np.asarray(raw, dtype=np.float32)
    axes = settings.get("axes", [0, 1, 2])
    fractions = settings.get("fractions", [0.25, 0.5, 0.75])
    if not axes or any(axis not in (0, 1, 2) for axis in axes):
        raise ValueError("View axes must be selected from 0=z, 1=y, 2=x")
    if not fractions or any(not 0 <= float(f) <= 1 for f in fractions):
        raise ValueError("View fractions must be nonempty and in [0, 1]")
    size = int(settings.get("size", 224))
    if size < 14:
        raise ValueError("DINO image size must be at least one model patch")
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    resize = settings.get("resize", "stretch")
    for axis in axes:
        for fraction in fractions:
            index = int(round(float(fraction) * (raw.shape[axis] - 1)))
            plane = torch.from_numpy(np.ascontiguousarray(np.take(raw, index, axis=axis)))[
                None, None
            ]
            if resize == "letterbox":
                scale = size / max(plane.shape[-2:])
                shape = tuple(max(1, round(s * scale)) for s in plane.shape[-2:])
                plane = functional.interpolate(
                    plane, size=shape, mode="bilinear", align_corners=False, antialias=True
                )
                dy, dx = size - shape[0], size - shape[1]
                plane = functional.pad(plane, (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2))
            elif resize == "stretch":
                plane = functional.interpolate(
                    plane, size=(size, size), mode="bilinear", align_corners=False, antialias=True
                )
            else:
                raise ValueError("resize must be stretch or letterbox")
            yield (plane[0].expand(3, -1, -1) - mean) / std


def object_readers(config):
    """Yield ID plus a lazy reader so individual object failures can be recorded."""
    data = config.get("data", {})
    if data.get("source") == "n5_masked_patches":
        from morphofeatures.real_mae import prepare_real_mae_data, resolve_real_mae_config

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            prepared = prepare_real_mae_data(resolve_real_mae_config(path))
        dataset = prepared.dataset(config.get("inference", {}).get("split", "all"), "encode")
        for i, label_id in enumerate(dataset.label_ids):

            def read(i=i):
                group = dataset.read_group(i)
                for patch_index in group["patch_indices"]:
                    raw = np.asarray(dataset._dataset()[int(patch_index)])
                    mask = (
                        raw != 0
                        if data.get("mask_mode", "nonzero") == "nonzero"
                        else np.ones_like(raw, dtype=bool)
                    )
                    yield raw, mask

            yield int(label_id), read
    else:
        from morphofeatures.mae3d import _load_label_ids

        if not data.get("crops"):
            raise ValueError("DINO extraction requires data.crops or an N5 patch source")
        crops = np.load(data["crops"], mmap_mode="r")
        ids = _load_label_ids(config, len(crops))
        masks = np.load(data["loss_masks"], mmap_mode="r") if data.get("loss_masks") else None
        for i, label_id in enumerate(ids):

            def read(i=i):
                raw = np.asarray(crops[i])
                if raw.ndim == 4 and raw.shape[0] == 1:
                    raw = raw[0]
                mask = (
                    np.asarray(masks[i]).reshape(raw.shape).astype(bool)
                    if masks is not None
                    else raw != 0
                )
                yield raw, mask

            yield int(label_id), read


def extract_dino(settings, progress=None, *, backbone=None):
    import torch

    from morphofeatures.training_runtime import resolve_device

    torch.manual_seed(int(settings.get("config", {}).get("seed", 42)))
    device = resolve_device(settings.get("config", {}).get("device", "auto"))
    model = (backbone if backbone is not None else load_backbone(settings)).to(device).eval()
    views = settings.get("views", {})
    size = int(views.get("size", 224))
    stride = 14 if settings["model"] == "dinov2" else 16
    if size % stride:
        raise ValueError(f"views.size must be divisible by the {stride}-pixel model patch size")
    batch_size = int(views.get("batch_size", 16))
    feature = views.get("feature", "cls")
    aggregation = views.get("aggregation", "mean")
    if batch_size < 1 or feature not in {"cls", "patch_mean"} or aggregation not in {"mean", "max"}:
        raise ValueError("Use positive batch_size, feature=cls/patch_mean, aggregation=mean/max")
    ids, embeddings, excluded = [], [], []
    with torch.inference_mode():
        for label_id, read in object_readers(settings["config"]):
            try:
                pending, vectors = [], []

                def flush(pending=pending, vectors=vectors):
                    result = model.forward_features(torch.stack(pending).to(device))
                    encoded = (
                        result["x_norm_clstoken"]
                        if feature == "cls"
                        else result["x_norm_patchtokens"].mean(1)
                    )
                    vectors.append(encoded.cpu().float().numpy())
                    pending.clear()

                for raw, mask in read():
                    for view in microscopy_views(raw, mask, views):
                        pending.append(view)
                        if len(pending) == batch_size:
                            flush()
                if pending:
                    flush()
                if not vectors:
                    raise ValueError("No usable views")
                vectors = np.concatenate(vectors)
                vector = vectors.mean(0) if aggregation == "mean" else vectors.max(0)
                if not np.isfinite(vector).all():
                    raise ValueError("Backbone produced nonfinite features")
                ids.append(label_id)
                embeddings.append(vector)
            except (ValueError, OSError) as error:
                excluded.append({"label_id": label_id, "reason": str(error)})
            if progress:
                progress.write(
                    "object_encoded", label_id=label_id, processed=len(ids), excluded=len(excluded)
                )
    if not ids:
        raise ValueError(f"No objects were encoded; failures: {excluded[:5]}")
    return np.asarray(ids, dtype=np.int64), np.stack(embeddings), excluded
