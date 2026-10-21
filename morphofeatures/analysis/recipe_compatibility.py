"""Explain representation compatibility separately from acquisition differences."""

import json
from copy import deepcopy


def canonical_recipe(value):
    """Read legacy fusion recipes without treating DINO input storage as a model."""
    if isinstance(value, list):
        return [canonical_recipe(v) for v in value]
    if not isinstance(value, dict):
        return value
    value = deepcopy(value)
    if value.get("model") in {"dinov2", "dinov3", "fusion"}:
        value.pop("input layout", None)
        value.pop("patch sampling", None)
    if value.get("model") == "fusion" and value.get("fusion"):
        value["checkpoint"] = "component recipes"
    return {k: canonical_recipe(v) for k, v in value.items()}


def differences(left, right, prefix=""):
    """Return every differing field, including absent fields in known recipes."""
    if isinstance(left, dict) and isinstance(right, dict):
        return [
            row
            for key in sorted(left.keys() | right.keys())
            for row in differences(left.get(key), right.get(key), f"{prefix}.{key}".strip("."))
        ]
    if (
        isinstance(left, list)
        and isinstance(right, list)
        and len(left) == len(right)
        and any(isinstance(v, dict) for v in left + right)
    ):
        return [
            row
            for i, (a, b) in enumerate(zip(left, right))
            for row in differences(a, b, f"{prefix}[{i}]")
        ]
    return [] if left == right else [{"parameter": prefix, "reference": left, "target": right}]


def preparation_recipe(metadata):
    """Human-readable acquisition details; file paths and object IDs are identities."""
    settings = metadata.get("settings", {})
    data = settings.get("config", {}).get("data", {})
    prep = metadata.get("preprocessing", {})
    recipe = {}
    if data.get("source"):
        recipe["input layout"] = data["source"]
    if data.get("source") == "n5_masked_patches":
        recipe.update(
            {
                "patches per object (cap)": data.get("group_size", 200),
                "patch shape Z/Y/X (voxels)": data.get("patch_shape_zyx"),
                "raw voxel spacing Z/Y/X (µm)": data.get("resolution_zyx_um"),
                "foreground mask": data.get("mask_mode", "nonzero"),
                "patch foreground cutoff": data.get("min_foreground_fraction", 0.01),
                "patch intensity normalization": data.get("normalization", "dtype"),
                "position stride Z/Y/X": data.get("position_stride_zyx", [8, 8, 8]),
            }
        )
    elif data.get("source") == "grouped_crops":
        recipe.update(data.get("patch_recipe", {}))
        recipe["raw voxel spacing Z/Y/X (µm)"] = data.get("volume", {}).get("raw_spacing_zyx")
    elif prep:
        factor = 0.001 if prep.get("unit") == "nm" else 1
        recipe.update(
            {
                "crop shape Z/Y/X (voxels)": prep.get("crop_shape"),
                "raw voxel spacing Z/Y/X (µm)": [
                    float(v) * factor for v in prep.get("spacing_zyx", [])
                ],
                "foreground mask": prep.get("mask"),
                "crop center": prep.get("settings", {}).get("center"),
                "oversized objects": prep.get("settings", {}).get("oversized"),
            }
        )
    elif data.get("source") == "physical_volume":
        source = data.get("volume", {})
        for key in ("raw_spacing_zyx", "segmentation_spacing_zyx"):
            recipe[key + " (µm)"] = source.get(key)
    return {k: v for k, v in recipe.items() if v is not None}


def _verified(signature):
    if not signature.get("model") or not signature.get("checkpoint"):
        return False
    if signature.get("model") == "spatial_context":
        return _verified((signature.get("context") or {}).get("input_recipe") or {})
    if signature.get("model") == "fusion":
        blocks = (signature.get("fusion") or {}).get("blocks", [])
        return bool(blocks) and all(_verified(b.get("recipe", {})) for b in blocks)
    return True


def compare_recipes(reference, target, reference_name="Reference", target_name="Target"):
    from morphofeatures.analysis.joint import _signature

    left, right = _signature(reference), _signature(target)
    # Unknown provenance may be explicitly acknowledged, but known differences
    # must still block even when a checkpoint hash is missing on one side.
    errors = [
        row
        for key in sorted(left.keys() | right.keys())
        if left.get(key) is not None and right.get(key) is not None
        for row in differences(left[key], right[key], key)
    ]
    if reference.get("dimensions") and target.get("dimensions"):
        errors += differences(reference["dimensions"], target["dimensions"], "feature dimensions")
    warnings = []
    if left.get("model") in {"dinov2", "dinov3", "fusion"} and left.get("model") == right.get(
        "model"
    ):
        warnings = differences(preparation_recipe(reference), preparation_recipe(target))
    return {
        "reference": reference_name,
        "target": target_name,
        "verified": _verified(left) and _verified(right),
        "compatible": not errors,
        "differences": errors,
        "preparation_differences": warnings,
    }


def compatibility_message(comparison):
    rows = comparison["differences"]

    def display(value):
        return "not recorded" if value is None else json.dumps(value, ensure_ascii=False)

    details = "\n".join(
        f"• different {row['parameter']}: {comparison['reference']} = {display(row['reference'])}; "
        f"{comparison['target']} = {display(row['target'])}. {parameter_meaning(row['parameter'])}"
        for row in rows
    )
    return (
        f"Incompatible representations: {comparison['reference']} and {comparison['target']}.\n"
        + details
        + "\nChoose embeddings made with matching checkpoint, view/pooling and combination settings, "
        "or re-extract the differing representation. Confirming missing provenance does not override known differences. "
        "Changing classifier or UMAP settings will not fix an extraction mismatch."
    )


def parameter_meaning(parameter):
    meanings = {
        "checkpoint": "This identifies the weight-file contents; equal filenames alone are insufficient.",
        "variant": "This selects the DINO architecture, including backbone size and register tokens.",
        "model": "This identifies the encoder family or the combined representation.",
        "feature": "CLS is a global image summary; patch_mean and foreground_mean pool spatial tokens differently.",
        "aggregation": "This controls how image vectors are pooled into one object vector.",
        "mode": "Stored crops, whole-label views and fixed physical tiles describe different spatial coverage.",
        "size": "This is the pixel width of each image passed to DINO.",
        "resize": "Stretching changes image proportions; letterboxing preserves them with padding.",
        "tile_anchor": "This changes where the physical tile grid is placed within each nucleus.",
        "tile_size_um": "This sets the physical amount of tissue represented by a fine tile.",
        "min_foreground_fraction": "This is the minimum nucleus volume fraction used to retain a fine tile.",
        "min_slice_foreground_fraction": "This is the minimum nucleus area fraction used to retain a 2D fine image.",
        "weight": "This controls the contribution of a feature block to the combined vector.",
        "feature dimensions": "The classifier requires the same number and meaning of feature columns.",
    }
    key = parameter.rsplit(".", 1)[-1].split("[", 1)[0]
    if key in meanings:
        return meanings[key]
    if parameter.startswith("adaptation"):
        return "Adapted coordinates must come from the same fitted transform or neural adapter."
    return "This setting changes the extracted representation."
