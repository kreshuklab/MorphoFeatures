"""Stream categorical labels back onto the original instance segmentation."""

import itertools
import json
from contextlib import ExitStack
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from morphofeatures.analysis.annotations import read_id_table
from morphofeatures.analysis.visualization import category_colors
from morphofeatures.artifacts import write_json_atomic
from morphofeatures.data.preprocessing import triple
from morphofeatures.data.volumes import open_volume
from morphofeatures.representations import fingerprint_file


def _zarr_backend(format):
    import zarr

    if format == "zarr3" and int(zarr.__version__.split(".")[0]) < 3:
        import inspect

        try:
            import z5py

            if "zarr_format" in inspect.signature(z5py.File).parameters:
                return z5py
        except ImportError:
            pass
        raise ValueError("Zarr v3 export requires zarr-python >=3 or z5py with Zarr v3 support")
    return zarr


def validate_volume_export(settings):
    format = settings.get("output_format", "h5")
    if format not in {"h5", "zarr2", "zarr3"}:
        raise ValueError("Volume output_format must be h5, zarr2, or zarr3")
    if format == "h5":
        import h5py  # noqa: F401
    else:
        _zarr_backend(format)
    for key in ("labels", "segmentation"):
        if not settings.get(key) or not Path(settings[key]).exists():
            raise ValueError(f"Volume export requires an existing {key} path")
    layers = settings.get("layers", ["known_label", "predicted_label", "cluster"])
    if not layers or not set(layers) <= {"known_label", "predicted_label", "cluster"}:
        raise ValueError("Choose known_label, predicted_label and/or cluster layers")
    triple(settings.get("block_shape", [64, 64, 64]), "block_shape", integer=True)
    triple(settings.get("spacing_zyx", [1, 1, 1]), "spacing_zyx")
    triple(settings.get("origin_zyx", [0, 0, 0]), "origin_zyx", positive=False)
    labels = read_id_table(settings["labels"])
    if not set(layers) <= set(labels):
        raise ValueError(f"Label table is missing requested layers: {set(layers) - set(labels)}")
    if settings.get("id_mapping"):
        mapping = read_id_table(settings["id_mapping"])
        column = settings.get("mapping_column", "segmentation_id")
        mapped = read_id_table(settings["id_mapping"], column)
        mapping[column] = mapped[column]
        labels = labels.merge(
            mapping[["label_id", column]], on="label_id", how="left", validate="one_to_one"
        )
        if labels[column].isna().any():
            raise ValueError("The ID mapping must cover every exported embedding object")
        labels["segmentation_id"] = labels[column].astype(np.int64)
    else:
        labels["segmentation_id"] = labels.label_id
    if labels.segmentation_id.duplicated().any():
        raise ValueError(
            "Multiple embedding objects map to one segmentation object; resolve this ambiguity before export"
        )
    with open_volume(
        settings["segmentation"],
        settings.get("segmentation_key"),
        settings.get("segmentation_axes", "zyx"),
        settings.get("segmentation_channel", 0),
    ) as volume:
        if not np.issubdtype(volume.dtype, np.integer):
            raise ValueError("The source segmentation must contain integer instance IDs")
    return labels.sort_values("segmentation_id").reset_index(drop=True)


def export_label_volumes(settings, destination, progress=None):
    labels = validate_volume_export(settings)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    format = settings.get("output_format", "h5")
    target = destination / ("labels.h5" if format == "h5" else "labels.zarr")
    if target.exists():
        raise FileExistsError(f"Volume export already exists: {target}")
    layers = list(
        dict.fromkeys(settings.get("layers", ["known_label", "predicted_label", "cluster"]))
    )
    class_colors = category_colors(
        [v for name in ("known_label", "predicted_label") if name in labels for v in labels[name]]
    )
    lut, codes = [], {}
    for layer in layers:
        values = labels[layer].astype("string")
        # TSV may represent integer clusters as floats when some IDs have no projection.
        if layer == "cluster":
            values = pd.to_numeric(labels[layer], errors="raise").astype("Int64").astype("string")
        palette = category_colors(values) if layer == "cluster" else class_colors
        categories = sorted(values.dropna().unique())
        numbers = {value: index + 1 for index, value in enumerate(categories)}
        codes[layer] = values.map(numbers).fillna(0).to_numpy(dtype=np.uint32)
        lut.append(
            {"layer": layer, "value": 0, "name": "Background / unassigned", "color": "#000000"}
        )
        lut.extend(
            {"layer": layer, "value": numbers[value], "name": value, "color": palette[value]}
            for value in categories
        )
        labels[layer + "_value"] = codes[layer]
    ids = labels.segmentation_id.to_numpy(dtype=np.int64)
    matched = np.zeros(len(ids), dtype=bool)
    blocks = triple(settings.get("block_shape", [64, 64, 64]), "block_shape", integer=True)
    with ExitStack() as stack:
        volume = stack.enter_context(
            open_volume(
                settings["segmentation"],
                settings.get("segmentation_key"),
                settings.get("segmentation_axes", "zyx"),
                settings.get("segmentation_channel", 0),
            )
        )
        shape = volume.shape
        chunks = tuple(min(int(a), int(b)) for a, b in zip(blocks, shape))
        if format == "h5":
            import h5py

            store = stack.enter_context(h5py.File(target, "x"))
            create = partial(
                store.create_dataset, shape=shape, chunks=chunks, dtype="uint32", compression="gzip"
            )
        else:
            backend = _zarr_backend(format)
            if backend.__name__ == "z5py":
                store = stack.enter_context(
                    backend.File(str(target), "x", use_zarr_format=True, zarr_format=3)
                )
                create = partial(
                    store.create_dataset,
                    shape=shape,
                    chunks=chunks,
                    dtype="uint32",
                    compression="gzip",
                )
            elif int(backend.__version__.split(".")[0]) >= 3:
                store = backend.open_group(
                    str(target), mode="w-", zarr_format=3 if format == "zarr3" else 2
                )
                create = partial(store.create_array, shape=shape, chunks=chunks, dtype="uint32")
            else:
                store = backend.open_group(str(target), mode="w-")
                create = partial(store.create_dataset, shape=shape, chunks=chunks, dtype="uint32")
        store.attrs.update(
            {
                "axes": "zyx",
                "spacing_zyx": list(settings.get("spacing_zyx", [1, 1, 1])),
                "origin_zyx": list(settings.get("origin_zyx", [0, 0, 0])),
                "unit": settings.get("unit", "voxel"),
                "schema": "morphofeatures.label_volumes.v1",
                "label_lookup": json.dumps(lut),
            }
        )
        arrays = {name: create(name) for name in layers}
        rgb_arrays, rgb_lookup = {}, {}
        if settings.get("include_rgb", False):
            for name in layers:
                rgb_arrays[name] = create(
                    name + "_rgb", shape=shape + (3,), chunks=chunks + (3,), dtype="uint8"
                )
                rgb_arrays[name].attrs.update(
                    {"axes": "zyxc", "_ARRAY_DIMENSIONS": ["z", "y", "x", "c"]}
                )
                colors = [row["color"].lstrip("#") for row in lut if row["layer"] == name]
                rgb_lookup[name] = np.asarray(
                    [[int(color[i : i + 2], 16) for i in (0, 2, 4)] for color in colors],
                    dtype=np.uint8,
                )
        for name, array in arrays.items():
            array.attrs.update(
                {
                    "axes": "zyx",
                    "_ARRAY_DIMENSIONS": ["z", "y", "x"],
                    "label_lookup": json.dumps([row for row in lut if row["layer"] == name]),
                }
            )
        total = int(np.prod(np.ceil(np.asarray(shape) / blocks)))
        for count, start in enumerate(
            itertools.product(*(range(0, s, int(b)) for s, b in zip(shape, blocks))), 1
        ):
            slices = tuple(slice(a, min(a + int(b), s)) for a, b, s in zip(start, blocks, shape))
            raw = volume.read(slices)
            if raw.min() < 0 or int(raw.max()) > np.iinfo(np.int64).max:
                raise ValueError("Source segmentation IDs must fit nonnegative int64")
            # Mixing uint64 voxels with int64 IDs promotes comparisons to float64,
            # which can merge neighboring IDs above 2**53. Validate, then match exactly.
            raw = raw.astype(np.int64, copy=False)
            values, inverse = np.unique(raw, return_inverse=True)
            positions = np.searchsorted(ids, values)
            valid = (positions < len(ids)) & (values != 0)
            valid[valid] &= ids[positions[valid]] == values[valid]
            matched[positions[valid]] = True
            for name, array in arrays.items():
                mapping = np.zeros(len(values), dtype=np.uint32)
                mapping[valid] = codes[name][positions[valid]]
                categorical = mapping[inverse].reshape(raw.shape)
                array[slices] = categorical
                if name in rgb_arrays:
                    rgb_arrays[name][slices + (slice(None),)] = rgb_lookup[name][categorical]
            if progress:
                progress(blocks=count, total_blocks=total)
    pd.DataFrame(lut).to_csv(destination / "label_lookup.tsv", sep="\t", index=False)
    labels["present_in_segmentation"] = matched
    labels.to_csv(destination / "object_mapping.tsv", sep="\t", index=False)
    return write_json_atomic(
        destination / "volume_export.json",
        {
            "schema": "morphofeatures.label_volumes.v1",
            "settings": settings,
            "source": fingerprint_file(settings["segmentation"]),
            "labels_source": fingerprint_file(settings["labels"]),
            "container": str(target),
            "layers": layers,
            "rgb_layers": [name + "_rgb" for name in rgb_arrays],
            "shape_zyx": list(shape),
            "axes": "zyx",
            "lookup": "label_lookup.tsv",
            "objects": "object_mapping.tsv",
            "matched_objects": int(matched.sum()),
            "absent_object_ids": labels.loc[~matched, "label_id"].tolist(),
            "semantics": "Categorical uint32 values, not instance IDs. Zero is background or unassigned; colors and original IDs are in the lookup tables. Predictions retain their held-out/fitted provenance in object_mapping.tsv.",
        },
    )
