"""Bounded object views and shareable nearest-neighbor image tables."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from morphofeatures.analysis.visualization import figure_bytes
from morphofeatures.data.crop_storage import load_crop_array, open_crop_array


def input_config_for_result(metadata, representation=None):
    from morphofeatures.configuration_editor import load_document

    explicit = metadata.get("settings", {}).get("input_config")
    if explicit:
        return load_document(explicit)
    entry = metadata.get("representations", {}).get(representation, metadata)
    return entry.get("extraction", {}).get("settings", {}).get("config", {})


def read_object_crops(config, object_ids, *, source="prepared", max_side=128):
    """Read selected crops only; original-volume views use the saved object index."""
    data = config.get("data", {})
    requested = [int(value) for value in object_ids]
    if not 1 <= len(requested) <= 25 or not 8 <= int(max_side) <= 256:
        raise ValueError("Inspect 1–25 objects with a view side between 8 and 256 voxels")
    if data.get("source") == "platybrowser":
        from morphofeatures.analysis.annotations import read_id_table
        from morphofeatures.analysis.mesh_inspection import object_id_mapping
        from morphofeatures.data.volumes import open_volume

        if source != "original":
            raise ValueError(
                "PlatyBrowser supplies original raw views; select original as the image source"
            )
        table = read_id_table(data["object_table"]).set_index("label_id")
        mapping = object_id_mapping(data) if data.get("id_mapping") else None
        spacing, origin = np.asarray(data["raw_spacing_zyx"]), np.asarray(data["raw_origin_zyx"])
        with open_volume(
            data["raw"], data["raw_key"], remote_options=data.get("remote_options")
        ) as raw:
            for object_id in requested:
                nucleus = mapping.get(object_id) if mapping is not None else object_id
                if nucleus not in table.index:
                    raise ValueError(
                        f"Object {object_id} has no nucleus in the published table/mapping"
                    )
                row = table.loc[nucleus]
                lo = row[["bb_min_" + a for a in "zyx"]].to_numpy(dtype=float)
                hi = row[["bb_max_" + a for a in "zyx"]].to_numpy(dtype=float)
                center = np.rint(((lo + hi) / 2 - origin) / spacing).astype(int)
                shape = np.minimum(np.ceil((hi - lo) / spacing).astype(int) + 5, int(max_side))
                start, stop = (
                    np.maximum(0, center - shape // 2),
                    np.minimum(raw.shape, center - shape // 2 + shape),
                )
                if np.any(stop <= start):
                    raise ValueError("Published object position is outside the raw volume")
                crop = raw.read(tuple(slice(int(a), int(b)) for a, b in zip(start, stop)))
                yield (
                    object_id,
                    crop,
                    "Streamed PlatyBrowser raw around published nucleus bounds (bounded field of view)",
                )
        return
    if source == "original" and data.get("source") != "n5_masked_patches":
        from morphofeatures.data.volumes import open_volume

        metadata_path = Path(data.get("preprocessing") or "")
        if not metadata_path.is_file():
            raise ValueError(
                "Original-volume inspection requires the preprocessing.json referenced by mae_config.yaml"
            )
        metadata = json.loads(metadata_path.read_text())
        settings = metadata["settings"]
        objects = pd.read_csv(
            metadata_path.parent / "objects.tsv", sep="\t", dtype={"label_id": np.int64}
        ).set_index("label_id")
        with open_volume(
            settings["raw"],
            settings.get("raw_key"),
            settings.get("raw_axes", "zyx"),
            settings.get("raw_channel", 0),
        ) as raw:
            for object_id in requested:
                if object_id not in objects.index:
                    raise ValueError(f"Object {object_id} is absent from the preprocessing index")
                row = objects.loc[object_id]
                center = np.asarray(
                    [(row[f"bbox_min_{a}"] + row[f"bbox_max_{a}"]) // 2 for a in "zyx"], dtype=int
                )
                shape = np.minimum(np.asarray(metadata["crop_shape"], dtype=int), max_side)
                start, stop = (
                    np.maximum(0, center - shape // 2),
                    np.minimum(raw.shape, center - shape // 2 + shape),
                )
                yield (
                    object_id,
                    raw.read(tuple(slice(int(a), int(b)) for a, b in zip(start, stop))),
                    "Original raw volume",
                )
        return
    if data.get("source") == "n5_masked_patches":
        import z5py

        from morphofeatures.data.n5 import load_patch_index
        from morphofeatures.data.volumes import open_volume

        index = load_patch_index(
            Path(data["positions_container"]),
            data.get("positions_key", "positions"),
            data.get("ids_key", "ids"),
        )
        if source == "original":
            if not data.get("qc_raw_container") or not data.get("position_to_raw_scale_zyx"):
                raise ValueError(
                    "Grouped original-volume inspection requires qc_raw_container, qc_raw_key and position_to_raw_scale_zyx in the data YAML"
                )
            scale = np.asarray(data["position_to_raw_scale_zyx"], dtype=float)
            with open_volume(
                data["qc_raw_container"],
                data.get("qc_raw_key"),
                "zyx",
                remote_options=data.get("remote_options"),
            ) as raw:
                for object_id in requested:
                    row = np.searchsorted(index.unique_label_ids, object_id)
                    if (
                        row == len(index.unique_label_ids)
                        or index.unique_label_ids[row] != object_id
                    ):
                        raise ValueError(
                            f"Object {object_id} is absent from the grouped patch index"
                        )
                    first, count = int(index.first_indices[row]), int(index.counts[row])
                    center = np.rint(
                        np.mean(index.positions_zyx[first : first + count], axis=0) * scale
                    ).astype(int)
                    start = np.maximum(0, center - max_side // 2)
                    stop = np.minimum(raw.shape, center - max_side // 2 + max_side)
                    if np.any(stop <= start):
                        raise ValueError(
                            "Mapped object center is outside the raw volume; check coordinate scaling"
                        )
                    crop = np.array(
                        raw.read(tuple(slice(int(a), int(b)) for a, b in zip(start, stop))),
                        copy=True,
                    )
                    yield (
                        object_id,
                        crop,
                        "Raw volume around mean patch position (bounded field of view)",
                    )
            return
        with z5py.File(data["patches_container"], "r") as store:
            patches = store[data.get("patches_key", "patches")]
            for object_id in requested:
                row = np.searchsorted(index.unique_label_ids, object_id)
                if row == len(index.unique_label_ids) or index.unique_label_ids[row] != object_id:
                    raise ValueError(f"Object {object_id} is absent from the grouped patch index")
                patch = int(index.first_indices[row] + index.counts[row] // 2)
                slices = tuple(
                    slice(
                        max(0, (s - max_side) // 2), max(0, (s - max_side) // 2) + min(s, max_side)
                    )
                    for s in patches.shape[-3:]
                )
                yield (
                    object_id,
                    np.asarray(patches[(patch,) + slices]),
                    "Representative stored patch (not the whole nucleus)",
                )
        return
    ids = load_crop_array(data, "label_ids")
    rows = {int(value): i for i, value in enumerate(ids)}
    with open_crop_array(data) as crops:
        for object_id in requested:
            if object_id not in rows:
                raise ValueError(f"Object {object_id} is absent from the crop IDs")
            shape = crops.shape[-3:]
            slices = tuple(
                slice(max(0, (s - max_side) // 2), min(s, (s - min(s, max_side)) // 2 + max_side))
                for s in shape
            )
            indexing = (rows[object_id],) + ((0,) if crops.ndim == 5 else ()) + slices
            yield (
                object_id,
                np.array(crops[indexing], copy=True),
                "Prepared crop" + (" (center clipped)" if max(shape) > max_side else ""),
            )


def _wrap_figure_text(value, width, renderer, font):
    """Wrap to a measured width in points, including long unbroken class names."""
    lines, current = [], ""
    for word in str(value).split():
        candidate = f"{current} {word}".strip()
        if renderer.get_text_width_height_descent(candidate, font, False)[0] <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = ""
        for character in word:
            if (
                current
                and renderer.get_text_width_height_descent(current + character, font, False)[0]
                > width
            ):
                lines.append(current)
                current = ""
            current += character
    return lines + ([current] if current else [])


def neighbor_figure(images, neighbors, labels=None):
    """An image grid with separate, measured header, annotation and caption bands."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D

    columns = min(4, len(images))
    if not columns:
        raise ValueError("Select at least one object for the neighbor figure")
    side, gap, left = 1.75, 0.18, 1.10
    width = left + columns * side + (columns - 1) * gap + 0.12
    # At 72 dpi the measurement renderer's pixels are typographic points.
    figure = Figure(figsize=(width, 1), dpi=72, facecolor="white")
    FigureCanvasAgg(figure)
    renderer = figure.canvas.get_renderer()
    font = FontProperties(family="DejaVu Sans", size=8)
    line_height = 11 / 72
    lookup = labels.set_index("label_id") if labels is not None else pd.DataFrame()
    fields = [("distance", "Distance")] if "distance" in neighbors else []
    fields.extend(
        (key, title)
        for key, title in (
            ("known_label", "Known type"),
            ("predicted_label", "Predicted type"),
            ("cluster", "Cluster"),
        )
        if key in lookup
    )
    cells = []
    for i, (object_id, _, _) in enumerate(images):
        item = lookup.loc[object_id] if object_id in lookup.index else {}
        values = []
        for key, _ in fields:
            value = item.get(key)
            if key == "distance":
                value = f"{float(neighbors.iloc[i].distance):.4g}"
            elif pd.isna(value):
                value = "Unlabeled" if key != "cluster" else "Unassigned"
            elif key == "cluster":
                value = str(int(value))
            values.append(_wrap_figure_text(value, (side - 0.06) * 72, renderer, font))
        cells.append(values)
    groups = [list(range(i, min(i + columns, len(images)))) for i in range(0, len(images), columns)]
    row_heights = [
        [max(len(cells[i][j]) for i in group) * line_height + 0.08 for j in range(len(fields))]
        for group in groups
    ]
    header_height, plane_gap, table_gap = 0.48, 0.08, 0.19
    image_height = 3 * side + 2 * plane_gap
    block_heights = [header_height + image_height + table_gap + sum(rows) for rows in row_heights]
    descriptions = "; ".join(dict.fromkeys(description for _, _, description in images))
    caption = (
        f"Source: {descriptions}. XY / XZ / YZ center slices. "
        "Contrast scaled per object (1st–99th percentile)."
    )
    caption_lines = _wrap_figure_text(caption, (width - 0.3) * 72, renderer, font)
    title_height, block_gap = 0.55, 0.32
    height = (
        title_height
        + sum(block_heights)
        + block_gap * (len(groups) - 1)
        + 0.24
        + len(caption_lines) * line_height
        + 0.15
    )
    figure.set_size_inches(width, height)

    def text(x, y, value, *, size=8, weight="normal", color="#30343b", **kwargs):
        return figure.text(
            x / width,
            1 - y / height,
            value,
            fontsize=size,
            fontfamily="DejaVu Sans",
            fontweight=weight,
            color=color,
            va="top",
            parse_math=False,
            **kwargs,
        )

    def rule(y, color="#d6d9dd"):
        figure.add_artist(
            Line2D(
                [0.15 / width, (width - 0.12) / width],
                [1 - y / height] * 2,
                transform=figure.transFigure,
                linewidth=0.5,
                color=color,
            )
        )

    text(
        0.15,
        0.12,
        "Selected objects" if "role" in neighbors else "Embedding neighborhood",
        size=12,
        weight="bold",
    )
    top = title_height
    for group, heights, block_height in zip(groups, row_heights, block_heights):
        for col, i in enumerate(group):
            object_id, crop, _ = images[i]
            x = left + col * (side + gap)
            text(
                x,
                top,
                neighbors.iloc[i]["role"]
                if "role" in neighbors
                else ("Selected" if i == 0 else f"Neighbor {i}"),
                size=10,
                weight="bold",
                color="#275f85" if i == 0 else "#20252b",
            )
            text(x, top + 0.23, f"ID {object_id}")
            crop = np.asarray(crop, dtype=np.float32)
            finite = crop[np.isfinite(crop)]
            low, high = np.percentile(finite, [1, 99]) if finite.size else (0, 1)
            if high <= low:
                low, high = float(finite.min()), float(finite.max()) + 1e-6
            for plane in range(3):
                y = top + header_height + plane * (side + plane_gap)
                axis = figure.add_axes(
                    [x / width, 1 - (y + side) / height, side / width, side / height]
                )
                axis.imshow(
                    np.take(crop, crop.shape[plane] // 2, axis=plane),
                    cmap="gray",
                    vmin=low,
                    vmax=high,
                    interpolation="nearest",
                )
                axis.set_axis_off()
        for plane, name in enumerate(("XY", "XZ", "YZ")):
            text(
                left - 0.15,
                top + header_height + plane * (side + plane_gap) + side / 2 - 0.06,
                name,
                weight="bold",
                ha="right",
            )
        table_top = top + header_height + image_height + table_gap
        rule(table_top - 0.1)
        for j, ((_, name), row_height) in enumerate(zip(fields, heights)):
            text(0.15, table_top, name, color="#626972")
            for col, i in enumerate(group):
                for line, value in enumerate(cells[i][j]):
                    text(left + col * (side + gap), table_top + line * line_height, value)
            table_top += row_height
        top += block_height + block_gap
    caption_top = top - block_gap + 0.24
    rule(caption_top - 0.1)
    for line, value in enumerate(caption_lines):
        text(0.15, caption_top + line * line_height, value, color="#626972")
    return figure


def neighbor_sheet(config, neighbors, labels=None, *, source="prepared", max_side=128):
    import matplotlib.pyplot as plt

    images = list(read_object_crops(config, neighbors.label_id, source=source, max_side=max_side))
    figure = neighbor_figure(images, neighbors, labels)
    try:
        # Keep SVG lettering editable and produce a print-resolution PNG.
        with plt.rc_context({"svg.fonttype": "none"}):
            return {
                extension: figure_bytes(figure, extension, dpi=300) for extension in ("svg", "png")
            }
    finally:
        plt.close(figure)
