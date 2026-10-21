"""Metadata-only connection of physical context graphs to streamed segmentations."""

import hashlib
from copy import deepcopy
from urllib.parse import urljoin

import numpy as np

from morphofeatures.analysis.platybrowser import (
    _atomic_bytes,
    _image_source,
    _linked_file,
    _project_sources,
)
from morphofeatures.data.ome_zarr import resolve_ome_zarr_source
from morphofeatures.workflow_validation import FieldError


def context_catalog(options):
    options, _, _, _, sources = _project_sources(options)
    formats = {"bdv.n5.s3", "ome.zarr", "ome.zarr.s3"}
    return {
        "options": options,
        "segmentation": sorted(
            name
            for name, entry in sources.items()
            if formats.intersection(entry.get("segmentation", {}).get("imageData", {}))
        ),
    }


def resolve_context_source(settings):
    settings = deepcopy(settings)
    source = settings.get("spatial_source", "manual")
    if source == "manual":
        return settings
    if source not in {"ome_zarr", "platybrowser"}:
        raise FieldError(
            "Choose manual, ome_zarr or platybrowser as the spatial source.", "spatial_source"
        )
    if settings.get("graph_mode", "adjacency_table") != "segmentation":
        raise FieldError(
            "Streaming volumes defines touching-label neighbors; choose segmentation graph mode, or Manual for a position/edge table.",
            "graph_mode",
            "spatial_source",
        )
    options = settings.get("remote_options", {})
    if source == "ome_zarr":
        spec = settings.get("ome_zarr", {})
        if not spec.get("url"):
            raise FieldError(
                "Enter the segmentation OME-Zarr root URL, then choose its resolution. A raw EM image cannot define instance contacts.",
                "ome_zarr.url",
            )
        resolved = resolve_ome_zarr_source(spec, options)
        options = {**options, "format": "ome_zarr"}
        provenance = {"format": "ome_zarr", "url": spec["url"], "selection": spec}
    else:
        selected, cache, base, dataset_url, sources = _project_sources(
            settings.get("platybrowser", {})
        )
        name = selected.get("segmentation_source", "cells")
        try:
            image_source = sources[name]["segmentation"]
            formats = image_source["imageData"]
        except KeyError as error:
            raise FieldError(
                "Choose a published instance segmentation from the context source list.",
                "platybrowser.segmentation_source",
            ) from error
        level = int(selected.get("segmentation_level", 0))
        if "bdv.n5.s3" in formats:
            resolved = _image_source(cache, base, image_source, level, selected)
            resolved.update(axes="zyx", channel=0, timepoint=0, unit="um")
            format_ = "n5"
        else:
            spec = formats.get("ome.zarr.s3", formats.get("ome.zarr", {}))
            url = spec.get("s3Address") or urljoin(base, spec.get("relativePath", ""))
            if url.startswith("s3://"):
                bucket, key = url[5:].split("/", 1)
                url = f"https://{bucket}.s3.amazonaws.com/{key}"
            resolved = resolve_ome_zarr_source(
                {"url": url, "level": level, "channel": spec.get("channel", 0)}, selected
            )
            format_ = "ome_zarr"
        options = {**selected, "format": format_}
        settings["platybrowser"] = selected
        provenance = {"format": format_, "dataset_url": dataset_url, "project": selected}
        if image_source.get("tableData", {}).get("tsv"):
            table_url = urljoin(
                base, image_source["tableData"]["tsv"]["relativePath"].rstrip("/") + "/default.tsv"
            )
            _, content = _linked_file(cache, table_url)
            table_path = (
                cache.root / "context_tables" / (hashlib.sha256(content).hexdigest() + ".tsv")
            )
            table_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_bytes(table_path, content)
            provenance["object_table"] = str(table_path)
        if not settings.get("source_version"):
            settings["source_version"] = selected["revision"]
    from morphofeatures.data.volumes import inspect_volume

    info = inspect_volume(
        resolved["url"],
        resolved["key"],
        resolved.get("axes", "zyx"),
        resolved.get("channel", 0),
        timepoint=resolved.get("timepoint", 0),
        remote_options=options,
    )
    if not np.issubdtype(np.dtype(info["dtype"]), np.integer):
        raise FieldError(
            "Choose integer instance labels, not a raw image or probability map.", "segmentation"
        )
    settings.update(
        segmentation=resolved["url"],
        segmentation_key=resolved["key"],
        segmentation_axes=resolved.get("axes", "zyx"),
        segmentation_channel=resolved.get("channel", 0),
        segmentation_timepoint=resolved.get("timepoint", 0),
        remote_options=options,
        segmentation_spacing_zyx=resolved["spacing_zyx"],
        segmentation_origin_zyx=resolved["origin_zyx"],
        segmentation_unit=resolved.get("unit", "um"),
        spatial_source_metadata={**provenance, "volume": info, "resolved": resolved},
    )
    return settings
