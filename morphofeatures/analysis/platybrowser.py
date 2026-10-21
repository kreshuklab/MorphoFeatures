"""Resolve versioned PlatyBrowser volumes and object bounds for inspection and crops."""

import hashlib
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit

import numpy as np

from morphofeatures.analysis.annotations import read_id_table
from morphofeatures.artifacts import write_json_atomic
from morphofeatures.data.remote_n5 import HttpCache, HttpN5Array, remote_url

DEFAULTS = {
    "project_url": "https://github.com/mobie/platybrowser-project",
    "revision": "2d231ac5dcd55d6d97436ad544f8fc5791fab3c7",
    "dataset": "1.0.1",
    "raw_level": 3,
    "segmentation_level": 0,
    "raw_source": "raw",
    "segmentation_source": "nuclei",
    "object_ids": "cell",
    "cache_size_mb": 1024,
    "cache_version": "1",
    "timeout_seconds": 20,
}


def _atomic_bytes(path, content):
    if path.is_file() and path.read_bytes() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _linked_file(cache, url):
    """GitHub raw endpoints return the text target of a repository symlink."""
    seen = set()
    for _ in range(12):
        if url in seen:
            raise ValueError("Cycle in published metadata links")
        seen.add(url)
        content = cache.get(url)
        target = content.decode("utf-8").strip()
        if target.startswith("../") and "\n" not in target and "\t" not in target:
            url = urljoin(url, target)
        else:
            return url, content
    raise ValueError("Too many linked metadata files")


def _image_source(cache, base, source, level, options):
    if int(level) != level or not 0 <= level <= 12:
        raise ValueError("Select an integer resolution level from 0 to 12")
    reference = source["imageData"]["bdv.n5.s3"]["relativePath"]
    xml_url, content = _linked_file(cache, urljoin(base, reference))
    if b"<!DOCTYPE" in content or b"<!ENTITY" in content:
        raise ValueError("Image XML must not contain DTD/entity declarations")
    xml = ET.fromstring(content)
    setups = xml.findall("./SequenceDescription/ViewSetups/ViewSetup")
    if len(setups) != 1 or setups[0].findtext("id") != "0":
        raise ValueError("Streaming currently supports one setup (0) per image")
    unit = setups[0].findtext("voxelSize/unit")
    factors = {"micrometer": 1.0, "um": 1.0, "µm": 1.0, "nanometer": 0.001, "nm": 0.001}
    if unit not in factors:
        raise ValueError("Image coordinates must declare micrometer or nanometer units")
    transforms = xml.findall("./ViewRegistrations/ViewRegistration/ViewTransform/affine")
    if len(transforms) != 1:
        raise ValueError("Expected one axis-aligned image transform")
    transform = np.asarray([float(v) for v in transforms[0].text.split()]).reshape(3, 4)
    linear = transform[:, :3]
    if (
        not np.isfinite(transform).all()
        or not np.allclose(linear, np.diag(np.diag(linear)))
        or np.any(np.diag(linear) <= 0)
    ):
        raise ValueError("Rotated/sheared image registrations need resampling before inspection")
    loader = xml.find("./SequenceDescription/ImageLoader")
    if loader is None or loader.get("format") != "bdv.n5.s3":
        raise ValueError("Select a published bdv.n5.s3 image source")
    root = (
        remote_url(loader.findtext("ServiceEndpoint"))
        + "/"
        + loader.findtext("BucketName").strip("/")
        + "/"
        + loader.findtext("Key").strip("/")
    )
    key = f"setup0/timepoint0/s{int(level)}"
    array = HttpN5Array(root, key, options)
    downsample = np.asarray(array.attrs.get("downsamplingFactors", []), dtype=float)
    if downsample.shape != (3,) or not np.isfinite(downsample).all() or np.any(downsample <= 0):
        raise ValueError("Resolution level must declare three positive downsampling factors")
    return {
        "url": root,
        "key": key,
        "shape_zyx": list(array.shape),
        "spacing_zyx": (np.diag(linear) * downsample * factors[unit])[::-1].tolist(),
        "origin_zyx": (transform[:, 3] * factors[unit])[::-1].tolist(),
        "xml_url": xml_url,
    }


def validate_index_reference(settings):
    """Prevent editable source fields from silently reusing another source's bounds."""
    reference = settings.get("index_reference")
    if reference:
        for field in ("segmentation", "segmentation_key", "spacing_zyx", "origin_zyx"):
            if settings.get(field) != reference[field]:
                raise ValueError(
                    "Published bounds belong to a different source/grid. Reconnect with the desired resolution settings"
                )
        if (
            settings.get("segmentation_axes", "zyx") != "zyx"
            or settings.get("segmentation_channel", 0) != 0
        ):
            raise ValueError("Published bounds require the resolved ZYX image axes and channel")


def _project_sources(options):
    options = {**DEFAULTS, **options}
    cache = HttpCache(options)
    options["cache_directory"] = str(cache.root)
    project = urlsplit(remote_url(options["project_url"]))
    parts = project.path.strip("/").split("/")
    if (
        project.hostname != "github.com"
        or len(parts) < 2
        or not all(re.fullmatch(r"[\w.-]+", p) for p in parts[:2])
    ):
        raise ValueError(
            "Enter a GitHub project URL such as https://github.com/mobie/platybrowser-project"
        )
    repository = "/".join(parts[:2]).removesuffix(".git")
    revision = str(options["revision"]).strip()
    if not revision:
        raise ValueError("A Git revision, tag or branch is required")
    if not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        revision = cache.json(
            f"https://api.github.com/repos/{repository}/commits/{quote(revision, safe='')}"
        )["sha"]
    if not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        raise ValueError("The project revision did not resolve to a commit")
    dataset = str(options["dataset"])
    if not re.fullmatch(r"[\w.-]+", dataset) or dataset in {".", ".."}:
        raise ValueError("Enter a dataset version, e.g. 1.0.1")
    base = f"https://raw.githubusercontent.com/{repository}/{revision}/data/{dataset}/"
    dataset_url, contents = _linked_file(cache, base + "dataset.json")
    sources = json.loads(contents)["sources"]
    options["revision"] = revision
    return options, cache, base, dataset_url, sources


def list_platybrowser_sources(options):
    """List published HTTP N5 images and segmentations without reading pixels."""
    options, _, _, _, sources = _project_sources(options)
    choices = {"image": [], "segmentation": []}
    for name, entry in sources.items():
        for kind in choices:
            source = entry.get(kind, {})
            if "bdv.n5.s3" in source.get("imageData", {}):
                choices[kind].append(name)
    return {"options": options, **{kind: sorted(names) for kind, names in choices.items()}}


def resolve_platybrowser(config, *, cache_directory=None):
    """Fetch metadata and tables only; never fetch image blocks or scan a volume."""
    config = deepcopy(config)
    options = {**DEFAULTS, **config.get("inspection", {}).get("platybrowser", {})}
    if cache_directory is not None:
        options.setdefault("cache_directory", str(cache_directory))
    options, cache, base, dataset_url, sources = _project_sources(options)
    revision, dataset = options["revision"], options["dataset"]
    try:
        instance_source = sources[options["segmentation_source"]]["segmentation"]
        raw_source = sources[options["raw_source"]]["image"]
        raw = _image_source(cache, base, raw_source, options["raw_level"], options)
        segmentation = _image_source(
            cache, base, instance_source, options["segmentation_level"], options
        )
        table_url, table_content = _linked_file(
            cache,
            urljoin(
                base,
                instance_source["tableData"]["tsv"]["relativePath"].rstrip("/") + "/default.tsv",
            ),
        )
    except KeyError as error:
        raise ValueError(
            "Choose published bdv.n5.s3 image/segmentation sources with an object TSV table"
        ) from error
    mapping_url = None
    if options["object_ids"] == "cell" and options["segmentation_source"] == "nuclei":
        folder = sources["cells"]["segmentation"]["tableData"]["tsv"]["relativePath"]
        mapping_url, mapping_content = _linked_file(
            cache, urljoin(base, folder.rstrip("/") + "/cells_to_nuclei.tsv")
        )
    elif options["object_ids"] not in {"nucleus", "segmentation"}:
        raise ValueError(
            "Use segmentation IDs for this source; cell-to-nucleus mapping requires nuclei"
        )
    signature = hashlib.sha256(
        json.dumps(
            [revision, dataset, segmentation, options["cache_version"]], sort_keys=True
        ).encode()
    ).hexdigest()
    directory = cache.root / "catalogs" / signature
    table_path = directory / "instances.tsv"
    _atomic_bytes(table_path, table_content)
    table = read_id_table(table_path)
    spacing, origin = (
        np.asarray(segmentation["spacing_zyx"]),
        np.asarray(segmentation["origin_zyx"]),
    )
    required = [prefix + axis for prefix in ("bb_min_", "bb_max_") for axis in "zyx"]
    if not set(required) <= set(table.columns):
        raise ValueError(
            "The selected segmentation needs a published table with bb_min/max_x/y/z physical object bounds"
        )
    lower = table[["bb_min_" + a for a in "zyx"]].to_numpy(dtype=float)
    upper = table[["bb_max_" + a for a in "zyx"]].to_numpy(dtype=float)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(upper < lower):
        raise ValueError("Published object bounds must be finite and ordered")
    # Published extrema are physical voxel-center coordinates. Include the last
    # voxel and tolerate decimal roundoff; mesh loading adds a padding voxel.
    starts = np.floor((lower - origin) / spacing + 1e-5).astype(np.int64)
    stops = np.ceil((upper - origin) / spacing - 1e-5).astype(np.int64) + 1
    if np.any(starts < -1) or np.any(stops > np.asarray(segmentation["shape_zyx"]) + 1):
        raise ValueError("Published table bounds are outside the segmentation grid")
    index = table[["label_id"]].copy()
    for dim, axis in enumerate("zyx"):
        index["bbox_min_" + axis] = np.maximum(0, starts[:, dim])
        index["bbox_max_" + axis] = np.minimum(segmentation["shape_zyx"][dim], stops[:, dim])
    index_path = directory / "objects.tsv"
    _atomic_bytes(index_path, index.to_csv(sep="\t", index=False).encode())
    mapping_path = None
    if mapping_url:
        mapping_path = directory / "cells-to-nuclei.tsv"
        _atomic_bytes(mapping_path, mapping_content)
        from morphofeatures.analysis.mesh_inspection import object_id_mapping

        object_id_mapping({"id_mapping": str(mapping_path)})  # Validate exact, unique assignments.
    remote_options = {
        key: options[key]
        for key in ("cache_directory", "cache_size_mb", "cache_version", "timeout_seconds")
    }
    provenance = {
        "project_url": options["project_url"],
        "revision": revision,
        "dataset": dataset,
        "dataset_url": dataset_url,
        "object_table_url": table_url,
        "raw_source": options["raw_source"],
        "segmentation_source": options["segmentation_source"],
        "mapping_url": mapping_url,
        "object_table_sha256": hashlib.sha256(table_content).hexdigest(),
        "raw": raw,
        "segmentation": segmentation,
    }
    if options["segmentation_source"] == "nuclei":
        provenance.update(
            nucleus_table_url=table_url, nucleus_table_sha256=provenance["object_table_sha256"]
        )
    write_json_atomic(directory / "source.json", provenance)
    mesh = {
        **config.get("inspection", {}).get("mesh", {}),
        "source": "segmentation",
        "segmentation": segmentation["url"],
        "segmentation_key": segmentation["key"],
        "segmentation_axes": "zyx",
        "segmentation_channel": 0,
        "label_kind": "instances",
        "spacing_zyx": segmentation["spacing_zyx"],
        "origin_zyx": segmentation["origin_zyx"],
        "unit": "um",
        "object_index": str(index_path),
        "index_ids": "segmentation",
        "id_mapping": str(mapping_path) if mapping_path else None,
        "remote_options": remote_options,
        "remote_provenance": provenance,
    }
    mesh["index_reference"] = {
        key: mesh[key] for key in ("segmentation", "segmentation_key", "spacing_zyx", "origin_zyx")
    }
    config["data"] = {
        "source": "platybrowser",
        "raw": raw["url"],
        "raw_key": raw["key"],
        "raw_spacing_zyx": raw["spacing_zyx"],
        "raw_origin_zyx": raw["origin_zyx"],
        "object_table": str(table_path),
        "id_mapping": mesh["id_mapping"],
        "remote_options": remote_options,
    }
    config.setdefault("inspection", {}).update(
        platybrowser={**options, "revision": revision}, mesh=mesh
    )
    return config


def resolve_platybrowser_preprocessing(settings):
    """Resolve a Workflow source into aligned, catalog-indexed crop preparation."""
    options = {**settings.get("platybrowser", {}), "object_ids": "segmentation"}
    resolved = resolve_platybrowser({"inspection": {"platybrowser": options}})
    mesh, data = resolved["inspection"]["mesh"], resolved["data"]
    raw = mesh["remote_provenance"]["raw"]
    segmentation = mesh["remote_provenance"]["segmentation"]
    if raw["shape_zyx"] != segmentation["shape_zyx"] or any(
        not np.allclose(raw[key], segmentation[key], rtol=1e-6, atol=1e-8)
        for key in ("spacing_zyx", "origin_zyx")
    ):
        raise ValueError(
            "The selected PlatyBrowser volumes are on different grids. "
            f"Raw: {raw['shape_zyx']} at {raw['spacing_zyx']} µm; "
            f"segmentation: {segmentation['shape_zyx']} at {segmentation['spacing_zyx']} µm. "
            "Choose aligned resolution levels (raw 3 / nuclei 0 for dataset 1.0.1), "
            "or resample the volumes locally first."
        )
    return {
        **deepcopy(settings),
        **{
            key: mesh[key]
            for key in (
                "segmentation",
                "segmentation_key",
                "segmentation_axes",
                "segmentation_channel",
                "spacing_zyx",
                "origin_zyx",
                "unit",
                "object_index",
                "index_reference",
                "remote_options",
                "remote_provenance",
            )
        },
        "source": "platybrowser",
        "raw": data["raw"],
        "raw_key": data["raw_key"],
        "raw_axes": "zyx",
        "raw_channel": 0,
        "raw_spacing_zyx": raw["spacing_zyx"],
        "raw_origin_zyx": raw["origin_zyx"],
        "segmentation_spacing_zyx": segmentation["spacing_zyx"],
        "segmentation_origin_zyx": segmentation["origin_zyx"],
        "platybrowser": resolved["inspection"]["platybrowser"],
        "inspection_config": resolved["inspection"],
    }
