"""Physical, bounded DINO views of indexed instances (coordinates are voxel centers)."""

from __future__ import annotations

import hashlib
import itertools
import json
import sqlite3
import time
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates

from morphofeatures.analysis.annotations import read_id_table
from morphofeatures.artifacts import write_json_atomic
from morphofeatures.data.bounded_cache import ObjectMaskVolume, ReadVolume
from morphofeatures.data.object_eligibility import (
    ObjectEligibilityError,
    object_geometry,
    validate_object_limits,
)
from morphofeatures.data.preprocessing import _index_objects, triple
from morphofeatures.data.volumes import open_volume
from morphofeatures.workflow_validation import FieldError, at_fields

VIEW_VERSION = "physical-dino-v4-boundary-policy"


def resolve_source(source):
    """Resolve metadata without reading pixels; unlike crop preparation, grids may differ."""
    s = deepcopy(source)
    if s.get("source") == "platybrowser":
        from morphofeatures.analysis.platybrowser import resolve_platybrowser

        config = resolve_platybrowser({"inspection": {"platybrowser": s.get("platybrowser", {})}})
        mesh = config["inspection"]["mesh"]
        s["platybrowser"] = config["inspection"]["platybrowser"]
        for key in ("object_index", "id_mapping", "remote_options", "remote_provenance"):
            s[key] = mesh.get(key)
        for kind in ("raw", "segmentation"):
            info = mesh["remote_provenance"][kind]
            s[kind], s[kind + "_key"] = info["url"], info["key"]
            s[kind + "_spacing_zyx"] = info["spacing_zyx"]
            s[kind + "_origin_zyx"] = info["origin_zyx"]
            s[kind + "_axes"] = "zyx"
            s[kind + "_channel"] = 0
            s[kind + "_timepoint"] = 0
        s["unit"] = "um"
        s["inspection_config"] = config
    elif s.get("source") == "ome_zarr":
        from morphofeatures.data.ome_zarr import resolve_ome_zarr_source

        options = {**s.get("remote_options", {}), "format": "ome_zarr"}
        for kind in ("raw", "segmentation"):
            info = resolve_ome_zarr_source(s.get("ome_zarr", {}).get(kind, {}), options)
            s[kind] = info["url"]
            for key in ("key", "axes", "channel", "timepoint", "spacing_zyx", "origin_zyx"):
                s[kind + "_" + key] = info[key]
            if info["unit"] != "um":
                raise ValueError(
                    "Physical DINO views require declared physical units in OME-Zarr metadata"
                )
        s.update(unit="um", remote_options=options)
    factor = {"um": 1.0, "micrometer": 1.0, "nm": 0.001}.get(s.get("unit"))
    if factor is None:
        raise FieldError(
            "Physical DINO inputs require unit=um or nm and measured voxel spacing", "unit"
        )
    for kind in ("raw", "segmentation"):
        if not s.get(kind):
            raise FieldError(f"Select a {kind} volume for physical DINO views", kind)
        s[kind + "_spacing_zyx"] = (
            triple(s.get(kind + "_spacing_zyx") or s.get("spacing_zyx", []), kind + " spacing")
            * factor
        ).tolist()
        s[kind + "_origin_zyx"] = (
            triple(
                s.get(kind + "_origin_zyx") or s.get("origin_zyx", [0, 0, 0]),
                kind + " origin",
                positive=False,
            )
            * factor
        ).tolist()
    s["unit"] = "um"
    return s


def source_identity(s):
    """Dataset namespace independent of a particular DINO view or checkpoint."""
    values = {
        k: s.get(k)
        for k in (
            "raw",
            "raw_key",
            "segmentation",
            "segmentation_key",
            "id_mapping",
            "data_version",
            "raw_channel",
            "raw_timepoint",
            "segmentation_channel",
            "segmentation_timepoint",
        )
    }
    for key in ("raw", "segmentation", "id_mapping"):
        path = Path(str(s.get(key) or ""))
        if s.get(key) and path.exists():
            stat = path.stat()
            values[key] = [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
    values["remote"] = s.get("remote_provenance", s.get("ome_zarr"))
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def open_source(stack, s, kind):
    return stack.enter_context(
        open_volume(
            s[kind],
            s.get(kind + "_key"),
            s.get(kind + "_axes", "zyx"),
            s.get(kind + "_channel", 0),
            timepoint=s.get(kind + "_timepoint", 0),
            remote_options=s.get("remote_options"),
        )
    )


def validate_source(source):
    from morphofeatures.data.object_selection import resolve_selection

    resolve_selection(source.get("object_selection"))
    validate_object_limits(source)
    s = resolve_source(source)
    with ExitStack() as stack:
        volumes = {}
        for kind in ("raw", "segmentation"):
            with at_fields(kind, kind + "_key", kind + "_axes"):
                volumes[kind] = open_source(stack, s, kind)
        if not np.issubdtype(volumes["segmentation"].dtype, np.integer):
            raise FieldError(
                "Instance segmentation must contain integer label IDs",
                "segmentation",
                "segmentation_key",
            )
        boxes = []
        for kind, volume in volumes.items():
            origin, spacing = np.array(s[kind + "_origin_zyx"]), np.array(s[kind + "_spacing_zyx"])
            boxes.append((origin - spacing / 2, origin + (np.array(volume.shape) - 0.5) * spacing))
        if np.any(np.maximum(boxes[0][0], boxes[1][0]) >= np.minimum(boxes[0][1], boxes[1][1])):
            raise FieldError(
                "Raw and segmentation physical extents do not overlap",
                "raw_origin_zyx",
                "segmentation_origin_zyx",
                "raw_spacing_zyx",
                "segmentation_spacing_zyx",
            )
        s["raw_shape_zyx"], s["segmentation_shape_zyx"] = (
            list(volumes[k].shape) for k in ("raw", "segmentation")
        )
    if (
        s.get("segmentation", "").startswith(("http:", "https:"))
        and not s.get("roi")
        and not s.get("object_index")
    ):
        raise FieldError(
            "Select a bounded ROI or an object index before indexing a remote segmentation",
            "roi",
            "object_index",
        )
    if s.get("roi") is not None:
        with at_fields("roi"):
            roi = np.asarray(s["roi"], dtype=float)
            if (
                roi.shape != (2, 3)
                or not np.isfinite(roi).all()
                or not np.equal(roi, np.rint(roi)).all()
                or np.any(roi[0] < 0)
                or np.any(roi[1] > s["segmentation_shape_zyx"])
                or np.any(roi[1] <= roi[0])
            ):
                raise ValueError("Use integer ZYX start/stop bounds inside the segmentation grid")
    for key in ("object_index", "id_mapping"):
        if s.get(key) and not Path(s[key]).is_file():
            raise FieldError("The selected table does not exist", key)
    if s.get("id_mapping"):
        from morphofeatures.analysis.mesh_inspection import object_id_mapping

        with at_fields("id_mapping"):
            object_id_mapping(s)
    return s


def index_source(s, destination, *, progress=None):
    """Reuse bounds or perform a worker-side chunk scan; choose the cohort once."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        seg = open_source(stack, s, "segmentation")
        roi = np.array(s.get("roi") or [[0, 0, 0], list(seg.shape)], dtype=int)
        if (
            roi.shape != (2, 3)
            or np.any(roi[0] < 0)
            or np.any(roi[1] > seg.shape)
            or np.any(roi[1] <= roi[0])
        ):
            raise ValueError("ROI must be nonempty ZYX bounds inside the segmentation grid")
        if s.get("object_index"):
            table = read_id_table(s["object_index"])
            if "segmentation_id" in table:
                # A worker's objects.tsv may already map nucleus IDs to annotation IDs.
                # Validate both ID columns exactly before reusing that cached index.
                table["segmentation_id"] = read_id_table(
                    s["object_index"], id_column="segmentation_id"
                ).segmentation_id
        else:
            if s.get("segmentation", "").startswith(("http:", "https:")) and not s.get("roi"):
                raise ValueError(
                    "Select a bounded ROI or supply an object index before scanning a remote segmentation"
                )
            with sqlite3.connect(":memory:") as db:
                _index_objects(seg, roi[0], roi[1], np.array([64, 64, 64]), db, progress)
                rows = db.execute("SELECT id,z0,y0,x0,z1,y1,x1 FROM objects ORDER BY id").fetchall()
            table = pd.DataFrame(
                rows,
                columns=["label_id"] + [p + a for p in ("bbox_min_", "bbox_max_") for a in "zyx"],
            )
        table = table[
            ["label_id"]
            + (["segmentation_id"] if "segmentation_id" in table else [])
            + [p + a for p in ("bbox_min_", "bbox_max_") for a in "zyx"]
        ].copy()
        lo, hi = (
            table[[p + a for a in "zyx"]].to_numpy(dtype=np.int64)
            for p in ("bbox_min_", "bbox_max_")
        )
        table = table[np.all(hi > roi[0], axis=1) & np.all(lo < roi[1], axis=1)]
    if "segmentation_id" not in table:
        table.insert(1, "segmentation_id", table.label_id.astype(np.int64))
    if s.get("id_mapping"):
        from morphofeatures.analysis.mesh_inspection import object_id_mapping

        mapping = object_id_mapping(s)
        reverse = {v: k for k, v in mapping.items()}
        table = table[table.segmentation_id.isin(reverse)].copy()
        table["label_id"] = table.segmentation_id.map(reverse).astype(np.int64)
    if s.get("object_ids") is not None:
        requested = set(int(v) for v in s["object_ids"])
        missing = requested - set(table.label_id)
        if missing:
            raise ValueError(
                f"Requested IDs are absent from the source index/ROI: {sorted(missing)[:10]}"
            )
        table = table[table.label_id.isin(requested)]
    from morphofeatures.data.object_selection import select_objects, write_selection

    table, selection_audit = select_objects(table, s.get("object_selection"), destination=destination)
    table = table.sort_values("label_id")
    limit = int(s.get("max_objects", 0))
    if limit < 0:
        raise ValueError("Maximum objects must be nonnegative")
    if limit and len(table) > limit:
        table = table.iloc[
            np.sort(
                np.random.default_rng(s.get("seed", 42)).choice(len(table), limit, replace=False)
            )
        ]
    if table.empty:
        raise ValueError("No objects selected in this source")
    write_selection(destination, selection_audit, selected_ids=table.label_id)
    table.to_csv(destination / "objects.tsv", sep="\t", index=False)
    write_json_atomic(destination / "source.json", s)
    return table


def sample_volume(volume, points, spacing, origin, *, label=None, sigma=None, budget=128**3):
    """Sample physical points, recursively partitioning reads to a voxel budget."""
    coordinates = (np.asarray(points) - origin) / spacing
    sigma = np.zeros(3) if sigma is None else np.asarray(sigma)
    halo = np.ceil(3 * sigma).astype(int) + 1
    lower = np.maximum(0, np.floor(coordinates.min(0)).astype(int) - halo)
    upper = np.minimum(volume.shape, np.ceil(coordinates.max(0)).astype(int) + halo + 1)
    if np.any(upper <= lower):
        return np.zeros(len(points), dtype=np.float32)
    if np.prod(upper - lower, dtype=np.int64) > budget:
        if len(points) < 2:
            raise ValueError(
                "Resampling filter exceeds the read budget; use a coarser raw resolution"
            )
        axis = int(np.argmax(np.ptp(coordinates, axis=0)))
        order = np.argsort(coordinates[:, axis], kind="stable")
        result = np.empty(len(points), dtype=np.float32)
        for part in np.array_split(order, 2):
            result[part] = sample_volume(
                volume,
                np.asarray(points)[part],
                spacing,
                origin,
                label=label,
                sigma=sigma,
                budget=budget,
            )
        return result
    data = volume.read(tuple(slice(int(a), int(b)) for a, b in zip(lower, upper)))
    data = (
        (data == label).astype(np.float32)
        if label is not None
        else np.asarray(data, dtype=np.float32)
    )
    if label is None and np.any(sigma > 0):
        data = gaussian_filter(data, sigma, mode="nearest")
    return map_coordinates(
        data,
        (coordinates - lower).T,
        order=0 if label is not None else 1,
        mode="constant",
        cval=0,
        prefilter=False,
    )


def sample_points_by_chunk(volume, points, spacing, origin):
    """Group scattered normalization reads without changing their order or interpolation."""
    chunks = np.asarray(getattr(volume, "chunks", (64, 64, 64)))
    cells = np.floor(((points - origin) / spacing) / chunks).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    result = np.empty(len(points), dtype=np.float32)
    for group in np.split(order, np.flatnonzero(np.diff(inverse[order])) + 1):
        result[group] = sample_volume(volume, points[group], spacing, origin)
    return result


def distributed_indices(centers, limit, center=None):
    if not limit or len(centers) <= limit:
        return np.arange(len(centers))
    centers = np.asarray(centers)
    first = int(
        np.argmin(np.square(centers - (centers.mean(0) if center is None else center)).sum(1))
    )
    chosen, distance = [], np.full(len(centers), np.inf)
    for _ in range(limit):
        chosen.append(first)
        distance = np.minimum(distance, np.square(centers - centers[first]).sum(1))
        distance[chosen] = -1
        first = int(np.argmax(distance))
    return np.sort(chosen)


def eligible_tiles(points, spacing, lower, upper, width, minimum, budget=250000):
    """Integrate foreground voxel boxes over the physical tile lattice exactly.

    Center counting would miss tiles narrower than a segmentation voxel and
    misestimate foreground fractions at lattice boundaries.
    """
    shape = np.ceil((upper - lower) / width - 1e-10).astype(int)
    if np.prod(shape, dtype=np.int64) > budget:
        raise ValueError(
            "Physical tile lattice exceeds max_candidate_tiles; increase tile width or the candidate budget"
        )
    starts = np.floor((points - spacing / 2 - lower) / width + 1e-10).astype(int)
    stops = np.ceil((points + spacing / 2 - lower) / width - 1e-10).astype(int)
    occupancy = np.zeros(int(np.prod(shape)), dtype=np.float64)
    for offset in itertools.product(*(range(int(n)) for n in (stops - starts).max(0))):
        cells = starts + offset
        valid = np.all((cells < stops) & (cells >= 0) & (cells < shape), axis=1)
        if not valid.any():
            continue
        cells, foreground = cells[valid], points[valid]
        tile_lower = lower + cells * width
        overlap = np.maximum(
            0,
            np.minimum(foreground + spacing / 2, tile_lower + width)
            - np.maximum(foreground - spacing / 2, tile_lower),
        )
        flat = np.ravel_multi_index(cells.T, shape)
        occupancy += np.bincount(flat, weights=overlap.prod(1), minlength=len(occupancy))
    # Intersections of translated physical voxel boxes accumulate roundoff;
    # a mathematically full tile must still pass a 100% cutoff.
    keep = (occupancy > 0) & (occupancy / np.prod(width) >= minimum - 1e-10)
    return np.column_stack(np.unravel_index(np.flatnonzero(keep), shape))


def tile_grid_origin(lower, upper, width, points, anchor="bbox_min"):
    """Place a complete lattice; a chosen center is the center of one tile."""
    if anchor == "bbox_min":
        return lower, None  # Preserve the original lattice and selection order.
    if anchor not in {"bbox_center", "centroid"}:
        raise ValueError("Tile grid anchor must be bbox_min, bbox_center or centroid")
    center = points.mean(0) if anchor == "centroid" else (lower + upper) / 2
    origin = center - width / 2
    origin += np.floor((lower - origin) / width + 1e-10) * width
    return origin, center


def physical_tiles(obj, views):
    """Return eligible-count and selected physical boxes using the shared fine sampler."""
    v = views
    lower, upper = obj["bounds"]
    points, spacing = obj["points"], obj["spacing"]
    width = np.broadcast_to(np.asarray(v.get("tile_size_um", 1.0), dtype=float), (3,))
    lower, selection_center = tile_grid_origin(
        lower, upper, width, points, v.get("tile_anchor", "bbox_min")
    )
    unique = eligible_tiles(
        points,
        spacing,
        lower,
        upper,
        width,
        v.get("min_foreground_fraction", 0.01),
        int(v.get("max_candidate_tiles", 250000)),
    )
    eligible = len(unique)
    centers = lower + (unique + 0.5) * width
    chosen = (
        distributed_indices(
            centers,
            0 if v.get("exhaustive", False) else int(v.get("max_tiles", 64)),
            center=selection_center,
        )
        if eligible
        else []
    )
    regions = []
    for i in chosen:
        lo, hi = lower + unique[i] * width, lower + (unique[i] + 1) * width
        intersects = np.all(
            (points + spacing / 2 > lo + 1e-10) & (points - spacing / 2 < hi - 1e-10),
            axis=1,
        )
        foreground = np.clip(points[intersects], lo + width * 1e-8, hi - width * 1e-8)
        regions.append((lo, hi, foreground))
    return regions, eligible, lower


class PhysicalObjects:
    """One bounded instance mask plus streamed physical planes at a time."""

    def __init__(self, source, objects, views):
        self.source, self.objects, self.views = source, objects, views
        self.excluded_views = []

    def __enter__(self):
        self.stack = ExitStack()
        try:
            self.raw = open_source(self.stack, self.source, "raw")
            self.seg = open_source(self.stack, self.source, "segmentation")
            limit = self.views.get("_runtime", {}).get("decoded_cache_mb", 128)
            self.raw, self.seg = ReadVolume(self.raw, limit), ReadVolume(self.seg, limit)
        except Exception:
            self.stack.close()
            raise
        return self

    def __exit__(self, *args):
        if hasattr(self.excluded_views, "close"):
            self.excluded_views.close()
        self.stack.close()

    def prepare(self, row):
        started = time.perf_counter()
        self.timings = {}
        if hasattr(self.excluded_views, "close"):
            self.excluded_views.close()
        self.excluded_views = self.exclusion_factory() if hasattr(self, "exclusion_factory") else []
        self.sampling_summary = {}
        s = self.source
        label = int(row["segmentation_id"])
        roi = np.array(s.get("roi") or [[0, 0, 0], list(self.seg.shape)])
        geometry, lo, hi, _, _ = object_geometry(
            {**s, "segmentation_shape_zyx": self.seg.shape}, row
        )
        if geometry["decision"] not in {"eligible", "partial"}:
            raise ObjectEligibilityError(geometry)
        mask = self.seg.read(tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))) == label
        self._object_mask = ObjectMaskVolume(mask, lo, self.seg.shape)
        coords = np.column_stack(np.nonzero(mask))
        if not len(coords):
            raise ValueError("Indexed object has no matching foreground in segmentation")
        if any(
            np.take(mask, edge, axis=axis).any() and unconstrained
            for axis in range(3)
            for edge, unconstrained in ((0, lo[axis] > roi[0, axis]), (-1, hi[axis] < roi[1, axis]))
        ):
            raise ValueError("Object extends beyond supplied bounds; refresh the object index")
        spacing, origin = (
            np.array(s["segmentation_spacing_zyx"]),
            np.array(s["segmentation_origin_zyx"]),
        )
        points = origin + (coords + lo) * spacing
        bounds = (points.min(0) - spacing / 2, points.max(0) + spacing / 2)
        raw_spacing, raw_origin = np.array(s["raw_spacing_zyx"]), np.array(s["raw_origin_zyx"])
        if np.any(bounds[0] < raw_origin - raw_spacing / 2) or np.any(
            bounds[1] > raw_origin + (np.array(self.raw.shape) - 0.5) * raw_spacing
        ):
            raise ValueError("Complete instance lies outside raw image coverage")
        rng = np.random.default_rng(int(s.get("seed", 42)) ^ int(row["label_id"]))
        sample = points[rng.choice(len(points), min(65536, len(points)), replace=False)]
        normalization_started = time.perf_counter()
        sampler = (
            sample_points_by_chunk
            if self.views.get("_runtime", {}).get("group_normalization_reads", True)
            else sample_volume
        )
        values = sampler(self.raw, sample, raw_spacing, raw_origin)
        if not np.isfinite(values).all():
            raise ValueError("Foreground intensity sample contains nonfinite values")
        low, high = np.percentile(values, self.views.get("percentiles", [1, 99]))
        if high <= low:
            raise ValueError("Foreground intensity sample has no usable percentile contrast")
        self.timings["normalization_seconds"] = time.perf_counter() - normalization_started
        self.timings["prepare_seconds"] = time.perf_counter() - started
        return {
            "row": row,
            "points": points,
            "bounds": bounds,
            "spacing": spacing,
            "normalization": [float(low), float(high)],
            "coverage_scope": "partial_object"
            if geometry["decision"] == "partial"
            else "complete_object",
            "volume_faces": geometry["volume_faces"],
            "roi_faces": geometry["roi_faces"],
            "required_voxels": geometry["required_voxels"],
        }

    def planes(self, obj, mode):
        started = time.perf_counter()
        v = self.views
        lower, upper = obj["bounds"]
        points = obj["points"]
        axes = v.get("axes", [0, 1, 2])
        if mode == "whole_label":
            regions = [(lower, upper, points)]
            eligible = 1
            fractions = v.get("coarse_fractions", [0.1, 0.3, 0.5, 0.7, 0.9])
        else:
            regions, eligible, lower = physical_tiles(obj, v)
            fractions = v.get("fine_fractions", [0.5])
        self.timings["tile_seconds"] = (
            self.timings.get("tile_seconds", 0) + time.perf_counter() - started
        )
        self.sampling_summary = {
            "coverage_scope": obj.get("coverage_scope", "complete_object"),
            "eligible_tiles": eligible,
            "selected_tiles": len(regions),
            "tile_anchor": v.get("tile_anchor", "bbox_min"),
            "min_foreground_fraction": v.get("min_foreground_fraction", 0.01),
            "min_slice_foreground_fraction": v.get("min_slice_foreground_fraction", 0.0),
        }
        if not regions:
            self.excluded_views.append(
                {
                    "label_id": int(obj["row"]["label_id"]),
                    "mode": mode,
                    "reason": f"No fine tiles meet the 3D nucleus-content cutoff ({v.get('min_foreground_fraction', 0.01):.1%}). "
                    "Lower Minimum nucleus content per 3D fine tile, reduce tile width, or preview another tile grid anchor.",
                    **self.sampling_summary,
                }
            )
        for tile, (lo, hi, foreground) in enumerate(regions):
            for axis in axes:
                locations = (
                    np.unique(
                        foreground[:, axis].min()
                        + np.asarray(fractions) * np.ptp(foreground[:, axis])
                    )
                    if mode == "whole_label"
                    else np.unique(np.quantile(foreground[:, axis], fractions))
                )
                for position in locations:
                    image, mask, pixel_size = self.plane(
                        lo, hi, axis, position, int(obj["row"]["segmentation_id"])
                    )
                    minimum_slice = (
                        float(v.get("min_slice_foreground_fraction", 0.0))
                        if mode == "fixed_tiles"
                        else 0.0
                    )
                    if not mask.any() or mask.mean() < minimum_slice:
                        self.excluded_views.append(
                            {
                                "label_id": int(obj["row"]["label_id"]),
                                "mode": mode,
                                "tile": tile,
                                "axis": int(axis),
                                "position_um": float(position),
                                "foreground_fraction": float(mask.mean()),
                                "reason": "Empty foreground cross-section"
                                if not mask.any()
                                else f"Fine image nucleus content {mask.mean():.1%} is below the 2D cutoff {minimum_slice:.1%}; lower Minimum nucleus content per 2D fine image.",
                            }
                        )
                        continue
                    low, high = obj["normalization"]
                    image = (
                        np.clip((image - low) / (high - low), 0, 1)
                        if high > low
                        else mask.astype(np.float32)
                    )
                    if v.get("mask", True):
                        image[~mask] = 0
                    manifest = {
                        "label_id": int(obj["row"]["label_id"]),
                        "mode": mode,
                        "tile": tile,
                        "axis": int(axis),
                        "position_um": float(position),
                        "lower_um": lo.tolist(),
                        "upper_um": hi.tolist(),
                        "pixel_size_um": pixel_size,
                        "foreground_fraction": float(mask.mean()),
                        "normalization": obj["normalization"],
                        "eligible_tiles": eligible,
                        "selected_tiles": len(regions),
                        "tile_anchor": v.get("tile_anchor", "bbox_min"),
                        "tile_grid_origin_um": lower.tolist(),
                        "coverage_scope": obj.get("coverage_scope", "complete_object"),
                        "volume_faces": obj.get("volume_faces", ""),
                        "roi_faces": obj.get("roi_faces", ""),
                    }
                    yield image.astype(np.float32), mask, manifest

    def plane(self, lo, hi, axis, position, label):
        size = int(self.views.get("size", 224))
        dims = [a for a in range(3) if a != axis]
        extent = (hi - lo)[dims]
        pixel = float(max(extent) / size)
        center = (hi + lo) / 2
        grid = np.indices((size, size)).reshape(2, -1).T
        points = np.broadcast_to(center, (size * size, 3)).copy()
        points[:, axis] = position
        points[:, dims] += (grid + 0.5 - size / 2) * pixel
        inside = np.all((points[:, dims] >= lo[dims]) & (points[:, dims] < hi[dims]), axis=1)
        s = self.source
        if s.get("roi") is not None:
            roi = np.asarray(s["roi"])
            seg_spacing, seg_origin = (
                np.asarray(s["segmentation_spacing_zyx"]),
                np.asarray(s["segmentation_origin_zyx"]),
            )
            inside &= np.all(
                (points >= seg_origin + (roi[0] - 0.5) * seg_spacing)
                & (points < seg_origin + (roi[1] - 0.5) * seg_spacing),
                axis=1,
            )
        raw_spacing, raw_origin = np.array(s["raw_spacing_zyx"]), np.array(s["raw_origin_zyx"])
        sigma = np.maximum(0, (pixel / raw_spacing - 1) / 2)
        sigma[axis] = 0
        image = sample_volume(self.raw, points, raw_spacing, raw_origin, sigma=sigma)
        reuse = self.views.get("_runtime", {}).get("reuse_object_mask", True) and hasattr(
            self, "_object_mask"
        )
        mask = (
            sample_volume(
                self._object_mask if reuse else self.seg,
                points,
                np.array(s["segmentation_spacing_zyx"]),
                np.array(s["segmentation_origin_zyx"]),
                label=True if reuse else label,
            )
            > 0.5
        )
        mask &= inside
        image[~inside] = 0
        return image.reshape(size, size), mask.reshape(size, size), pixel


def inspection_config(s, objects_path):
    if s.get("inspection_config"):
        return s["inspection_config"]
    mesh = {
        **s,
        "source": "segmentation",
        "spacing_zyx": s["segmentation_spacing_zyx"],
        "origin_zyx": s["segmentation_origin_zyx"],
        "object_index": str(objects_path),
        "index_ids": "embedding",
        "unit": "um",
    }
    return {"data": {"source": "physical_volume", "volume": s}, "inspection": {"mesh": mesh}}
