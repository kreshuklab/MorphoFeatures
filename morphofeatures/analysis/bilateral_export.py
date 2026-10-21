"""Pair-level 3D exports with exact IDs and explicit physical coordinates.

A cell may participate in many pairs. Geometry is therefore an edge list, and
optional volume masks have one query/partner layer per pair, never a conflicting
single 'pair ID' segmentation. All selected masks share a bounded source ROI.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from morphofeatures.analysis.bilateral_comparison import inside, read_ranks, read_report
from morphofeatures.analysis.mesh_inspection import object_id_mapping
from morphofeatures.analysis.relations import integer_ids, read_table
from morphofeatures.artifacts import write_json_atomic
from morphofeatures.representations import fingerprint_file
from morphofeatures.workflow_validation import FieldError


def bilateral_entries(path, metadata, settings):
    selected = settings.get("bilateral_representation")
    if selected and selected not in metadata["representations"]:
        raise FieldError(
            "Select a representation present in this bilateral report.", "bilateral_representation"
        )
    for name, entry in metadata["representations"].items():
        if selected and name != selected:
            continue
        folder = inside(path, entry["directory"])
        variant = settings.get("bilateral_variant")
        if variant is not None and (
            isinstance(variant, bool)
            or int(variant) != variant
            or not 0 <= int(variant) < len(entry["variants"])
        ):
            raise FieldError("Choose a probe variant listed in this report.", "bilateral_variant")
        for index, value in enumerate(entry["variants"]):
            if variant is not None and index != int(variant):
                continue
            frame = read_ranks(inside(path, str(Path(entry["directory"]) / value["ranks"])))
            yield f"{name} / {value['settings']['name']}", folder, frame


def choose_pairs(frame, settings):
    maximum = settings.get("max_pairs", 20)
    if isinstance(maximum, bool) or int(maximum) != maximum or not 1 <= maximum <= 1000:
        raise FieldError(
            "Maximum pairs must be an integer from 1 to 1000; start with 20 for 3D inspection.",
            "max_pairs",
        )
    selected = settings.get("query_ids") or []
    if len(selected) > maximum:
        raise FieldError(
            "More query IDs were selected than max_pairs. Raise the pair limit or choose fewer cells.",
            "query_ids",
            "max_pairs",
        )
    if settings.get("pair_order", "best") not in {"best", "worst", "id"}:
        raise FieldError("Choose best, worst or id for automatic pair selection.", "pair_order")
    valid = frame.dropna(subset=["rank", "best_partner_id"])
    if selected:
        selected = integer_ids(selected)
        missing = set(selected) - set(valid.label_id)
        if missing:
            raise FieldError(
                f"These queries have no evaluable retrieved partner in the selected result: {sorted(missing)[:10]}",
                "query_ids",
            )
        valid = valid[valid.label_id.isin(selected)]
    order = settings.get("pair_order", "best")
    return valid.sort_values(
        ["label_id"] if order == "id" else ["rank", "label_id"],
        ascending=True if order != "worst" else [False, True],
    ).head(int(maximum))


def validate_bilateral_export(settings, *, require_report=True):
    pair_count = None
    if require_report:
        path, report = read_report(settings.get("analysis", ""), field="analysis")
        entries = list(bilateral_entries(path, report, settings))
        pair_count = sum(len(choose_pairs(frame, settings)) for _, _, frame in entries)
        if not pair_count:
            raise FieldError(
                "This selection has no evaluable pairs to export. Choose a different representation/variant or check its exclusions.",
                "bilateral_representation",
                "bilateral_variant",
            )
    else:
        choose_pairs(
            pd.DataFrame(columns=["label_id", "rank", "best_partner_id"]),
            {**settings, "query_ids": []},
        )
    if settings.get("export_pair_masks") and settings.get("mode", "portable") != "portable":
        raise FieldError(
            "Pair masks need a bounded portable ROI. Use portable mode and the segmentation URL/key to stream that region.",
            "mode",
            "export_pair_masks",
        )
    if settings.get("make_mobie", True) or settings.get("export_pair_masks"):
        from morphofeatures.analysis.mobie_export import resolve_mobie_export

        resolved = resolve_mobie_export(settings, require_analysis=require_report)
        if resolved["mode"] == "portable":
            if resolved.get("unit", "voxel") not in {"um", "nm"} and (
                settings.get("export_pair_masks") or not settings.get("positions")
            ):
                raise FieldError(
                    "Physical links and pair masks require calibrated um/nm coordinates. Set volume units and voxel spacing. A position table can supply coordinates for links, but cannot calibrate a mask volume.",
                    "unit",
                    "positions",
                )
            if settings.get("export_pair_masks") and pair_count is not None:
                roi = np.asarray(resolved["roi"], dtype=np.int64)
                total = pair_count * int(np.prod(roi[1] - roi[0], dtype=object))
                if total > settings.get("max_mask_voxels", 256**3):
                    raise FieldError(
                        f"Pair masks require up to {total:,} voxels across {pair_count} layers. Reduce the ROI/pairs or raise the total mask budget before exporting.",
                        "max_mask_voxels",
                        "max_pairs",
                        "roi",
                    )
    elif not settings.get("positions") or not Path(settings["positions"]).is_file():
        raise FieldError(
            "Non-MoBIE geometry needs a physical position table. Alternatively enable MoBIE export to obtain positions from the selected segmentation/catalogue.",
            "positions",
        )
    if settings.get("positions_unit", "um") not in {"um", "nm"}:
        raise FieldError("Position-table units must be um or nm.", "positions_unit")
    if settings.get("positions_ids", "cell") not in {"cell", "segmentation"}:
        raise FieldError("Position IDs must be cell or segmentation.", "positions_ids")
    for key, default in (("max_mask_voxels", 256**3),):
        value = settings.get(key, default)
        if not np.isfinite(value) or value < 1:
            raise FieldError("Set a positive total pair-mask voxel budget.", key)


def _positions(settings, default=None, *, default_unit="um"):
    if settings.get("positions"):
        table = read_table(settings["positions"])
        columns = settings.get("position_columns", ["anchor_z", "anchor_y", "anchor_x"])
        unit = settings.get("positions_unit", "um")
        namespace = settings.get("positions_ids", "cell")
        source = fingerprint_file(settings["positions"])
    else:
        table, columns, unit, namespace = (
            default,
            ["anchor_z", "anchor_y", "anchor_x"],
            default_unit,
            "segmentation",
        )
        source = {"basis": "MoBIE exported segmentation/object catalogue", "unit": unit}
    if (
        table is None
        or "label_id" not in table
        or len(columns) != 3
        or not set(columns) <= set(table)
    ):
        raise FieldError(
            "Position table needs label_id and three configured Z/Y/X physical columns.",
            "positions",
            "position_columns",
        )
    if unit not in {"um", "nm"}:
        raise FieldError(
            "3D pair geometry requires physical units (um/nm), not uncalibrated voxels.",
            "unit",
            "positions_unit",
        )
    ids = integer_ids(table.label_id)
    xyz = table[columns].to_numpy(float) * (0.001 if unit == "nm" else 1.0)
    if len(set(ids)) != len(ids) or not np.isfinite(xyz).all():
        raise FieldError("Positions need unique IDs and finite physical coordinates.", "positions")
    return dict(zip(map(int, ids), xyz)), namespace, source


def _write_geometry(folder, pairs, endpoints):
    endpoints = np.asarray(endpoints, dtype=np.float64).reshape(-1, 2, 3)
    pairs.to_csv(folder / "pairs.tsv", sep="\t", index=False)
    np.savez_compressed(
        folder / "pairs.npz",
        endpoints_zyx_um=endpoints,
        vectors_zyx_um=np.stack([endpoints[:, 0], endpoints[:, 1] - endpoints[:, 0]], axis=1),
        query_ids=pairs.label_id.to_numpy(dtype=np.int64),
        partner_ids=pairs.best_partner_id.to_numpy(dtype=np.int64),
        ranks=pairs["rank"].to_numpy(float),
    )
    # VTK legacy POLYDATA: XYZ physical coordinates, one directed query->partner edge.
    with (folder / "pairs.vtk").open("w") as stream:
        stream.write(
            f"# vtk DataFile Version 3.0\nBilateral candidate retrieval; coordinates in micrometers\nASCII\nDATASET POLYDATA\nPOINTS {2 * len(pairs)} double\n"
        )
        for point in endpoints.reshape(-1, 3)[:, ::-1]:
            stream.write(" ".join(format(v, ".17g") for v in point) + "\n")
        stream.write(f"LINES {len(pairs)} {len(pairs) * 3}\n")
        for i in range(len(pairs)):
            stream.write(f"2 {2 * i} {2 * i + 1}\n")
        stream.write(
            f"CELL_DATA {len(pairs)}\nSCALARS retrieval_rank double 1\nLOOKUP_TABLE default\n"
        )
        stream.write("\n".join(str(float(v)) for v in pairs["rank"]) + "\n")


def _write_masks(settings, groups, destination, progress):
    import h5py

    start, stop = np.asarray(settings["roi"], dtype=np.int64)
    shape = stop - start
    count = sum(len(frame) for _, frame, _ in groups)
    total = int(np.prod(shape, dtype=object)) * count
    if total > settings.get("max_mask_voxels", 256**3):
        raise FieldError(
            f"Pair masks require {total:,} voxels across {count} layers. Reduce the ROI/pairs or raise the total mask budget.",
            "max_mask_voxels",
            "max_pairs",
            "roi",
        )
    from morphofeatures.analysis.mobie_export import _blocks, _volume

    block = np.minimum(settings.get("block_shape", [32, 32, 32]), shape)
    with (
        _volume(settings, "segmentation") as volume,
        h5py.File(destination / "pair_masks.h5", "w") as store,
    ):
        store.attrs.update(
            axes="pzyx",
            unit=settings["unit"],
            spacing_zyx=settings["spacing_zyx"],
            origin_zyx=(
                np.array(settings["origin_zyx"]) + start * settings["spacing_zyx"]
            ).tolist(),
            roles="0:background, 1:query, 2:retrieved candidate",
        )
        pairs = pd.concat(
            [f.assign(representation=name) for name, f, _ in groups], ignore_index=True
        )
        masks = store.create_dataset(
            "roles",
            shape=(count, *map(int, shape)),
            dtype="uint8",
            chunks=(1, *map(int, block)),
            compression="gzip",
        )
        store["query_cell_ids"] = pairs.label_id.to_numpy(dtype=np.int64)
        store["partner_cell_ids"] = pairs.best_partner_id.to_numpy(dtype=np.int64)
        seen = np.zeros((count, 2), dtype=bool)
        for block_index, target in enumerate(_blocks(shape, block)):
            labels = volume.read(
                tuple(
                    slice(int(offset + sl.start), int(offset + sl.stop))
                    for offset, sl in zip(start, target)
                )
            )
            for i, row in enumerate(pairs.itertuples()):
                query, partner = (
                    labels == row.query_segmentation_id,
                    labels == row.partner_segmentation_id,
                )
                seen[i] |= [query.any(), partner.any()]
                masks[(i, *target)] = query.astype(np.uint8) + 2 * partner.astype(np.uint8)
            if progress:
                progress(phase="pair_masks", block=block_index + 1)
        pairs["query_present_in_roi"], pairs["partner_present_in_roi"] = seen[:, 0], seen[:, 1]
        pairs.to_csv(destination / "pair_mask_layers.tsv", sep="\t", index=False)
    return int((~seen.all(axis=1)).sum())


def export_bilateral(settings, destination, progress=None):
    validate_bilateral_export(settings)
    source, metadata = read_report(settings["analysis"], field="analysis")
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            "Bilateral export destination is not empty; choose a new run directory"
        )
    destination.mkdir(parents=True, exist_ok=True)
    default, default_unit, mobie = None, "um", None
    resolved = dict(settings)
    if settings.get("make_mobie", True) or settings.get("export_pair_masks"):
        from morphofeatures.analysis.mobie_export import (
            _portable,
            export_mobie,
            resolve_mobie_export,
        )

        resolved = resolve_mobie_export(settings)
        if settings.get("make_mobie", True):
            mobie = export_mobie(resolved, destination / "mobie", progress)
            report = json.loads(mobie.read_text())
            default = read_table(
                mobie.parent / "project" / report["dataset"] / "tables/objects/default.tsv"
            )
            default_unit = report["spatial"].get("unit", "um")
        elif not settings.get("positions"):
            # Indexing a bounded ROI supplies measured positions when no table was given.
            spatial = destination / "spatial"
            spatial.mkdir()
            _, default, info = _portable(resolved, spatial, progress)
            default_unit = info["unit"]
    positions, namespace, position_source = _positions(resolved, default, default_unit=default_unit)
    mapping = object_id_mapping(resolved)
    groups, records, exclusions = [], [], []
    for index, (name, _, frame) in enumerate(bilateral_entries(source, metadata, resolved)):
        selected = choose_pairs(frame, resolved)
        kept, endpoints = [], []
        for row in selected.to_dict("records"):
            query, partner = int(row["label_id"]), int(row["best_partner_id"])
            segids = [mapping.get(i) if resolved.get("id_mapping") else i for i in (query, partner)]
            pointids = [query, partner] if namespace == "cell" else segids
            if any(i is None or i not in positions for i in pointids) or (
                settings.get("export_pair_masks") and any(i is None for i in segids)
            ):
                exclusions.append(
                    {
                        "representation": name,
                        "label_id": query,
                        "best_partner_id": partner,
                        "reason": "Missing physical position or required segmentation ID mapping",
                    }
                )
                continue
            xyz = np.array([positions[i] for i in pointids])
            row.update(query_segmentation_id=segids[0], partner_segmentation_id=segids[1])
            for kind, point in zip(("query", "partner"), xyz):
                row.update({f"{kind}_{axis}_um": float(v) for axis, v in zip("zyx", point)})
            kept.append(row)
            endpoints.append(xyz)
        if not kept:
            continue
        folder = destination / f"pairs-{index}"
        folder.mkdir()
        pairs = pd.DataFrame(kept)
        for key in (
            "label_id",
            "best_partner_id",
            "native_id",
            "query_segmentation_id",
            "partner_segmentation_id",
        ):
            if key in pairs:
                pairs[key] = pd.array([row.get(key) for row in kept], dtype="Int64")
        _write_geometry(folder, pairs, endpoints)
        groups.append((name, pairs, folder))
        records.append({"name": name, "directory": folder.name, "pairs": len(pairs)})
    pd.DataFrame(
        exclusions, columns=["representation", "label_id", "best_partner_id", "reason"]
    ).to_csv(destination / "excluded_pairs.tsv", sep="\t", index=False)
    if not groups:
        raise FieldError(
            f"No pairs have both endpoints in the spatial data; inspect {destination / 'excluded_pairs.tsv'} and check cell/nucleus IDs, ROI and positions.",
            "positions",
            "id_mapping",
            "roi",
        )
    if exclusions and settings.get("require_all_ids"):
        raise FieldError(
            "Some selected pairs lack spatial endpoints; inspect excluded_pairs.tsv or disable strict coverage for an ROI export.",
            "require_all_ids",
        )
    incomplete = (
        _write_masks(resolved, groups, destination, progress)
        if settings.get("export_pair_masks")
        else 0
    )
    if incomplete and settings.get("require_all_ids"):
        raise FieldError(
            "Selected pair members are absent from the mask ROI; enlarge it or disable strict coverage. See pair_mask_layers.tsv.",
            "roi",
            "require_all_ids",
        )
    report = {
        "schema": "morphofeatures.bilateral_export.v1",
        "source": fingerprint_file(source),
        "settings": resolved,
        "position_source": position_source,
        "groups": records,
        "excluded_pairs": len(exclusions),
        "pairs_missing_mask_endpoint": incomplete,
        "mobie": str(mobie.relative_to(destination)) if mobie else None,
        "interpretation": "Lines connect query objects to their closest supplied bilateral candidate in feature space. They are hypotheses, not validated anatomical connections. Coordinates are physical Z/Y/X in µm (VTK uses X/Y/Z). Pair masks use separate layers because objects can belong to multiple pairs.",
    }
    path = write_json_atomic(destination / "bilateral_export.json", report)
    from morphofeatures.representation_analysis import package_results

    package_results(destination)
    return path
