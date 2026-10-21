"""Spatial graphs and the published self + neighborhood-mean context recipe.

Contact graphs use positive instance labels and explicit 6/18/26-connectivity.
Chunk halos preserve cross-block contacts. Radius/KNN graphs are alternatives in
physical centroid space, not claims of membrane contact. Bilateral partners and
embedding-space neighbors must never be used as a shortcut for spatial adjacency.
"""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import normalize

from morphofeatures.analysis.joint import _signature, extraction_metadata
from morphofeatures.analysis.relations import cell_identity, integer_ids, read_relations, read_table
from morphofeatures.artifacts import write_json_atomic
from morphofeatures.data.io import export_embeddings, load_embeddings
from morphofeatures.data.remote_n5 import is_remote_url
from morphofeatures.data.volumes import open_volume
from morphofeatures.representations import fingerprint_file
from morphofeatures.workflow_validation import FieldError


def validate_context(settings, *, check_files=True):
    mode = settings.get("graph_mode", "adjacency_table")
    if mode not in {"adjacency_table", "segmentation", "radius", "knn"}:
        raise FieldError("Choose adjacency_table, segmentation, radius or knn.", "graph_mode")
    required = {
        "adjacency_table": "adjacency_table",
        "segmentation": "segmentation",
        "radius": "positions",
        "knn": "positions",
    }[mode]
    if not settings.get(required):
        raise FieldError(f"Select {required} for this spatial-neighborhood method.", required)
    if (
        check_files
        and not is_remote_url(settings[required])
        and not Path(settings[required]).exists()
    ):
        raise FieldError(f"Spatial source does not exist: {settings[required]}", required)
    if mode == "adjacency_table" and Path(settings[required]).suffix == ".pkl":
        raise FieldError(
            "Bilateral candidate pickles are evaluation data, not physical adjacency. Select a source_id/target_id TSV/CSV.",
            required,
        )
    if settings.get("id_namespace", "cell") not in {"cell", "nucleus"}:
        raise FieldError("Choose cell or nucleus embedding IDs.", "id_namespace")
    if settings.get("graph_id_namespace", "cell") not in {"cell", "nucleus"}:
        raise FieldError("Choose cell or nucleus graph IDs.", "graph_id_namespace")
    if (
        settings.get("id_namespace") == "nucleus" or settings.get("graph_id_namespace") == "nucleus"
    ) and not settings.get("id_mapping"):
        raise FieldError(
            "Map nucleus embeddings to cell IDs before using a cell graph.", "id_mapping"
        )
    if check_files and settings.get("id_mapping") and not Path(settings["id_mapping"]).is_file():
        raise FieldError("ID mapping file does not exist.", "id_mapping")
    for key, choices, default in [
        ("missing_neighbors", {"error", "available"}, "error"),
        ("isolated", {"self", "skip", "error"}, "self"),
        ("roi_boundary", {"skip", "allow"}, "skip"),
        ("combination", {"raw", "balanced"}, "raw"),
        ("reduction", {"none", "agglomeration"}, "none"),
    ]:
        if settings.get(key, default) not in choices:
            raise FieldError(f"{key}: choose one of {sorted(choices)}.", key)
    for key, default in [
        ("neighbors", 8),
        ("n_features", 200),
        ("max_scan_voxels", 100_000_000),
        ("max_edges", 5_000_000),
    ]:
        value = settings.get(key, default)
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise FieldError(f"{key} must be a positive integer.", key)
    for key, default in [("radius_um", 10), ("self_weight", 0.5), ("minimum_coverage", 1.0)]:
        value = float(settings.get(key, default))
        if not np.isfinite(value) or (value <= 0 if key == "radius_um" else not 0 <= value <= 1):
            raise FieldError(
                f"Invalid {key}; use a positive radius or a fraction between 0 and 1.", key
            )
    if settings.get("connectivity", 1) not in (1, 2, 3):
        raise FieldError(
            "Connectivity 1/2/3 means face / face+edge / face+edge+corner contacts.", "connectivity"
        )
    if settings.get("unit", "um") not in {"um", "nm"}:
        raise FieldError("Spatial coordinates need measured um or nm units.", "unit")
    block = np.asarray(settings.get("block_shape", [64, 64, 64]))
    if (
        block.shape != (3,)
        or not np.isfinite(block).all()
        or np.any(block < 1)
        or not np.equal(block, np.floor(block)).all()
        or np.prod(block + 2) > 256**3
    ):
        raise FieldError(
            "Use three positive block dimensions with at most 256³ voxels including halos.",
            "block_shape",
        )


def contact_graph(volume, settings, progress=None):
    """Return undirected unique edges, observed labels and boundary labels."""
    if not np.issubdtype(volume.dtype, np.integer):
        raise FieldError("Contact graphs require an integer instance segmentation.", "segmentation")
    roi = np.asarray(settings.get("roi") or [[0, 0, 0], list(volume.shape)])
    if (
        roi.shape != (2, 3)
        or not np.isfinite(roi).all()
        or not np.equal(roi, np.floor(roi)).all()
        or np.any(roi[0] < 0)
        or np.any(roi[1] > volume.shape)
        or np.any(roi[1] <= roi[0])
    ):
        raise FieldError("ROI must be [start_zyx, stop_zyx] inside the segmentation.", "roi")
    roi = roi.astype(int)
    count = int(np.prod(roi[1] - roi[0]))
    if count > settings.get("max_scan_voxels", 100_000_000):
        raise FieldError(
            f"This graph needs {count:,} segmentation voxels. Select a smaller ROI or explicitly raise max_scan_voxels; the full volume will not be loaded into RAM.",
            "roi",
            "max_scan_voxels",
        )
    offsets = [
        np.array(o)
        for o in itertools.product([-1, 0, 1], repeat=3)
        if 0 < sum(abs(i) for i in o) <= settings.get("connectivity", 1)
        and next(i for i in o if i) > 0
    ]
    edges, observed, boundary = set(), set(), set()
    block = np.asarray(settings.get("block_shape", [64, 64, 64]), dtype=int)
    blocks = itertools.product(*(range(a, b, s) for a, b, s in zip(roi[0], roi[1], block)))
    block_count = int(np.prod(np.ceil((roi[1] - roi[0]) / block)))
    for block_index, start in enumerate(blocks):
        start = np.asarray(start)
        stop = np.minimum(start + block, roi[1])
        low, high = np.maximum(start - 1, roi[0]), np.minimum(stop + 1, roi[1])
        data = volume.read(tuple(slice(int(a), int(b)) for a, b in zip(low, high)))
        if np.any(data < 0) or (data.size and int(data.max()) > np.iinfo(np.int64).max):
            raise FieldError("Segmentation IDs must be nonnegative int64 values.", "segmentation")
        core = data[tuple(slice(int(a), int(b)) for a, b in zip(start - low, stop - low))]
        observed.update(int(i) for i in np.unique(core) if i)
        for axis in range(3):
            for is_low in (True, False):
                at_boundary = start[axis] == roi[0, axis] if is_low else stop[axis] == roi[1, axis]
                if at_boundary:
                    boundary.update(
                        int(i)
                        for i in np.unique(np.take(core, 0 if is_low else -1, axis=axis))
                        if i
                    )
        for offset in offsets:
            a = np.maximum(start, roi[0] - offset)
            b = np.minimum(stop, roi[1] - offset)
            if np.any(a >= b):
                continue
            left = data[tuple(slice(int(x), int(y)) for x, y in zip(a - low, b - low))]
            right = data[
                tuple(slice(int(x), int(y)) for x, y in zip(a + offset - low, b + offset - low))
            ]
            mask = (left > 0) & (right > 0) & (left != right)
            pairs = np.sort(np.column_stack([left[mask], right[mask]]).astype(np.int64), axis=1)
            edges.update(map(tuple, np.unique(pairs, axis=0)))
            if len(edges) > settings.get("max_edges", 5_000_000):
                raise FieldError(
                    "Graph exceeds max_edges; reduce the ROI or explicitly raise the edge budget.",
                    "max_edges",
                    "roi",
                )
        if progress:
            progress(block=block_index + 1, blocks=block_count, edges=len(edges))
    return np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2), observed, boundary


def spatial_graph(settings, progress=None):
    mode = settings.get("graph_mode", "adjacency_table")
    if mode == "adjacency_table":
        edges = read_relations(settings["adjacency_table"])
        if len(edges) > settings.get("max_edges", 5_000_000):
            raise FieldError(
                "Imported graph exceeds max_edges; supply a smaller graph or explicitly raise the edge budget.",
                "adjacency_table",
                "max_edges",
            )
        nodes_path = Path(settings["adjacency_table"]).with_name("graph_nodes.tsv")
        if nodes_path.is_file():
            nodes = read_table(nodes_path)
            if not {"label_id", "boundary"} <= set(nodes):
                raise FieldError(
                    "The sibling graph_nodes.tsv requires label_id and boundary (True/False or 1/0) columns.",
                    "adjacency_table",
                )
            ids = integer_ids(nodes.label_id)
            flags = nodes.boundary.str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            )
            if (
                len(set(ids)) != len(ids)
                or np.any(ids == 0)
                or flags.isna().any()
                or not set(edges.ravel()) <= set(ids)
            ):
                raise FieldError(
                    "The sibling graph_nodes.tsv must list unique positive IDs, valid boundary flags and every endpoint of the supplied edges. Use matching graph exports.",
                    "adjacency_table",
                )
            boundary = set(ids[flags.to_numpy(dtype=bool)])
            return (
                edges,
                set(ids),
                boundary,
                {
                    "source": fingerprint_file(settings["adjacency_table"]),
                    "nodes_source": fingerprint_file(nodes_path),
                },
            )
        return (
            edges,
            set(edges.ravel()),
            set(),
            {
                "source": fingerprint_file(settings["adjacency_table"]),
                "boundary_coverage": "unknown for imported edge table",
            },
        )
    if mode == "segmentation":
        source = (
            {"url": settings["segmentation"]}
            if is_remote_url(settings["segmentation"])
            else fingerprint_file(settings["segmentation"])
        )
        with open_volume(
            settings["segmentation"],
            settings.get("segmentation_key"),
            settings.get("segmentation_axes", "zyx"),
            settings.get("segmentation_channel", 0),
            timepoint=settings.get("segmentation_timepoint", 0),
            remote_options=settings.get("remote_options"),
        ) as volume:
            identity = {
                "algorithm": "spatial_contacts.v1",
                "source": source,
                "shape": list(volume.shape),
                "dtype": str(volume.dtype),
                "settings": {
                    k: settings.get(k)
                    for k in (
                        "segmentation_key",
                        "segmentation_axes",
                        "segmentation_channel",
                        "segmentation_timepoint",
                        "roi",
                        "connectivity",
                        "source_version",
                        "remote_options",
                    )
                },
            }
            cache = settings.get("graph_cache")
            cache_path = None
            if cache:
                key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                cache_path = Path(cache) / (key + ".npz")
                if cache_path.is_file():
                    with np.load(cache_path, allow_pickle=False) as saved:
                        if len(saved["edges"]) > settings.get("max_edges", 5_000_000):
                            raise FieldError(
                                "Cached graph exceeds max_edges; raise the edge budget or select a smaller region.",
                                "max_edges",
                            )
                        return (
                            saved["edges"],
                            set(saved["nodes"]),
                            set(saved["boundary"]),
                            {**identity, "cache_hit": True},
                        )
            edges, nodes, boundary = contact_graph(volume, settings, progress)
            if cache_path:
                import os
                import tempfile

                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    dir=cache_path.parent, suffix=".npz", delete=False
                ) as stream:
                    temporary = Path(stream.name)
                try:
                    np.savez_compressed(
                        temporary,
                        edges=edges,
                        nodes=np.array(sorted(nodes), dtype=np.int64),
                        boundary=np.array(sorted(boundary), dtype=np.int64),
                    )
                    os.replace(temporary, cache_path)
                finally:
                    temporary.unlink(missing_ok=True)
            return edges, nodes, boundary, {**identity, "cache_hit": False}
    frame = read_table(settings["positions"])
    columns = settings.get("position_columns", ["z", "y", "x"])
    if "label_id" not in frame or len(columns) != 3 or not set(columns) <= set(frame):
        raise FieldError(
            "Position table requires label_id and three configured physical coordinate columns (default z, y, x).",
            "positions",
            "position_columns",
        )
    ids = integer_ids(frame.label_id)
    xyz = frame[columns].to_numpy(dtype=float) * (
        0.001 if settings.get("unit", "um") == "nm" else 1
    )
    if len(set(ids)) != len(ids) or np.any(ids <= 0) or not np.isfinite(xyz).all():
        raise FieldError(
            "Position rows require unique positive cell IDs and finite physical coordinates.",
            "positions",
        )
    order = np.argsort(ids)
    ids, xyz = ids[order], xyz[order]
    tree = cKDTree(xyz)
    edges = set()
    for i in range(len(ids)):
        if mode == "radius":
            neighbors = tree.query_ball_point(xyz[i], float(settings.get("radius_um", 10)))
        else:
            _, neighbors = tree.query(
                xyz[i], k=min(len(ids), int(settings.get("neighbors", 8)) + 1)
            )
            neighbors = np.atleast_1d(neighbors)
        edges.update(tuple(sorted((int(ids[i]), int(ids[j])))) for j in neighbors if j != i)
        if len(edges) > settings.get("max_edges", 5_000_000):
            raise FieldError(
                "Spatial graph exceeds max_edges; reduce radius/neighbors or raise the edge budget.",
                "max_edges",
                "radius_um",
                "neighbors",
            )
    return (
        np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2),
        set(ids),
        set(),
        {
            "source": fingerprint_file(settings["positions"]),
            "coordinates_unit": "um",
            "knn_symmetrization": "union (degree can exceed k)" if mode == "knn" else None,
        },
    )


def build_context(settings, destination, progress=None):
    from morphofeatures.analysis.context_sources import resolve_context_source

    settings = resolve_context_source(settings)
    settings = {
        "graph_mode": "adjacency_table",
        "id_namespace": "cell",
        "graph_id_namespace": "cell",
        "include_self": True,
        "missing_neighbors": "error",
        "minimum_coverage": 1.0,
        "isolated": "self",
        "roi_boundary": "skip",
        "combination": "raw",
        "self_weight": 0.5,
        "reduction": "none",
        "n_features": 200,
        "connectivity": 1,
        "neighbors": 8,
        "radius_um": 10.0,
        "unit": "um",
        **settings,
    }
    validate_context(settings)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    table, bridge = cell_identity(load_embeddings(settings["embedding"]), settings)
    table = table.sorted()
    positions = {int(i): n for n, i in enumerate(table.label_ids)}
    edges, nodes, boundary, graph_metadata = spatial_graph(settings, progress)
    if settings.get("graph_id_namespace") == "nucleus":
        from morphofeatures.data.contracts import EmbeddingTable

        ordered = np.array(sorted(nodes), dtype=np.int64)
        _, graph_bridge = cell_identity(
            EmbeddingTable(ordered, np.zeros((len(ordered), 1))),
            {**settings, "id_namespace": "nucleus"},
        )
        mapping = dict(zip(graph_bridge.native_id, graph_bridge.label_id))
        edges = np.asarray([[mapping[a], mapping[b]] for a, b in edges], dtype=np.int64).reshape(
            -1, 2
        )
        nodes = {int(mapping[i]) for i in nodes}
        boundary = {int(mapping[i]) for i in boundary}
        graph_bridge.to_csv(destination / "graph_id_mapping.tsv", sep="\t", index=False)
    neighbors = {int(i): set() for i in nodes}
    for a, b in edges:
        neighbors.setdefault(int(a), set()).add(int(b))
        neighbors.setdefault(int(b), set()).add(int(a))
    pd.DataFrame(edges, columns=["source_id", "target_id"]).to_csv(
        destination / "adjacency.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        {"label_id": sorted(nodes), "boundary": [i in boundary for i in sorted(nodes)]}
    ).to_csv(destination / "graph_nodes.tsv", sep="\t", index=False)
    bridge.to_csv(destination / "id_mapping.tsv", sep="\t", index=False)
    ids, own, means, coverage = [], [], [], []
    for i in table.label_ids:
        all_neighbors = neighbors.get(int(i), set())
        available = all_neighbors & positions.keys()
        fraction = len(available) / len(all_neighbors) if all_neighbors else 1.0
        reason = ""
        if int(i) not in nodes:
            reason = "Object absent from spatial graph/selected region"
        elif i in boundary and settings.get("roi_boundary", "skip") == "skip":
            reason = "Object touches the segmentation/ROI boundary; neighborhood may be incomplete"
        elif (
            len(available) < len(all_neighbors)
            and settings.get("missing_neighbors", "error") == "error"
        ):
            raise FieldError(
                f"Cell {i} has {len(all_neighbors) - len(available)} spatial neighbors without embeddings. Embed those cells, or explicitly choose available-neighbor averaging and a minimum coverage.",
                "missing_neighbors",
                "embedding",
            )
        elif fraction < float(settings.get("minimum_coverage", 1.0)):
            reason = "Available neighbor fraction is below minimum_coverage"
        elif not available and settings.get("isolated", "self") == "error":
            raise FieldError(
                f"Cell {i} has no available spatial neighbors. Select self-only or skip under isolated cells.",
                "isolated",
            )
        elif not available and settings.get("isolated", "self") == "skip":
            reason = "Isolated cell skipped"
        if not available and not settings.get("include_self", True) and not reason:
            reason = "Empty neighborhood when focal cell is excluded"
        coverage.append(
            {
                "label_id": int(i),
                "graph_neighbors": len(all_neighbors),
                "available_neighbors": len(available),
                "coverage": fraction,
                "boundary": i in boundary,
                "reason": reason,
            }
        )
        if reason:
            continue
        members = sorted(available | ({int(i)} if settings.get("include_self", True) else set()))
        ids.append(int(i))
        own.append(table.features[positions[int(i)]])
        means.append(table.features[[positions[j] for j in members]].mean(axis=0))
    pd.DataFrame(coverage).to_csv(destination / "coverage.tsv", sep="\t", index=False)
    if not ids:
        raise FieldError(
            f"No cells retained for spatial context; inspect {destination / 'coverage.tsv'}. Check ID mapping, ROI boundary policy and minimum neighbor coverage.",
            "id_mapping",
            "roi_boundary",
            "minimum_coverage",
        )
    own, means = np.asarray(own), np.asarray(means)
    export_embeddings(destination / "self.npz", ids, own)
    export_embeddings(destination / "neighborhood_mean.npz", ids, means)
    if settings.get("combination", "raw") == "balanced":
        if np.any(np.linalg.norm(own, axis=1) == 0) or np.any(np.linalg.norm(means, axis=1) == 0):
            raise FieldError(
                "Balanced context requires nonzero self and mean vectors; select raw concatenation or revise the cohort.",
                "combination",
            )
        weight = float(settings.get("self_weight", 0.5))
        full = np.column_stack(
            [normalize(own) * np.sqrt(weight), normalize(means) * np.sqrt(1 - weight)]
        )
    else:
        full = np.column_stack([own, means])
    export_embeddings(destination / "full_context.npz", ids, full)
    features = full
    if settings.get("reduction", "none") == "agglomeration":
        count = int(settings.get("n_features", 200))
        if count > full.shape[1]:
            raise FieldError(
                f"n_features={count} exceeds the {full.shape[1]} concatenated dimensions.",
                "n_features",
            )
        reducer = FeatureAgglomeration(n_clusters=count).fit(full)
        features = reducer.transform(full)
        np.savez_compressed(destination / "agglomeration.npz", feature_groups=reducer.labels_)
    output = export_embeddings(destination / "embeddings.npz", ids, features)
    metadata = extraction_metadata(settings["embedding"])
    # Preserve dataset identity; change the representation recipe, never the dataset.
    report = {
        "schema": "morphofeatures.context.v1",
        "model": "spatial_context",
        "settings": settings,
        "source": fingerprint_file(settings["embedding"]),
        "graph": graph_metadata,
        "spatial_source_metadata": settings.get("spatial_source_metadata"),
        "embedding": output.name,
        "coverage": "coverage.tsv",
        "matched_objects": len(ids),
        "input_dimensions": own.shape[1],
        "full_dimensions": full.shape[1],
        "dimensions": features.shape[1],
        "graph_nodes": len(nodes),
        "graph_edges": len(edges),
        "isolated_graph_nodes": sum(not neighbors.get(int(i)) for i in nodes),
        "warnings": (
            [
                "No physical edges were found. Self-only vectors contain no additional neighborhood information; check the segmentation/graph, or use measured centroid neighborhoods for separated nuclei."
            ]
            if not len(edges)
            else []
        ),
        "id_namespace": "cell",
        "input_id_namespace": settings.get("id_namespace", "cell"),
        "dataset_identity": metadata.get("dataset_identity"),
        "sources": metadata.get("sources", {}),
        "representation_recipe": {
            "algorithm": "self_and_spatial_mean.v1",
            "settings": {
                k: settings[k]
                for k in (
                    "graph_mode",
                    "include_self",
                    "combination",
                    "self_weight",
                    "reduction",
                    "n_features",
                    "connectivity",
                    "neighbors",
                    "radius_um",
                    "missing_neighbors",
                    "minimum_coverage",
                    "isolated",
                    "roi_boundary",
                )
            },
            "input_recipe": _signature(metadata),
            "input_dimensions": own.shape[1],
        },
        "interpretation": "Self features concatenated with spatial-neighborhood mean. No labels or bilateral candidates enter feature construction. Saved agglomeration, if enabled, is fitted on this cohort; fit reduction within training folds for predictive validation.",
    }
    recipe = report["representation_recipe"]
    unused = []
    if settings["combination"] == "raw":
        unused.append("self_weight")
    if settings["reduction"] == "none":
        unused.append("n_features")
    else:
        recipe["feature_groups_sha256"] = hashlib.sha256(
            np.asarray(reducer.labels_, dtype="<i8").tobytes()
        ).hexdigest()
    for key, mode in (
        ("connectivity", "segmentation"),
        ("neighbors", "knn"),
        ("radius_um", "radius"),
    ):
        if settings["graph_mode"] != mode:
            unused.append(key)
    for key in unused:
        recipe["settings"].pop(key, None)
    if settings.get("id_mapping"):
        report["mapping_source"] = fingerprint_file(settings["id_mapping"])
    write_json_atomic(output.with_suffix(".metadata.json"), report)
    write_json_atomic(destination / "context.json", report)
    from morphofeatures.representation_analysis import package_results

    package_results(destination)
    return output
