"""Exact, bounded-memory bilateral-candidate retrieval on matched object cohorts.

Ranks are one-based after removing the query itself. Equal-distance matches use
the best possible rank, with tie counts exported. This explicit convention agrees
with the historical all-cell code in the absence of ties; historical opposite-side
indices were zero-based. Candidate sets are evaluation inputs, never feature inputs.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from threadpoolctl import threadpool_limits

from morphofeatures.analysis.projection import normalize_features
from morphofeatures.analysis.relations import cell_identity, integer_ids, read_relations, read_table
from morphofeatures.artifacts import write_json_atomic
from morphofeatures.config import repository_root
from morphofeatures.data.io import export_embeddings, load_embeddings
from morphofeatures.representations import fingerprint_file
from morphofeatures.workflow_validation import FieldError


def defaults():
    root = repository_root() / "analysis/data"
    return {
        "candidates": str(root / "bilateral_neighbors.pkl"),
        "sides": str(root / "distance_from_midline_cells_1_0_1.tsv"),
        "id_namespace": "cell",
        "batch_size": 128,
        "workers": 2,
        "top_k": [1, 5, 10, 50, 100],
        "variants": [
            {
                "name": "Published protocol",
                "normalization": "standardize",
                "metric": "euclidean",
                "search": "all",
                "pca": None,
            }
        ],
    }


def validate_bilateral(settings, *, check_files=True, representation_count=1):
    if (representation_count > 1 or settings.get("compare_saved_runs")) and not settings.get(
        "same_specimen"
    ):
        raise FieldError(
            "Bilateral-partner retrieval: confirm these representations describe the same specimen "
            "and corresponding cells. On Analyze, open the bilateral-partner retrieval settings "
            "and select 'These representations describe the same specimen and cells'. "
            "Raw and balanced combinations of the same source cells can be compared; matching "
            "numeric IDs from different specimens is not sufficient. If you only want fusion, "
            "projections and classification, turn off 'Evaluate bilateral-partner retrieval'.",
            "same_specimen",
        )
    if settings.get("compare_saved_runs"):
        from morphofeatures.analysis.bilateral_comparison import saved_inputs

        saved_inputs(settings, check_files=check_files)
    for key in ("batch_size", "workers"):
        value = settings.get(key, defaults()[key])
        if (
            isinstance(value, bool)
            or int(value) != value
            or not 1 <= value <= (4096 if key == "batch_size" else 32)
        ):
            raise FieldError(
                f"{key} must be a positive integer within the bounded CPU/read limit.", key
            )
    top = settings.get("top_k", [1, 5, 10, 50, 100])
    if not top or any(int(k) != k or k < 1 for k in top):
        raise FieldError(
            "Retrieval cutoffs must be positive integers, such as [1, 5, 10, 50, 100].", "top_k"
        )
    variants = settings.get("variants", defaults()["variants"])
    if not isinstance(variants, list) or not 1 <= len(variants) <= 32:
        raise FieldError("Add between 1 and 32 bilateral variants.", "variants")
    names = []
    for variant in variants:
        if not isinstance(variant, dict):
            raise FieldError("Each variant must be a settings mapping.", "variants")
        if not str(variant.get("name", "")).strip():
            raise FieldError("Give every bilateral variant a descriptive name.", "variants")
        names.append(str(variant["name"]))
        if variant.get("metric", "euclidean") not in {"euclidean", "cosine"}:
            raise FieldError("Bilateral metric must be euclidean or cosine.", "variants")
        if variant.get("search", "all") not in {"all", "opposite"}:
            raise FieldError("Search must be all or opposite.", "variants")
        if variant.get("normalization", "standardize") not in {"none", "l2", "standardize"}:
            raise FieldError("Choose none, l2 or standardize normalization.", "variants")
        if variant.get("pca") is not None and (
            int(variant["pca"]) != variant["pca"] or variant["pca"] < 1
        ):
            raise FieldError(
                "PCA dimensions must be a positive integer, or null for full features.", "variants"
            )
    if len(names) != len(set(names)):
        raise FieldError("Give each bilateral variant a unique name.", "variants")
    for name, identity in [("default", settings)] + list(settings.get("identities", {}).items()):
        if identity.get("id_namespace", "cell") not in {"cell", "nucleus"}:
            raise FieldError(f"{name}: choose cell or nucleus IDs.", "id_namespace", "identities")
        if identity.get("id_namespace") == "nucleus" and not identity.get("id_mapping"):
            raise FieldError(
                f"{name}: nucleus embeddings require a cell-to-nucleus mapping.",
                "id_mapping",
                "identities",
            )
        if (
            check_files
            and identity.get("id_mapping")
            and not Path(identity["id_mapping"]).is_file()
        ):
            raise FieldError(f"{name}: ID mapping file does not exist.", "id_mapping", "identities")
    if check_files:
        for key in ["candidates"] + (
            ["sides"] if any(v.get("search") == "opposite" for v in variants) else []
        ):
            if not settings.get(key) or not Path(settings[key]).is_file():
                raise FieldError(f"Select an existing {key} table for the bilateral probe.", key)


def _side_values(path, ids):
    frame = read_table(path)
    if not {"label_id", "side"} <= set(frame):
        raise FieldError("Body-side table needs label_id and side (True/False or 1/0).", "sides")
    frame["label_id"] = integer_ids(frame.label_id)
    if frame.label_id.duplicated().any():
        raise FieldError("Body-side table contains duplicate cell IDs.", "sides")
    values = (
        frame.set_index("label_id")
        .side.str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .reindex(ids)
    )
    if values.isna().any():
        raise FieldError(
            "Body-side table must provide a valid side for every cell in the shared search population.",
            "sides",
        )
    return values.to_numpy(dtype=bool)


def rank_candidates(features, ids, candidates, variant, *, sides=None, batch_size=128, workers=2):
    """Compute all exact ranks in blocks, without sorting a full n-by-n matrix."""
    values = normalize_features(features, variant.get("normalization", "standardize"))
    components = variant.get("pca")
    if components is not None:
        if components > min(values.shape):
            raise FieldError(
                f"PCA requests {components} components, but this cohort permits at most {min(values.shape)}.",
                "variants",
            )
        values = PCA(
            n_components=int(components), random_state=42, svd_solver="full"
        ).fit_transform(values)
    metric = variant.get("metric", "euclidean")
    if metric == "cosine" and np.any(np.linalg.norm(values, axis=1) == 0):
        raise FieldError(
            "Cosine distance is undefined for zero-norm vectors after preprocessing; choose Euclidean or another normalization.",
            "variants",
        )
    opposite = variant.get("search", "all") == "opposite"
    if opposite and sides is None:
        raise FieldError("Opposite-side search requires a body-side table.", "sides")
    positions = {int(label): i for i, label in enumerate(ids)}
    rows = []
    with threadpool_limits(limits=int(workers)):
        for start in range(0, len(ids), int(batch_size)):
            distances = pairwise_distances(
                values[start : start + batch_size], values, metric=metric, n_jobs=1
            )
            for local, distance in enumerate(distances):
                i = start + local
                candidates_all = candidates.get(int(ids[i]), ())
                present = [
                    positions[int(k)]
                    for k in candidates_all
                    if int(k) in positions and int(k) != int(ids[i])
                ]
                eligible = np.arange(len(ids)) != i
                if opposite:
                    eligible &= sides != sides[i]
                reason = ""
                if not len(candidates_all):
                    reason = "No candidate partners listed"
                elif not present:
                    reason = "No candidate partners remain in the shared cohort"
                elif opposite and any(sides[j] == sides[i] for j in present):
                    reason = "Candidate set includes same-side cells; excluded by historical opposite-side rule"
                row = {
                    "label_id": int(ids[i]),
                    "candidate_count_original": len(candidates_all),
                    "candidate_count_present": len(present),
                    "search_objects": int(eligible.sum()),
                    "reason": reason,
                    "best_partner_id": None,
                    "rank": None,
                    "tied_at_best_distance": None,
                    "feature_distance": None,
                }
                if not reason:
                    best_distance = float(np.min(distance[present]))
                    # Report ties explicitly; self is always excluded, even for identical vectors.
                    ties = np.isclose(distance, best_distance, rtol=1e-12, atol=1e-12)
                    best = min(int(ids[j]) for j in present if ties[j])
                    row.update(
                        best_partner_id=best,
                        rank=1 + int(np.sum(eligible & (distance < best_distance) & ~ties)),
                        tied_at_best_distance=int(np.sum(eligible & ties)),
                        feature_distance=best_distance,
                    )
                rows.append(row)
    frame = pd.DataFrame(rows)
    for column in ("best_partner_id", "rank", "tied_at_best_distance"):
        # Construct from Python integers, not an intermediate float column (large IDs).
        frame[column] = pd.array([row[column] for row in rows], dtype="Int64")
    return frame


def _chance(m, c, k):
    k = min(k, m)
    if k > m - c:
        return 1.0
    return 1 - math.exp(sum(math.log((m - k - j) / (m - j)) for j in range(c)))


def run_bilateral(embeddings, destination, settings):
    settings = {**defaults(), **settings}
    validate_bilateral(settings, representation_count=len(embeddings))
    if not embeddings:
        raise FieldError("Add at least one named embedding file.", "embeddings")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    tables, bridges = {}, {}
    from morphofeatures.analysis.bilateral_comparison import saved_inputs

    saved = saved_inputs(settings)
    if set(saved) & set(embeddings):
        raise FieldError(
            "A saved-run representation name duplicates a new input. Rename the comparison run label.",
            "comparison_reports",
        )
    for name, path in embeddings.items():
        identity = {**settings, **settings.get("identities", {}).get(name, {})}
        tables[name], bridges[name] = cell_identity(load_embeddings(path), identity)
        tables[name] = tables[name].sorted()
    for name, entry in saved.items():
        tables[name] = load_embeddings(entry["embedding"]).sorted()
        bridge = read_table(entry["bridge"])
        for key in ("label_id", "native_id"):
            bridge[key] = integer_ids(bridge[key])
        if (
            bridge.label_id.duplicated().any()
            or bridge.native_id.duplicated().any()
            or not set(tables[name].label_ids) <= set(bridge.label_id)
        ):
            raise FieldError(
                f"{name}: saved ID bridge is incomplete or ambiguous.", "comparison_reports"
            )
        bridges[name] = bridge
    if saved and not settings.get("same_specimen"):
        raise FieldError(
            "Confirm the saved runs and new vectors describe the same specimen/cells.",
            "same_specimen",
        )
    ids = np.asarray(
        sorted(set.intersection(*(set(t.label_ids) for t in tables.values()))), dtype=np.int64
    )
    if len(ids) < 2:
        raise FieldError(
            "Fewer than two shared cell IDs. Check specimen identity and the cell/nucleus mappings.",
            "embeddings",
            "id_mapping",
            "identities",
        )
    shared_ids = set(ids)
    edges = read_relations(settings["candidates"], bilateral=True)
    candidates = {}
    for a, b in edges:
        candidates.setdefault(int(a), []).append(int(b))
    sides = (
        _side_values(settings["sides"], ids)
        if any(v.get("search") == "opposite" for v in settings["variants"])
        else None
    )
    report = {
        "schema": "morphofeatures.bilateral.v1",
        "settings": settings,
        "matched_objects": len(ids),
        "candidate_source": fingerprint_file(settings["candidates"]),
        "representations": {},
        "comparison_protocol": "All new and saved vectors are re-evaluated on their shared cell population using this run's candidate table and variants. Historical scores are retained only as provenance; no old curves are overlaid.",
        "rank_convention": "one-based, query excluded, best rank for ties (tie counts exported)",
        "interpretation": "Candidate-partner retrieval, not verified one-to-one pairing or classification accuracy. Lower ranks are better. All methods use the same shared cell search population.",
    }
    if sides is not None:
        report["side_source"] = fingerprint_file(settings["sides"])
    summary, plots = [], []
    report["warnings"] = []
    for index, (name, table) in enumerate(tables.items()):
        folder = destination / f"representation-{index}"
        folder.mkdir(exist_ok=True)
        x = table.features[np.searchsorted(table.label_ids, ids)]
        export_embeddings(folder / "embeddings.npz", ids, x)
        bridges[name].to_csv(folder / "id_mapping.tsv", sep="\t", index=False)
        result = {
            "directory": folder.name,
            "source": saved[name]["source"]
            if name in saved
            else fingerprint_file(embeddings[name]),
            "excluded_from_matching": [int(i) for i in table.label_ids if i not in shared_ids],
            "variants": [],
        }
        if name in saved:
            result["saved_run"] = saved[name]["saved_run"]
            result["saved_vectors_source"] = fingerprint_file(saved[name]["embedding"])
        for n, variant in enumerate(settings["variants"]):
            frame = rank_candidates(
                x,
                ids,
                candidates,
                variant,
                sides=sides,
                batch_size=settings["batch_size"],
                workers=settings["workers"],
            )
            frame = frame.merge(bridges[name], on="label_id", validate="one_to_one")
            path = folder / f"ranks-{n}.tsv"
            frame.to_csv(path, sep="\t", index=False)
            good = frame[frame["rank"].notna()]
            row = {
                "representation": name,
                "variant": variant["name"],
                "evaluated": len(good),
                "excluded": len(frame) - len(good),
                "median_rank": float(good["rank"].median()) if len(good) else None,
                "mean_rank": float(good["rank"].mean()) if len(good) else None,
                "queries_with_ties": int((good.tied_at_best_distance > 1).sum()),
            }
            for k in settings["top_k"]:
                row[f"recall_at_{k}"] = float((good["rank"] <= k).mean()) if len(good) else None
                row[f"chance_at_{k}"] = (
                    float(
                        np.mean(
                            [
                                _chance(int(r.search_objects), int(r.candidate_count_present), k)
                                for r in good.itertuples()
                            ]
                        )
                    )
                    if len(good)
                    else None
                )
            summary.append(row)
            if not len(good):
                report["warnings"].append(
                    f"{name} / {variant['name']}: no evaluable queries. Check the ID namespace, candidate coverage and opposite-side exclusions in {folder.name}/{path.name}."
                )
            elif row["queries_with_ties"]:
                report["warnings"].append(
                    f"{name} / {variant['name']}: {row['queries_with_ties']} queries have distance ties at their best candidate. Best-possible tied ranks can make identical or collapsed vectors look artificially successful; inspect tied_at_best_distance."
                )
            result["variants"].append({"settings": variant, "ranks": path.name, "summary": row})
            if len(good):
                ranks = np.sort(good["rank"].to_numpy(dtype=int))
                plots.append((name + " · " + variant["name"], ranks))
        report["representations"][name] = result
    pd.DataFrame(summary).to_csv(destination / "summary.tsv", sep="\t", index=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for label, ranks in plots:
        ax.step(
            np.r_[0, ranks],
            np.r_[0, np.arange(1, len(ranks) + 1) / len(ranks)],
            where="post",
            label=label,
        )
    ax.set(
        xlabel="Nearest candidate partner rank (lower is better)",
        ylabel="Fraction of evaluated cells",
        ylim=(0, 1),
        xlim=(0, min(max(settings["top_k"]), len(ids) - 1)),
    )
    if plots:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, frameon=False)
    for suffix in ("svg", "png"):
        fig.savefig(destination / f"bilateral.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    path = write_json_atomic(destination / "bilateral.json", report)
    from morphofeatures.representation_analysis import package_results

    package_results(destination)
    return path
