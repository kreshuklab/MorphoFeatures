"""Exact, reusable object cohorts selected before image reads or embedding inference.

Selection uses the workflow's output IDs by default. A physical source can instead
match segmentation IDs. An optional one-to-one table translates selected IDs into
that namespace; matching numbers alone never imply cell/nucleus correspondence.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from morphofeatures.analysis.relations import integer_ids, read_table
from morphofeatures.artifacts import write_json_atomic
from morphofeatures.workflow_validation import FieldError, prefix_fields


def selection_active(options):
    if options is not None and not isinstance(options, dict):
        raise FieldError(
            "Target objects must be a YAML mapping with mode: all, ids, or table.",
            "object_selection",
        )
    return bool(options and options.get("mode", "all") != "all")


def _fingerprint(path):
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def validate_extraction_selection(settings):
    """A draft switched away from DINO must not silently encode all objects."""
    if selection_active(settings.get("object_selection")) and settings.get("model", "mae") not in {
        "dinov2",
        "dinov3",
    }:
        raise FieldError(
            "Target objects on an extraction stage currently requires DINO. For MAE, select the cohort on Prepare data first, then set this extraction's Objects to process to All. Alternatively switch the embedding model back to DINO.",
            "object_selection.mode",
            "model",
        )


def resolve_selection(options):
    """Read only ID/annotation metadata. Return exact requested→target pairs and audit."""
    if not selection_active(options):
        return None, None
    with prefix_fields("object_selection"):
        mode = options.get("mode")
        if mode not in {"ids", "table"}:
            raise FieldError("Choose all objects, an ID list, or an ID table.", "mode")
        if options.get("missing", "error") not in {"error", "skip"}:
            raise FieldError("Choose error or skip for unavailable IDs.", "missing")
        if options.get("id_space", "output") not in {"output", "segmentation"}:
            raise FieldError("Choose output or segmentation IDs.", "id_space")
        sources = {}
        if mode == "ids":
            values = options.get("ids", [])
            if not isinstance(values, (list, tuple)):
                raise FieldError("Enter a list of exact IDs, for example [12, 35, 81].", "ids")
            total_rows = len(values)
        else:
            if (
                not isinstance(options.get("ids_file"), (str, Path))
                or not str(options["ids_file"]).strip()
            ):
                raise FieldError(
                    "Choose a CSV/TSV file containing the target IDs, or switch to Enter an ID list.",
                    "ids_file",
                )
            with prefix_fields("ids_file"):
                frame = read_table(Path(options.get("ids_file") or "").expanduser())
                sources["ids_file"] = _fingerprint(options["ids_file"])
            total_rows = len(frame)
            column = options.get("id_column", "label_id")
            if column not in frame:
                raise FieldError(
                    f"ID column {column!r} is absent. Available columns: {list(frame)}", "id_column"
                )
            filtering = options.get("filter_column")
            if filtering:
                if filtering not in frame:
                    raise FieldError(
                        f"Filter column {filtering!r} is absent. Available columns: {list(frame)}",
                        "filter_column",
                    )
                categories = frame[filtering].fillna("").str.strip()
                if options.get("require_nonempty", True):
                    frame = frame.loc[categories.ne("")]
                choices = options.get("filter_values", [])
                if not isinstance(choices, list):
                    raise FieldError(
                        "Filter values must be a list of category names.", "filter_values"
                    )
                if choices:
                    frame = frame.loc[
                        categories.reindex(frame.index).isin([str(v).strip() for v in choices])
                    ]
            values = frame[column].tolist()
        with prefix_fields("ids" if mode == "ids" else "id_column"):
            if any(isinstance(value, (float, np.floating, bool)) for value in values):
                raise ValueError(
                    "Use exact integer IDs or integer strings, not floating-point values or booleans."
                )
            parsed = integer_ids(values, "Selected object ID")
        requested = np.unique(parsed)
        if not len(requested):
            raise FieldError(
                "No IDs remain in Target objects. Check the ID list, table or category filter. "
                "An empty selection never means process everything.",
                "ids",
                "ids_file",
                "filter_values",
            )
        targets = {int(i): int(i) for i in requested}
        if options.get("mapping_file"):
            with prefix_fields("mapping_file"):
                mapping = read_table(Path(options["mapping_file"]).expanduser())
                sources["mapping_file"] = _fingerprint(options["mapping_file"])
                a, b = (
                    options.get("mapping_from", "label_id"),
                    options.get("mapping_to", "nucleus_id"),
                )
                if a not in mapping or b not in mapping:
                    raise ValueError(
                        f"Mapping requires columns {a!r} and {b!r}; available: {list(mapping)}"
                    )
                left, right = integer_ids(mapping[a]), integer_ids(mapping[b])
                keep = (left > 0) & (right > 0)
                left, right = left[keep], right[keep]
                if len(np.unique(left)) != len(left) or len(np.unique(right)) != len(right):
                    raise ValueError(
                        "Positive ID mappings must be one-to-one; duplicate IDs are ambiguous."
                    )
                lookup = {int(a): int(b) for a, b in zip(left, right)}
            targets = {int(i): lookup.get(int(i)) for i in requested}
        rows = [
            {
                "requested_id": i,
                "target_id": target,
                "label_id": None,
                "status": "requested" if target is not None else "missing mapping",
            }
            for i, target in targets.items()
        ]
        audit = {
            "schema": "morphofeatures.object_selection.v1",
            "settings": options,
            "sources": sources,
            "table_rows": total_rows,
            "filtered_rows": len(parsed),
            "duplicate_ids_removed": len(parsed) - len(requested),
            "requested": len(requested),
            "interpretation": "Explicit ID selection before image encoding. ROI, object limits and extraction-quality checks can further exclude objects.",
            "rows": rows,
        }
        return targets, audit


def select_objects(table, options, *, destination=None, report=None):
    """Filter a label_id[/segmentation_id] table without loading any image pixels."""
    targets, audit = resolve_selection(options)
    if targets is None:
        return table, None
    column = (
        "segmentation_id" if options.get("id_space", "output") == "segmentation" else "label_id"
    )
    if column not in table:
        raise FieldError(
            "These prepared inputs expose only their saved object IDs. Select 'Workflow output IDs', "
            "and supply an ID mapping if your annotation IDs differ from the saved label_ids.",
            "object_selection.id_space",
        )
    if table[column].duplicated().any() or table.label_id.duplicated().any():
        raise FieldError(
            "Input IDs must be unique and map one-to-one to output IDs before targeting objects.",
            "object_selection",
        )
    lookup = dict(zip(map(int, table[column]), map(int, table.label_id)))
    for row in audit["rows"]:
        if row["target_id"] is not None:
            row["label_id"] = lookup.get(row["target_id"])
            row["status"] = "selected" if row["label_id"] is not None else "absent from input / ROI"
    selected = table[
        table.label_id.isin([r["label_id"] for r in audit["rows"] if r["label_id"] is not None])
    ]
    missing = [r["requested_id"] for r in audit["rows"] if r["label_id"] is None]
    audit.update(matched=len(selected), missing=len(missing))
    if report is not None:
        report.update(audit)
    if destination is not None:
        write_selection(destination, audit)
    if missing and options.get("missing", "error") == "error":
        raise FieldError(
            f"Target objects: {len(missing)} of {len(targets)} requested IDs are unavailable; examples: {missing[:10]}. "
            "Check the cell/nucleus namespace, mapping, ROI and input ID array. To process only available "
            "objects, choose 'Skip unavailable IDs and report them'.",
            "object_selection.missing",
            "object_selection.ids_file",
            "object_selection.mapping_file",
        )
    if selected.empty:
        raise FieldError(
            "Target objects matched no input objects. Check the ID namespace, mapping and ROI; no images will be encoded.",
            "object_selection",
        )
    return selected, audit


def write_selection(destination, audit, *, selected_ids=None):
    if audit is None:
        return
    if selected_ids is not None:
        selected_ids = set(map(int, selected_ids))
        for row in audit["rows"]:
            if row["status"] == "selected" and row["label_id"] not in selected_ids:
                row["status"] = "excluded by object limit or other selection"
    audit["selected"] = sum(r["status"] == "selected" for r in audit["rows"])
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            key: pd.array([r[key] for r in audit["rows"]], dtype="Int64")
            for key in ("requested_id", "target_id", "label_id")
        }
    )
    frame["status"] = [r["status"] for r in audit["rows"]]
    for key in ("preparation_status", "preparation_reason"):
        if any(key in row for row in audit["rows"]):
            frame[key] = [row.get(key, "") for row in audit["rows"]]
    frame.to_csv(root / "selection.tsv", sep="\t", index=False)
    write_json_atomic(root / "selection.json", audit)


def resolve_selection_paths(document, base):
    """Resolve selection-specific paths wherever they occur in a workflow copy."""
    if isinstance(document, dict):
        options = document.get("object_selection") or {}
        selection_active(options)  # Validate type before resolving paths.
        for key in ("ids_file", "mapping_file"):
            if options.get(key):
                options[key] = str((Path(base) / Path(options[key]).expanduser()).resolve())
        for value in document.values():
            resolve_selection_paths(value, base)
    elif isinstance(document, list):
        for value in document:
            resolve_selection_paths(value, base)


def selection_sources(document):
    """Fingerprint selection files without making the cohort part of the model recipe."""
    result = []
    if isinstance(document, dict):
        options = document.get("object_selection")
        if selection_active(options):
            _, audit = resolve_selection(options)
            result.append(audit["sources"])
        for key, value in document.items():
            if key != "object_selection":
                result.extend(selection_sources(value))
    elif isinstance(document, list):
        for value in document:
            result.extend(selection_sources(value))
    return result
