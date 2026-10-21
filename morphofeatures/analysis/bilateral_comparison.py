"""Reuse immutable saved vectors, then recompute ranks on a common cohort."""

import json
from pathlib import Path

from morphofeatures.analysis.relations import integer_ids, read_table
from morphofeatures.representations import fingerprint_file
from morphofeatures.workflow_validation import FieldError


def inside(report, relative):
    root = Path(report).resolve().parent
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Bilateral artifact paths must remain within the saved result directory")
    return path


def read_report(path, *, field="comparison_reports"):
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size > 64 * 1024**2:
        raise FieldError("Choose an existing bilateral.json (at most 64 MiB).", field)
    try:
        report = json.loads(path.read_text())
    except (ValueError, OSError) as error:
        raise FieldError(f"Cannot read bilateral report {path}: {error}", field) from error
    if report.get("schema") != "morphofeatures.bilateral.v1" or not report.get("representations"):
        raise FieldError(f"{path.name} is not a saved bilateral-probe result.", field)
    return path, report


def saved_inputs(settings, *, check_files=True):
    """Namespaced saved inputs retain the old cell-to-native ID bridge."""
    if not settings.get("compare_saved_runs"):
        return {}
    reports = settings.get("comparison_reports", {})
    if not isinstance(reports, dict) or not reports:
        raise FieldError(
            "Select at least one saved bilateral run, or disable saved-run comparison.",
            "comparison_reports",
        )
    inputs = {}
    for label, path in reports.items():
        source, report = read_report(path)
        for name, entry in report["representations"].items():
            directory = inside(source, entry["directory"])
            embedding = directory / "embeddings.npz"
            bridge = directory / "id_mapping.tsv"
            if check_files and (not embedding.is_file() or not bridge.is_file()):
                raise FieldError(
                    f"{label} / {name}: saved vectors or ID bridge are missing. Restore the whole result folder; historical rank tables alone cannot be fairly recomputed on a different cohort.",
                    "comparison_reports",
                )
            key = f"{label} / {name}"
            if key in inputs:
                raise FieldError(
                    "Saved-run labels produce duplicate representation names; rename a comparison entry.",
                    "comparison_reports",
                )
            inputs[key] = {
                "embedding": str(embedding),
                "bridge": str(bridge),
                "source": entry["source"],
                "saved_run": {
                    "report": fingerprint_file(source),
                    "representation": name,
                    "original_settings": report["settings"],
                    "original_matched_objects": report["matched_objects"],
                    "original_variants": entry["variants"],
                },
            }
    return inputs


def read_ranks(path):
    frame = read_table(path)
    for key in ("label_id", "native_id", "best_partner_id"):
        if key in frame:
            import pandas as pd

            frame[key] = pd.array(
                [None if pd.isna(v) else int(integer_ids([v])[0]) for v in frame[key]],
                dtype="Int64",
            )
    for key in ("rank", "feature_distance", "search_objects", "candidate_count_present"):
        if key in frame:
            import pandas as pd

            frame[key] = pd.to_numeric(frame[key])
    return frame
