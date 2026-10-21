"""Metadata-only eligibility checks shared by physical DINO previews and workers.

Bounds use segmentation voxel coordinates and exclusive stops. A read includes
a one-voxel halo except at an explicitly permitted volume/ROI edge. The budget
counts the whole read box, including background; it is not a RAM allocation.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from morphofeatures.artifacts import write_json_atomic
from morphofeatures.workflow_validation import FieldError

POLICIES = {
    "boundary_policy": {"skip", "error", "allow_partial"},
    "oversized_policy": {"skip", "error"},
    "on_object_error": {"skip", "error"},
}


def validate_object_limits(source):
    for key, choices in POLICIES.items():
        if source.get(key, "skip") not in choices:
            raise FieldError(f"Choose one of: {', '.join(sorted(choices))}", key)
    value = source.get("max_object_voxels", 256**3)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FieldError(
            "Maximum object-box voxels must be a positive whole number", "max_object_voxels"
        )


def object_geometry(source, row):
    """Assess one indexed box without reading a segmentation or raw image chunk."""
    shape = np.asarray(source["segmentation_shape_zyx"], dtype=np.int64)
    roi = np.asarray(source.get("roi") or [[0, 0, 0], shape.tolist()], dtype=np.int64)
    lower, upper = (
        np.asarray([row[p + a] for a in "zyx"], dtype=np.int64) for p in ("bbox_min_", "bbox_max_")
    )
    visible_lo, visible_hi = np.maximum(lower, roi[0]), np.minimum(upper, roi[1])
    read_lo, read_hi = np.maximum(visible_lo - 1, roi[0]), np.minimum(visible_hi + 1, roi[1])
    required = math.prod(map(int, np.maximum(0, read_hi - read_lo)))
    budget = int(source.get("max_object_voxels", 256**3))
    volume_faces = [
        a + side
        for i, a in enumerate("ZYX")
        for side, contact in ((" minimum", lower[i] <= 0), (" maximum", upper[i] >= shape[i]))
        if contact
    ]
    roi_faces = [
        a + side
        for i, a in enumerate("ZYX")
        for side, contact in (
            (" minimum", roi[0, i] > 0 and lower[i] <= roi[0, i]),
            (" maximum", roi[1, i] < shape[i] and upper[i] >= roi[1, i]),
        )
        if contact
    ]
    boundary = bool(volume_faces or roi_faces)
    record = {
        "label_id": int(row["label_id"]),
        "decision": "eligible",
        "reason_code": "eligible",
        "reason": "",
        "touches_boundary": boundary,
        "volume_faces": ", ".join(volume_faces),
        "roi_faces": ", ".join(roi_faces),
        "required_voxels": required,
        "max_object_voxels": budget,
        "over_budget": required > budget,
        "read_shape_zyx": " × ".join(map(str, read_hi - read_lo)),
    }
    if np.any(lower < 0) or np.any(upper > shape) or np.any(upper <= lower):
        record.update(
            decision=source.get("on_object_error", "skip"),
            reason_code="invalid_bounds",
            reason="Object index has invalid bounds for this segmentation; select a matching bounds table or rebuild it.",
        )
    elif np.any(visible_hi <= visible_lo):
        record.update(
            decision="skip",
            reason_code="outside_roi",
            reason="Object does not intersect the selected segmentation ROI.",
        )
    elif boundary and source.get("boundary_policy", "skip") != "allow_partial":
        contacts = "; ".join(
            filter(
                None,
                [
                    "volume " + ", ".join(volume_faces) if volume_faces else "",
                    "ROI " + ", ".join(roi_faces) if roi_faces else "",
                ],
            )
        )
        record.update(
            decision=source.get("boundary_policy", "skip"),
            reason_code="boundary",
            reason=f"Object touches the {contacts} boundary; complete label coverage is unavailable. Prepare data → Object eligibility, boundaries and memory → Boundary objects: keep Skip, stop with an error, or explicitly use the available part. Expanding an ROI cannot restore tissue missing from the source volume.",
        )
    elif required > budget:
        record.update(
            decision=source.get("oversized_policy", "skip"),
            reason_code="mask_budget",
            reason=f"Object mask exceeds max_object_voxels: read box {record['read_shape_zyx']} requires {required:,} voxels; limit is {budget:,}. Prepare data → Object eligibility, boundaries and memory → Maximum object-box voxels: raise the limit if RAM permits, or use a coarser segmentation level. The object is never cropped to meet this budget.",
        )
    elif boundary:
        record.update(
            decision="partial",
            reason_code="partial",
            reason="Only the object portion inside the source volume and selected ROI will be encoded; fine and coarse outputs are explicitly marked partial.",
        )
    return record, read_lo, read_hi, visible_lo, visible_hi


def geometry_fields(record):
    return {
        "boundary": ("boundary_policy", "roi"),
        "mask_budget": ("max_object_voxels", "oversized_policy"),
        "invalid_bounds": ("object_index",),
        "outside_roi": ("roi",),
    }.get(record.get("reason_code"), ("on_object_error",))


class ObjectEligibilityError(FieldError):
    def __init__(self, record):
        self.record = record
        super().__init__(record["reason"], *geometry_fields(record))


def assess_objects(source, objects):
    records = [object_geometry(source, row)[0] for row in objects.to_dict("records")]
    table = pd.DataFrame(records)
    counts = Counter(r["reason_code"] for r in records)
    acceptable = [r for r in records if r["decision"] in {"eligible", "partial"}]
    over = [
        r["required_voxels"]
        for r in records
        if r["over_budget"]
        and (not r["touches_boundary"] or source.get("boundary_policy") == "allow_partial")
    ]
    return table, {
        "selected_objects": len(records),
        "eligible_objects": len(acceptable),
        "partial_objects": sum(r["decision"] == "partial" for r in records),
        "skip_objects": sum(r["decision"] == "skip" for r in records),
        "error_objects": sum(r["decision"] == "error" for r in records),
        "reasons": dict(counts),
        "minimum_budget_for_an_oversized_object": min(over) if over else None,
        "budget_for_all_oversized_objects": max(over) if over else None,
        "configured_budget": int(source.get("max_object_voxels", 256**3)),
        "interpretation": "Bounds-only screening: no image chunks or DINO weights are read. Contrast, stale bounds and tile cutoffs can still exclude objects during extraction. Counts use the selected cohort; skipped objects are not replaced.",
    }


def eligibility_message(summary):
    message = f"{summary['eligible_objects']} of {summary['selected_objects']} objects pass the boundary and mask-box checks. "
    names = {
        "boundary": "touch a volume/ROI boundary",
        "mask_budget": "exceed the mask-box budget",
        "invalid_bounds": "have invalid indexed bounds",
        "outside_roi": "are outside the ROI",
        "partial": "will use partial coverage",
        "eligible": "are eligible complete objects",
    }
    message += "; ".join(f"{n} {names.get(k, k)}" for k, n in summary["reasons"].items()) + ". "
    if summary["minimum_budget_for_an_oversized_object"]:
        message += f"Oversized boxes need {summary['minimum_budget_for_an_oversized_object']:,}–{summary['budget_for_all_oversized_objects']:,} voxels versus the configured {summary['configured_budget']:,}. "
    return (
        message
        + "See Prepare data → this dataset → Object eligibility, boundaries and memory. This limit covers the whole segmentation box, not the 1 µm fine tile or GPU batch."
    )


def write_eligibility(source, objects, destination):
    table, summary = assess_objects(source, objects)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination / "eligibility.tsv", sep="\t", index=False)
    write_json_atomic(destination / "eligibility.json", summary)
    blocked = table[~table.decision.isin(["eligible", "partial"])]
    if not summary["eligible_objects"] or summary["error_objects"]:
        blocked.to_csv(destination / "excluded.tsv", sep="\t", index=False)
        fields = list(
            dict.fromkeys(
                field for record in blocked.to_dict("records") for field in geometry_fields(record)
            )
        )
        raise FieldError(
            eligibility_message(summary)
            + f" Reports: {destination / 'eligibility.tsv'} and {destination / 'excluded.tsv'}. No DINO inference was started.",
            *fields,
        )
    return table, summary
