"""Reconcile worker outcomes with cached scheduler observations.

A cached RUNNING/PENDING value is not proof that a finished worker still runs.
Keep the old observation in provenance, while respecting live scheduler checks
made after the worker finished. Age alone and arbitrary log text are not evidence
of failure. Deletion and the UI use the same persisted resolution.
"""

import json
from datetime import datetime
from pathlib import Path

from morphofeatures.metrics import utc_now
from morphofeatures.slurm import normalize_slurm_state

TERMINAL = frozenset({"completed", "failed", "cancelled"})
PROTECTED = frozenset({"deleting", "deletion_failed", "deleted", "dry_run"})


def _time(value):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        # sacct times can be naive cluster-local times. Do not compare them as
        # UTC against the worker's explicitly offset timestamp.
        return result if result.tzinfo is not None else None
    except (ValueError, TypeError):
        return None


def worker_outcome(record):
    """Read only this workspace worker's small, atomically written final status."""
    if record.workflow != "workspace_run":
        return {}
    path = Path(record.working_directory) / "status.json"
    try:
        if path.stat().st_size > 8 * 1024**2:
            return {}
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(value, dict) or value.get("state") not in {"failed", "completed"}:
        return {}
    if value.get("record_id") not in (None, record.id):
        return {}
    ended = _time(value.get("completed"))
    start = _time(record.started_at) or _time(record.submitted_at) or _time(record.created_at)
    if ended is None or (start and ended < start):
        return {}
    return value


def _confirmed_active(record, ended):
    observation = record.provenance.get("scheduler_observation", {})
    checked = _time(observation.get("checked_at"))
    return bool(
        normalize_slurm_state(record.raw_scheduler_state or "") in {"queued", "running"}
        and observation.get("raw_state") == record.raw_scheduler_state
        and checked
        and ended
        and checked >= ended
    )


def resolved_changes(record):
    if record.application_state in PROTECTED:
        return {}
    if record.provenance.get("local_stop"):
        from morphofeatures.job_control import local_stop_changes

        return {k: v for k, v in local_stop_changes(record).items() if v != getattr(record, k)}
    worker = worker_outcome(record)
    raw_state = normalize_slurm_state(record.raw_scheduler_state or "")
    changes = {}
    if worker:
        # A scheduler failure after image processing (e.g. OOM during shutdown)
        # remains a failed allocation even if scientific outputs were produced.
        state = raw_state if raw_state in {"failed", "cancelled"} else worker["state"]
        changes.update(
            application_state=state,
            completed_at=record.completed_at or worker["completed"],
            exit_status=record.exit_status or ("1" if state == "failed" else "0"),
        )
        if worker.get("error"):
            changes["error_message"] = str(worker["error"])
        elif state == "completed":
            changes["error_message"] = None
        artifacts = worker.get("artifacts", [])
        if isinstance(artifacts, list) and all(isinstance(p, str) for p in artifacts):
            changes["artifacts"] = tuple(dict.fromkeys((*record.artifacts, *artifacts)))
        ended = _time(worker["completed"])
    elif (
        record.application_state in TERMINAL
        and record.completed_at
        and record.exit_status is not None
    ):
        ended = _time(record.completed_at)
    elif raw_state in TERMINAL:
        changes.update(application_state=raw_state)
        if raw_state in {"failed", "cancelled"} and not record.error_message:
            changes["error_message"] = (
                f"SLURM ended with {record.raw_scheduler_state}. See the job logs."
            )
        ended = _time(record.completed_at)
    else:
        ended = None
    if ended and raw_state in {"queued", "running"} and not _confirmed_active(record, ended):
        changes["raw_scheduler_state"] = None
        changes["provenance"] = {
            **record.provenance,
            "status_resolution": {
                "resolved_at": utc_now(),
                "previous_scheduler_state": record.raw_scheduler_state,
                "worker_state": changes.get("application_state", record.application_state),
                "finished_at": ended.isoformat(),
                "source": str(Path(record.working_directory) / "status.json")
                if worker
                else "registry exit status",
                "reason": "Cached scheduler state predates the recorded worker outcome; it is not a current activity check.",
            },
        }
    return {key: value for key, value in changes.items() if value != getattr(record, key)}


def reconcile_job_states(registry, *, connection=None, record_ids=None):
    """Repair saved status for the UI and deletion, including downstream jobs.

    With an existing transaction this is also used by deletion's final check.
    Never infer that a job stopped from its age, missing accounting, or an error
    string alone. In-progress cancellation still needs scheduler confirmation.
    """
    if connection is None:
        with registry._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            return reconcile_job_states(registry, connection=connection, record_ids=record_ids)
    from morphofeatures.registry import JSON_FIELDS

    changed = 0
    for row in connection.execute("SELECT * FROM jobs ORDER BY id").fetchall():
        if record_ids is not None and row["id"] not in record_ids:
            continue
        record = registry._record(row)
        changes = resolved_changes(record)
        if not changes:
            continue
        encoded = {
            k: json.dumps(v, sort_keys=True) if k in JSON_FIELDS else v for k, v in changes.items()
        }
        connection.execute(
            "UPDATE jobs SET " + ", ".join(f"{key} = ?" for key in encoded) + " WHERE id = ?",
            (*encoded.values(), record.id),
        )
        changed += 1
    return changed


def scheduler_display(record):
    if record.raw_scheduler_state:
        return record.raw_scheduler_state
    previous = record.provenance.get("status_resolution", {}).get("previous_scheduler_state")
    return f"{previous} (outdated; worker finished)" if previous else "—"
