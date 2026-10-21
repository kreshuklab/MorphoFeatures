"""Reviewed deletion of owned run directories; never traverse shared artifact paths.

Registry tombstones preserve dependency IDs and prevent accidental reuse of a
deleted run ID. A same-filesystem rename separates a committed deletion from slow
file removal. Interrupted cleanup can be retried without touching a new directory.
"""

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path

from morphofeatures.metrics import utc_now

FINISHED = frozenset({"completed", "failed", "cancelled", "dry_run"})
DELETING = frozenset({"deleting", "deletion_failed", "deleted"})


def _run_path(output_root, run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("Invalid run ID; deletion is restricted to a named experiment directory.")
    root = Path(output_root).resolve()
    path = root / "experiments" / run_id
    if path.parent.is_symlink() or path.is_symlink() or path.resolve() != path:
        raise ValueError("The run directory must not be a symlink or redirect outside experiments.")
    return path


@contextmanager
def run_lock(output_root, run_id):
    directory = Path(output_root).resolve() / ".morphofeatures" / "run-locks"
    directory.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(run_id.encode()).hexdigest() + ".lock"
    with (directory / name).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def run_write_guard(artifact):
    """Serialize interactive analysis reservation with deletion of its owning run."""
    from morphofeatures.registry import JobRegistry

    path = Path(artifact).absolute()
    for run in path.parents:
        if run.parent.name != "experiments":
            continue
        root = run.parent.parent
        registry_path = root / ".morphofeatures/registry.sqlite3"
        if not registry_path.exists():
            continue
        with run_lock(root, run.name):
            registry = JobRegistry(registry_path)
            records = registry.list(run_id=run.name, include_deleted=True)
            if not run.is_dir() or any(r.application_state in DELETING for r in records):
                raise ValueError(
                    "This run has been deleted or is being deleted. Open another result."
                )
            yield
        return
    yield


def _metadata(path):
    import yaml

    if path.stat().st_size > 8 * 1024**2:
        raise ValueError(f"Cannot safely check an oversized metadata file before deletion: {path}")
    try:
        value = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ValueError(f"Unreadable metadata at {path}: {error}") from error
    return value if isinstance(value, dict) else {}


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child)


def _references(record):
    snapshot = Path(record.config_snapshot)
    paths = [snapshot] if snapshot.is_file() else []
    if snapshot.is_dir():
        paths = sorted(
            [*snapshot.glob("*.yaml"), *snapshot.glob("*.yml"), *snapshot.glob("*.json")]
        )
    values = [*record.artifacts, record.checkpoint_path, record.embedding_path, *record.command]
    warnings = []
    for path in paths:
        try:
            values.extend(_strings(_metadata(path)))
        except (OSError, ValueError) as error:
            warnings.append(
                f"Cannot check all input references for {record.run_id}: {path}: {error}"
            )
    result = set()
    for value in values:
        if isinstance(value, str) and "/" in value and "://" not in value:
            path = Path(value).expanduser()
            path = Path(os.path.abspath(Path(record.working_directory) / path))
            # A link inside the deleted run can be a dependency even when its
            # target is a shared external file that will be preserved.
            result.update((str(path), str(path.resolve())))
    return result, warnings


def _scheduler_active(record):
    from morphofeatures.slurm import normalize_slurm_state

    return record.raw_scheduler_state == "CANCEL_REQUESTED" or bool(
        record.slurm_job_id
        and normalize_slurm_state(record.raw_scheduler_state or "") in {"queued", "running"}
    )


def _inventory(directory):
    digest = hashlib.sha256()
    count, size, links = 0, 0, 0
    top = []
    if not directory.exists():
        return {"files": 0, "bytes": 0, "symlinks": 0, "top_level": [], "fingerprint": "absent"}
    if not directory.is_dir() or directory.is_symlink() or os.path.ismount(directory):
        raise ValueError(
            f"Deletion requires an ordinary run directory, not a file, symlink or mount: {directory}"
        )
    device = directory.stat().st_dev

    def visit(path):
        nonlocal count, size, links
        info = path.lstat()
        relative = str(path.relative_to(directory))
        digest.update(
            repr(
                (
                    relative,
                    info.st_dev,
                    info.st_ino,
                    info.st_mode,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                )
            ).encode()
        )
        if stat.S_ISLNK(info.st_mode):
            links += 1
        elif stat.S_ISDIR(info.st_mode):
            if info.st_dev != device or (path != directory and os.path.ismount(path)):
                raise ValueError(
                    f"A mounted directory is inside this run; it will not be deleted: {path}"
                )
            for child in sorted(path.iterdir()):
                if path == directory:
                    top.append(str(child))
                visit(child)
        elif stat.S_ISREG(info.st_mode):
            count += 1
            size += info.st_size
            if path.name in {"analysis_variants.json", "variant.json"}:
                metadata = _metadata(path)
                entries = (
                    metadata.get("variants", [])
                    if path.name == "analysis_variants.json"
                    else [metadata]
                )
                if any(
                    e.get("origin") != "workflow" and e.get("state") in {"queued", "running"}
                    for e in entries
                ):
                    raise ValueError(
                        f"An interactive analysis is still queued/running. Wait for it to finish before deleting this run: {path}"
                    )
        else:
            raise ValueError(
                f"Unexpected special file inside run; automatic deletion refused: {path}"
            )

    visit(directory)
    return {
        "files": count,
        "bytes": size,
        "symlinks": links,
        "top_level": top,
        "fingerprint": digest.hexdigest(),
    }


def _plan(registry, output_root, run_id, connection, *, records=None, reference_cache=None):
    root = _run_path(output_root, run_id)
    if records is None:
        from morphofeatures.job_status import reconcile_job_states

        reconcile_job_states(registry, connection=connection)
        records = [
            registry._record(row) for row in connection.execute("SELECT * FROM jobs ORDER BY id")
        ]
    if reference_cache is None:
        reference_cache = {}
    selected = [r for r in records if r.run_id == run_id]
    if not selected:
        raise ValueError("This run is not registered in the selected workspace.")
    if all(r.application_state == "deleted" for r in selected):
        raise ValueError("This run has already been deleted.")
    if any(r.application_state not in FINISHED | {"deleting", "deletion_failed"} for r in selected):
        raise ValueError(
            "Only finished runs can be deleted. Wait for every job in this run to finish."
        )
    if any(r.raw_scheduler_state == "CANCEL_REQUESTED" for r in selected):
        raise ValueError(
            "SLURM cancellation is still pending. Refresh SLURM until cancellation is confirmed before deleting."
        )
    if any(_scheduler_active(r) for r in selected):
        raise ValueError(
            "SLURM still records an active job. Refresh SLURM to confirm it has finished before deleting."
        )
    for record in selected:
        workdir = Path(record.working_directory)
        if not workdir.resolve().is_relative_to(root):
            raise ValueError(
                f"Registered working directory is outside this run; automatic deletion refused: {workdir}"
            )
    cleanup = root
    pending = {
        r.provenance.get("deletion", {}).get("cleanup_directory")
        for r in selected
        if r.application_state in DELETING
    }
    if pending:
        if len(pending) != 1 or None in pending:
            raise ValueError(
                "The interrupted deletion record is inconsistent; no files were removed."
            )
        cleanup = Path(pending.pop())
        expected = Path(output_root).resolve() / ".morphofeatures/deleted_runs"
        if (
            cleanup.parent != expected
            or not cleanup.name.startswith(run_id + "-")
            or cleanup.resolve() != cleanup
        ):
            raise ValueError("The interrupted cleanup path is outside the deletion area.")
    dependencies, preserved, warnings = [], set(), []
    ids, scheduler_ids = (
        {r.id for r in selected},
        {r.slurm_job_id for r in selected if r.slurm_job_id},
    )
    for record in records:
        if record.application_state == "deleted":
            continue
        if record.id not in reference_cache:
            reference_cache[record.id] = _references(record)
        references, notices = reference_cache[record.id]
        warnings.extend(notices)
        if record.run_id == run_id:
            preserved.update(p for p in references if not Path(p).is_relative_to(root))
            continue
        if Path(record.working_directory).resolve().is_relative_to(root):
            raise ValueError(
                f"Another registered run owns files inside this directory: {record.run_id}"
            )
        matching = sorted(p for p in references if Path(p).is_relative_to(root))
        linked = record.parent_job_id in ids or record.dependency_job_id in scheduler_ids
        if matching or linked:
            if record.application_state not in FINISHED | DELETING or _scheduler_active(record):
                raise ValueError(
                    f"Run {record.display_name} ({record.run_id}) is still active and depends on this run. Finish it before deleting these inputs."
                )
            dependencies.append(
                {
                    "run_id": record.run_id,
                    "name": record.display_name,
                    "state": record.application_state,
                    "references": matching,
                }
            )
    plan = {
        "run_id": run_id,
        "run_directory": str(root),
        "cleanup_directory": str(cleanup),
        "jobs": [
            {"id": r.id, "workflow": r.workflow, "state": r.application_state} for r in selected
        ],
        "inventory": _inventory(cleanup),
        "dependent_runs": dependencies,
        "preserved_paths": sorted(preserved),
        "warnings": warnings,
    }
    identity = {"plan": plan, "records": [registry._values(r) for r in selected]}
    plan["token"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return plan


def prepare_run_deletion(registry, output_root, run_id):
    with run_lock(output_root, run_id), registry._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        return _plan(registry, output_root, run_id, connection)


def _batch_plan(registry, output_root, run_ids, connection):
    from morphofeatures.job_status import reconcile_job_states

    reconcile_job_states(registry, connection=connection)
    records = [
        registry._record(row) for row in connection.execute("SELECT * FROM jobs ORDER BY id")
    ]
    cache, plans = {}, []
    for run_id in run_ids:
        try:
            plans.append(
                _plan(
                    registry,
                    output_root,
                    run_id,
                    connection,
                    records=records,
                    reference_cache=cache,
                )
            )
        except ValueError as error:
            raise ValueError(f"Run {run_id}: {error}") from error
    return {
        "runs": plans,
        "token": hashlib.sha256(json.dumps(plans, sort_keys=True).encode()).hexdigest(),
    }


@contextmanager
def _selection_locks(output_root, run_ids):
    # Stable lock order prevents two overlapping batch requests from deadlocking.
    run_ids = sorted(set(run_ids))
    if not run_ids:
        raise ValueError("Select at least one run to delete.")
    with ExitStack() as stack:
        for run_id in run_ids:
            _run_path(output_root, run_id)
            stack.enter_context(run_lock(output_root, run_id))
        yield run_ids


def prepare_runs_deletion(registry, output_root, run_ids):
    """Review a deduplicated selection without removing any files."""
    with _selection_locks(output_root, run_ids) as run_ids, registry._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        return _batch_plan(registry, output_root, run_ids, connection)


def _stage_deletion(connection, output_root, plan, moved):
    run_id = plan["run_id"]
    root, cleanup = Path(plan["run_directory"]), Path(plan["cleanup_directory"])
    if cleanup == root:
        trash = Path(output_root).resolve() / ".morphofeatures/deleted_runs"
        trash.mkdir(parents=True, exist_ok=True)
        if trash.is_symlink() or trash.resolve() != trash:
            raise ValueError("The deletion area must not redirect to another directory.")
        cleanup = trash / (run_id + "-" + uuid.uuid4().hex)
        if root.exists():
            root.rename(cleanup)
            moved.append((root, cleanup))
    for row in connection.execute("SELECT * FROM jobs WHERE run_id = ?", (run_id,)).fetchall():
        provenance = json.loads(row["provenance"])
        provenance["deletion"] = {
            "requested_at": utc_now(),
            "cleanup_directory": str(cleanup),
            "run_directory": str(root),
            "review": plan,
        }
        connection.execute(
            "UPDATE jobs SET application_state = 'deleting', provenance = ? WHERE id = ?",
            (json.dumps(provenance), row["id"]),
        )
    return cleanup


def _cleanup_deletion(registry, plan, cleanup):
    run_id = plan["run_id"]
    try:
        if cleanup.exists():
            # Never follow child symlinks to shared files outside the owned folder.
            if cleanup.is_symlink() or not shutil.rmtree.avoids_symlink_attacks:
                raise ValueError("Safe directory removal is unavailable for this path/platform.")
            shutil.rmtree(cleanup)
    except Exception as error:
        with registry._connect() as connection:
            connection.execute(
                "UPDATE jobs SET application_state = 'deletion_failed', error_message = ? WHERE run_id = ?",
                (f"Cleanup incomplete at {cleanup}: {error}. Review deletion to retry.", run_id),
            )
        raise ValueError(
            f"Deletion cleanup is incomplete at {cleanup}. The run remains listed so you can retry. {error}"
        ) from error
    with registry._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        for row in connection.execute(
            "SELECT id, provenance FROM jobs WHERE run_id = ?", (run_id,)
        ).fetchall():
            provenance = json.loads(row["provenance"])
            provenance["deletion"]["deleted_at"] = utc_now()
            connection.execute(
                "UPDATE jobs SET application_state = 'deleted', artifacts = '[]', error_message = NULL, provenance = ? WHERE id = ?",
                (json.dumps(provenance), row["id"]),
            )


def _delete_reviewed_runs(registry, output_root, run_ids, *, confirmation_token, single=False):
    with _selection_locks(output_root, run_ids) as run_ids:
        with registry._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            review = _batch_plan(registry, output_root, run_ids, connection)
            expected = review["runs"][0]["token"] if single else review["token"]
            if not confirmation_token or confirmation_token != expected:
                raise ValueError(
                    "The selected runs or their files changed after review. Review deletion again and confirm the updated list."
                )
            # All plans are checked before moving any folder. Stage the entire
            # selection in one transaction, then perform slow cleanup outside it.
            moved, cleanups = [], []
            try:
                for plan in review["runs"]:
                    cleanups.append(_stage_deletion(connection, output_root, plan, moved))
                connection.commit()
            except Exception:
                connection.rollback()
                for root, cleanup in reversed(moved):
                    cleanup.rename(root)
                raise
        outcome = {"deleted": [], "failed": []}
        for plan, cleanup in zip(review["runs"], cleanups):
            try:
                _cleanup_deletion(registry, plan, cleanup)
            except Exception as error:
                # Keep trying the rest; each unfinished cleanup stays listed for retry.
                outcome["failed"].append({"run_id": plan["run_id"], "error": str(error)})
            else:
                outcome["deleted"].append(plan)
        return outcome


def delete_runs(registry, output_root, run_ids, *, confirmation_token):
    """Validate the whole selection, then delete; report cleanup failures per run."""
    return _delete_reviewed_runs(
        registry, output_root, run_ids, confirmation_token=confirmation_token
    )


def delete_run(registry, output_root, run_id, *, confirmation_token):
    """Recheck the reviewed plan, tombstone every job, then remove only owned files."""
    outcome = _delete_reviewed_runs(
        registry, output_root, [run_id], confirmation_token=confirmation_token, single=True
    )
    if outcome["failed"]:
        raise ValueError(outcome["failed"][0]["error"])
    return outcome["deleted"][0]


def forget_deleted_run(state, run_id, job_ids, directory):
    """Discard stale navigation/confirmation pointers without changing user drafts."""

    def refers(value):
        return any(
            text in job_ids
            or text == run_id
            or text == str(directory)
            or text.startswith(str(directory) + "/")
            for text in _strings(value)
        )

    for key in (
        "runs:selected",
        "selected_run_id",
        "results:registered",
        "value:results:path",
        "widget:results:path",
    ):
        if key in state and refers(state[key]):
            state.pop(key, None)
    if "runs:selected_ids" in state:
        state["runs:selected_ids"] = [
            job_id for job_id in state["runs:selected_ids"] if job_id not in job_ids
        ]
    state["navigation:history"] = [
        screen for screen in state.get("navigation:history", []) if not refers(screen)
    ]
    if refers(state.get("navigation:current", {})):
        state.pop("navigation:current", None)
    state["runs:table_epoch"] = state.get("runs:table_epoch", 0) + 1
