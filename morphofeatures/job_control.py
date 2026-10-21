"""Stop registered workers without signalling unrelated or recycled local PIDs."""

import json
import os
import signal
import socket
from datetime import datetime
from pathlib import Path

import psutil

from morphofeatures.metrics import utc_now
from morphofeatures.slurm import normalize_slurm_state

ACTIVE = frozenset({"created", "submitting", "queued", "running", "unknown", "cancelling"})
REMOVED = frozenset({"dry_run", "deleting", "deletion_failed", "deleted"})


def can_stop_slurm(record):
    return bool(
        record.slurm_job_id
        and record.application_state not in REMOVED
        and (
            record.application_state in ACTIVE
            or record.raw_scheduler_state == "CANCEL_REQUESTED"
            or normalize_slurm_state(record.raw_scheduler_state or "") in {"queued", "running"}
        )
    )


def process_metadata(record, pid):
    """Capture identity at launch; the worker owns its POSIX session/group."""
    data = {
        "pid": pid,
        "pgid": pid,
        "hostname": socket.gethostname(),
        "record_id": record.id,
        "started": utc_now(),
    }
    try:
        data["create_time"] = psutil.Process(pid).create_time()
    except psutil.NoSuchProcess:
        pass  # A very short worker can have finished before its parent records it.
    return data


def _live(process):
    return process.is_running() and process.status() not in {
        psutil.STATUS_ZOMBIE,
        psutil.STATUS_DEAD,
    }


def _members(pgid):
    if not hasattr(os, "getpgid"):
        raise ValueError("Local job stopping requires a POSIX host (Linux/macOS).")
    members = []
    for process in psutil.process_iter():
        try:
            if os.getpgid(process.pid) == pgid and _live(process):
                if os.getsid(process.pid) != pgid or process.uids().real != os.getuid():
                    raise ValueError("The saved process group does not belong to this worker/user.")
                members.append(process)
        except (ProcessLookupError, psutil.NoSuchProcess):
            continue
    return members


def _same_process(process, identity):
    return (
        process.pid == identity["pid"]
        and abs(process.create_time() - identity["create_time"]) < 0.01
    )


def local_target(record):
    """Verify launch host, session, PID creation time, argv and working directory.

    Legacy PID-only files require matching argv/cwd and a launch-time check. A
    stopped leader's remaining children can be targeted only with a previously
    verified cancellation record; that record also guards against PID reuse.
    """
    path = Path(record.working_directory) / "process.json"
    try:
        data = json.loads(path.read_text())
        pid = int(data["pid"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Cannot read the local worker identity at {path}: {error}") from error
    if pid <= 1 or pid == os.getpid() or data.get("record_id") not in (None, record.id):
        raise ValueError("The saved local process identity does not match this job.")
    host = data.get("hostname")
    if host and host != socket.gethostname():
        raise ValueError(
            f"This local job was launched on {host}. Open the UI on that host to stop it."
        )
    if int(data.get("pgid", pid)) != pid:
        raise ValueError("This local worker does not own a separate process group.")
    members = _members(pid)
    if not members:
        if not host and not record.provenance.get("local_stop"):
            raise ValueError(
                "This older local run has no recorded launch host and no matching process here. Its stopped state cannot be confirmed from this host."
            )
        return data, []
    pending = record.provenance.get("local_stop", {})
    if pending:
        if pending.get("hostname") != socket.gethostname() or pending.get("pgid") != pid:
            raise ValueError(
                "The local cancellation record belongs to a different host or process group."
            )
        if not any(_same_process(p, ident) for p in members for ident in pending["members"]):
            raise ValueError(
                "The old worker identity cannot be verified; no signal was sent to this process group."
            )
        return data, members
    leader = next((p for p in members if p.pid == pid), None)
    if leader is None:
        raise ValueError(
            "The worker leader has exited but its process group still exists. Its identity cannot be verified safely."
        )
    created = leader.create_time()
    if data.get("create_time") is not None:
        if abs(created - float(data["create_time"])) >= 0.01:
            raise ValueError(
                "The saved PID has been reused by another process; no signal was sent."
            )
    else:
        try:
            started = datetime.fromisoformat(data["started"])
            if started.tzinfo is None or not -10 <= created - started.timestamp() <= 2:
                raise ValueError()
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError("Cannot verify the launch time of this older local worker.") from error
    if (
        leader.cmdline()[1:] != list(record.command)[1:]
        or Path(leader.cwd()).resolve() != Path(record.working_directory).resolve()
    ):
        raise ValueError(
            "The saved PID's command or working directory differs from this job; no signal was sent."
        )
    return data, members


def local_stop_changes(record):
    """Confirm cancellation only once the verified worker group has stopped."""
    pending = record.provenance.get("local_stop")
    if not pending:
        return None
    if pending.get("confirmed_at"):
        return {"application_state": "cancelled", "raw_scheduler_state": "LOCAL_CANCELLED"}
    try:
        _, members = local_target(record)
    except (ValueError, OSError, psutil.Error):
        return {"application_state": "cancelling"}
    if members:
        return {"application_state": "cancelling", "raw_scheduler_state": "LOCAL_CANCEL_REQUESTED"}
    finished = utc_now()
    return {
        "application_state": "cancelled",
        "raw_scheduler_state": "LOCAL_CANCELLED",
        "completed_at": finished,
        "provenance": {**record.provenance, "local_stop": {**pending, "confirmed_at": finished}},
    }


def stop_local_job(record, registry, *, force=False):
    """Request SIGTERM (or explicit SIGKILL) for this job's verified process group."""
    from morphofeatures.job_status import reconcile_job_states
    from morphofeatures.run_deletion import run_lock

    with run_lock(registry.path.parent.parent, record.run_id):
        record = registry.get(record.id)
        if record.slurm_job_id or record.application_state in REMOVED:
            raise ValueError("Select an active local job to stop.")
        data, members = local_target(record)
        previous = record.provenance.get("local_stop", {})
        if previous.get("confirmed_at"):
            return record
        requested_signal = signal.SIGKILL if force else signal.SIGTERM
        pending = {
            **previous,
            "requested_at": utc_now(),
            "hostname": socket.gethostname(),
            "pgid": data["pid"],
            "signal": requested_signal.name,
            "members": [{"pid": p.pid, "create_time": p.create_time()} for p in members],
        }
        if members:
            # Recheck immediately before signalling, including a reused leader PID.
            _, checked = local_target(record)
            if checked:
                try:
                    os.killpg(data["pid"], requested_signal)
                except ProcessLookupError:
                    pass
        registry.update(
            record.id,
            application_state="cancelling",
            completed_at=None,
            raw_scheduler_state="LOCAL_CANCEL_REQUESTED",
            provenance={**record.provenance, "local_stop": pending},
        )
        reconcile_job_states(registry, record_ids=(record.id,))
        return registry.get(record.id)
