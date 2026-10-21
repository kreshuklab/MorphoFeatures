"""Explicit workspace-local relocation records; original reports are immutable."""

from contextvars import ContextVar
from pathlib import Path

_locations = ContextVar("artifact_locations", default=None)


def configure_locations(root):
    import json

    _locations.set({})
    file = Path(root) / ".morphofeatures" / "artifact_locations.json"
    try:
        if file.stat().st_size > 1024**2:
            raise ValueError("Artifact relocation index is too large")
        values = json.loads(file.read_text())
        locations = values.get("locations", {})
        if not isinstance(locations, dict) or any(not isinstance(v, dict) or not isinstance(v.get("path"), str) for v in locations.values()):
            raise ValueError("Invalid relocation index")
    except FileNotFoundError:
        locations = {}
    _locations.set(locations)


def artifact_path(value):
    path = Path(value).expanduser()
    entry = (_locations.get() or {}).get(str(path.resolve()))
    return Path(entry["path"]) if entry else path


def relocated_metadata(value):
    if isinstance(value, dict):
        return {k: relocated_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [relocated_metadata(v) for v in value]
    if isinstance(value, str) and value in (_locations.get() or {}):
        return _locations.get()[value]["path"]
    return value


def record_location(root, original, replacement, *, reason):
    """Record the user's identity assertion; it never overrides recipe validation."""
    import fcntl
    import json

    from morphofeatures.artifact_availability import check_selected
    from morphofeatures.artifacts import write_json_atomic
    from morphofeatures.metrics import utc_now

    original, replacement = Path(original).resolve(), Path(replacement).expanduser().resolve()
    if original == replacement:
        raise ValueError("Choose the artifact's new location.")
    status = check_selected(replacement)
    if not status.ready:
        raise ValueError(status.reason or "The replacement is not usable.")
    if original.suffix.lower() != replacement.suffix.lower():
        raise ValueError("Relinking preserves the saved format. Import a converted file as a new input instead.")
    file = Path(root) / ".morphofeatures" / "artifact_locations.json"
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        values = json.loads(file.read_text()) if file.exists() else {"schema": "morphofeatures.locations.v1", "locations": {}, "history": []}
        entry = {"original": str(original), "path": str(replacement), "verification": "user asserted same artifact",
                 "reason": reason, "recorded_at": utc_now(), "kind": status.kind}
        values["locations"][str(original)] = entry
        values["history"].append(entry)
        write_json_atomic(file, values)
    configure_locations(root)
