"""Append-only structured training events readable during interrupted writes."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


METRICS_ENV = "MORPHOFEATURES_METRICS_PATH"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configured_metrics_path(config: Mapping[str, Any], fallback: Path) -> Path:
    environment = os.environ.get(METRICS_ENV)
    if environment:
        return Path(environment)
    runtime = config.get("runtime", {}) if isinstance(config, Mapping) else {}
    return Path(runtime.get("metrics_path", fallback))


class MetricWriter:
    def __init__(self, path: Path):
        self.path = Path(path)

    def write(self, event: str, **values: Any) -> None:
        payload = {"timestamp": utc_now(), "event": str(event)}
        payload.update({key: value for key, value in values.items() if value is not None})
        encoded = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(str(self.path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(descriptor, encoded)
        finally:
            os.close(descriptor)


def read_metric_events(path: Path) -> List[Dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    raw = source.read_bytes()
    lines = raw.splitlines(keepends=True)
    events = []
    for index, encoded in enumerate(lines):
        complete = encoded.endswith((b"\n", b"\r"))
        try:
            event = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if index == len(lines) - 1 and not complete:
                break
            raise ValueError(
                "Invalid metrics JSONL at line {} in {}".format(index + 1, source)
            ) from error
        if not isinstance(event, dict) or "event" not in event or "timestamp" not in event:
            raise ValueError("Metrics line {} is missing event or timestamp".format(index + 1))
        events.append(event)
    return events


def metric_series(events: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        dict(event)
        for event in events
        if any(key in event for key in ("train_loss", "validation_loss", "learning_rate"))
    ]
