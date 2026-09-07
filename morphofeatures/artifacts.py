"""Bounded artifact and log inspection based on explicit run metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

from morphofeatures.data.io import load_embeddings


@dataclass(frozen=True)
class EmbeddingSummary:
    path: str
    rows: int
    features: int
    minimum_label: int
    maximum_label: int
    finite: bool
    unique_labels: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def inspect_embedding(path: Path) -> EmbeddingSummary:
    table = load_embeddings(Path(path), mmap=True)
    if len(table.label_ids) == 0:
        raise ValueError("Embedding is empty")
    return EmbeddingSummary(
        str(Path(path)),
        len(table.label_ids),
        int(table.features.shape[1]),
        int(table.label_ids.min()),
        int(table.label_ids.max()),
        bool(np.all(np.isfinite(table.features))),
        len(np.unique(table.label_ids)) == len(table.label_ids),
    )


def inspect_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Checkpoint inspection requires Torch") from error
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        return {"path": str(path), "format": type(payload).__name__}
    model = payload.get("model", {})
    return {
        "path": str(path),
        "epoch": payload.get("epoch"),
        "step": payload.get("step"),
        "metrics": payload.get("metrics", {}),
        "config": payload.get("config", {}),
        "parameter_tensors": len(model) if isinstance(model, dict) else None,
    }


def tail_text(path: Path, *, max_bytes: int = 65_536, max_lines: int = 200) -> str:
    source = Path(path)
    if not source.exists():
        return ""
    size = source.stat().st_size
    with source.open("rb") as stream:
        stream.seek(max(0, size - max(1, int(max_bytes))))
        data = stream.read(max(1, int(max_bytes)))
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    prefix = "… bounded tail …\n" if size > max_bytes else ""
    return prefix + "\n".join(lines[-max(1, int(max_lines)) :])


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(destination))
    return destination


def read_run_metadata(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or "run_id" not in value or "artifacts" not in value:
        raise ValueError("Run metadata is missing run_id or artifacts")
    return value


def existing_artifacts(paths: Iterable[Optional[Path]]) -> Tuple[str, ...]:
    return tuple(str(Path(path)) for path in paths if path is not None and Path(path).exists())
