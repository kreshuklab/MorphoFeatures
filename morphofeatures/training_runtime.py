"""Torch runtime, checkpoint, and optional experiment logging utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def resolve_device(requested: str = "auto"):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA runtime is available")
    return device


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def portable_state_dict(model) -> Dict[str, Any]:
    return unwrap_model(model).state_dict()


def normalize_state_dict_keys(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return dict(state_dict)


def save_checkpoint(
    path: Path,
    model,
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
    step: int = 0,
    config: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, float]] = None,
) -> Path:
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "step": int(step),
        "model": portable_state_dict(model),
        "config": dict(config or {}),
        "metrics": dict(metrics or {}),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(str(temporary), str(destination))
    return destination


def load_checkpoint(path: Path, model, device="cpu", optimizer=None, scheduler=None, strict=True):
    import torch

    payload = torch.load(Path(path), map_location=device)
    state_dict = payload.get("model", payload)
    unwrap_model(model).load_state_dict(normalize_state_dict_keys(state_dict), strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return payload


class ExperimentLogger:
    """No-op by default; initializes WandB only when explicitly enabled."""

    def __init__(self, enabled: bool = False, project: str = "MorphoFeatures", **kwargs):
        self.run = None
        if enabled:
            try:
                import wandb
            except ImportError as error:
                raise RuntimeError("WandB logging requires morphofeatures[wandb]") from error
            self.run = wandb.init(project=project, **kwargs)

    def log(self, values: Mapping[str, Any], step: Optional[int] = None) -> None:
        if self.run is not None:
            self.run.log(dict(values), step=step)

    def close(self) -> None:
        if self.run is not None:
            self.run.finish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
