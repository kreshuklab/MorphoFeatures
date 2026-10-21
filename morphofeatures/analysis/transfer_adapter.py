"""Small shared neural adapters on frozen embeddings, with optional Deep CORAL loss.

Training needs PyTorch; exported numeric models support NumPy-only inference.
Target annotations are deliberately absent from the training API.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from morphofeatures.artifacts import write_json_atomic

ADAPTER_DEFAULTS = {
    "hidden_dim": 128,
    "latent_dim": 64,
    "epochs": 100,
    "batch_size": 128,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "coral_weight": 1.0,
    "standardize": True,
    "device": "cpu",
}


def validate_adapter_settings(settings):
    from morphofeatures.workflow_validation import FieldError

    if not isinstance(settings, dict):
        raise FieldError("Adapter settings must be a mapping", "adapter")
    options = {**ADAPTER_DEFAULTS, **settings}
    for name, minimum in (("hidden_dim", 1), ("latent_dim", 2), ("epochs", 1), ("batch_size", 2)):
        value = options[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise FieldError(f"Use an integer of at least {minimum}", "adapter." + name)
    for name in ("learning_rate", "weight_decay", "coral_weight"):
        try:
            value = float(options[name])
            valid = np.isfinite(value) and value >= 0
        except (ValueError, TypeError):
            valid = False
        if not valid or (name == "learning_rate" and value == 0):
            raise FieldError(
                "Use a finite positive learning rate or nonnegative loss/decay weight",
                "adapter." + name,
            )
    if options["device"] not in {"cpu", "cuda", "auto"}:
        raise FieldError("Choose cpu, cuda or auto", "adapter.device")
    if not isinstance(options["standardize"], bool):
        raise FieldError("Choose whether to standardize adapter inputs", "adapter.standardize")
    return options


def coral_loss(source, target):
    """Sun & Saenko (2016): ||Cs - Ct||²_F / (4 d²), unbiased batch covariances."""
    if min(len(source), len(target)) < 2 or source.shape[1] != target.shape[1]:
        raise ValueError("CORAL loss needs two objects per domain and equal feature dimensions")
    source = source - source.mean(dim=0)
    target = target - target.mean(dim=0)
    difference = source.T @ source / (len(source) - 1) - target.T @ target / (len(target) - 1)
    return difference.square().sum() / (4 * source.shape[1] ** 2)


class NumericAdapter:
    """Numeric export of Linear → ReLU → Linear latent features → linear softmax head."""

    def __init__(self, arrays, metadata):
        self.arrays = arrays
        self.metadata = metadata
        self.classes_ = np.asarray(metadata["classes"])

    def transform(self, values):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.arrays["input_mean"]):
            raise ValueError("Adapter input feature dimensions do not match the saved model")
        if not np.isfinite(values).all():
            raise ValueError("Adapter inputs must be finite")
        blocks = []
        for start in range(0, len(values), 1024):
            x = (values[start : start + 1024] - self.arrays["input_mean"]) / self.arrays[
                "input_scale"
            ]
            x = np.maximum(x @ self.arrays["hidden_weight"].T + self.arrays["hidden_bias"], 0)
            blocks.append(x @ self.arrays["latent_weight"].T + self.arrays["latent_bias"])
        return (
            np.concatenate(blocks)
            if blocks
            else np.empty((0, self.arrays["latent_weight"].shape[0]), np.float32)
        )

    def probabilities_from_latent(self, features):
        logits = features @ self.arrays["head_weight"].T + self.arrays["head_bias"]
        logits -= logits.max(axis=1, keepdims=True)
        scores = np.exp(logits)
        return scores / scores.sum(axis=1, keepdims=True)

    def predict_proba(self, values):
        return self.probabilities_from_latent(self.transform(values))

    def save(self, destination, history):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / "adapter_parameters.npz", **self.arrays)
        write_json_atomic(destination / "adapter.json", self.metadata)
        pd.DataFrame(history).to_csv(destination / "training_history.tsv", sep="\t", index=False)

    @classmethod
    def load(cls, path):
        path = Path(path)
        metadata = json.loads(path.read_text())
        if metadata.get("schema") != "morphofeatures.transfer_adapter.v1":
            raise ValueError("Select an adapter.json created by Transfer labels")
        with np.load(path.with_name("adapter_parameters.npz"), allow_pickle=False) as arrays:
            return cls(dict(arrays), metadata)


def fit_adapter(source, labels, source_pool, target, settings, *, seed, with_coral):
    """Use labeled source objects for classification and unlabeled pools for alignment.

    Both conditions share source batches and initialization. Separate random streams
    keep target sampling from changing the supervised training schedule.
    """
    import torch
    from torch import nn

    options = validate_adapter_settings(settings)
    weight = float(options["coral_weight"]) if with_coral else 0.0
    classes, y = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    if len(classes) < 2:
        raise ValueError("Adapter training needs at least two reference classes")
    source = np.asarray(source, dtype=np.float32)
    source_pool = np.asarray(source_pool, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if weight and min(len(source_pool), len(target)) < 2:
        raise ValueError("Deep CORAL needs at least two reference and target objects")
    if (
        not np.isfinite(source).all()
        or not np.isfinite(source_pool).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("Adapter inputs must be finite")
    device = options["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "Adapter requested CUDA, but no CUDA device is available; select CPU or request a GPU"
        )
    mean = source.mean(0) if options["standardize"] else np.zeros(source.shape[1], np.float32)
    scale = source.std(0) if options["standardize"] else np.ones(source.shape[1], np.float32)
    scale = np.where(scale > 1e-6, scale, 1).astype(np.float32)
    # Keep complete cohorts on the CPU; only bounded mini-batches move to the GPU.
    source, source_pool, target = [(v - mean) / scale for v in (source, source_pool, target)]
    rng_labels, rng_pool, rng_target = [
        np.random.default_rng(seed + offset) for offset in (0, 1, 2)
    ]
    history = []
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        # Layers initialize on CPU; avoid changing unrelated CUDA RNG streams.
        torch.random.default_generator.manual_seed(int(seed))
        hidden = nn.Linear(source.shape[1], options["hidden_dim"])
        latent = nn.Linear(options["hidden_dim"], options["latent_dim"])
        head = nn.Linear(options["latent_dim"], len(classes))
        encoder = nn.Sequential(hidden, nn.ReLU(), latent).to(device)
        head = head.to(device)
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(head.parameters()),
            lr=float(options["learning_rate"]),
            weight_decay=float(options["weight_decay"]),
        )
        class_weights = len(y) / (len(classes) * np.bincount(y))
        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
        )
        batch_size = options["batch_size"]

        def tensor(array):
            return torch.as_tensor(array, dtype=torch.float32, device=device)

        for epoch in range(options["epochs"]):
            order = rng_labels.permutation(len(y))
            totals = np.zeros(3)
            for start in range(0, len(y), batch_size):
                rows = order[start : start + batch_size]
                optimizer.zero_grad()
                supervised = loss_fn(
                    head(encoder(tensor(source[rows]))), torch.tensor(y[rows], device=device)
                )
                alignment = supervised.new_zeros(())
                if weight:
                    sr = rng_pool.choice(
                        len(source_pool), min(batch_size, len(source_pool)), replace=False
                    )
                    tr = rng_target.choice(len(target), min(batch_size, len(target)), replace=False)
                    alignment = coral_loss(
                        encoder(tensor(source_pool[sr])), encoder(tensor(target[tr]))
                    )
                loss = supervised + weight * alignment
                if not torch.isfinite(loss):
                    raise ValueError(
                        "Adapter loss became nonfinite; reduce learning rate or CORAL weight"
                    )
                loss.backward()
                optimizer.step()
                totals += np.array([supervised.item(), alignment.item(), loss.item()]) * len(rows)
            history.append(
                dict(
                    epoch=epoch + 1,
                    classification_loss=totals[0] / len(y),
                    coral_loss=totals[1] / len(y),
                    total_loss=totals[2] / len(y),
                )
            )
        arrays = {"input_mean": mean, "input_scale": scale}
        for name, layer in (("hidden", hidden), ("latent", latent), ("head", head)):
            arrays[name + "_weight"] = layer.weight.detach().cpu().numpy().copy()
            arrays[name + "_bias"] = layer.bias.detach().cpu().numpy().copy()
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Adapter weights became nonfinite; reduce learning rate or CORAL weight")
    return NumericAdapter(
        arrays,
        {
            "schema": "morphofeatures.transfer_adapter.v1",
            "classes": classes.tolist(),
            "settings": options,
            "seed": int(seed),
            "effective_coral_weight": weight,
            "target_features_used": bool(weight),
            "target_labels_used": False,
            "device": device,
            "torch_version": str(torch.__version__),
            "architecture": "Linear-ReLU-Linear latent features; linear softmax classifier",
            "input_normalization": "l2_then_reference_training_standardization"
            if options["standardize"]
            else "l2",
            "loss": "class-balanced cross entropy + weight * ||Cs-Ct||_F^2 / (4*d^2)",
        },
    ), history
