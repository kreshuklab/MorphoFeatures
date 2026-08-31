"""Small deterministic 3D fixture used by tests and examples."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from .io import export_embeddings


@dataclass(frozen=True)
class SyntheticDataset:
    raw: np.ndarray
    cells: np.ndarray
    nuclei: np.ndarray
    label_ids: np.ndarray
    nucleus_ids: np.ndarray
    embeddings: np.ndarray


def make_synthetic_dataset(seed: int = 42, shape: Tuple[int, int, int] = (24, 32, 32)) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.1, 0.02, size=shape).astype(np.float32)
    cells = np.zeros(shape, dtype=np.uint16)
    nuclei = np.zeros(shape, dtype=np.uint16)
    boxes = [
        (slice(2, 11), slice(2, 14), slice(2, 14)),
        (slice(2, 11), slice(18, 30), slice(18, 30)),
        (slice(13, 22), slice(2, 14), slice(18, 30)),
        (slice(13, 22), slice(18, 30), slice(2, 14)),
    ]
    for index, box in enumerate(boxes, start=1):
        cells[box] = index
        z, y, x = box
        nucleus_id = index + 100
        nucleus_box = (
            slice(z.start + 2, z.stop - 2),
            slice(y.start + 3, y.stop - 3),
            slice(x.start + 3, x.stop - 3),
        )
        nuclei[nucleus_box] = nucleus_id
        raw[box] += np.float32(index * 0.12)
        raw[nucleus_box] += np.float32(0.2)
    ids = np.arange(1, 5, dtype=np.int64)
    nucleus_ids = ids + 100
    embeddings = rng.normal(size=(4, 8)).astype(np.float32)
    embeddings[:, 0] += ids
    return SyntheticDataset(raw, cells, nuclei, ids, nucleus_ids, embeddings)


def save_synthetic_dataset(output_dir: Path, seed: int = 42) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dataset = make_synthetic_dataset(seed=seed)
    np.save(destination / "raw.npy", dataset.raw)
    np.save(destination / "cells.npy", dataset.cells)
    np.save(destination / "nuclei.npy", dataset.nuclei)
    export_embeddings(destination / "embeddings.npy", dataset.label_ids, dataset.embeddings)
    with (destination / "cell_to_nucleus.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["cell_id", "nucleus_id"])
        writer.writerows(zip(dataset.label_ids, dataset.nucleus_ids))
    with (destination / "metadata.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["label_id", "cell_type"])
        writer.writerows(zip(dataset.label_ids, ["A", "A", "B", "B"]))
    return destination
