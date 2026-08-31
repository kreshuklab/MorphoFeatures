"""Runtime-checked contracts for volumes, tables, and exported embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np


ZYX = ("z", "y", "x")


@dataclass(frozen=True)
class VolumeSpec:
    path: Path
    dataset: str
    resolution: Tuple[float, float, float]
    coordinate_order: Tuple[str, str, str] = ZYX
    unit: str = "micrometer"
    kind: str = "raw"

    def __post_init__(self) -> None:
        if tuple(self.coordinate_order) != ZYX:
            raise ValueError("Volumes must be described in z, y, x order")
        if len(self.resolution) != 3 or any(value <= 0 for value in self.resolution):
            raise ValueError("resolution must contain three positive z, y, x values")
        if self.kind not in {"raw", "cell_segmentation", "nucleus_segmentation"}:
            raise ValueError("Unsupported volume kind: {}".format(self.kind))


@dataclass(frozen=True)
class TableContract:
    name: str
    required_columns: Tuple[str, ...]

    def validate(self, columns: Iterable[str]) -> None:
        available = set(columns)
        missing = [column for column in self.required_columns if column not in available]
        if missing:
            raise ValueError("{} is missing columns: {}".format(self.name, ", ".join(missing)))


CELL_NUCLEUS_MAPPING = TableContract(
    "cell-to-nucleus mapping", ("cell_id", "nucleus_id")
)
BOUNDING_BOX_TABLE = TableContract(
    "bounding-box table",
    (
        "label_id",
        "bb_min_z",
        "bb_min_y",
        "bb_min_x",
        "bb_max_z",
        "bb_max_y",
        "bb_max_x",
    ),
)
METADATA_TABLE = TableContract("cell metadata", ("label_id",))
MOBIE_TABLE = TableContract("MoBIE table", ("label_id",))


@dataclass(frozen=True)
class EmbeddingTable:
    """Embedding rows with integer label ids separate from numeric features."""

    label_ids: np.ndarray
    features: np.ndarray

    def __post_init__(self) -> None:
        ids = np.asarray(self.label_ids)
        features = np.asarray(self.features)
        if ids.ndim != 1:
            raise ValueError("label_ids must be one-dimensional")
        if features.ndim != 2:
            raise ValueError("features must be a two-dimensional matrix")
        if len(ids) != len(features):
            raise ValueError("label_ids and features must have the same row count")
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("label_ids must use an integer dtype")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("label_ids must be unique")
        if not np.issubdtype(features.dtype, np.number):
            raise ValueError("features must be numeric")

    @classmethod
    def from_array(cls, array: np.ndarray, require_finite: bool = True) -> "EmbeddingTable":
        matrix = np.asarray(array)
        if matrix.ndim != 2 or matrix.shape[1] < 2:
            raise ValueError("Embedding arrays must have shape (n_cells, 1 + n_features)")
        raw_ids = matrix[:, 0]
        if not np.all(np.isfinite(raw_ids)) or not np.allclose(raw_ids, np.rint(raw_ids)):
            raise ValueError("Column 0 must contain finite integer-valued label_id values")
        features = np.asarray(matrix[:, 1:])
        if require_finite and not np.all(np.isfinite(features)):
            raise ValueError("Embedding features contain NaN or infinite values")
        return cls(np.rint(raw_ids).astype(np.int64), features)

    def as_array(self) -> np.ndarray:
        return np.column_stack((self.label_ids, self.features))

    def sorted(self) -> "EmbeddingTable":
        order = np.argsort(self.label_ids, kind="stable")
        return EmbeddingTable(self.label_ids[order], self.features[order])


def normalize_mapping_columns(columns: Sequence[str]) -> Mapping[str, str]:
    """Return aliases needed to normalize common legacy mapping headers."""
    aliases = {
        "label_id": "cell_id",
        "cell_label_id": "cell_id",
        "nucleus_label_id": "nucleus_id",
        "nucl_id": "nucleus_id",
    }
    return {column: aliases[column] for column in columns if column in aliases}
