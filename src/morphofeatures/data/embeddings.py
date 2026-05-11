"""Read, write, and align MorphoFeatures embedding tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(slots=True)
class EmbeddingTable:
    """A cell embedding matrix aligned to integer label IDs."""

    ids: np.ndarray
    features: np.ndarray

    def __post_init__(self) -> None:
        """Validate shape compatibility after dataclass construction."""

        self.ids = np.asarray(self.ids).astype(int)
        self.features = np.asarray(self.features, dtype=float)
        if self.features.ndim != 2:
            raise ValueError("features must be a 2D matrix.")
        if self.ids.ndim != 1:
            raise ValueError("ids must be a 1D array.")
        if self.ids.shape[0] != self.features.shape[0]:
            raise ValueError("ids and features must have the same number of rows.")

    def as_matrix(self) -> np.ndarray:
        """Return the standard ``label_id + features`` matrix format."""

        return np.column_stack((self.ids, self.features))


def reorder_by_id(features: np.ndarray, ids: np.ndarray) -> EmbeddingTable:
    """Sort an embedding matrix by ascending label ID.

    Args:
        features: Feature matrix.
        ids: Label IDs matching rows in ``features``.

    Returns:
        A sorted embedding table.
    """

    order = np.argsort(np.asarray(ids).astype(int))
    return EmbeddingTable(ids=np.asarray(ids)[order], features=np.asarray(features)[order])


def _read_h5_embedding(path: Path) -> EmbeddingTable:
    """Read embeddings from an HDF5 file with ``embed`` and ``label_ids`` datasets."""

    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Reading .h5 embeddings requires h5py.") from exc

    with h5py.File(path, "r") as handle:
        return EmbeddingTable(ids=handle["label_ids"][:], features=handle["embed"][:])


def read_embedding_table(path: str | Path) -> EmbeddingTable:
    """Read an embedding file in npy, text, TSV, or HDF5 format.

    Args:
        path: Embedding file path. The first column must contain ``label_id``
            for text, ``.np``, ``.npy``, and ``.tsv`` inputs.

    Returns:
        A sorted embedding table.
    """

    embedding_path = Path(path)
    suffix = embedding_path.suffix.lower()

    if suffix == ".h5":
        table = _read_h5_embedding(embedding_path)
    elif suffix == ".npy":
        matrix = np.load(embedding_path)
        table = EmbeddingTable(ids=matrix[:, 0], features=matrix[:, 1:])
    elif suffix == ".tsv":
        frame = pd.read_csv(embedding_path, sep="\t")
        if "label_id" in frame.columns:
            ids = frame["label_id"].to_numpy()
            features = frame.drop(columns=["label_id"]).to_numpy(dtype=float)
        else:
            matrix = frame.to_numpy(dtype=float)
            ids = matrix[:, 0]
            features = matrix[:, 1:]
        table = EmbeddingTable(ids=ids, features=features)
    else:
        matrix = np.loadtxt(embedding_path)
        table = EmbeddingTable(ids=matrix[:, 0], features=matrix[:, 1:])

    return reorder_by_id(table.features, table.ids)


def merge_embedding_tables(paths: Iterable[str | Path], scale: bool = False) -> EmbeddingTable:
    """Merge multiple embedding tables by column after validating IDs.

    Args:
        paths: Embedding files to merge.
        scale: Whether to standardize each merged feature column.

    Returns:
        A single embedding table with horizontally concatenated features.

    Raises:
        ValueError: If no files are supplied or label IDs do not match.
    """

    tables = [read_embedding_table(path) for path in paths]
    if not tables:
        raise ValueError("At least one embedding file is required.")

    base_ids = tables[0].ids
    for table in tables[1:]:
        if not np.array_equal(base_ids, table.ids):
            raise ValueError("Embedding files must contain identical sorted label IDs.")

    features = np.column_stack([table.features for table in tables])
    if scale:
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        features = (features - mean) / std
    return EmbeddingTable(ids=base_ids, features=features)


def write_embedding_table(table: EmbeddingTable, path: str | Path, delimiter: str = "\t") -> None:
    """Write an embedding table using the standard first-column ID format.

    Args:
        table: Embedding table to write.
        path: Output file path.
        delimiter: Text delimiter used by ``numpy.savetxt``.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, table.as_matrix(), delimiter=delimiter)
