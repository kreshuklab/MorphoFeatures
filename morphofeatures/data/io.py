"""I/O helpers preserving the legacy label_id-in-column-zero convention."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .contracts import EmbeddingTable


def load_embeddings(path: Path, mmap: bool = False) -> EmbeddingTable:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        matrix = np.load(source, mmap_mode="r" if mmap else None)
    elif suffix in {".tsv", ".csv"}:
        separator = "\t" if suffix == ".tsv" else ","
        frame = pd.read_csv(source, sep=separator)
        if "label_id" not in frame.columns:
            raise ValueError("{} has no label_id column".format(source))
        feature_columns = [column for column in frame.columns if column != "label_id"]
        matrix = np.column_stack((frame["label_id"].to_numpy(), frame[feature_columns].to_numpy()))
    elif suffix in {".np", ".txt"}:
        matrix = np.loadtxt(source)
    else:
        raise ValueError("Unsupported embedding format: {}".format(source.suffix))
    return EmbeddingTable.from_array(matrix)


def merge_embeddings(paths: Sequence[Path], standardize: bool = False) -> EmbeddingTable:
    if not paths:
        raise ValueError("At least one embedding path is required")
    tables = [load_embeddings(path).sorted() for path in paths]
    reference_ids = tables[0].label_ids
    for table in tables[1:]:
        if not np.array_equal(reference_ids, table.label_ids):
            raise ValueError("Embedding files do not contain identical sorted label ids")
    features = np.column_stack([table.features for table in tables])
    if standardize:
        from sklearn.preprocessing import StandardScaler

        features = StandardScaler().fit_transform(features)
    return EmbeddingTable(reference_ids, features)


def export_embeddings(
    path: Path,
    label_ids: Iterable[int],
    features: np.ndarray,
    feature_names: Iterable[str] = (),
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = EmbeddingTable(np.asarray(list(label_ids), dtype=np.int64), np.asarray(features)).sorted()
    if destination.suffix.lower() == ".npy":
        np.save(destination, table.as_array())
    elif destination.suffix.lower() in {".tsv", ".csv"}:
        names = list(feature_names)
        if names and len(names) != table.features.shape[1]:
            raise ValueError("feature_names length does not match the feature matrix")
        if not names:
            names = [str(index) for index in range(table.features.shape[1])]
        frame = pd.DataFrame(table.features, columns=names)
        frame.insert(0, "label_id", table.label_ids)
        frame.to_csv(destination, sep="\t" if destination.suffix == ".tsv" else ",", index=False)
    else:
        raise ValueError("Export path must end in .npy, .tsv, or .csv")
    return destination
