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
    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            table = EmbeddingTable(archive["label_ids"], archive["features"])
        if not np.isfinite(table.features).all():
            raise ValueError("Embedding features contain NaN or infinite values")
        return table
    if suffix == ".npy":
        matrix = np.load(source, mmap_mode="r" if mmap else None)
    elif suffix in {".tsv", ".csv"}:
        separator = "\t" if suffix == ".tsv" else ","
        frame = pd.read_csv(source, sep=separator)
        if "label_id" not in frame.columns:
            raise ValueError("{} has no label_id column".format(source))
        feature_columns = [column for column in frame.columns if column != "label_id"]
        ids = frame["label_id"].to_numpy()
        if not np.issubdtype(ids.dtype, np.integer):
            if not np.isfinite(ids).all() or not np.equal(ids, np.rint(ids)).all() or np.any(np.abs(ids) > 2**53):
                raise ValueError("label_id requires exact integers; store large IDs as integer text")
            ids = ids.astype(np.int64)
        features = frame[feature_columns].to_numpy()
        if not np.isfinite(features).all():
            raise ValueError("Embedding features contain NaN or infinite values")
        return EmbeddingTable(ids, features)
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
    if destination.suffix.lower() == ".npz":
        np.savez_compressed(destination, label_ids=table.label_ids, features=table.features)
    elif destination.suffix.lower() == ".npy":
        if np.any(table.label_ids > 2**53) or np.any(table.label_ids < -(2**53)):
            raise ValueError("Label-first NPY cannot preserve IDs above 2**53; export .npz or .tsv")
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
        raise ValueError("Export path must end in .npz, .npy, .tsv, or .csv")
    return destination
