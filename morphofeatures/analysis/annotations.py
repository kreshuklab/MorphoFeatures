"""Exact object-ID joins and explicit handling of missing biological labels."""

from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd


def read_id_table(path, id_column="label_id", *, require_unique=True):
    path = Path(path)
    frame = pd.read_csv(
        path, sep="\t" if path.suffix.lower() == ".tsv" else ",", dtype={id_column: "string"}
    )
    if id_column not in frame:
        raise ValueError(f"Table requires {id_column}")
    ids = []
    for value in frame[id_column]:
        try:
            number = Decimal(str(value))
            if (
                not number.is_finite()
                or number != number.to_integral_value()
                or not 0 <= number <= np.iinfo(np.int64).max
            ):
                raise ValueError
            ids.append(int(number))
        except (InvalidOperation, ValueError):
            raise ValueError(f"{id_column} must contain exact nonnegative int64 IDs") from None
    frame[id_column] = np.asarray(ids, dtype=np.int64)
    if require_unique and frame[id_column].duplicated().any():
        raise ValueError(f"Table requires unique {id_column} values")
    return frame


def read_annotations(path, settings=None):
    settings = settings or {}
    frame = read_id_table(path)
    column = settings.get("label_column", "auto")
    if column == "auto":
        column = next((c for c in ("cell_type", "label", "cell_label") if c in frame), None)
    if column not in frame:
        raise ValueError(f"Missing annotation column: {column}; choose label_column explicitly")
    labels = frame[column].astype("string").str.strip()
    missing = {
        str(v).strip().casefold()
        for v in settings.get(
            "unlabeled_values",
            ["", "unknown", "unlabeled", "unlabelled", "unassigned", "none", "nan"],
        )
    }
    frame["known_label"] = labels.mask(labels.str.casefold().isin(missing))
    return frame


def join_labels(ids, settings):
    frame = pd.DataFrame({"label_id": np.asarray(ids, dtype=np.int64)})
    frame["known_label"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    if settings.get("annotations"):
        annotations = read_annotations(settings["annotations"], settings)
        frame["known_label"] = (
            annotations.set_index("label_id")["known_label"].reindex(ids).to_numpy()
        )
    return frame
