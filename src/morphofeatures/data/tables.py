"""Tabular data helpers for cell and nucleus metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_tsv(path: str | Path) -> pd.DataFrame:
    """Read a tab-separated table.

    Args:
        path: Table path.

    Returns:
        Parsed pandas DataFrame.
    """

    return pd.read_csv(path, sep="\t")


def read_cell_to_nucleus_table(path: str | Path) -> dict[int, int]:
    """Read a two-column cell-to-nucleus mapping table.

    Args:
        path: TSV-like file whose first row is a header and whose columns are
            cell label ID and nucleus label ID.

    Returns:
        Mapping from cell ID to nucleus ID, excluding zero nucleus IDs.
    """

    values = np.loadtxt(path, skiprows=1)
    return {int(cell_id): int(nucleus_id) for cell_id, nucleus_id in values if nucleus_id != 0}


def load_cell_nucleus_tables(cell_table_path: str | Path, nucleus_table_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load paired cell and nucleus metadata tables.

    Args:
        cell_table_path: Cell metadata table.
        nucleus_table_path: Nucleus metadata table.

    Returns:
        ``(cell_table, nucleus_table)``.
    """

    return read_tsv(cell_table_path), read_tsv(nucleus_table_path)
