"""Fast structural validation for artifacts distributed with the paper."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


EXPECTED_ARRAYS: Dict[str, Tuple[int, int]] = {
    "morphofeatures_all_cells.npy": (11382, 481),
    "morphocontextfeatures_all_cells_agglomerated.npy": (10391, 201),
    "manually_defined_features.npy": (11348, 141),
}


@dataclass(frozen=True)
class ArtifactValidation:
    path: str
    observed_shape: Tuple[int, ...]
    expected_shape: Tuple[int, ...]
    label_ids_integral: bool
    passed: bool

    def to_dict(self):
        return asdict(self)


def validate_bundled_artifacts(analysis_data: Path) -> Dict[str, ArtifactValidation]:
    results: Dict[str, ArtifactValidation] = {}
    for name, expected_shape in EXPECTED_ARRAYS.items():
        path = Path(analysis_data) / name
        if not path.exists():
            results[name] = ArtifactValidation(str(path), (), expected_shape, False, False)
            continue
        matrix = np.load(path, mmap_mode="r")
        ids = np.asarray(matrix[:, 0])
        integral = bool(np.all(np.isfinite(ids)) and np.allclose(ids, np.rint(ids)))
        observed = tuple(int(value) for value in matrix.shape)
        results[name] = ArtifactValidation(
            str(path), observed, expected_shape, integral, observed == expected_shape and integral
        )
    return results


def validate_mobie_tables(mobie_data: Path) -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    for path in sorted(Path(mobie_data).glob("*.tsv")):
        columns = pd.read_csv(path, sep="\t", nrows=0).columns
        results[path.name] = "label_id" in columns
    return results
