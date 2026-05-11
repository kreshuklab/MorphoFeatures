"""Dataset splitting utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def train_val_split(
    labels: Sequence[T],
    validation_fraction: float = 0.2,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split labels into train and validation arrays without mutating input.

    Args:
        labels: Ordered labels or IDs to split.
        validation_fraction: Fraction assigned to validation.
        seed: Optional random seed.

    Returns:
        A ``(train, validation)`` tuple.

    Raises:
        ValueError: If ``validation_fraction`` is outside ``[0, 1)``.
    """

    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in the range [0, 1).")

    labels_array = np.asarray(labels).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(labels_array)
    split_at = int(np.floor(len(labels_array) * validation_fraction))
    return labels_array[split_at:], labels_array[:split_at]
