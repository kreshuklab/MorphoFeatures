"""Typed configuration objects used by pipeline modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TextureDataPaths:
    """Paths and constants required to open texture volumes and tables."""

    root_dir: Path
    raw_data: Path
    cell_segmentation_xml: Path
    nucleus_segmentation_xml: Path
    cell_to_nucleus_table: Path
    cell_table: Path
    nucleus_table: Path
    raw_dataset: str = "setup0/timepoint0/s3"
    cell_dataset: str = "setup0/timepoint0/s2"
    nucleus_dataset: str = "setup0/timepoint0/s0"
    resolution_um: tuple[float, float, float] = (0.025, 0.01, 0.01)
    high_resolution_shape: tuple[int, int, int] = (11416, 25916, 27499)


@dataclass(slots=True)
class LoaderSettings:
    """Generic PyTorch dataloader settings."""

    batch_size: int = 1
    shuffle: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_kwargs(self) -> dict[str, Any]:
        """Return settings as ``DataLoader`` keyword arguments."""

        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        kwargs.update(self.extra)
        return kwargs


@dataclass(slots=True)
class EmbeddingPaths:
    """Common paths for embedding workflows."""

    input_files: list[Path]
    output_file: Path | None = None
