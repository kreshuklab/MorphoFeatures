"""Loader construction for texture MorphoFeatures datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from morphofeatures.config.loading import load_yaml_config
from morphofeatures.data.splits import train_val_split
from morphofeatures.data.tables import load_cell_nucleus_tables, read_cell_to_nucleus_table
from morphofeatures.data.volumes import open_z5_dataset, resolve_bdv_data_path
from morphofeatures.texture.datasets import (
    RawAutoencoderContrastiveCellDataset,
    TextPatchContrastiveCellDataset,
)
from morphofeatures.texture.transforms import get_transforms


def collate_contrastive(batch: list[Any]) -> tuple[Any, Any]:
    """Flatten two-view contrastive examples into a single batch."""

    try:
        import torch
    except ImportError as exc:
        raise ImportError("Contrastive collation requires torch.") from exc

    inputs = torch.cat([item[0] for item in batch])
    targets = torch.cat([item[1] for item in batch])
    if len(batch[0]) == 3:
        targets2 = torch.cat([item[2] for item in batch])
        return inputs, [targets, targets2]
    return inputs, targets


def _legacy_paths(data_config: dict[str, Any]) -> dict[str, Any]:
    """Build original Platyneris paths from a compact legacy config."""

    root_dir = Path(data_config.get("root_dir", "/scratch/zinchenk/cell_match/data/platy_data"))
    version = data_config.get("version")
    if version is None:
        raise KeyError("Texture data config requires either explicit paths or a 'version'.")
    return {
        "raw_data": root_dir / "rawdata/sbem-6dpf-1-whole-raw.n5",
        "cell_segmentation_xml": root_dir / version / "images/local/sbem-6dpf-1-whole-segmented-cells.xml",
        "nucleus_segmentation_xml": root_dir / version / "images/local/sbem-6dpf-1-whole-segmented-nuclei.xml",
        "cell_to_nucleus_table": root_dir
        / version
        / "tables/sbem-6dpf-1-whole-segmented-cells/cells_to_nuclei.tsv",
        "cell_table": root_dir / version / "tables/sbem-6dpf-1-whole-segmented-cells/default.tsv",
        "nucleus_table": root_dir / version / "tables/sbem-6dpf-1-whole-segmented-nuclei/default.tsv",
    }


def _resolve_data_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve texture data paths from new or legacy config keys."""

    data_config = dict(config.get("data", config.get("data_config", {})))
    paths = dict(data_config.get("paths", {}))
    explicit_path_keys = {
        "raw_data",
        "cell_segmentation_xml",
        "nucleus_segmentation_xml",
        "cell_to_nucleus_table",
        "cell_table",
        "nucleus_table",
    }
    if not paths and explicit_path_keys.issubset(data_config):
        paths = {key: data_config[key] for key in explicit_path_keys}
    if not paths:
        paths = _legacy_paths(data_config)

    resolved = {**data_config, **paths}
    resolved.setdefault("raw_dataset", "setup0/timepoint0/s3")
    resolved.setdefault("cell_dataset", "setup0/timepoint0/s2")
    resolved.setdefault("nucleus_dataset", "setup0/timepoint0/s0")
    resolved.setdefault("resolution_um", (0.025, 0.01, 0.01))
    resolved.setdefault("high_resolution_shape", (11416, 25916, 27499))
    return resolved


class CellLoaders:
    """Factory for texture train, validation, and prediction dataloaders."""

    def __init__(self, configuration_file: str | Path | dict[str, Any]) -> None:
        """Load config, open volumes, and prepare dataset settings."""

        if isinstance(configuration_file, dict):
            self.config = configuration_file
        else:
            self.config = load_yaml_config(configuration_file)

        self.data_config = _resolve_data_config(self.config)
        self.raw_vol, self.cell_vol, self.nuclei_vol = self._open_volumes()
        self.nucleus_by_cell = read_cell_to_nucleus_table(self.data_config["cell_to_nucleus_table"])
        self.tables = load_cell_nucleus_tables(self.data_config["cell_table"], self.data_config["nucleus_table"])
        self.validation_fraction = self.data_config.get("split", self.data_config.get("validation_fraction", 0.2))
        self.seed = self.data_config.get("seed")
        self.other_kwargs: dict[str, Any] = dict(self.config.get("other", {}))

        if self.config.get("contrastive", False):
            self.dataset_class = RawAutoencoderContrastiveCellDataset
        elif self.config.get("texture_contrastive", False):
            self.dataset_class = TextPatchContrastiveCellDataset
            raw_level = self.data_config.get("raw_level")
            if raw_level is not None:
                self.raw_vol = open_z5_dataset(self.data_config["raw_data"], f"setup0/timepoint0/s{raw_level}")
                self.other_kwargs["cell_hr_vol"] = open_z5_dataset(
                    resolve_bdv_data_path(self.data_config["cell_segmentation_xml"]),
                    f"setup0/timepoint0/s{raw_level - 1}",
                )
        else:
            raise ValueError("Texture config must set 'contrastive' or 'texture_contrastive'.")

        self.transforms = get_transforms(self.config.get("transforms")) if self.config.get("transforms") else None
        self.transforms_sim = (
            get_transforms(self.config.get("transforms_sim")) if self.config.get("transforms_sim") else None
        )

    def _open_volumes(self) -> tuple[Any, Any, Any]:
        """Open raw, cell, and nucleus volumes from configured paths."""

        raw = open_z5_dataset(self.data_config["raw_data"], self.data_config["raw_dataset"])
        cell = open_z5_dataset(
            resolve_bdv_data_path(self.data_config["cell_segmentation_xml"]),
            self.data_config["cell_dataset"],
        )
        nuclei = open_z5_dataset(
            resolve_bdv_data_path(self.data_config["nucleus_segmentation_xml"]),
            self.data_config["nucleus_dataset"],
        )
        return raw, cell, nuclei

    def _dataset(self, indices: np.ndarray | None = None, predict: bool = False) -> Any:
        """Instantiate the configured texture dataset."""

        return self.dataset_class(
            self.tables,
            self.nucleus_by_cell,
            self.cell_vol,
            self.nuclei_vol,
            self.raw_vol,
            indices=indices,
            transforms=self.transforms,
            transforms_sim=self.transforms_sim,
            predict=predict,
            resolution_um=self.data_config["resolution_um"],
            high_resolution_shape=self.data_config["high_resolution_shape"],
            **self.other_kwargs,
        )

    def get_train_loaders(self) -> tuple[Any, Any]:
        """Build train and validation dataloaders."""

        try:
            from torch.utils.data.dataloader import DataLoader
        except ImportError as exc:
            raise ImportError("Texture dataloaders require torch.") from exc

        train_ids, val_ids = train_val_split(
            list(self.nucleus_by_cell.keys()),
            validation_fraction=float(self.validation_fraction),
            seed=self.seed,
        )
        train_loader = DataLoader(
            self._dataset(train_ids),
            collate_fn=collate_contrastive,
            **self.config.get("loader_config", self.config.get("loader", {})),
        )
        val_loader = DataLoader(
            self._dataset(val_ids),
            collate_fn=collate_contrastive,
            **self.config.get("val_loader_config", self.config.get("validation_loader", {})),
        )
        return train_loader, val_loader

    def get_predict_loaders(self) -> Any:
        """Build the prediction dataloader."""

        try:
            from torch.utils.data.dataloader import DataLoader
        except ImportError as exc:
            raise ImportError("Texture dataloaders require torch.") from exc

        return DataLoader(
            self._dataset(predict=True),
            **self.config.get("pred_loader_config", self.config.get("prediction_loader", {})),
        )
