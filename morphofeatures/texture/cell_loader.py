"""Config-driven texture data loading without private paths or Inferno transforms."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.ndimage import gaussian_filter, map_coordinates
from torch.utils.data import DataLoader

from morphofeatures.texture.cell_dset import RawAEContrCellDataset, TextPatchContrCellDataset


def get_train_val_split(labels, split=0.2, r_seed=42):
    labels = np.asarray(labels, dtype=np.int64)
    labels = np.random.default_rng(r_seed).permutation(labels)
    spl = int(np.floor(len(labels)*split))
    return labels[spl:], labels[:spl]


def _size(config):
    if isinstance(config, (list, tuple)):
        return tuple(int(value) for value in config)
    for key in ('shape', 'size', 'target_size', 'output_shape'):
        if key in config:
            return tuple(int(value) for value in config[key])
    raise ValueError('Crop transform requires shape, size, target_size, or output_shape')


def _crop_or_pad(array, target, random_crop=False):
    padding = [(max(0, size - current) // 2,
                max(0, size - current) - max(0, size - current) // 2)
               for current, size in zip(array.shape, target)]
    array = np.pad(array, padding, mode='constant')
    starts = []
    for current, size in zip(array.shape, target):
        maximum = current - size
        starts.append(np.random.randint(0, maximum + 1) if random_crop and maximum else maximum // 2)
    return array[tuple(slice(start, start + size) for start, size in zip(starts, target))]


class VolumeTransform:
    def __init__(self, config=None):
        self.config = config or {}

    def __call__(self, data):
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        array = np.asarray(data)
        if self.config.get('crop_pad_to_size'):
            array = _crop_or_pad(array, _size(self.config['crop_pad_to_size']))
        if self.config.get('random_crop'):
            array = _crop_or_pad(array, _size(self.config['random_crop']), random_crop=True)
        if self.config.get('rotate90'):
            axes = [(0, 1), (0, 2), (1, 2)][np.random.randint(0, 3)]
            array = np.rot90(array, k=np.random.randint(0, 4), axes=axes)
        if self.config.get('elastic_transform'):
            elastic = self.config['elastic_transform']
            alpha, sigma = float(elastic.get('alpha', 1.0)), float(elastic.get('sigma', 4.0))
            displacements = [gaussian_filter(np.random.normal(size=array.shape), sigma) * alpha
                             for _ in range(3)]
            coordinates = np.meshgrid(*[np.arange(size) for size in array.shape], indexing='ij')
            array = map_coordinates(array, [axis + delta for axis, delta in zip(coordinates, displacements)],
                                    order=int(elastic.get('order', 3)), mode='reflect')
        if self.config.get('normalize_range'):
            normalize = self.config['normalize_range'] or {}
            source_min = float(normalize.get('min', np.min(array)))
            source_max = float(normalize.get('max', np.max(array)))
            array = (array - source_min) / max(source_max - source_min, np.finfo(np.float32).eps)
        array = np.ascontiguousarray(array, dtype=np.float32)
        tensor = torch.from_numpy(array)
        return tensor.unsqueeze(0) if tensor.ndim == 3 else tensor


def get_transforms(transform_config):
    return VolumeTransform(transform_config)


def collate_contrastive(batch):
    inputs = torch.cat([i[0] for i in batch])
    targets = torch.cat([i[1] for i in batch])
    if len(batch[0]) == 3:
        targets2 = torch.cat([i[2] for i in batch])
        targets = [targets, targets2]
    return inputs, targets


class CellLoaders(object):
    def __init__(self, configuration_file):
        config_path = Path(configuration_file).resolve()
        with config_path.open('r', encoding='utf-8') as stream:
            self.config = yaml.safe_load(stream) or {}
        data_config = self.config.get('data_config')
        configured_root = data_config.get('data_root') or os.environ.get('MORPHOFEATURES_DATA_ROOT')
        if not configured_root:
            raise ValueError('Set data_config.data_root or MORPHOFEATURES_DATA_ROOT')
        root_path = Path(configured_root).expanduser()
        self.PATH = (root_path if root_path.is_absolute() else config_path.parent / root_path).resolve()
        version = data_config.get("version")
        base = self.PATH / version if version else self.PATH

        def configured(name, default):
            value = Path(data_config.get(name, default))
            return value if value.is_absolute() else base / value

        raw_data = configured('raw_volume', '../rawdata/sbem-6dpf-1-whole-raw.n5')
        cell_segm = configured('cell_segmentation', 'images/local/sbem-6dpf-1-whole-segmented-cells.xml')
        nucl_segm = configured('nucleus_segmentation', 'images/local/sbem-6dpf-1-whole-segmented-nuclei.xml')
        cell_to_nucl = configured('cell_to_nucleus', 'tables/sbem-6dpf-1-whole-segmented-cells/cells_to_nuclei.tsv')
        cell_default = configured('cell_table', 'tables/sbem-6dpf-1-whole-segmented-cells/default.tsv')
        nucl_default = configured('nucleus_table', 'tables/sbem-6dpf-1-whole-segmented-nuclei/default.tsv')

        raw_file = self._open_container(raw_data)
        cell_file = self._open_container(self._resolve_bdv(cell_segm))
        nucl_file = self._open_container(self._resolve_bdv(nucl_segm))
        self.raw_vol = raw_file[data_config.get('raw_dataset', 'setup0/timepoint0/s3')]
        self.cell_vol = cell_file[data_config.get('cell_dataset', 'setup0/timepoint0/s2')]
        self.nuclei_vol = nucl_file[data_config.get('nucleus_dataset', 'setup0/timepoint0/s0')]

        mapping = pd.read_csv(cell_to_nucl, sep='\t')
        mapping_columns = data_config.get('mapping_columns', list(mapping.columns[:2]))
        self.nucl_dict = {int(cell): int(nucleus) for cell, nucleus in
                          mapping[mapping_columns].itertuples(index=False, name=None) if nucleus != 0}
        self.tables = [pd.read_csv(f, sep='\t') for f in [cell_default, nucl_default]]

        self.split = data_config.get('split', 0.2)
        self.seed = data_config.get('seed', 42)

        self.other_kwargs = self.config['other'] if 'other' in self.config else {}

        if self.config.get('contrastive', False):
            self.dset = RawAEContrCellDataset
        elif self.config.get('texture_contrastive', False):
            self.dset = TextPatchContrCellDataset
            raw_level = data_config.get("raw_level")
            self.raw_vol = raw_file[data_config.get('fine_raw_dataset',
                                                    'setup0/timepoint0/s{}'.format(raw_level))]
            self.other_kwargs['cell_hr_vol'] = cell_file[data_config.get(
                'fine_cell_dataset', 'setup0/timepoint0/s{}'.format(raw_level - 1))]

        self.transf = get_transforms(self.config.get('transforms'))
        self.trans_sim = get_transforms(self.config.get('transforms_sim'))

    @staticmethod
    def _resolve_bdv(path):
        path = Path(path)
        if path.suffix.lower() != '.xml':
            return path
        try:
            from pybdv.metadata import get_data_path
        except ImportError as error:
            raise RuntimeError('BDV XML inputs require pybdv from the legacy-training extra') from error
        return Path(get_data_path(str(path), True))

    @staticmethod
    def _open_container(path):
        try:
            import z5py

            return z5py.File(str(path), 'r')
        except ImportError:
            try:
                import zarr

                return zarr.open(str(path), mode='r')
            except ImportError as error:
                raise RuntimeError('N5/Zarr inputs require zarr or legacy z5py') from error

    def get_train_loaders(self):
        labels = get_train_val_split(list(self.nucl_dict.keys()),
                                     split=self.split, r_seed=self.seed)
        cell_dsets = [self.dset(self.tables, self.nucl_dict,
                                self.cell_vol, self.nuclei_vol, self.raw_vol,
                                indices=i, transforms=self.transf,
                                transforms_sim=self.trans_sim,
                                **self.other_kwargs) for i in labels]

        train_loader = DataLoader(cell_dsets[0], collate_fn=collate_contrastive,
                                  **self.config.get('loader_config'))
        val_loader = DataLoader(cell_dsets[1], collate_fn=collate_contrastive,
                                **self.config.get('val_loader_config'))
        return train_loader, val_loader

    def get_predict_loaders(self):
        pred_dataset = self.dset(self.tables, self.nucl_dict,
                                 self.cell_vol, self.nuclei_vol, self.raw_vol,
                                 transforms=self.transf, predict=True,
                                 **self.other_kwargs)
        pred_loader = DataLoader(pred_dataset, **self.config.get('pred_loader_config'))
        return pred_loader
