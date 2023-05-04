import os
import numpy as np
import pandas as pd
import z5py
import torch
from torch.utils.data.dataloader import DataLoader

from inferno.io.transform import Compose
from inferno.io.transform.generic import NormalizeRange, Cast, AsTorchBatch
from inferno.io.transform.volume import CropPad2Size, VolumeRandomCrop, RandomRot903D
from inferno.io.transform.image import ElasticTransform
from inferno.utils.io_utils import yaml2dict

from pybdv.metadata import get_data_path

from cell_dset import RawAEContrCellDataset, TextPatchContrCellDataset


def get_train_val_split(labels, split=0.2, r_seed=None):
    np.random.seed(seed=r_seed)
    np.random.shuffle(labels)
    spl = int(np.floor(len(labels)*split))
    return labels[spl:], labels[:spl]


def get_transforms(transform_config):
    order = 3
    transforms = Compose()
    if transform_config.get('crop_pad_to_size'):
        crop_pad_to_size = transform_config.get('crop_pad_to_size')
        transforms.add(CropPad2Size(**crop_pad_to_size))
    if transform_config.get('random_crop'):
        random_crop = transform_config.get('random_crop')
        transforms.add(VolumeRandomCrop(**random_crop))
    if transform_config.get('cast'):
        transforms.add(Cast('float32'))
    if transform_config.get('normalize_range'):
        normalize_range_config = transform_config.get('normalize_range')
        transforms.add(NormalizeRange(**normalize_range_config))
    if transform_config.get('rotate90'):
        transforms.add(RandomRot903D())
    if transform_config.get('elastic_transform'):
        elastic_config = transform_config.get('elastic_transform')
        transforms.add(ElasticTransform(order=order, **elastic_config))
    if transform_config.get('torch_batch'):
        transforms.add(AsTorchBatch(3))
    return transforms


def collate_contrastive(batch):
    inputs = torch.cat([i[0] for i in batch])
    targets = torch.cat([i[1] for i in batch])
    if len(batch[0]) == 3:
        targets2 = torch.cat([i[2] for i in batch])
        targets = [targets, targets2]
    return inputs, targets


class CellLoaders(object):
    def __init__(self, configuration_file):
        self.config = yaml2dict(configuration_file)
        data_config = self.config.get('data_config')

        self.PATH = "/scratch/zinchenk/cell_match/data/platy_data"
        version = data_config.get("version")

        raw_data = os.path.join(self.PATH, "rawdata/sbem-6dpf-1-whole-raw.n5")
        cell_segm = os.path.join(self.PATH, version, "images/local",
                                 "sbem-6dpf-1-whole-segmented-cells.xml")
        nucl_segm = os.path.join(self.PATH, version, "images/local",
                                 "sbem-6dpf-1-whole-segmented-nuclei.xml")
        cell_to_nucl = os.path.join(self.PATH, version, "tables",
                                    "sbem-6dpf-1-whole-segmented-cells/cells_to_nuclei.tsv")
        cell_default = os.path.join(self.PATH, version, "tables",
                                    "sbem-6dpf-1-whole-segmented-cells/default.tsv")
        nucl_default = os.path.join(self.PATH, version, "tables",
                                    "sbem-6dpf-1-whole-segmented-nuclei/default.tsv")

        cell_file = z5py.File(get_data_path(cell_segm, True), 'r')
        nucl_file = z5py.File(get_data_path(nucl_segm, True), 'r')
        raw_file = z5py.File(raw_data, 'r')

        self.raw_vol = raw_file['setup0/timepoint0/s3']
        self.cell_vol = cell_file['setup0/timepoint0/s2']
        self.nuclei_vol = nucl_file['setup0/timepoint0/s0']

        self.nucl_dict = {int(k): int(v)
                          for k, v in np.loadtxt(cell_to_nucl, skiprows=1)
                          if v != 0}
        self.tables = [pd.read_csv(f, sep='\t') for f in [cell_default, nucl_default]]

        self.split = data_config.get('split', None)
        self.seed = data_config.get('seed', None)

        self.other_kwargs = self.config['other'] if 'other' in self.config else {}

        if self.config.get('contrastive', False):
            self.dset = RawAEContrCellDataset
        elif self.config.get('texture_contrastive', False):
            self.dset = TextPatchContrCellDataset
            raw_level = data_config.get("raw_level")
            self.raw_vol = raw_file['setup0/timepoint0/s{}'.format(raw_level)]
            self.other_kwargs['cell_hr_vol'] = cell_file['setup0/timepoint0/s{}'\
                                               .format(raw_level - 1)]

        self.transf = get_transforms(self.config.get('transforms')) \
                      if self.config.get('transforms') else None
        self.trans_sim = get_transforms(self.config.get('transforms_sim')) \
                         if self.config.get('transforms_sim') else None

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
