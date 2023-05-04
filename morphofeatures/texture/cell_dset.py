import numpy as np
import z5py
import torch
from torch.utils.data.dataset import Dataset
from skimage.transform import rescale


class CellDataset(Dataset):
    def __init__(self, cell_nucl_tables, nucl_dict, cell_data, nucl_data, raw_data,
                 predict=False, indices=None, transforms=None, transforms_sim=None,
                 size_cut=200):
        self.RES = [0.025, 0.01, 0.01]
        self.HR_SHAPE = [11416, 25916, 27499]
        self.cell_table = cell_nucl_tables[0]
        self.nucl_table = cell_nucl_tables[1]
        self.nucl_dict = nucl_dict
        self.cell_data = cell_data
        self.nucl_data = nucl_data
        self.raw_data = raw_data

        self.predict = predict

        self.tfs = transforms
        self.tfs_sim = transforms_sim
        if indices is None:
            indices = list(nucl_dict.keys())
        assert isinstance(indices, (list, tuple, np.ndarray)), \
            ("The index list is of class {}, "
             "not list, tuple or numpy array").format(type(indices).__name__)
        self.indices = np.array(indices)
        self.res_diff = self.get_resolution_diff()
        self.cell_ref_bbs = self.get_bbs(self.cell_table)
        self.size_cut = size_cut

    def __len__(self):
        return len(self.indices)

    def transform(self, data, transforms=None):
        if transforms is not None:
            return transforms(data)
        else:
            return data

    def get_bbs(self, table):
        bbs = [[slice(int(np.rint(row['bb_min_{}'.format(ax)] / self.RES[n])),
                      int(np.rint(row['bb_max_{}'.format(ax)] / self.RES[n])))
                for n, ax in enumerate(['z', 'y', 'x'])]
               for _, row in table.iterrows()]
        upd_bbs = [self.update_bb(bb) for bb in bbs]
        upd_bbs = [[]] + upd_bbs
        return upd_bbs

    def get_resolution_diff(self):
        act_shape = self.cell_data.shape
        res_diff = np.array(act_shape) / np.array(self.HR_SHAPE)
        return res_diff

    def update_bb(self, bb):
        new_bb = []
        for axis, old_slice in enumerate(bb):
            res_diff_axis = self.res_diff[axis]
            new_slice = slice(int(np.floor(old_slice.start * res_diff_axis)),
                              int(np.ceil(old_slice.stop * res_diff_axis)))
            new_bb.append(new_slice)
        return tuple(new_bb)

    def center_of_mass(self, idx, cell=False):
        table = self.cell_table if cell else self.nucl_table
        id_data = table[table['label_id'] == idx]
        com_um = [id_data['anchor_{}'.format(ax)].values.item()
                  for ax in ['z', 'y', 'x']]
        com_pxl = [int(np.rint(i / self.RES[n] * self.res_diff[n]))
                   for n, i in enumerate(com_um)]
        return com_pxl

    def cut_to_size(self, cell_id):
        bb = self.cell_ref_bbs[cell_id]
        nucl_id = self.nucl_dict[cell_id]
        nucl_center = [int(i) for i in self.center_of_mass(nucl_id)]
        cut = int(self.size_cut/2)
        cut_box = [slice(max(0, axis_center - cut), axis_center + cut)
                   for axis_center in nucl_center]
        bb_upd = [slice(max(sl1.start, sl2.start), min(sl2.stop, sl1.stop))
                  for sl1, sl2 in zip(bb, cut_box)]
        return tuple(bb_upd)


class RawAEContrCellDataset(CellDataset):
    def __init__(self, *super_args, **super_kwargs):
        self.remove_nucl = super_kwargs.pop('remove_nucl', False)
        self.only_nucl = super_kwargs.pop('only_nucl', False)
        self.dilate_mask = super_kwargs.pop('dilate_mask', False)
        super().__init__(*super_args, **super_kwargs)

    def get_data_stack(self, cell_idx):
        cell_bb = self.cut_to_size(cell_idx)
        cell_mask = (self.cell_data[cell_bb] == cell_idx)
        raw_mask = self.raw_data[cell_bb] * cell_mask
        if self.remove_nucl:
            nucl_idx = self.nucl_dict[cell_idx]
            nucleus_mask = (self.nucl_data[cell_bb] == nucl_idx)
            raw_mask = raw_mask * np.invert(nucleus_mask)
        elif self.only_nucl:
            nucl_idx = self.nucl_dict[cell_idx]
            nucleus_mask = (self.nucl_data[cell_bb] == nucl_idx)
            raw_mask = raw_mask * nucleus_mask
        return raw_mask

    def __getitem__(self, idx):
        cell_index = self.indices[idx]
        data_stack = self.get_data_stack(cell_index)

        if self.predict:
            return self.transform(data_stack, self.tfs)

        target_data = [self.transform(data_stack.copy(), self.tfs) for i in range(2)]
        input_data = [self.transform(i.clone(), self.tfs_sim) for i in target_data]
        return torch.stack(input_data), torch.stack(target_data)


class TextPatchContrCellDataset(CellDataset):
    def __init__(self, *super_args, **super_kwargs):
        self.hr_cell_data = super_kwargs.pop('cell_hr_vol')
        self.radius = super_kwargs.pop('radius')
        if super_kwargs.get('crops_file', None):
            positions_file = z5py.File(super_kwargs.pop('crops_file'))
            self.positions = positions_file['positions']
            self.all_ids = positions_file['ids'][:]
        else:
            self.positions, self.all_ids = None, None

        self.remove_nucl = super_kwargs.pop('remove_nucl', False)
        self.only_nucl = super_kwargs.pop('only_nucl', False)
        self.take_every = super_kwargs.pop('take_every', 1)
        super().__init__(*super_args, **super_kwargs)
        self.bb_scale = int(np.rint(self.raw_data.shape[0] / self.cell_data.shape[0]))

    def __len__(self):
        return int(self.positions.shape[0] / self.take_every)

    def get_data_stack(self, cell_idx, loc):
        rand_bb = [slice(i - self.radius, i + self.radius) for i in loc]

        cell_bb = self.cut_to_size(cell_idx)
        crop_bb = [slice(sl1.start + sl2.start, sl1.start + sl2.stop)
                   for sl1, sl2 in zip(cell_bb, rand_bb)]
        hr_bb = tuple([slice(sl.start * self.bb_scale, sl.stop * self.bb_scale)
                       for sl in crop_bb])

        hr_crop = self.raw_data[hr_bb] * (self.hr_cell_data[hr_bb] == cell_idx)
        nucl_crop = self.nucl_data[tuple(crop_bb)] == self.nucl_dict[cell_idx]
        if self.remove_nucl and np.any(nucl_crop):
            ups_nucl = rescale(nucl_crop, self.bb_scale, multichannel=False, order=0)
            hr_crop = hr_crop * (1 - ups_nucl)
        if self.only_nucl:
            hr_crop = self.raw_data[hr_bb]
            ups_nucl = rescale(nucl_crop, self.bb_scale, multichannel=False, order=0)
            hr_crop = hr_crop * ups_nucl

        return hr_crop

    def __getitem__(self, idx):
        pos = self.positions[idx * self.take_every]
        cell_idx = pos[0]
        pos = pos[1:]

        if self.predict:
            input_data = self.get_data_stack(cell_idx, pos)
            return self.transform(input_data, self.tfs)

        data_stack = self.get_data_stack(cell_idx, pos)
        target_data = [self.transform(data_stack.copy(), self.tfs) for i in range(2)]
        input_data = [self.transform(i.clone(), self.tfs_sim) for i in target_data]
        return torch.stack(input_data), torch.stack(target_data)
