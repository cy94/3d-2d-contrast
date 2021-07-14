'''
3D semantic segmentation on ScanNet occupancy voxel grids
'''
from datasets.scannet.common import load_ply
import random
import os
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.scannet.common import nyu40_to_continuous, read_label_mapping, read_list
from transforms.grid_3d import pad_volume

def collate_func(sample_list):
    return {
        'path': [s['path'] for s in sample_list],
        'x': torch.Tensor([s['x'] for s in sample_list]),
        'y': torch.LongTensor([s['y'] for s in sample_list]),
    }

class ScanNetSemSegOccGrid(Dataset):
    '''
    ScanNet 3d semantic segmentation on voxel grid

    x: (l, b, h) binary grid
    labels: 0-19 (20 classes) + target padding/ignore label 

    labels as given here here: http://kaldir.vc.in.tum.de/scannet_benchmark/
    '''
    def __init__(self, cfg, transform=None, split=None, full_scene=False):
        '''
        data_cfg:
            see configs/occgrid_train.yml
            root: root of scannet dataset
            limit_scans: read only these many scans
            subvol_size: size of subvolumes to sample
            subvols_per_scene: sample these many subvolumes per scene
        transform: apply on each subvol
        split: name of the split, used to read the list from cfg
        full_scene: return the full scene
        '''
        self.root_dir = Path(cfg['root'])
        
        self.use_rgb = cfg.get('use_rgb', False)

        self.transform = transform
        self.full_scene = full_scene

        # sample subvolumes
        self.subvols_per_scene = cfg.get('subvols_per_scene', None)
        self.subvol_size = np.array(cfg.get('subvol_size', None))
        self.target_padding = cfg.get('target_padding', None)

        self.scannet_to_nyu40 = read_label_mapping(cfg['label_file'])

        self.num_classes = cfg.get('num_classes', 20)

        if split:
            # read train/val/test list
            self.scans = read_list(cfg[f'{split}_list'])
        else:
            self.scans = sorted(os.listdir(self.root_dir))

        if cfg['limit_scans']:
            self.scans = self.scans[:cfg['limit_scans']]

        self.paths = self.get_paths()

    def get_paths(self):
        '''
        Paths to files to scene files - 1 file per scene
        '''
        paths = []
        for scan_id in self.scans:
            path = self.root_dir / scan_id / f'{scan_id}_occ_grid.pth'

            if path.exists():
                paths.append(path)

        return paths

    def __len__(self):
        # vols per scene * num scenes
        if self.full_scene:
            return len(self.paths)
        return self.subvols_per_scene * len(self.paths)

    def sample_subvol(self, x, y):
        '''
        x, y - volumes of the same size
        '''
        # pad the input volume for these reasons
        # 1. if the volume is is smaller than the subvol size
        #    pad it along the required dimensions so that a proper subvol can be created
        # 2. need to learn the padding, which is needed later during inference
        # 3. augmentation
        # 
        # for 2+3 - apply padding to the whole scene, then sample subvolumes as usual
        # so that subvols on the edge of the scene get padding
        #
        # result: left+right padding = max(subvol size, padding required to reach subvol size)
        # then apply half of this padding on each side 

        # the padding required for small scenes (left+right)
        small_scene_pad = self.subvol_size - x.shape
        small_scene_pad[small_scene_pad < 0] = 0

        # augmentation padding for all other scenes (left+right)
        aug_pad = self.subvol_size

        # final scene size
        pad = np.maximum(small_scene_pad, aug_pad)
        scene_size = np.array(x.shape) + pad
        # splits the padding equally on both sides and applies it
        x, y = pad_volume(x, scene_size), pad_volume(y, scene_size, pad_val=self.target_padding)

        # now x, y are atleast the size of subvol in each dimension
        # sample subvols as usual
        while 1:
            # the max value at which the subvol index can start 
            max_start = np.array(x.shape) - self.subvol_size
            # pick an index between 0 and the max value along each dimension
            # add 1 to max_start because its exclusive
            start = np.random.randint((0, 0, 0), max_start + 1, dtype=np.uint16)
            end = start + self.subvol_size
            # extract the subvol from the full scene
            x_sub = x[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            y_sub = y[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

            # classes 0,1 = wall, floor
            # if: the subvol has only these 2 classes -> keep only 5% of such subvols
            # or: other classes with index >2? keep the subvol
            if (y_sub.max() == 1 and random.random() > 0.95) or (y_sub.max() > 1):
                break

        return x_sub, y_sub

    def get_scene_grid(self, scene_ndx):
        path = self.paths[scene_ndx]
        # load the full scene
        data = torch.load(path)
        # labels are scannet IDs
        x, y_nyu = data['x'], data['y']

        return x, y_nyu

    def __getitem__(self, ndx):
        if not self.full_scene:
            # get the scene ndx for this subvol 
            scene_ndx = ndx // self.subvols_per_scene
        else:
            scene_ndx = ndx
        
        path = self.paths[scene_ndx]

        x, y_nyu = self.get_scene_grid(scene_ndx)

        # convert bool x to float
        x = x.astype(float)
        # dont use int8 anywhere, avoid possible overflow with more than 128 classes
        y = nyu40_to_continuous(y_nyu, ignore_label=self.target_padding, 
                                num_classes=self.num_classes).astype(np.int16)
        if self.full_scene:
            xval, yval = x, y
        else:
            xval, yval = self.sample_subvol(x, y)

        sample = {'path': path, 'x': xval, 'y': yval}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

class ScanNetPLYDataset(ScanNetSemSegOccGrid):
    '''
    Read voxelized ScanNet PLY files which contain
    vertices, colors and labels

    Create dense grid containing these voxelized vertices and labels
    and sample subvolumes from it
    '''
    def get_paths(self):
        '''
        Paths to files to scene files - 1 file per scene
        '''
        paths = []
        for scan_id in self.scans:
            path = self.root_dir / scan_id / f'{scan_id}_voxelized.ply'

            if path.exists():
                paths.append(path)

        return paths

    def get_scene_grid(self, scene_ndx):
        path = self.paths[scene_ndx]
        # load the full scene
        coords, rgb, labels = load_ply(path, read_label=True)
        coords = coords.astype(np.int32)
        # translate the points to start at 0
        t = coords.min(axis=0)
        coords_new = coords - t
        # integer coordinates, get the grid size from this
        grid_size = tuple(coords_new.max(axis=0).astype(np.int32) + 1)

        if self.use_rgb:
            # use RGB values as grid features
            x = np.zeros(grid_size + (3,))
            x[coords_new[:, 0], coords_new[:, 1], coords_new[:, 2]] = rgb
        else:
            # binary occupancy grid
            x = np.zeros(grid_size)
            x[coords_new[:, 0], coords_new[:, 1], coords_new[:, 2]] = 1

        # fill Y with negative ints
        y_nyu = np.ones(grid_size, dtype=np.int16) * -1
        y_nyu[coords_new[:, 0], coords_new[:, 1], coords_new[:, 2]] = labels

        return x, y_nyu

class ScanNetGridTestSubvols:
    '''
    Take a full scannet scene, pad it to multiple of subvolumes
    Read all non overlapping subvolumes
    '''
    def __init__(self, scene, subvol_size, target_padding, transform=None):
        '''
        scene: a full_scene sample from the above dataset
        subvol_size: size of the subvolumes
        '''
        x = scene['x']
        y = scene['y']
        self.path = scene['path']
        self.subvol_size = subvol_size
        self.target_padding = target_padding

        self.transform = transform 

        # pad the scene to multiples of subvols
        padded_size = ((np.array(x.shape) // self.subvol_size) + 1) * self.subvol_size
        self.x = pad_volume(x, padded_size)
        self.y = pad_volume(y, padded_size, self.target_padding)

        # mapping from ndx to subvol slices
        self.mapping = OrderedDict()
        ndx = 0
        # height
        for k in range(0, self.x.shape[2], self.subvol_size[2]):
            # width
            for j in range(0, self.x.shape[1], self.subvol_size[1]):
                # length
                for i in range(0, self.x.shape[0], self.subvol_size[0]):
                    slice = np.s_[
                        i : i+self.subvol_size[0], 
                        j : j+self.subvol_size[1], 
                        k : k+self.subvol_size[2], 
                    ]
                    self.mapping[ndx] = slice
                    ndx += 1

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, ndx):
        slice = self.mapping[ndx]
        sub_x = self.x[slice]
        sub_y = self.y[slice]

        sample = {'x': sub_x, 'y': sub_y, 'path': self.path}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


