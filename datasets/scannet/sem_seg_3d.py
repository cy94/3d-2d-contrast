'''
3D semantic segmentation on ScanNet occupancy voxel grids
'''
import random
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.scannet.utils import nyu40_to_continuous
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
    labels: 0-20 

    labels as given here here: http://kaldir.vc.in.tum.de/scannet_benchmark/
    '''
    def __init__(self, cfg, transform=None):
        '''
        data_cfg:
            see configs/occgrid_train.yml
            root: root of scannet dataset
            limit_scans: read only these many scans
            subvol_size: size of subvolumes to sample
            subvols_per_scene: sample these many subvolumes per scene
            transform: apply on each subvol
        '''
        root_dir = Path(cfg['root'])
        
        self.subvols_per_scene = cfg['subvols_per_scene']
        self.paths = []
        self.transform = transform
        self.subvol_size = np.array(cfg['subvol_size'])

        scans = sorted(os.listdir(root_dir))

        if cfg['limit_scans']:
            scans = scans[:cfg['limit_scans']]

        for scan_id in scans:
            path = root_dir / scan_id / f'{scan_id}_occ_grid.pth'

            if path.exists():
                self.paths.append(path)

        # important: shuffle the scenes because they were recorded in a certain
        # order and are not completely independent!
        random.shuffle(self.paths)

    def __len__(self):
        # vols per scene * num scenes
        return self.subvols_per_scene * len(self.paths)

    def sample_subvol(self, x, y):
        '''
        x, y - volumes of the same size
        '''
        # pad the input volume for 2 reasons
        # 1. if the volume is is smaller than the subvol size
        #    pad it along the required dimensions so that a proper subvol can be created
        # 2. need to learn the padding, which is done later during inference
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
        x, y = pad_volume(x, scene_size), pad_volume(y, scene_size, 120)

        # now x, y are atleast the size of subvol in each dimension
        # sample subvols as usual
        while 1:
            # pick a random subvolume
            max_start = np.array(x.shape) - self.subvol_size
            # add 1 to max_start because its exclusive
            start = np.random.randint((0, 0, 0), max_start + 1, dtype=np.uint16)
            end = start + self.subvol_size

            x_sub = x[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            y_sub = y[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

            # classes 0,1,2 = none, wall, floor
            # only these 3 -> keep only 5% of such subvols
            # other classes >2? keep this subvol 
            if (y_sub.max() == 2 and random.random() > 0.95) or (y_sub.max() > 2):
                break

        return x_sub, y_sub

    def __getitem__(self, ndx):
        # pick the scene
        scene_ndx = ndx // self.subvols_per_scene
        path = self.paths[scene_ndx]
        # load the full scene
        data = torch.load(path)
        x, y_nyu = data['x'], data['y']
        # map nyu40 labels to continous labels
        y = nyu40_to_continuous(y_nyu).astype(np.int8)
        
        x_sub, y_sub = self.sample_subvol(x, y)

        sample = {'path': path, 'x': x_sub, 'y': y_sub}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample