'''
3D semantic segmentation on ScanNet occupancy voxel grids
'''

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.scannet.utils import nyu40_to_continuous

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
        self.subvol_size = cfg['subvol_size']

        scans = sorted(os.listdir(root_dir))

        if cfg['limit_scans']:
            scans = scans[:cfg['limit_scans']]

        for scan_id in scans:
            path = root_dir / scan_id / f'{scan_id}_occ_grid.pth'

            if path.exists():
                self.paths.append(path)

    def __len__(self):
        # vols per scene * num scenes
        return self.subvols_per_scene * len(self.paths)

    def __getitem__(self, ndx):
        # pick the scene
        scene_ndx = ndx // self.subvols_per_scene
        path = self.paths[scene_ndx]
        # load the full scene
        data = torch.load(path)
        x, y = data['x'], data['y']
        # pick a random subvolume
        max_start = np.array(x.shape) - self.subvol_size
        start = np.random.randint((0, 0, 0), max_start, dtype=np.uint8)
        end = start + self.subvol_size

        x = x[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        y = y[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        y_cts = nyu40_to_continuous(y).astype(np.int8)

        sample = {'path': path, 'x': x, 'y': y_cts}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample