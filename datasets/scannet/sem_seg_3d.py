'''
3D semantic segmentation on ScanNet occupancy voxel grids
'''

import os
from pathlib import Path

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
    def __init__(self, root_dir, limit_scans=None, transform=None):
        self.root_dir = Path(root_dir)
        self.paths = []
        self.transform = transform

        scans = sorted(os.listdir(self.root_dir))
        if limit_scans:
            scans = scans[:limit_scans]

        for scan_id in scans:
            path = self.root_dir / scan_id / f'{scan_id}_occ_grid.pth'

            if path.exists():
                self.paths.append(path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, ndx):
        path = self.paths[ndx]
        data = torch.load(path)
        x, y_orig = data['x'], data['y']

        y_cts = nyu40_to_continuous(y_orig)

        sample = {'path': path, 'x': x, 'y': y_cts}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample