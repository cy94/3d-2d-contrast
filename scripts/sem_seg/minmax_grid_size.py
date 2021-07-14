'''
Find the maximum voxel grid size over all scans
'''
import argparse

from pathlib import Path

import os
from tqdm import tqdm
import numpy as np

import torch


def main(args):
    root = Path(args.scannet_dir)

    grid_sizes = []

    for scan_id in tqdm(sorted(os.listdir(root)), desc='scan'):
        scan_dir = root / scan_id
        
        in_file = f'{scan_id}_occ_grid.pth'
        data = torch.load(scan_dir / in_file)
        grid = data['y']
        grid_sizes.append(grid.shape)
    
    grid_sizes = np.array(grid_sizes)
    
    print(f'Max grid size: {grid_sizes.max(axis=0)}')
    print(f'Max grid size: {grid_sizes.min(axis=0)}')


if __name__ == '__main__':
    # params
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('scannet_dir', help='path to scannet root dir file to read')

    args = parser.parse_args()

    main(args)