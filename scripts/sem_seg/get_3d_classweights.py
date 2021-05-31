'''
Get class weights for 3D voxel grid
'''
import os 
import argparse
from datetime import datetime as dt

import numpy as np

from tqdm import tqdm

from lib.misc import read_config
from datasets.scannet.sem_seg_3d import ScanNetSemSegOccGrid
from datasets.scannet.utils import CLASS_NAMES

def main(args):
    cfg = read_config(args.cfg_path)
    dataset = ScanNetSemSegOccGrid(cfg['data'], full_scene=True)
    print(f'Dataset: {len(dataset)}')
    
    num_classes = len(CLASS_NAMES)
    counts = np.zeros((num_classes,))
    
    for ndx, sample in enumerate(tqdm(dataset)):
        counts += np.bincount(sample['y'].flatten(), minlength=num_classes)

    fraction = counts/counts.sum()
    print('Class distrib:', fraction)
    print('1/fraction:', 1/fraction)
    print('1/log(freq):', 1/np.log(counts))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cfg_path', help='Path to cfg')
    args = p.parse_args()

    main(args)