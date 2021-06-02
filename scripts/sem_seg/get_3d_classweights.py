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

    # remove none and target padding counts, 20 classes left
    counts = counts[1:-1]
    print('Counts:', counts)
    fraction = counts/counts.sum() 
    print('Class distrib: ',fraction)
    
    inv_frac = 1/fraction
    print('1/fraction: ', inv_frac.tolist())
    print('Normed: ', (inv_frac / inv_frac.sum()).tolist())

    inv_log = 1/np.log(counts)
    print('1/log(freq): ', inv_log.tolist())
    print('normed: ', (inv_log / inv_log.sum()).tolist())

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cfg_path', help='Path to cfg')
    args = p.parse_args()

    main(args)