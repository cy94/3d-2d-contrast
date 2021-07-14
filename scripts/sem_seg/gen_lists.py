'''
Generate train val lists
'''
import argparse

from pathlib import Path

import os
from tqdm import tqdm
import numpy as np

import random

def write_list(path, l):
    '''
    path: Path object
    l: list of strings 
    '''
    print('Writing to', path)
    with open(path, 'w') as f:
        f.writelines((f'{line}\n' for line in l))

def main(args):
    root = Path(args.scannet_dir)
    train_frac = 0.7
    val_frac = 0.2

    scans = os.listdir(root)
    random.shuffle(scans)

    n_train = int(train_frac * len(scans))
    n_val = int(val_frac * len(scans))
    
    train_scans = scans[:n_train]
    val_scans = scans[n_train:n_train+n_val]
    test_scans = scans[n_train+n_val:]
    
    write_list(root.parent / 'train.txt', train_scans)
    write_list(root.parent / 'val.txt', val_scans)
    write_list(root.parent / 'test.txt', test_scans)

if __name__ == '__main__':
    # params
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('scannet_dir', help='path to scannet root dir')

    args = parser.parse_args()

    main(args)