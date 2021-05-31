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
    train_frac = 0.8

    scans = os.listdir(root)
    random.shuffle(scans)
    n_train = int(train_frac * len(scans))
    
    train_scans = scans[:n_train]
    val_scans = scans[n_train:]
    
    write_list(root / 'train.txt', train_scans)
    write_list(root / 'val.txt', val_scans)

if __name__ == '__main__':
    # params
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('scannet_dir', help='path to scannet root dir')

    args = parser.parse_args()

    main(args)