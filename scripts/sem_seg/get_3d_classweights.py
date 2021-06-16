'''
Get class weights for 3D voxel grid
'''
import argparse

import numpy as np

from tqdm import tqdm

from lib.misc import read_config
from datasets.scannet.common import CLASS_NAMES
from datasets.scannet.utils import get_trainval_sets

def main(args):
    cfg = read_config(args.cfg_path)
    dataset, _ = get_trainval_sets(cfg)
    print(f'Train set: {len(dataset)}')
    
    num_classes = 20 + 1
    counts = np.zeros((num_classes,))
    
    for _, sample in enumerate(tqdm(dataset)):
        counts += np.bincount(sample[2].flatten(), minlength=num_classes)

    # remove the last ignored class
    counts = counts[:-1]
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