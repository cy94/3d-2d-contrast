'''
Get class weights for 3D voxel grid
'''
import argparse
from datasets.scannet.sem_seg_2d import ScanNetSemSeg2D
from datasets.scannet.sem_seg_3d import ScanNetSemSegOccGrid

import numpy as np

from tqdm import tqdm

from lib.misc import read_config
from datasets.scannet.utils import get_trainval_sets
from models.sem_seg.utils import SPARSE_MODELS

def main(args):
    cfg = read_config(args.cfg_path)

    num_classes = cfg['data'].get('num_classes', 20)
    print(f'Num classes: {num_classes}')

    if args.rgb:
        # use 2D dataset
        dataset = ScanNetSemSeg2D(cfg, split='train')
    else:
        model_name = cfg['model']['name']
        is_sparse = model_name in SPARSE_MODELS
        if is_sparse:
            dataset, _ = get_trainval_sets(cfg)
        else:
            # read full grids
            dataset = ScanNetSemSegOccGrid(cfg['data'], split='train', full_scene=True)

    print(f'Train set: {len(dataset)}')

    # all NYU40 classes, or only those relevant to scannet?
    counts = np.zeros((num_classes + 1,))
    
    for _, sample in enumerate(tqdm(dataset)):
        if args.rgb:
            y = sample['label']
        else:
            if is_sparse:
                y = sample[2].flatten()
            else:
                y = sample['y']

        counts += np.bincount(y.flatten(), minlength=num_classes+1)

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
    p.add_argument('--rgb', action='store_true', dest='rgb', 
                default=False, help='Use 2D RGB dataset')     
    args = p.parse_args()

    main(args)