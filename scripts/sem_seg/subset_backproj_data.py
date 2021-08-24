'''
Pick a random subset of the backproj h5 dataset
write it to a different file
'''

import argparse
from lib.misc import read_config
from scripts.sem_seg.prep_backproj_data import create_datasets
import random
from tqdm import tqdm
import h5py

def main(args):
    cfg = read_config(args.cfg_path)

    with h5py.File(args.in_path) as f:
        indices = list(range(len(f['x'])))
        random.shuffle(indices)
        out_indices = sorted(indices[:args.n_select])
        keys = f.keys()
        with h5py.File(args.out_path, 'w') as outf:
            create_datasets(outf, len(out_indices), tuple(cfg['data']['subvol_size']), 
                            cfg['data']['num_nearest_images'])
            for out_ndx, ndx in enumerate(tqdm(out_indices)):
                for key in keys:
                    outf[key][out_ndx] = f[key][ndx]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('in_path', help='Path to input h5 file')
    p.add_argument('n_select', type=int, help='Num samples to select')
    p.add_argument('out_path', help='Path to output h5 file')
    p.add_argument('cfg_path', help='Path to cfg')

    args = p.parse_args()

    main(args)