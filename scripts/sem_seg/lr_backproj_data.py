'''
Create a limited reconstructions version of a backproj dataset
write it to a different file
'''

import argparse
from lib.misc import read_config
from datasets.scannet.common import read_list
from scripts.sem_seg.prep_backproj_data import create_datasets
import random
from tqdm import tqdm
import h5py

def get_scene_scan_ids(scan_name):
    '''
    scan_name: scene0673_05
    output: 673, 05 (ints)
    '''
    return int(scan_name[5:9]), int(scan_name[10:12])

def main(args):
    cfg = read_config(args.cfg_path)
    scan_list = read_list(args.list_path)
    scene_scan_ids = list(map(get_scene_scan_ids, scan_list))

    with h5py.File(args.in_path) as f:
        keys = f.keys()

        # how many subvols per scan? pick one and check
        scene, scan = scene_scan_ids[0]
        subvols_per_scan = sum((f['scene_id'][:] == scene) & (f['scan_id'][:] == scan))

        total_subvols = len(scan_list) * subvols_per_scan

        with h5py.File(args.out_path, 'w') as outf:
            create_datasets(outf, total_subvols, tuple(cfg['data']['subvol_size']), 
                            cfg['data']['num_nearest_images'])
            out_ndx = 0
            for ndx in tqdm(range(len(f['x']))):
                if (f['scene_id'][ndx], f['scan_id'][ndx]) in scene_scan_ids:
                    for key in keys:
                        outf[key][out_ndx] = f[key][ndx]
                    out_ndx += 1

    print(f'Wrote {out_ndx} subvols')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('in_path', help='Path to input h5 file')
    p.add_argument('list_path', help='Path to list of scans to use')
    p.add_argument('out_path', help='Path to output h5 file')
    p.add_argument('cfg_path', help='Path to cfg')

    args = p.parse_args()

    main(args)