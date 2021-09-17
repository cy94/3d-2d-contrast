'''
Create a limited reconstructions version of a backproj dataset
write it to a different file
'''

import argparse
from scripts.sem_seg.prep_backproj_data import create_datasets
from tqdm import tqdm
import h5py
import numpy as np

def get_scene_scan_ids(scan_name):
    '''
    scan_name: scene0673_05
    output: 673, 05 (ints)
    '''
    return int(scan_name[5:9]), int(scan_name[10:12])

def get_rot_mat(num_rots, subvol_size):
    '''
    num_rots: number of rotations by np.rot90 in the direction X->Y axis, about the Z axis
    subvol_size: (W, H, D) size
    '''
    # rotate about the Z axis num_rots times
    rot90 = np.eye(4)
    rot90[0, 0] = 0
    rot90[1, 1] = 0
    rot90[0, 1] = -1
    rot90[1, 0] = 1

    rot_n = np.linalg.matrix_power(rot90, num_rots)

    return rot_n 

def main(args):
    assert args.in_path != args.out_path

    rng = np.random.default_rng()

    with h5py.File(args.in_path) as f:
        # keys to copy as-is
        copy_keys = 'scan_id', 'scene_id', 'frames'
        total_subvols = len(f['x']) 
        subvol_size = f['x'][0].shape
        # create rotation matrices once
        rot_mats = {n: get_rot_mat(n, subvol_size) for n in (0, 1, 2, 3)}
        num_nearest_images = len(f['frames'][0])

        with h5py.File(args.out_path, 'w') as outf:
            create_datasets(outf, total_subvols, subvol_size, num_nearest_images)

            for ndx in tqdm(range(len(f['x']))):
                for key in copy_keys:
                    outf[key][ndx] = f[key][ndx]
                x, y, world_to_grid = f['x'][ndx], f['y'][ndx], f['world_to_grid'][ndx] 
                # rotate x, y around the Z axis
                num_rots = rng.integers(0, 3, endpoint=True)
                aug_x = np.rot90(x, k=num_rots)
                aug_y = np.rot90(y, k=num_rots)
                # change world_to_grid accordingly
                aug_world_to_grid = rot_mats[num_rots] @ world_to_grid
                outf['x'][ndx] = aug_x
                outf['y'][ndx] = aug_y
                outf['world_to_grid'][ndx] = aug_world_to_grid

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('in_path', help='Path to input h5 file')
    p.add_argument('out_path', help='Path to output h5 file')

    args = p.parse_args()

    main(args)