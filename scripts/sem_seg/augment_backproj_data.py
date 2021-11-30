'''
Create 4 rotations for each sample 
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
        subvol_size = f['x'][0].shape
        # create rotation matrices once
        rot_mats = {n: get_rot_mat(n, subvol_size) for n in (0, 1, 2, 3)}
        # num rots * num samples
        in_samples = len(f['x']) 
        total_aug_subvols = len(rot_mats) * in_samples
        print(f'In samples: {in_samples}, out samples: {total_aug_subvols}')
        num_nearest_images = len(f['frames'][0])

        with h5py.File(args.out_path, 'w') as outf:
            create_datasets(outf, total_aug_subvols, subvol_size, num_nearest_images)
            
            for ndx in tqdm(range(len(f['x']))):
                # read everything once
                # these will change
                x, y, world_to_grid = f['x'][ndx], f['y'][ndx], f['world_to_grid'][ndx] 
                # these wont change
                scan_id, scene_id, frames = f['scan_id'][ndx], f['scene_id'][ndx], f['frames'][ndx]

                for num_rots, rot_mat in rot_mats.items():
                    out_ndx = 4*ndx + num_rots

                    # copy existing fields
                    outf['scan_id'][out_ndx] = scan_id
                    outf['scene_id'][out_ndx] = scene_id
                    outf['frames'][out_ndx] = frames

                    # rotate x, y around the Z axis
                    aug_x = np.rot90(x, k=num_rots)
                    aug_y = np.rot90(y, k=num_rots)
                    # change world_to_grid accordingly
                    aug_world_to_grid = rot_mat @ world_to_grid
                    # put these 3 into the file
                    outf['x'][out_ndx] = aug_x
                    outf['y'][out_ndx] = aug_y
                    outf['world_to_grid'][out_ndx] = aug_world_to_grid

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('in_path', help='Path to input h5 file')
    p.add_argument('out_path', help='Path to output h5 file')

    args = p.parse_args()

    main(args)