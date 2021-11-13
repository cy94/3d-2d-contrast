'''
join 2 h5 files with backproj data
'''

import argparse
from scripts.sem_seg.prep_backproj_data import create_datasets
from tqdm import tqdm
import h5py

def get_scene_scan_ids(scan_name):
    '''
    scan_name: scene0673_05
    output: 673, 05 (ints)
    '''
    return int(scan_name[5:9]), int(scan_name[10:12])

def main(args):
    assert args.in_path1 != args.out_path
    assert args.in_path2 != args.out_path

    f1 = h5py.File(args.in_path1)
    f2 = h5py.File(args.in_path2)
    subvol_size = f1['x'][0].shape
    # keys to copy as-is
    copy_keys = 'scan_id', 'scene_id', 'frames', 'world_to_grid', 'x', 'y'
    f1_subvols = len(f1['x'])
    f2_subvols = len(f2['x'])
    total_subvols = f1_subvols + f2_subvols
    num_nearest_images = len(f1['frames'][0])

    with h5py.File(args.out_path, 'w') as outf:
        create_datasets(outf, total_subvols, subvol_size, num_nearest_images)

        for ndx in tqdm(range(len(f1['x']))):
            for key in copy_keys:
                outf[key][ndx] = f1[key][ndx]
                
        for ndx in tqdm(range(len(f2['x']))):
            for key in copy_keys:
                outf[key][f1_subvols + ndx] = f2[key][ndx]
        
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('in_path1', help='Path to input h5 file 1')
    p.add_argument('in_path2', help='Path to input h5 file 2')
    p.add_argument('out_path', help='Path to output h5 file')

    args = p.parse_args()

    main(args)