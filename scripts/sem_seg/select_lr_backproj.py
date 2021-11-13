'''
select the LR dataset samples from the full backproj dataset
'''

import argparse
from scripts.sem_seg.prep_backproj_data import create_datasets
from datasets.scannet.common import read_list, get_scene_scan_ids
from tqdm import tqdm
import h5py
import wandb

def main(args):
    assert args.in_path != args.out_path
    lr_list = read_list(args.list_path)
    lr_ids_list = list(map(get_scene_scan_ids, lr_list))

    f = h5py.File(args.in_path)
    subvol_size = f['x'][0].shape
    # keys to copy as-is
    copy_keys = 'scan_id', 'scene_id', 'frames', 'world_to_grid', 'x', 'y'
    num_nearest_images = len(f['frames'][0])

    # get indices of required subvols
    scene_ids = f['scene_id'][:].tolist()
    scan_ids = f['scan_id'][:].tolist()
    ids = zip(scene_ids, scan_ids)
    lr_subvols_ndx = [ndx for ndx, id in enumerate(ids) if id in lr_ids_list]
    out_subvols = len(lr_subvols_ndx)

    with h5py.File(args.out_path, 'w') as outf:
        create_datasets(outf, out_subvols, subvol_size, num_nearest_images)

        for out_ndx, ndx in enumerate(tqdm(lr_subvols_ndx)):
            for key in copy_keys:
                outf[key][out_ndx] = f[key][ndx]
    
    f.close()
        
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('in_path', help='Path to input h5 file (full)')
    p.add_argument('list_path', help='Path to LR list.txt')
    p.add_argument('out_path', help='Path to output h5 file (LR)')

    args = p.parse_args()

    main(args)