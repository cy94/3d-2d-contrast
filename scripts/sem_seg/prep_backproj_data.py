'''
prepare 2d+3d dataset (like 3DMV)
3d: 32^3 subvolumes x and y sampled from dense grid
2d: indices of 5 nearest images to each subvolume
'''
from datasets.scannet.sem_seg_3d import ScanNetPLYDataset
from tqdm import tqdm
import argparse
from pathlib import Path
import h5py
from lib.misc import read_config
import numpy as np

def create_datasets(out_file, n_samples, subvol_size, num_nearest_images):
    '''
    create the datasets in the output hdf5 file

    out_file: h5py.File object opened in 'w' mode
    '''
    # input subvolume
    out_file.create_dataset('x', (n_samples,) + subvol_size, dtype=np.float32)
    # label subvolume
    out_file.create_dataset('y', (n_samples,) + subvol_size, dtype=np.int16)
    # id of the scene that the volume came from (0000, 0002 ..)
    out_file.create_dataset('scene_id', (n_samples,), dtype=np.uint16)
    # id of the scan within the scene: 00, 01, 02..
    out_file.create_dataset('scan_id', (n_samples,), dtype=np.uint8)
    # world to grid transformation for this subvolume
    out_file.create_dataset('world_to_grid', (n_samples, 4, 4), dtype=np.float32)
    # indices of the corresponding frames
    out_file.create_dataset('frames', (n_samples, 1), dtype=np.uint16)

def main(args):
    cfg = read_config(args.cfg_path)

    dataset = ScanNetPLYDataset(cfg['data'], split=args.split,
                                  full_scene=True)
    print(f'Dataset: {len(dataset)}')

    # create dir if it doesn't exist
    out_path = Path(args.out_path)
    out_path.parent.mkdir(exist_ok=True)

    subvols_per_scene = cfg['data']['subvols_per_scene']
    n_samples = len(dataset) * subvols_per_scene
    subvol_size = tuple(cfg['data']['subvol_size'])
    num_nearest_images = cfg['data']['num_nearest_images']

    outfile = h5py.File(out_path, 'w')
    create_datasets(outfile, n_samples, subvol_size, num_nearest_images)

    # iterate over each scene, read it only once
    data_ndx = 0

    for _, scene in enumerate(tqdm(dataset)):
        scene_x, scene_y, path = scene['x'], scene['y'], scene['path']
        scene_T = scene['translation']
        scene_id, scan_id = get_scene_scan_ids(path)
        world_to_scene = get_world_to_scene(cfg['data']['voxel_size'], scene_T)

        # sample N subvols from this scene
        for _ in tqdm(range(subvols_per_scene), leave=False):
            subvol_x, subvol_y, subvol_t = dataset.sample_subvol(scene_x, scene_y,
                                                    return_start_ndx=True)
            # add the location of the grid                                                    
            world_to_grid = add_translation(world_to_scene, subvol_t)

            outfile['x'][data_ndx] = subvol_x                                                    
            outfile['y'][data_ndx] = subvol_y
            outfile['scene_id'][data_ndx] = scene_id
            outfile['scan_id'][data_ndx] = scan_id 
            outfile['world_to_grid'][data_ndx] = world_to_grid
            
            outfile['frames'][data_ndx] = 1
            data_ndx += 1
            break 
        break

    outfile.close()

def add_translation(transform, t):
    new_transform = transform.copy()
    new_transform[:3, 3] += t
    return new_transform

def get_world_to_scene(voxel_size, translation):
    t = np.eye(4, dtype=np.float32) / voxel_size
    # insert the translation
    t[:3, 3] = translation
    t[3, 3] = 1

    return t

def get_scene_scan_ids(path):
    '''
    path: /mnt/data/scannet/scans/scene0673_05/scene0673_05_voxelized.ply
    output: 673, 05 (ints)
    '''
    return int(path.stem[5:9]), int(path.stem[10:12])

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cfg_path', help='Path to backproj_prep cfg')
    p.add_argument('split', help='Split to be used: train/val')
    p.add_argument('out_path', help='Path to output hdf5 file')
    args = p.parse_args()

    main(args)