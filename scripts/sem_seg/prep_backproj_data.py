'''
prepare 2d+3d dataset (like 3DMV)
3d: 32^3 subvolumes x and y sampled from dense grid
2d: indices of 5 nearest images to each subvolume
'''
import os, os.path as osp
import argparse
from pathlib import Path

from transforms.grid_3d import AddChannelDim, TransposeDims

import torch
from tqdm import tqdm
import h5py
import numpy as np

from lib.misc import read_config
from datasets.scannet.sem_seg_3d import ScanNetPLYDataset
from datasets.scannet.utils_3d import ProjectionHelper, adjust_intrinsic, load_depth, load_intrinsic, load_pose, make_intrinsic

def create_datasets(out_file, n_samples, subvol_size, num_nearest_images):
    '''
    create the datasets in the output hdf5 file

    out_file: h5py.File object opened in 'w' mode
    '''
    # input subvolume
    out_file.create_dataset('x', (n_samples, 1) + subvol_size, dtype=np.float32)
    # label subvolume
    out_file.create_dataset('y', (n_samples,) + subvol_size, dtype=np.int16)
    # id of the scene that the volume came from (0000, 0002 ..)
    out_file.create_dataset('scene_id', (n_samples,), dtype=np.uint16)
    # id of the scan within the scene: 00, 01, 02..
    out_file.create_dataset('scan_id', (n_samples,), dtype=np.uint8)
    # world to grid transformation for this subvolume
    out_file.create_dataset('world_to_grid', (n_samples, 4, 4), dtype=np.float32)
    # indices of the corresponding frames
    out_file.create_dataset('frames', (n_samples, num_nearest_images), dtype=np.int16)

def add_translation(transform, t):
    '''
    transform: existing transformation matrix
    t: additional translation
    
    add t to the existing translation
    '''
    new_transform = transform.copy()
    new_transform[:3, 3] += t
    return new_transform

def get_world_to_scene(voxel_size, translation):
    '''
    standard world to scene transformation matrix
    voxel_size: size of the grid voxels
    translation: translation of the scene from the origin
    '''
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

def get_scan_name(path):
    '''
    path: /mnt/data/scannet/scans/scene0673_05/scene0673_05_voxelized.ply
    output: scene0673_05
    '''
    return path.stem[:12]

def get_nearest_images(world_to_grid, num_nearest_imgs, scan_name, root_dir,
                        frame_skip, image_dims, projector):
    '''
    world_to_grid: location of the grid
    num_nearest_imgs: only 1 supported now
    scan_name: scene0000_00
    root_dir: dir containing all scans with RGB and depth
    frame_skip: number of frames to skip
    image_dims: dims of the image used for projection 
    projector: ProjectionHelper object
    '''
    if num_nearest_imgs != 1:
        raise NotImplementedError

    root = Path(root_dir)
    scan_dir = root / scan_name
    pose_dir = scan_dir / 'pose'
    depth_dir = scan_dir / 'depth'
    intrinsic_path = root / scan_name / 'intrinsic/intrinsic_color.txt'

    # set the intrinsic
    intrinsic = load_intrinsic(intrinsic_path)
    intrinsic = adjust_intrinsic(intrinsic, [1296, 968], image_dims)
    projector.update_intrinsic(intrinsic)

    # list all the camera poses
    all_pose_files = sorted(os.listdir(pose_dir), key=lambda f: int(osp.splitext(f)[0]))
    # indices into all_pose_files of poses considered
    pose_indices = range(0, len(all_pose_files), frame_skip)
    pose_files = [all_pose_files[ndx] for ndx in pose_indices]
    # ndx of the image where the coverage came from
    # 1 coverage int for each pose
    coverages = np.zeros(len(pose_files), dtype=np.uint16)

    world_to_grid = torch.Tensor(world_to_grid)

    # iterate over camera pose files
    for file_ndx, pose_fname in enumerate(tqdm(pose_files, leave=False, desc='pose')):
        # N.txt
        pose_path = pose_dir / pose_fname
        # just N
        ndx = Path(pose_fname).stem
        depth_path = depth_dir / f'{ndx}.png'
        # read pose and depth
        depth = torch.Tensor(load_depth(depth_path))
        pose = torch.Tensor(load_pose(pose_path))

        # store the coverage of each pose
        coverages[file_ndx] = projector.get_coverage(depth, pose, world_to_grid)
    
    # some image covers this subvol
    if coverages.max() > 0:
        # pick the image with max coverage N.txt
        nearest_pose = all_pose_files[pose_indices[np.argmax(coverages)]]
        # return its index N
        return int(osp.splitext(nearest_pose)[0])
    # nothing covers this subvol
    else:
        return None

def main(args):
    cfg = read_config(args.cfg_path)

    # get full scene grid, extract subvols later
    dataset = ScanNetPLYDataset(cfg['data'], split=args.split,
                                  full_scene=True)
    print(f'Dataset: {len(dataset)}')

    # create dir if it doesn't exist
    out_path = Path(args.out_path)
    out_path.parent.mkdir(exist_ok=True)

    subvols_per_scene = cfg['data']['subvols_per_scene']
    n_samples = len(dataset) * subvols_per_scene
    subvol_size = tuple(cfg['data']['subvol_size'])
    num_nearest_imgs = cfg['data']['num_nearest_images']
    img_size = cfg['data']['img_size']

    outfile = h5py.File(out_path, 'w')
    create_datasets(outfile, n_samples, subvol_size, num_nearest_imgs)

    data_ndx = 0

    # intrinsic of the color camera from scene0001_00
    intrinsic = make_intrinsic(1170.187988, 1170.187988, 647.75, 483.75)
    # adjust for smaller image size
    intrinsic = adjust_intrinsic(intrinsic, [1296, 968], cfg['data']['img_size'])
    projector = ProjectionHelper(intrinsic, 
                                cfg['data']['depth_min'], cfg['data']['depth_max'], 
                                img_size,
                                subvol_size,
                                cfg['data']['voxel_size'])

    # iterate over each scene, read it only once
    for _, scene in enumerate(tqdm(dataset, desc='scene')):
        scene_x, scene_y, path = scene['x'], scene['y'], scene['path']
        # translation to go from scaled world coords (original->voxelized)
        # to scene voxel coords
        scene_T = scene['translation']
        # eg: 0000, 01 (int)
        scene_id, scan_id = get_scene_scan_ids(path)
        # eg: scene0000_00 (str)
        scan_name = get_scan_name(path)
        world_to_scene = get_world_to_scene(cfg['data']['voxel_size'], scene_T)

        # sample N subvols from this scene
        for _ in tqdm(range(subvols_per_scene), leave=False, desc='subvol'):
            subvol_x, subvol_y, start_ndx = dataset.sample_subvol(scene_x, scene_y,
                                                    return_start_ndx=True)
            # need to subtract the start index from scene coords to get grid coords                                                    
            subvol_t = - start_ndx.astype(np.int16)                                                    
            # add the additional translation to scene transform                                                    
            world_to_grid = add_translation(world_to_scene, subvol_t)

            nearest_images = get_nearest_images(world_to_grid, 
                                                    num_nearest_imgs,
                                                    scan_name, cfg['data']['root'],
                                                    cfg['data']['frame_skip'],
                                                    img_size,
                                                    projector)
            
            if nearest_images is None:
                # discard this subvol
                continue

            # find 1 nearest image to this grid
            outfile['frames'][data_ndx] = nearest_images
            
            # transform the volumes to the format used while training
            x_final = AddChannelDim.apply(subvol_x)
            x_final, y_final = TransposeDims.apply(x_final, subvol_y)

            outfile['x'][data_ndx] = x_final
            outfile['y'][data_ndx] = y_final
            outfile['scene_id'][data_ndx] = scene_id
            outfile['scan_id'][data_ndx] = scan_id 
            outfile['world_to_grid'][data_ndx] = world_to_grid

            data_ndx += 1
    print(f'Good subvols: {data_ndx}')
    outfile.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cfg_path', help='Path to backproj_prep cfg')
    p.add_argument('split', help='Split to be used: train/val')
    p.add_argument('out_path', help='Path to output hdf5 file')
    args = p.parse_args()

    main(args)