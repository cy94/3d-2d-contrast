'''
prepare 2d+3d dataset (like 3DMV)
3d: 32^3 subvolumes x and y sampled from dense grid
2d: indices of 5 nearest images to each subvolume
'''


import os, os.path as osp
import argparse
from pathlib import Path
from functools import partial

from torch.multiprocessing import Pool

import torch
from tqdm import tqdm
import h5py
import numpy as np

from lib.misc import read_config
from datasets.scannet.sem_seg_3d import ScanNetGridTestSubvols, ScanNetPLYDataset, ScanNetSemSegOccGrid
from datasets.scannet.utils_3d import ProjectionHelper, adjust_intrinsic, \
    load_depth_multiple, load_intrinsic, load_pose_multiple, make_intrinsic


# number of processes to prepare data
N_PROC = 8
# number of samples handled at a time
CHUNK_SIZE = 16

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

def get_nearest_images(world_to_grid, poses, depths, num_nearest_imgs, projector, 
                        ):
    '''
    world_to_grid: location of the grid
    num_nearest_imgs: only 1 supported now
    projector: ProjectionHelper object
    device: run on cuda/cpu
    '''
    projections = []

    for pose, depth in zip(poses, depths):
        projection = projector.compute_projection(depth, pose, world_to_grid)
        projections.append(projection)

    # initially none of the voxels are covered
    covered_voxels = set()
    # initially no frames selected
    frames = []

    for _ in range(num_nearest_imgs):
        current_coverages = np.zeros(len(projections), dtype=np.int16)

        # get the number of extra voxels that each unselected pose adds
        for pose_ndx, projection in enumerate(projections):
            # should have a valid projection, not be selected already
            if projection is not None and pose_ndx not in frames:
                ind3d, _ = projection
                num_ind = ind3d[0]
                covered_by_this_pose = set(ind3d[1:1+num_ind].tolist())
                newly_covered = len(covered_by_this_pose - covered_voxels)
                current_coverages[pose_ndx] = newly_covered
        
        # find the current best pose
        best_pose = np.argmax(current_coverages)
        if (projections[best_pose] is not None) and (current_coverages[best_pose] > 0):
            # add to frames
            frames.append(best_pose)
            # update voxels covered so far
            ind3d, _ = projections[best_pose]
            num_ind = ind3d[0]
            covered_by_this_pose = set(ind3d[1:1+num_ind].tolist())
            covered_voxels = covered_voxels.union(covered_by_this_pose)

    # some image covers this subvol
    if len(frames) > 0:
        return np.array(frames, dtype=np.int16)
    # nothing covers this subvol
    else:
        # -1 for each index
        return -np.ones(num_nearest_imgs, dtype=np.int16)

def main(args):
    cfg = read_config(args.cfg_path)
    
    device = torch.device('cuda' if args.gpu else 'cpu')
    print('Using device:', device)

    # get full scene grid, extract subvols later
    dataset = ScanNetSemSegOccGrid(cfg['data'], split=args.split, full_scene=True)
    print(f'Dataset: {len(dataset)}')

    # create dir if it doesn't exist
    out_path = Path(args.out_path)
    out_path.parent.mkdir(exist_ok=True)

    # subvol dims W, H, D
    subvol_size = tuple(cfg['data']['subvol_size'])

    if args.full_scene:
        print('Get total number of subvols in val/test set')
        # actual number of subvols in each scene
        padval = cfg['data']['target_padding']
        subvols_per_scene_list = [len(ScanNetGridTestSubvols(scene, subvol_size, padval)) \
                             for scene in tqdm(dataset)]
        n_samples = sum(subvols_per_scene_list)
    else:
        subvols_per_scene = cfg['data']['subvols_per_scene']
        # total number of subvols
        n_samples = len(dataset) * subvols_per_scene

    print(f'Total subvols in output file: {n_samples}')

    # images per subvol 
    num_nearest_imgs = cfg['data']['num_nearest_images']
    # W, H size of the image that looks at the subvol
    img_size = cfg['data']['img_size']

    outfile = h5py.File(out_path, 'w')
    # create the datasets in the h5 file
    create_datasets(outfile, n_samples, subvol_size, num_nearest_imgs)

    data_ndx = 0

    # intrinsic of the color camera from scene0001_00
    # start with this intrinsic which can work for all scenes (approx)
    # later adjust the instrinsic if needed
    intrinsic = make_intrinsic(1170.187988, 1170.187988, 647.75, 483.75)
    # adjust for smaller image size
    intrinsic = adjust_intrinsic(intrinsic, [1296, 968], cfg['data']['img_size'])
    projector = ProjectionHelper(intrinsic, 
                                cfg['data']['depth_min'], cfg['data']['depth_max'], 
                                img_size,
                                subvol_size,
                                cfg['data']['voxel_size']).to(device)

    # number of subvols to compute projection in parallel
    if args.full_scene:
        batch_size = N_PROC * CHUNK_SIZE
    else:
        batch_size = min(N_PROC * CHUNK_SIZE, subvols_per_scene)

    # store subvols and w2g in tensors, distribute to processes    
    subvol_x_batch = np.empty((batch_size,) + subvol_size, dtype=np.float32)
    subvol_y_batch = np.empty((batch_size,) + subvol_size, dtype=np.int16)
    world_to_grid_batch = np.empty((batch_size,) + (4,4), dtype=np.float32)

    bad_subvols = 0

    # iterate over each scene, read it only once
    for scene_ndx, scene in enumerate(tqdm(dataset, desc='scene')):
        scene_x, scene_y, path = scene['x'], scene['y'], scene['path']
        # translation to go from scaled world coords (original->voxelized)
        # to scene voxel coords
        scene_T = scene['translation']
        # eg: 0000, 01 (int)
        scene_id, scan_id = get_scene_scan_ids(path)
        # eg: scene0000_00 (str)
        scan_name = get_scan_name(path)
        # the scaling and translation applied during voxelization
        world_to_scene = get_world_to_scene(cfg['data']['voxel_size'], scene_T)

        # load poses, depths, set intrinsic 
        root = Path(cfg['data']['root'])
        scan_dir = root / scan_name
        pose_dir = scan_dir / 'pose'
        depth_dir = scan_dir / 'depth'
        intrinsic_path = root / scan_name / 'intrinsic/intrinsic_color.txt'

        # set the intrinsic -> once per scene
        intrinsic = load_intrinsic(intrinsic_path)
        intrinsic = adjust_intrinsic(intrinsic, [1296, 968], img_size)
        projector.update_intrinsic(intrinsic)

        # list all the camera poses
        all_pose_files = sorted(os.listdir(pose_dir), key=lambda f: int(osp.splitext(f)[0]))
        # indices into all_pose_files of poses considered
        pose_indices = range(0, len(all_pose_files), cfg['data']['frame_skip'])
        pose_files = [all_pose_files[ndx] for ndx in pose_indices]
        pose_file_ndxs = [int(osp.splitext(f)[0]) for f in pose_files]

        pose_paths = (pose_dir / f for f in pose_files)
        depth_paths = (depth_dir / f'{Path(f).stem}.png' for f in pose_files)

        # load all depth images once
        # tensor contains H, W images -> conv2d convention
        depths = torch.empty(len(pose_files), img_size[1], img_size[0])
        poses = torch.empty(len(pose_files), 4, 4)

        load_depth_multiple(depth_paths, img_size, depths)
        depths = depths.to(device)
        load_pose_multiple(pose_paths, poses)
        poses = poses.to(device)

        subvols_found = 0

        # actual number of subvols in this scene, need to get all of them
        if args.full_scene:
            # create the test subvols dataset once
            subvols_dataset = iter(ScanNetGridTestSubvols(scene, subvol_size, padval))
            subvols_per_scene = subvols_per_scene_list[scene_ndx]

        pbar = tqdm(total=subvols_per_scene, desc='subvol', leave=False)

        while subvols_found < subvols_per_scene:
            # full scene -> pick all the subvols
            if args.full_scene:
                if subvols_per_scene_list[scene_ndx] < batch_size:
                    # only 1 small batch
                    num_in_batch = subvols_per_scene_list[scene_ndx]
                else:
                    # not the first batch - min(bsize, whatever is left)
                    num_in_batch = min(batch_size, subvols_per_scene_list[scene_ndx] - subvols_found)
            # randomly sample a batch of subvols
            else:
                num_in_batch = batch_size
            
            for ndx in tqdm(range(num_in_batch), desc='sample_subvol', leave=False):
                if args.full_scene:
                    # full scene dataset? take the next subvol
                    sample = next(subvols_dataset)
                    subvol_x, subvol_y, start_ndx = sample['x'], sample['y'], \
                                                    sample['start_ndx']
                else:
                    # randomly sample a subvol
                    subvol_x, subvol_y, start_ndx = dataset.sample_subvol(scene_x, scene_y,
                                                            return_start_ndx=True)
                # need to subtract the start index from scene coords to get grid coords                                                    
                subvol_t = - start_ndx.astype(np.int16)                                                    
                # add the additional translation to scene transform                                                    
                world_to_grid = add_translation(world_to_scene, subvol_t)
                # store everything in the batch tensor
                subvol_x_batch[ndx] = subvol_x
                subvol_y_batch[ndx] = subvol_y
                world_to_grid_batch[ndx] = world_to_grid

            # make a tensor, used for projection
            # handle the case when actual subvols < batch_size, take only a subset
            world_to_grid_batch_tensor = torch.Tensor(world_to_grid_batch[:num_in_batch]).to(device)

            # projection expects origin of chunk in a corner
            # but w2g is wrt center of the chunk -> add 16 to its "grid coords" 
            # to get the required grid indices
            # ie 0,0,0 becomes 16,16,16
            # add an additional translation to existing one 
            t = torch.eye(4).to(device)
            t[:3, -1] = torch.Tensor(subvol_size) / 2
            world_to_grid_batch_tensor = t @ world_to_grid_batch_tensor

            # compute projection for the whole batch in parallel
            task_func = partial(get_nearest_images, poses=poses, depths=depths,
                                    num_nearest_imgs=num_nearest_imgs,
                                    projector=projector) 

            nearest_imgs_all = []

            if args.multiproc:
                with Pool(processes=N_PROC) as pool:
                    for nearest_imgs in tqdm(pool.imap(task_func, world_to_grid_batch_tensor,
                                                        CHUNK_SIZE),
                                        total=len(world_to_grid_batch_tensor),
                                        leave=False, desc='projection'
                                    ):
                        nearest_imgs_all.append(nearest_imgs)
            else:
                for world_to_grid_tensor in tqdm(world_to_grid_batch_tensor, 
                            desc='projection', leave=False):
                    nearest_imgs = get_nearest_images(world_to_grid_tensor,
                                        poses, depths, num_nearest_imgs, projector)
                    nearest_imgs_all.append(nearest_imgs)
            
            good_in_batch = 0

            # check the valid ones and save them to file
            # full scene - save all to file, -1 as nearest image
            for ndx, nearest_imgs in enumerate(nearest_imgs_all):
                # not full scene, and didnt get coverage -> discard
                if (-1 in nearest_imgs) and (not args.full_scene):
                    bad_subvols += 1
                else:
                    # update the number found
                    # full_scene -> every subvol is a "good" one
                    good_in_batch += 1
                    subvols_found += 1
                    
                    # full scene: insert -1 as the pose
                    # didnt get coverage -> insert -1 as it is 
                    if -1 in nearest_imgs:
                        nearest_poses = -np.ones(num_nearest_imgs, dtype=np.int16)
                    else:
                        # nearest imgs is a list of ints
                        # its length is not necessarily num_nearest_imgs
                        # -> rest should be -1
                        # get the corresponding pose for each of these 
                        nearest_poses = -np.ones(num_nearest_imgs, dtype=np.int16)
                        for i, pose_ndx in enumerate(nearest_imgs):
                            # get the "N".txt number
                            nearest_poses[i] = pose_file_ndxs[pose_ndx]
                
                    # find nearest images to this grid
                    outfile['frames'][data_ndx] = nearest_poses
                    outfile['x'][data_ndx] = subvol_x_batch[ndx]
                    outfile['y'][data_ndx] = subvol_y_batch[ndx]
                    outfile['scene_id'][data_ndx] = scene_id
                    outfile['scan_id'][data_ndx] = scan_id 
                    outfile['world_to_grid'][data_ndx] = world_to_grid_batch[ndx]
                    # update ndx into the file
                    data_ndx += 1

                    # have enough, dont write here onwards to file
                    if subvols_found == subvols_per_scene:
                        break
            
            pbar.update(good_in_batch)
        pbar.close()

    print('Good subvols:', data_ndx)
    print('Bad subvols:', bad_subvols)
    
    outfile.close()

if __name__ == '__main__':
    from torch.multiprocessing import set_start_method
    set_start_method('spawn')

    p = argparse.ArgumentParser()
    p.add_argument('cfg_path', help='Path to backproj_prep cfg')
    p.add_argument('split', help='Split to be used: train/val')
    p.add_argument('out_path', help='Path to output hdf5 file')
    p.add_argument('--multiproc', action='store_true', default=False,
                    dest='multiproc', help='Use multiprocessing?')
    p.add_argument('--gpu', action='store_true', default=False,
                    dest='gpu', help='Use GPU?')
    p.add_argument('--full-scene', action='store_true', default=False,
                    dest='full_scene', help='Sliding window of subvols over the whole scene')
    args = p.parse_args()

    main(args)