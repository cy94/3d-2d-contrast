'''
Input: ScanNet gt and label PLY files
Output: GT Binary occupancy voxel grid and label grid
'''
import argparse
import torch

import numpy as np
from scipy.spatial.distance import cdist

from tqdm import tqdm

import os, os.path as osp
from pathlib import Path

import trimesh
from trimesh.exchange.ply import parse_header, ply_binary

from datasets.scannet.utils import VALID_CLASSES

def read_gt(gt_path):
    '''
    get vertices, rgb and labels from scannet GT PLY file such as 
    {scan_id}_vh_clean_2.labels.ply

    vertices: n, 3
    rgb: n, 3
    labels: n,
    '''
    with open(gt_path, 'rb') as f:
        elements, _, _ = parse_header(f)
        ply_binary(elements, f)
        
    labels = elements['vertex']['data']['label']

    x, y, z = elements['vertex']['data']['x'], elements['vertex']['data']['y'], \
                elements['vertex']['data']['z']
    r, g, b = elements['vertex']['data']['red'], elements['vertex']['data']['green'], \
                elements['vertex']['data']['blue']

    gt_vertices = np.stack((x, y, z), axis=-1)
    gt_rgb = np.stack((r, g, b), axis=-1)

    return gt_vertices, gt_rgb, labels

def get_label_grid(input_grid, gt_vertices, gt_vtx_labels, voxel_size=None, method='nearest'):
    '''
    input_grid:  the input trimesh.VoxelGrid (l, h, b)
    gt_vertices: (n, 3) vertices of the GT mesh 
    gt_vtx_labels: (n, ) labels of these vertices

    return: (l, h, b) array of labels for each grid cell
    '''
    centers = input_grid.points
    indices = input_grid.points_to_indices(centers)
    pairs = list(zip(centers, indices))
    label_grid = np.zeros_like(input_grid.matrix, dtype=np.uint8)

    for center, ndx in tqdm(pairs, leave=False, desc='nearest_point'):
        if method == 'nearest':
            # distance from this voxel center to all vertices
            dist = cdist(np.expand_dims(center, 0), gt_vertices).flatten()
            # closest vertex
            closest_vtx_ndx = dist.argmin()
            # label of this vertex
            label = gt_vtx_labels[closest_vtx_ndx]
        elif method == 'voting':
            # find indices all vertices within this voxel
            low, high = center - voxel_size, center + voxel_size
            vtx_in_voxel = np.all(np.logical_and((gt_vertices >= low), (gt_vertices <= high)), axis=1)
            # labels of these vertices
            labels = gt_vtx_labels[vtx_in_voxel]
            # most common label
            try:
                label = np.bincount(labels).argmax()
            except ValueError:
                label = None
        
        # assign to label and color grid
        if label is not None and label in VALID_CLASSES:
            label_grid[ndx[0], ndx[1], ndx[2]] = label

    return label_grid

def main(args):
    root = Path(args.scannet_dir)
    voxel_size = args.voxel_size
    print(f'Using voxel size: {voxel_size}')

    for scan_id in tqdm(sorted(os.listdir(root))[:5], desc='scan'):
        scan_dir = root / scan_id

        input_file = f'{scan_id}_vh_clean_2.ply' 
        gt_file = f'{scan_id}_vh_clean_2.labels.ply' 

        # read input mesh and voxelize
        input_mesh = trimesh.load(scan_dir / input_file)
        input_grid = input_mesh.voxelized(pitch=voxel_size) 
        
        # read GT mesh, get vertex coordinates and labels
        gt_vertices, _, labels = read_gt(scan_dir / gt_file)
        gt_vtx_labels = np.array([l if l in VALID_CLASSES else 0 for l in labels.tolist()], 
                                dtype=np.uint8)

        label_grid = get_label_grid(input_grid, gt_vertices, gt_vtx_labels)

        x, y = input_grid.matrix, label_grid
        out_file = f'{scan_id}_occ_grid.pth'

        data = {'x': x, 'y': y}
        torch.save(data, scan_dir / out_file)

if __name__ == '__main__':
    # params
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('scannet_dir', help='path to scannet root dir file to read')
    parser.add_argument('--voxel-size', type=float, dest='voxel_size', default=0.05)

    args = parser.parse_args()

    main(args)