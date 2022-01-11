from collections import defaultdict
from tqdm import tqdm

import trimesh
import torch
import numpy as np

from datasets.scannet.common import create_color_palette

def save_preds(preds, val_set, out_path):
    '''
    scene_scan: sceneid, scanid
    preds: list of preds on batches
    val_loader: loader on val set
    '''
    palette = np.array(create_color_palette(), dtype=int)

    points_all, labels_all, colors_all = [], [], []

    # go through each pred
    print('Processing predictions ..')
    for pred_out, sample in tqdm(zip(preds, val_set), total=len(val_set)):
        # permute x and preds to WHD
        x = torch.Tensor(sample['x']).squeeze().permute(2, 1, 0)
        pred = pred_out.permute(2, 1, 0)
        # pick the occupied locations in X
        occupied = (x == 1)
        grid_coords = occupied.nonzero()
        # get XYZ->label in the grid
        labels = pred[occupied.nonzero(as_tuple=True)]
        w2g = torch.Tensor(sample['world_to_grid']).inverse()
        # homogenous coords, add 1 at the end
        grid_coords_homo = torch.cat([grid_coords, torch.ones(grid_coords.shape[0], 1)], dim=1)
        # map grid coords to world coords
        world_coords = (w2g @ grid_coords_homo.T).T[:, :-1]
        # map back to NYU40 labels
        # 0-39 -> 1-40
        labels_nyu = labels + 1
        # add to list of points
        points_all.append(world_coords.numpy())
        labels_all.append(labels_nyu.numpy())
        # get vertex colors
        colors = palette[labels_nyu]
        colors_all.append(colors)
    
    points_all = np.vstack(points_all)
    labels_all = np.concatenate(labels_all)
    colors_all = np.vstack(colors_all)

    # save list of points, labels and colors as PLY
    mesh = trimesh.Trimesh(vertices=points_all, vertex_colors=colors_all)
    mesh.export(out_path)

def get_dataset_split_scans(dataset):
    '''
    dataset: ScanNet2D3DH5 with 'scan_id' and 'scene_id' fields

    return: dict with key = (scene_id, scan_id): [list of sample indices for this scan]
    '''
    splits = defaultdict(list)
    print('Split dataset into scenes')
    for ndx, sample in enumerate(tqdm(dataset)):
        key = (sample['scene_id'], sample['scan_id'])
        splits[key].append(ndx)

    return splits


def gen_predictions(model, loader, ckpt_path):
    '''
    get predictions of a sem seg 3d model on a loader
    model: the model object
    loader: 3d/2d3d data loader
    ckpt_path: path to checkpoint
    '''
    # load the ckpt
    model.load_state_dict(torch.load(ckpt_path, map_location=model.device)['state_dict'])
    # eval mode
    model.eval()
    # store all preds
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            preds = model.common_step(batch)[0]
            all_preds.append(preds)

    return all_preds