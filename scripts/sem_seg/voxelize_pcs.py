'''
Input: ScanNet gt and label PLY files
Output: single voxelized PLY file
'''
import argparse
import os

import numpy as np

from tqdm import tqdm

from pathlib import Path

from MinkowskiEngine.utils import sparse_quantize
from plyfile import PlyData, PlyElement
from datasets.scannet.common import load_ply


def main(args):
    root = Path(args.scannet_dir)
    voxel_size = args.voxel_size
    print(f'Using voxel size: {voxel_size}')

    for scan_id in tqdm(sorted(os.listdir(root)), desc='scan'):
        scan_dir = root / scan_id

        input_file = f'{scan_id}_vh_clean_2.ply' 
        gt_file = f'{scan_id}_vh_clean_2.labels.ply' 

        _, rgb, _ = load_ply(scan_dir / input_file)
        # read coords and labels from GT file
        coords, _, labels = load_ply(scan_dir / gt_file, read_label=True)

        coords_vox, rgb_vox, labels_vox = sparse_quantize(coords, rgb, labels, 
                                                quantization_size=voxel_size,
                                                device='cuda')
        arr = np.array([tuple(coords_vox[i]) + tuple(rgb_vox[i]) + (labels_vox[i],) \
                        for i in range(len(coords_vox))], 
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                                ('red', 'f4'), ('green', 'f4'), ('blue', 'f4'), 
                                ('label', 'i4')])
        elem = PlyElement.describe(arr, 'vertex')

        out_file = scan_dir / f'{scan_id}_voxelized.ply'
        PlyData([elem]).write(out_file)                                                               


if __name__ == '__main__':
    # params
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('scannet_dir', help='path to scannet root dir file to read')
    parser.add_argument('--voxel-size', type=float, dest='voxel_size', default=0.05)

    args = parser.parse_args()

    main(args)