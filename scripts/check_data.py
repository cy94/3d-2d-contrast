'''
Check scans for which depth data, etc has completely been extracted
color: jpg files
depth: png files
'''


import argparse
import os, os.path as osp
from pathlib import Path
from tqdm import tqdm


def main(args):
    root = Path(args.scannet_dir)

    for scan_ndx, scan_id in enumerate(tqdm(sorted(os.listdir(root)))):
        scan_dir = root / scan_id
        color_dir = scan_dir / 'color'
        depth_dir = scan_dir / 'depth'
        color_files = sorted(os.listdir(color_dir), key=lambda f: int(osp.splitext(f)[0]))

        for color_file in tqdm(color_files, leave=False):
                # just N
                ndx = Path(color_file).stem
                # full label path
                depth_path = depth_dir / f'{ndx}.png'

                if not depth_path.exists():
                    print(f'{depth_path} missing, scan ndx: {scan_ndx}')
                    return

if __name__ == '__main__':
    # params
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('scannet_dir', help='path to sens file to read')

    args = parser.parse_args()

    main(args)