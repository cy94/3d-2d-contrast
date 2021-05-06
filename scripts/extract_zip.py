import argparse
import os, os.path as osp
from pathlib import Path

from tqdm import tqdm 
import subprocess

ALLOWED_SUFFIXES = ('_2d-label-filt.zip', '_2d-label.zip', 
                    '_2d-instance-filt.zip', '_2d-instance.zip')

def main(args):
    root = Path(args.scannet_dir)
    suffix = args.suffix

    assert suffix in ALLOWED_SUFFIXES

    for scan_id in tqdm(os.listdir(root)):
        in_file = root / scan_id / f'{scan_id}{suffix}'
        
        cmd = ['unzip', '-n', str(in_file), '-d', str(root / scan_id)]
        subprocess.run(cmd, stdout=subprocess.DEVNULL)



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('scannet_dir', help='Path to ScanNet root')
    p.add_argument('suffix', help='Suffix of the filename. Options: _2d-label-filt.zip')
    args = p.parse_args()

    main(args)