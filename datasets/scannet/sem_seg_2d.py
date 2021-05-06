'''
2D semantic segmentation on ScanNet images
'''

from pathlib import Path
import os, os.path as osp

from torch.utils.data import Dataset


class ScanNetSemSeg2D(Dataset):
    '''
    ScanNet 2d semantic segmentation dataset

    images: RGB 
    labels: 0-20 

    labels as reported here: http://kaldir.vc.in.tum.de/scannet_benchmark/
    '''
    def __init__(self, root_dir, label_file, limit_scans=None):
        '''
        root_dir: contains scan dirs scene0000_00, etc
        img_size: resize images to this size
        label_file: path to scannetv2-labels.combined.tsv
        '''
        self.root_dir = Path(root_dir)
        self.rgb_paths = []
        self.label_paths = []

        scans = sorted(os.listdir(self.root_dir))
        if limit_scans:
            scans = scans[:limit_scans]

        for scan_id in scans:
            scan_dir = self.root_dir / scan_id
            color_dir = scan_dir / 'color'
            label_dir = scan_dir / 'label-filt'

            for rgb_fname in os.listdir(color_dir):
                ndx = Path(rgb_fname).stem

                rgb_path = color_dir / rgb_fname
                label_path = label_dir / f'{ndx}.png'

                if label_path.exists():
                    self.rgb_paths.append(rgb_path)
                    self.label_paths.append(label_path)

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        sample = {
            'rgb_path': self.rgb_paths[idx],
            'label_path': self.label_paths[idx]
        }

        return sample