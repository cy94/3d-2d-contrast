'''
2D semantic segmentation on ScanNet images
'''

from pathlib import Path
import os, os.path as osp

import numpy as np
import imageio

import torch
from torch.utils.data import Dataset

from datasets.scannet.utils import read_label_mapping, map_labels


def collate_func(sample_list):
    return {
        'img_path': [s['img_path'] for s in sample_list],
        'label_path': [s['label_path'] for s in sample_list],
        'img': torch.Tensor([s['img'] for s in sample_list]),
        'label': torch.Tensor([s['label'] for s in sample_list]),
    }

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
        self.img_paths = []
        self.label_paths = []

        self.mapping = read_label_mapping(label_file)

        scans = sorted(os.listdir(self.root_dir))
        if limit_scans:
            scans = scans[:limit_scans]

        for scan_id in scans:
            scan_dir = self.root_dir / scan_id
            color_dir = scan_dir / 'color'
            label_dir = scan_dir / 'label-filt'

            for img_fname in os.listdir(color_dir):
                ndx = Path(img_fname).stem

                img_path = color_dir / img_fname
                label_path = label_dir / f'{ndx}.png'

                if label_path.exists():
                    self.img_paths.append(img_path)
                    self.label_paths.append(label_path)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label_path = self.img_paths[idx], self.label_paths[idx]
        
        img = np.array(imageio.imread(img_path))
        
        label_scannet = np.array(imageio.imread(label_path))
        label_nyu40 = map_labels(label_scannet, self.mapping)
        # TODO: labels from 0-20
        label_selected = label_nyu40

        sample = {
            'img_path': img_path,
            'label_path': label_path,
            'img': img,
            'label': label_selected
        }

        return sample