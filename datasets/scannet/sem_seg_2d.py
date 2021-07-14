'''
2D semantic segmentation on ScanNet images
'''

from datasets.scannet.common import read_list
from pathlib import Path
import os, os.path as osp

import numpy as np
import imageio

import torch
from torch.utils.data import Dataset

from datasets.scannet.common import read_label_mapping, map_labels, nyu40_to_continuous


def collate_func(sample_list):
    return {
        'img_path': [s['img_path'] for s in sample_list],
        'label_path': [s['label_path'] for s in sample_list],
        'x': torch.Tensor([s['img'] for s in sample_list]),
        'y': torch.LongTensor([s['label'] for s in sample_list]),
    }

class ScanNetSemSeg2D(Dataset):
    '''
    ScanNet 2d semantic segmentation dataset

    images: RGB 
    labels: 0-19 (20 classes) + 20 (ignored classes) 

    labels as reported here: http://kaldir.vc.in.tum.de/scannet_benchmark/
    '''
    def __init__(self, cfg, split, transform=None):
        '''
        cfg: the full train cfg
        split: train/val/None (all)
        transform: apply on each sample
        '''
        self.root_dir = Path(cfg['data']['root'])
        # store data paths
        self.img_paths = []
        self.label_paths = []

        self.transform = transform
        self.frame_skip = cfg['data'].get('frame_skip', 1)
        self.scannet_to_nyu40 = read_label_mapping(cfg['data']['label_file'])

        self.num_classes = cfg['data'].get('num_classes', 20)

        # use a fixed set of scans
        if split:
            scans = read_list(cfg['data'][f'{split}_list'])
        else:
        # use all scans in this dir
            scans = sorted(os.listdir(self.root_dir))

        self.ignore_label = cfg['data']['ignore_label']

        for scan_id in scans:
            scan_dir = self.root_dir / scan_id
            color_dir = scan_dir / 'color'
            label_dir = scan_dir / 'label-filt'

            # sort color files by ndx - 0,1,2.jpg ...
            color_files = sorted(os.listdir(color_dir), key=lambda f: int(osp.splitext(f)[0]))
            # skip frames?
            for file_ndx in range(0, len(color_files), self.frame_skip):
                # N.jpg
                img_fname = color_files[file_ndx]
                # just N
                ndx = Path(img_fname).stem
                # full image path
                img_path = color_dir / img_fname
                # full label path
                label_path = label_dir / f'{ndx}.png'
                # use this sample only if label exists
                if label_path.exists():
                    self.img_paths.append(img_path)
                    self.label_paths.append(label_path)

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path, label_path = self.img_paths[idx], self.label_paths[idx]
        # read the image as float
        img = np.array(imageio.imread(img_path)).astype(np.float32)
        # read the scannet label image as int
        label_scannet = np.array(imageio.imread(label_path))
        # map from scannet to nyu40 labels
        label_nyu40 = map_labels(label_scannet, self.scannet_to_nyu40)
        # map from NYU40 to 0-19 + 20 (ignored) labels
        label_selected = nyu40_to_continuous(label_nyu40, ignore_label=self.ignore_label, 
                                            num_classes=self.num_classes)

        sample = {
            'img_path': img_path,
            'label_path': label_path,
            'img': img,
            'label': label_selected
        }

        if self.transform is not None:
            sample = self.transform(sample) 

        return sample