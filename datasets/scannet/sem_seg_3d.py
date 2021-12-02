'''
3D semantic segmentation on ScanNet occupancy voxel grids
'''
from datasets.scannet.common import load_ply
import random
import os
from pathlib import Path
from collections import OrderedDict

import h5py

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.scannet.common import nyu40_to_continuous, read_label_mapping, read_list, get_scene_scan_ids
from transforms.grid_3d import pad_volume

def collate_func(sample_list):
    return {
        # 'path': [s['path'] for s in sample_list],
        'x': torch.Tensor([s['x'] for s in sample_list]),
        'y': torch.LongTensor([s['y'] for s in sample_list]),
    }

class ScanNetOccGridH5(Dataset):
    '''
    Read x,y samples from a h5 file
    '''
    def __init__(self, cfg, split, transform=None):
        '''
        cfg: train_cfg['data']
        split: train/val/test
        transform: callable Object
        '''
        self.data = None
        self.transform = transform
        self.file_path = cfg[f'{split}_file'] 
        self.split = split
        
        # get the length once
        with h5py.File(self.file_path, 'r') as f:
            self.length = len(f['x'])

    def __len__(self):
        return self.length

    def __getitem__(self, ndx):
        # open once in each worker, allow multiproc
        if self.data is None:
            self.data = h5py.File(self.file_path, 'r') 

        x, y = self.data['x'][ndx], self.data['y'][ndx]

        sample = {'x': x, 'y': y}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __del__(self):
        if self.data is not None:
            self.data.close()

class ScanNet2D3DH5(ScanNetOccGridH5):
    '''
    Read 3D+2D dataset from file (like 3DMV)
    '''
    def __init__(self, cfg, split, transform=None):
        super().__init__(cfg, split, transform)
        self.num_images = cfg['num_nearest_images']

        self.lr_subvols_ndx = None
        self.target_padding = cfg['target_padding']
        self.labeled_samples_ndx = None
        
        # keep labels only for a subset scenes
        if self.split == 'train' and 'filter_train_label' in cfg:
            # read list of scenes
            lr_list = read_list(cfg['filter_train_label'])
            lr_ids_list = list(map(get_scene_scan_ids, lr_list))

            # open the file once in each worker, allow multiproc
            if self.data is None:
                self.data = h5py.File(self.file_path, 'r')

            # create tuples
            scene_ids = self.data['scene_id'][:].tolist()
            scan_ids = self.data['scan_id'][:].tolist()
            ids = zip(scene_ids, scan_ids)
            # get the indices of the samples where labels should be kept
            self.labeled_samples_ndx = [ndx for ndx, id in enumerate(ids) if id in lr_ids_list]

            print(f'Keeping 3D labels for {len(self.labeled_samples_ndx)} train samples')
        
    @staticmethod
    def collate_func(samples):
        # get the key in the first sample
        have_keys = set(samples[0].keys())
        # set only the keys which are there in the sample
        floats = list(set(('x', 'world_to_grid')).intersection(have_keys))
        ints = list(set(('y', 'frames', 'scene_id', 'scan_id', 'has_label')).intersection(have_keys))
        stack_tensors = list(set(('depths', 'rgbs', 'poses', 'labels2d')).intersection(have_keys))

        batch = {}

        # these are np arrays, create a tensor from the list of arrays
        for key in floats:
            batch[key] = torch.Tensor([s[key] for s in samples]) 
        for key in ints:
            batch[key] = torch.LongTensor([s[key] for s in samples])             
        # these are already tensors, stack them
        for key in stack_tensors:
            batch[key] = torch.stack([s[key] for s in samples])

        return batch

    def __getitem__(self, ndx):
        # open once in each worker, allow multiproc
        if self.data is None:
            self.data = h5py.File(self.file_path, 'r') 

        keys = 'x', 'y', 'world_to_grid', 'frames', 'scene_id', 'scan_id'

        sample = {key: self.data[key][ndx] for key in keys} 
        # keep only the required frames
        sample['frames'] = sample['frames'][:self.num_images]

        # by default, the sample has a label y
        sample['has_label'] = 1

        if self.labeled_samples_ndx is not None and ndx not in self.labeled_samples_ndx:
            sample['y'].fill(self.target_padding)
            # sample has no label
            sample['has_label'] = 0

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

class ScanNetSemSegOccGrid(Dataset):
    '''
    ScanNet 3d semantic segmentation on voxel grid

    x: (l, b, h) binary grid
    labels: 0-19 (20 classes) + target padding/ignore label 

    labels as given here here: http://kaldir.vc.in.tum.de/scannet_benchmark/
    '''
    def __init__(self, cfg, transform=None, split=None, full_scene=False):
        '''
        data_cfg:
            see configs/occgrid_train.yml
            root: root of scannet dataset
            limit_scans: read only these many scans
            subvol_size: size of subvolumes to sample
            subvols_per_scene: sample these many subvolumes per scene
        transform: apply on each subvol
        split: name of the split, used to read the list from cfg
        full_scene: return the full scene
        '''
        self.root_dir = Path(cfg['root'])
        
        self.use_rgb = cfg.get('use_rgb', False)

        self.transform = transform
        self.full_scene = full_scene

        # sample subvolumes
        self.subvols_per_scene = cfg.get('subvols_per_scene', None)
        self.subvol_size = np.array(cfg.get('subvol_size', None))
        self.target_padding = cfg.get('target_padding', None)

        self.scannet_to_nyu40 = read_label_mapping(cfg['label_file'])

        self.num_classes = cfg.get('num_classes', 20)

        if split:
            # read train/val/test list
            self.scans = read_list(cfg[f'{split}_list'])
        else:
            self.scans = sorted(os.listdir(self.root_dir))

        if cfg.get('limit_scans', False):
            self.scans = self.scans[:cfg['limit_scans']]

        self.paths = self.get_paths()

    def get_paths(self):
        '''
        Paths to scene files - 1 file per scene
        '''
        paths = []
        for scan_id in self.scans:
            path = self.root_dir / scan_id / f'{scan_id}_occ_grid.pth'

            if path.exists():
                paths.append(path)

        return paths

    def __len__(self):
        # vols per scene * num scenes
        if self.full_scene:
            return len(self.paths)
        return self.subvols_per_scene * len(self.paths)

    def sample_subvol(self, x, y, return_start_ndx=False, min_occ=0.2, num_retries=20):
        '''
        x, y - volumes of the same size
        return_start_ndx: return the start index of the subvol within the 
                              whole scene grid
        '''
        # pad the input volume for these reasons
        # 1. if the volume is is smaller than the subvol size
        #    pad it along the required dimensions so that a proper subvol can be created
        # 2. need to learn the padding, which is needed later during inference
        # 3. augmentation
        # 
        # for 2+3 - apply padding to the whole scene, then sample subvolumes as usual
        # so that subvols on the edge of the scene get padding
        #
        # result: right padding = max(subvol size, padding required to reach subvol size)

        # the padding required for small scenes (left+right)
        small_scene_pad = self.subvol_size - x.shape
        small_scene_pad[small_scene_pad < 0] = 0

        # augmentation padding for all other scenes (left+right)
        aug_pad = self.subvol_size

        # final scene size
        pad = np.maximum(small_scene_pad, aug_pad)
        scene_size = np.array(x.shape) + pad
        # pad only on the right side, no need to change start_ndx then
        x = pad_volume(x, scene_size, pad_end='right')
        y = pad_volume(y, scene_size, pad_val=self.target_padding, pad_end='right')

        num_voxels = np.prod(self.subvol_size)

        # min frac of occupied voxels in the chunk
        current_min_occ = min_occ
        n_attempts = 0
        # now x, y are atleast the size of subvol in each dimension
        # sample subvols as usual
        while 1:
            # the max value at which the subvol index can start 
            max_start = np.array(x.shape) - self.subvol_size
            # pick an index between 0 and the max value along each dimension
            # add 1 to max_start because its exclusive
            start = np.random.randint((0, 0, 0), max_start + 1, dtype=np.uint16)
            end = start + self.subvol_size
            # extract the subvol from the full scene
            x_sub = x[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            y_sub = y[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

            occupied = (x_sub == 1).sum() / num_voxels
            
            # first check if atleast 50% of the subvol is occupied
            # then check class distribution
            # classes 0,1 = wall, floor
            # if: the subvol has only these 2 classes -> keep only 5% of such subvols
            # or: other classes with index >2? keep the subvol
            if (occupied >= current_min_occ) and \
                ((y_sub.max() == 1 and random.random() > 0.95) or (y_sub.max() > 1)):
                break
            n_attempts += 1
            if n_attempts % num_retries == 0:
                # try to get lesser occupancy
                current_min_occ -= 0.01
                # can get atleast this much 
                current_min_occ = max(current_min_occ, 0.01)

        retval = (x_sub, y_sub)
        
        if return_start_ndx:
            retval += (start,)
        
        return retval

    def get_scene_grid(self, scene_ndx):
        '''
        get the full scene at scene_ndx (1..N)
        return 
            x, y of same shape
            world to grid translation of the scene / None if not available
        '''
        path = self.paths[scene_ndx]
        # load the full scene
        data = torch.load(path)
        # labels are scannet IDs
        x, y_nyu = data['x'], data['y']
        # the translation to be applied to 0,0,0 
        # to get the center of the origin voxel
        t = -data['start_ndx']

        return x, y_nyu, t

    def __getitem__(self, ndx):
        if not self.full_scene:
            # get the scene ndx for this subvol 
            scene_ndx = ndx // self.subvols_per_scene
        else:
            scene_ndx = ndx
        
        path = self.paths[scene_ndx]

        x, y_nyu, translation = self.get_scene_grid(scene_ndx)

        # convert bool x to float
        x = x.astype(np.float32)
        # dont use int8 anywhere, avoid possible overflow with more than 128 classes
        y = nyu40_to_continuous(y_nyu, ignore_label=self.target_padding, 
                                num_classes=self.num_classes).astype(np.int16)
        if self.full_scene:
            xval, yval = x, y
        else:
            xval, yval = self.sample_subvol(x, y)

        sample = {'path': path, 'x': xval, 'y': yval}
        
        if translation is not None:
            sample['translation'] = translation
            
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

class ScanNetPLYDataset(ScanNetSemSegOccGrid):
    '''
    Read voxelized ScanNet PLY files which contain
    vertices, colors and labels

    Create dense grid containing these voxelized vertices and labels
    and sample subvolumes from it
    '''
    def get_paths(self):
        '''
        Paths to files to scene files - 1 file per scene
        '''
        paths = []
        for scan_id in self.scans:
            path = self.root_dir / scan_id / f'{scan_id}_voxelized.ply'

            if path.exists():
                paths.append(path)

        return paths

    def get_scene_grid(self, scene_ndx):
        '''
        get the full scene at scene_ndx (1..N)
        return 
            x, y of same shape
            world to grid translation of the scene / None if not available
        '''
        path = self.paths[scene_ndx]
        # load the full scene
        coords, rgb, labels = load_ply(path, read_label=True)
        coords = coords.astype(np.int32)
        # translate the points to start at 0
        t = coords.min(axis=0)
        coords_new = coords - t
        # integer coordinates, get the grid size from this
        grid_size = tuple(coords_new.max(axis=0).astype(np.int32) + 1)

        if self.use_rgb:
            # use RGB values as grid features
            x = np.zeros(grid_size + (3,), dtype=np.float32)
            x[coords_new[:, 0], coords_new[:, 1], coords_new[:, 2]] = rgb
        else:
            # binary occupancy grid
            x = np.zeros(grid_size, np.float32)
            x[coords_new[:, 0], coords_new[:, 1], coords_new[:, 2]] = 1

        # fill Y with negative ints
        y_nyu = np.ones(grid_size, dtype=np.int16) * -1
        y_nyu[coords_new[:, 0], coords_new[:, 1], coords_new[:, 2]] = labels

        translation = -t

        return x, y_nyu, translation 

class ScanNetGridTestSubvols:
    '''
    Take a full scannet scene, pad it to multiple of subvolumes
    Read all non overlapping subvolumes
    '''
    def __init__(self, scene, subvol_size, target_padding, transform=None):
        '''
        scene: a full_scene sample from the above dataset
        subvol_size: size of the subvolumes
        '''
        x = scene['x']
        y = scene['y']
        self.path = scene['path']
        self.subvol_size = subvol_size
        self.target_padding = target_padding

        self.transform = transform 

        # pad the scene on the right to reach nearest multiple of subvols
        padded_size = ((np.array(x.shape) // self.subvol_size) + 1) * self.subvol_size
        self.x = pad_volume(x, padded_size, pad_end='right')
        self.y = pad_volume(y, padded_size, self.target_padding, pad_end='right')

        # mapping from ndx to subvol slices
        self.mapping = OrderedDict()
        self.start_ndx = OrderedDict()
        ndx = 0

        # depth
        for k in range(0, self.x.shape[2], self.subvol_size[2]):
            # height
            for j in range(0, self.x.shape[1], self.subvol_size[1]):
                # width
                for i in range(0, self.x.shape[0], self.subvol_size[0]):
                    slice = np.s_[
                        i : i+self.subvol_size[0], 
                        j : j+self.subvol_size[1], 
                        k : k+self.subvol_size[2], 
                    ]
                    # if subvol is not occupied at all, discard
                    if (self.x[slice] == 1).sum() > 0:
                        self.mapping[ndx] = slice
                        self.start_ndx[ndx] = (i, j, k)
                        ndx += 1

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, ndx):
        slice = self.mapping[ndx]
        # index where the subvol starts
        start_ndx = np.array(self.start_ndx[ndx], dtype=np.uint16)
        
        sub_x = self.x[slice]
        sub_y = self.y[slice]

        sample = {'x': sub_x, 'y': sub_y, 'path': self.path, 'start_ndx': start_ndx}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample



