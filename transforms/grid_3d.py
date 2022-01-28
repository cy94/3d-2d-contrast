from datasets.scannet.common import map_labels, nyu40_to_continuous, read_label_mapping
import imageio
import cv2
from datasets.scannet.utils_3d import load_depth_multiple, load_pose_multiple, load_rgbs_multiple
from pathlib import Path
from abc import ABC
from copy import deepcopy

import numpy as np

import torch

def pad_volume(vol, size, pad_val=-100, pad_end=False):
    '''
    vol: (l, b, h) array
    size: (3,) array
    pad_val: value to pad
    '''
    # how much to pad
    diff = size - np.array(vol.shape)
    # left and right padding for 3 dims (ie l/r, front/back, top/bottom)
    # pad only at right end
    if pad_end == 'right':
        padvals = (np.zeros(3), diff)
    # pad only at left end
    elif pad_end == 'left':
        padvals = (diff, np.zeros(3))
    # pad half at each end
    else:
        padvals = (np.floor(diff/2), np.ceil(diff/2))

    pad = np.stack(padvals, axis=-1).astype(np.uint8).tolist()
    padded = np.pad(vol, pad, constant_values=pad_val)

    return padded

class JitterOccupancy:
    '''
    Jitter the occupancy grid -> set a few 1s to 0s, few 0s to 1s randomly
    '''
    def __init__(self, prob=0.05):
        # change 0->1 and 1-> with this probability
        self.prob = prob

    def __call__(self, sample):
        # change only x
        occupied = (sample['x'] == 1)
        empty = (sample['x'] == 0)

        rnd = np.random.rand(*sample['x'].shape)

        # change these locations
        change_locs = rnd < self.prob
        sample['x'][change_locs & occupied] = 0
        sample['x'][change_locs & empty] = 1

        return sample

class RandomPadding:
    '''
    Randomly change a few edge layers of the input to padding
    set x along the edges of the volume to 0, but dont change y
    no need to change w2g transformation
    '''
    def __init__(self, max_pad=3, pad_x=-100, pad_y=40):
        self.max_pad = max_pad
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.rng = np.random.default_rng()
    
    def __call__(self, sample):
        # how many slices of the input to change?
        # along each axis, each end = 3x2 = 6
        xl, xr, yl, yr, zl, zr = self.rng.integers(0, self.max_pad, 6, endpoint=True)
        slices = (
            np.s_[:xl, :, :],
            np.s_[-xr:, :, :],
            np.s_[:, :yl, :], 
            np.s_[:, -yr:, :],
            np.s_[:, :, :zl],
            np.s_[:, :, -zr:],
        )

        for slice in slices:
            sample['x'][slice] = self.pad_x
            sample['y'][slice] = self.pad_y

        return sample

class RandomInputZero:
    '''
    Randomly change a few edge layers of the input to padding
    set x along the edges of the volume to 0, but dont change y
    no need to change w2g transformation
    '''
    def __init__(self, max_pad=3, x_val=0, y_val=40):
        self.max_pad = max_pad
        self.x_val = x_val
        self.y_val = y_val
        self.rng = np.random.default_rng()
    
    def __call__(self, sample):
        # how many slices of the input to change?
        # along each axis, each end = 3x2 = 6
        xl, xr, yl, yr, zl, zr = self.rng.integers(0, self.max_pad, 6, endpoint=True)
        slices = (
            np.s_[:xl, :, :],
            np.s_[-xr:, :, :],
            np.s_[:, :yl, :], 
            np.s_[:, -yr:, :],
            np.s_[:, :, :zl],
            np.s_[:, :, -zr:],
        )

        for slice in slices:
            sample['x'][slice] = self.x_val
            sample['y'][slice] = self.y_val

        return sample

def get_rot_mat(num_rots):
    '''
    num_rots: number of rotations by np.rot90 in the direction X->Y axis, about the Z axis
    subvol_size: (W, H, D) size
    '''
    # rotate about the Z axis num_rots times
    rot90 = np.eye(4)
    rot90[0, 0] = 0
    rot90[1, 1] = 0
    rot90[0, 1] = -1
    rot90[1, 0] = 1

    rot_n = np.linalg.matrix_power(rot90, num_rots)

    return rot_n 


class RandomRotate:
    '''
    Randomly rotate the scene by 90, 180 or 270 degrees 
    '''
    def __init__(self, aug_w2g=False):
        self.rng = np.random.default_rng()
        self.aug_w2g = aug_w2g

        if self.aug_w2g:
            self.rot_mats = {n: get_rot_mat(n) for n in (0, 1, 2, 3)}

    def __call__(self, sample):
        '''
        sample with x and y
        rotate both of them
        '''
        # rotate 0, 1, 2 or 3 times
        num_rots = self.rng.integers(0, 3, endpoint=True)
        sample['x'] = np.rot90(sample['x'], k=num_rots)
        sample['y'] = np.rot90(sample['y'], k=num_rots)

        if self.aug_w2g:
            sample['world_to_grid'] = self.rot_mats[num_rots] @ sample['world_to_grid']
        return sample

class RandomTranslate:
    '''
    Randomly translate the whole scene
    '''
    def __init__(self, max_shift=(10, 10, 3)):
        self.max_shift = np.array(max_shift)
        self.rng = np.random.default_rng()

    def __call__(self, sample):
        new_sample = deepcopy(sample)
        # generate one shift
        shift = self.rng.integers(-self.max_shift, self.max_shift, endpoint=True)
        new_sample['coords'] = new_sample['coords'] + shift

        return new_sample

class JitterCoords:
    '''
    Jitter each coordinate
    '''
    def __init__(self, max_shift=(2, 2, 2)):
        self.max_shift = np.array(max_shift)
        self.rng = np.random.default_rng()
    
    def __call__(self, sample):
        new_sample = deepcopy(sample)

        num_points = len(new_sample['coords'])
        # generate one shift for each point
        shift = self.rng.integers(-self.max_shift, self.max_shift, (num_points, 3), endpoint=True)
        new_sample['coords'] = new_sample['coords'] + shift

        return new_sample

class DenseToSparse:
    '''
    Convert dense grid to sparse coords, features and labels
    '''
    def __init__(self):
        pass

    def __call__(self, sample):
        new_sample = deepcopy(sample)
        # coords of occupied grid cells
        locs = np.nonzero(new_sample['x'])
        x, y, z = locs
        coords = np.transpose(locs)

        # coords - N, 3
        new_sample['coords'] = coords
        # const feature for each of these cells
        new_sample['feats'] = np.ones((len(coords), 1))
        # pick the labels of these cells
        new_sample['labels'] = new_sample['y'][x, y, z]

        return new_sample

class MapClasses:
    '''
    Ignore the none class, set it to a different value
    '''
    def __init__(self, class_map):
        '''
        class_map: dict int -> int
        '''
        self.class_map = class_map
    
    def __call__(self, sample):
        new_sample = deepcopy(sample)
        y = new_sample['y']

        for old, new in self.class_map.items():
            y[y == old] = new

        new_sample['y'] = y

        return new_sample
        
class Pad:
    '''
    Pad (l,b,h) grid to max_size
    '''
    def __init__(self, size):
        self.size = np.array(size)

    def __call__(self, sample):
        '''
        Assume sample is smaller than self.size in all dims
        '''
        new_sample = deepcopy(sample)
        
        new_sample['x'] = pad_volume(new_sample['x'], self.size)
        new_sample['y'] = pad_volume(new_sample['y'], self.size)

        return new_sample

class LoadData(ABC):
    '''
    Base class for transforms that load data related to a subvolume
    in backproj 2D+3D models
    '''
    def __init__(self, cfg):
        self.data_dir = cfg['data']['root']

    def get_scan_name(self, scene_id, scan_id):
        return f'scene{str(scene_id).zfill(4)}_{str(scan_id).zfill(2)}' 

class LoadDepths(LoadData):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.img_size = cfg['data']['proj_img_size']

    def __call__(self, sample):
        # create all paths for depths
        scan_name = self.get_scan_name(sample['scene_id'], sample['scan_id'])
        frames = sample['frames']
        depths = torch.zeros(len(frames), self.img_size[1], self.img_size[0])
        
        # check if this sample has frames
        if -1 not in frames:
            paths = [Path(self.data_dir) / scan_name / 'depth' / f'{i}.png' for i in frames]
            # invert dims in the tensor
            # N, H, W -> torch nn convention
            # all the paths should exist
            if all([path.exists() for path in paths]):
                load_depth_multiple(paths, self.img_size, depths)
            else:
                sample['frames'][:] = -1
        else:
            sample['frames'][:] = -1

        sample['depths'] = depths
        
        return sample
    
class LoadPoses(LoadData):
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, sample):
        # create all paths for depths
        scan_name = self.get_scan_name(sample['scene_id'], sample['scan_id'])
        frames = sample['frames']
        poses = torch.zeros(len(frames), 4, 4)

        if -1 not in frames:
            paths = [Path(self.data_dir) / scan_name / 'pose' / f'{i}.txt' for i in frames]
            # all the paths should exist
            if all([path.exists() for path in paths]):
                load_pose_multiple(paths, poses)
            else:
                sample['frames'][:] = -1
        else:
            sample['frames'][:] = -1

        sample['poses'] = poses

        return sample

class LoadLabels2D(LoadData):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.img_size = tuple(cfg['data']['rgb_img_size'])
        self.ignore_label = cfg['data']['target_padding']
        self.num_classes = cfg['data']['num_classes']
        self.scannet_to_nyu40 = read_label_mapping(cfg['data']['label_file'])

    def __call__(self, sample):
        scan_name = self.get_scan_name(sample['scene_id'], sample['scan_id'])
        frames = sample['frames']

        # N, H, W -> torch nn convention
        labels2d = torch.zeros(len(frames), self.img_size[1], self.img_size[0], dtype=torch.int64)

        paths = [Path(self.data_dir) / scan_name / 'label-filt' / f'{i}.png' for i in frames]

        # fill the 2d label only if 3d label exists
        if sample['has_label']:
            for path_ndx, path in enumerate(paths):
                if path.exists():
                    # read the scannet label image as int, H,W
                    label_scannet = np.array(imageio.imread(path))
                    # map from scannet to nyu40 labels 0-40, H,W
                    label_nyu40 = map_labels(label_scannet, self.scannet_to_nyu40)
                    # map from NYU40 labels to 0-39 + 40 (ignored) labels, H,W
                    y = nyu40_to_continuous(label_nyu40, ignore_label=self.ignore_label, 
                                                        num_classes=self.num_classes)
                    # resize label image here using the proper interpolation - no artifacts  
                    # dims: H,W                                     
                    y = cv2.resize(y, self.img_size, interpolation=cv2.INTER_NEAREST)
                    labels2d[path_ndx] = torch.LongTensor(y.astype(np.int64))
                else:
                    # no label image
                    labels2d[path_ndx] = self.ignore_label
        else:
            labels2d.fill_(self.ignore_label)

        sample['labels2d'] = labels2d
        return sample

class LoadRGBs(LoadData):
    def __init__(self, cfg, transform=None):
        super().__init__(cfg)
        self.img_size = tuple(cfg['data']['rgb_img_size'])
        self.transform = transform

    def __call__(self, sample):
        # create all paths for images
        scan_name = self.get_scan_name(sample['scene_id'], sample['scan_id'])
        frames = sample['frames']

        # N, C, H, W -> torch nn convention
        rgbs = torch.zeros(len(frames), 3, self.img_size[1], self.img_size[0])

        if -1 not in frames:
            paths = [Path(self.data_dir) / scan_name / 'color' / f'{i}.jpg' for i in frames]
            # all the paths should exist
            if all([path.exists() for path in paths]):
                load_rgbs_multiple(paths, self.img_size, rgbs, self.transform)
            else:
                sample['frames'][:] = -1
        else:
            sample['frames'][:] = -1

        sample['rgbs'] = rgbs

        return sample
    
class AddChannelDim:
    '''
    Add a "1" dimension for the channel
    input: x=W, H, D
    output: x=W, H, D, 1

    '''
    def __init__(self):
        pass
    
    @staticmethod
    def apply(x):
        return np.expand_dims(x, axis=-1) 

    def __call__(self, sample):
        new_sample = deepcopy(sample)
        new_sample['x'] = np.expand_dims(new_sample['x'], axis=-1) 

        return new_sample

class TransposeDims:
    '''
    Change the order of dims to match conv3d's expected input
    input:  x: W, H, D, C
            y: W, H, D
    output: x: C, D, H, W
            y: D, H, W
    '''
    def __init__(self):
        pass

    @staticmethod
    def apply(x, y):
        x_new = x.transpose((3, 2, 1, 0))
        y_new = y.transpose((2, 1, 0))
        return x_new, y_new

    def __call__(self, sample):
        new_sample = deepcopy(sample)
        new_sample['x'] = new_sample['x'].transpose((3, 2, 1, 0))
        new_sample['y'] = new_sample['y'].transpose((2, 1, 0))

        return new_sample

        