from transforms.image_2d import Normalize
from datasets.scannet.utils_3d import load_depth_multiple, load_pose_multiple, load_rgbs_multiple
from pathlib import Path
from abc import ABC
from copy import deepcopy

import numpy as np

import torch
import torchvision.transforms as transforms

def pad_volume(vol, size, pad_val=-100):
    '''
    vol: (l, b, h) array
    size: (3,) array
    pad_val: value to pad
    '''
    diff = size - np.array(vol.shape)
    # left and right padding for 3 dims (ie l/r, front/back, top/bottom)
    pad = np.stack((np.floor(diff/2), np.ceil(diff/2)), axis=-1).astype(np.uint8).tolist()
    
    padded = np.pad(vol, pad, constant_values=pad_val)

    return padded

class RandomRotate:
    '''
    Randomly rotate the scene by 90, 180 or 270 degrees 
    '''
    def __init__(self):
        self.rng = np.random.default_rng()

    def __call__(self, sample):
        '''
        sample with x and y
        rotate both of them
        '''
        new_sample = deepcopy(sample)

        # rotate 0, 1, 2 or 3 times
        num_rots = self.rng.integers(0, 3, endpoint=True)
        new_sample['x'] = np.rot90(new_sample['x'], k=num_rots)
        new_sample['y'] = np.rot90(new_sample['y'], k=num_rots)

        return new_sample

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
    
class LoadRGBs(LoadData):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.img_size = tuple(cfg['data']['rgb_img_size'])
        t = Normalize()
        # transform to operate on arrays, not dicts
        self.transform = lambda img: Normalize.apply(img.astype(np.float32), mean=t.mean, std=t.std)

    def __call__(self, sample):
        # create all paths for depths
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

        