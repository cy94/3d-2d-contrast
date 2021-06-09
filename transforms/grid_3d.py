from copy import deepcopy

import numpy as np

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

class AddChannelDim:
    '''
    Add a "1" dimension for the channel
    input: x=W, H, D
    output: x=W, H, D, 1

    '''
    def __init__(self):
        pass

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

    def __call__(self, sample):
        new_sample = deepcopy(sample)
        new_sample['x'] = new_sample['x'].transpose((3, 2, 1, 0))
        new_sample['y'] = new_sample['y'].transpose((2, 1, 0))

        return new_sample

        