from copy import deepcopy

import numpy as np

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
        
        diff = self.size - np.array(new_sample['x'].shape)
        # left and right padding for 3 dims (ie l/r, front/back, top/bottom)
        pad = np.stack((np.floor(diff/2), np.ceil(diff/2)), axis=-1).astype(np.uint8).tolist()
        
        new_sample['x'] = np.pad(new_sample['x'], pad, constant_values=-100)
        new_sample['y'] = np.pad(new_sample['y'], pad, constant_values=-100)

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

        