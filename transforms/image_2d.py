from copy import deepcopy

import numpy as np
import skimage.transform as sktf

class Resize:
    def __init__(self, size=(640, 480)):
        # height, width
        self.size = size[::-1]

    def __call__(self, sample):
        new_sample = deepcopy(sample)
        # height, width
        new_sample['img'] = sktf.resize(new_sample['img'], self.size, order=0, 
                                        preserve_range=True)
        new_sample['label'] = sktf.resize(new_sample['label'], self.size, order=0, 
                                        preserve_range=True)                                        
        return new_sample                                

class TransposeChannels:
    def __init__(self):
        pass

    def __call__(self, sample):
        new_sample = deepcopy(sample)
        # move channels to the front
        new_sample['img'] = new_sample['img'].transpose((2, 0, 1))

        return new_sample

class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, sample):
        '''
        input: rgb in (0,255)
        output: normalize image
        '''
        sample['img'] /= 255.0
        sample['img'] = (sample['img'] - self.mean) / self.std

        return sample
