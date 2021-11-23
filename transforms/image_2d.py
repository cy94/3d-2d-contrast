import albumentations as A

import cv2
import numpy as np

class TransformX:
    '''
    Base class that transfroms only X using albumentations
    '''
    def __call__(self, sample):
        aug = self.t(image=sample['x'])
        sample['x'] = aug['image']

        return sample

class TransformXY:
    '''
    Base class that transforms only X using albumentations
    Can transform only X as well if Y is missing
    '''
    def __call__(self, sample):
        if 'y' in sample:
            aug = self.t(image=sample['x'], mask=sample['y'])
            sample['x'] = aug['image']
            sample['y'] = aug['mask']
        else:
            aug = self.t(image=sample['x'])
            sample['x'] = aug['image']

        return sample

#############
# transform both X and Y
#############

class CoarseDropout(TransformXY):
    def __init__(self):
        self.t = A.CoarseDropout()

class LRFlip(TransformXY):
    def __init__(self):
        self.t = A.HorizontalFlip()

#############
# transform only X 
#############

class Downscale(TransformX):
    def __init__(self):
        self.t = A.Downscale()

class GaussianBlur(TransformX):
    def __init__(self):
        self.t = A.GaussianBlur()

class ColorJitter(TransformX):
    def __init__(self):
        self.t = A.ColorJitter()

class HueSaturationValue(TransformX):
    def __init__(self):
        self.t = A.HueSaturationValue()

class Blur(TransformX):
    def __init__(self):
        self.t = A.Blur()

class RandomBrightnessContrast(TransformX):
    def __init__(self):
        self.t = A.RandomBrightnessContrast()

class GaussNoise(TransformX):
    def __init__(self):
        self.t = A.GaussNoise()

class RGBShift(TransformX):
    def __init__(self):
        self.t = A.RGBShift()

class Resize:
    def __init__(self, size=(640, 480)):
        # height, width
        self.size = tuple(size)

    def __call__(self, sample):
        # height, width
        sample['x'] = cv2.resize(sample['img'], self.size) 
        # use nearest method - keep valid labels!
        if 'y' in sample:
            sample['y'] = cv2.resize(sample['label'], self.size, 
                                                interpolation=cv2.INTER_NEAREST)
        return sample                                

class TransposeChannels:
    def __init__(self):
        pass

    def __call__(self, sample):
        # move channels to the front
        sample['x'] = sample['x'].transpose((2, 0, 1))

        return sample

class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean)
        self.std = np.array(std)

    @staticmethod
    def apply(img, mean, std):
        '''
        img: H,W,3
        '''
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std

        return img

    def __call__(self, sample):
        '''
        input: H,W,3 in (0,255)
        output: normalize image
        '''
        sample['x'] = sample['x'].astype(np.float32) / 255.0
        sample['x'] = (sample['x'] - self.mean) / self.std

        return sample
