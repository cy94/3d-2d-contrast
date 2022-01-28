
from datasets.scannet.common import CLASS_WEIGHTS_ALL_2D

import torch

class SemSegNet2D:
    def init_class_weights(self, cfg):
        if cfg['train']['class_weights']:
            print('Using class weights for 2D model')
            weights = {
                40: CLASS_WEIGHTS_ALL_2D
            }
            if self.num_classes in weights:
                self.class_weights = torch.Tensor(weights[self.num_classes])
            else:
                raise NotImplementedError(f'Add class weights for {self.num_classes} classes')
        else: 
            self.class_weights = None