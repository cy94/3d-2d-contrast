'''
3D fully conv network
'''
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from eval.sem_seg_2d import miou

class FCN3D(pl.LightningModule):
    '''
    Dense 3d convs on a volume grid
    '''
    def __init__(self, in_channels, num_classes, cfg=None):
        '''
        in_channels: number of channels in input

        '''
        super().__init__()
        self.cfg = cfg

        self.layers = nn.ModuleList([
            # args: inchannels, outchannels, kernel, stride, padding
            # 1->1/2
            nn.Conv3d(in_channels, 32, 3, 2, 1),
            # same
            nn.Conv3d(32, 64, 3, 1, 1),
            nn.ReLU(),

            # 1/2->1/4
            nn.Conv3d(64, 64, 3, 2, 1),
            # same
            nn.Conv3d(64, 128, 3, 1, 1),
            nn.ReLU(),
            
            # 1.4->1/8
            nn.Conv3d(128, 128, 3, 2, 1),
            # same
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.ReLU(),
            
            # inchannels, outchannels, kernel, stride, padding, output_padding
            # 1/8->1/4
            nn.ConvTranspose3d(128, 128, 4, 2, 1),
            nn.ReLU(),
            # 1/4->1/2
            nn.ConvTranspose3d(128, 64, 4, 2, 1),
            nn.ReLU(),
            # 1/2->original shape
            nn.ConvTranspose3d(64, num_classes, 4, 2, 1),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def common_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        out = self(x)
        loss = F.cross_entropy(out, y)
        return out, loss

    def training_step(self, batch, batch_idx):
        out, loss = self.common_step(batch, batch_idx)
        self.log('train_loss', loss)

        # miou only for some batches
        if random.random() > 0.8:
            m = self.get_miou(batch, out)
            self.log('miou/train', m)

        return loss

    def get_miou(self, batch, out):
        preds = out.argmax(dim=1)
        m = miou(preds, batch['y'], 21)
        return m if not np.isnan(m).any() else None
            
    def validation_step(self, batch, batch_idx):
        out, loss = self.common_step(batch, batch_idx)

        if random.random() > 0.8:
            m = self.get_miou(batch, out)
            self.log('miou/val', m)

        return loss
    
    def validation_epoch_end(self, validation_step_outputs):
        val_loss = torch.Tensor(validation_step_outputs).mean()
        self.log('val_loss', val_loss)
        
