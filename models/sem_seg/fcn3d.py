'''
3D fully conv network
'''
import random

import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import torchmetrics as tmetrics 
import torchmetrics.functional as tmetricsF

from eval.vis import confmat_to_fig, fig_to_arr
from eval.iou_confmat import iou_confmat
from datasets.scannet.utils import CLASS_NAMES, CLASS_WEIGHTS

class FCN3D(pl.LightningModule):
    '''
    Dense 3d convs on a volume grid
    '''
    def __init__(self, in_channels, num_classes, cfg=None):
        '''
        in_channels: number of channels in input

        '''
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        print(f'FCN with {self.num_classes} classes')
        
        self.target_padding = cfg['data']['target_padding']
        
        if cfg['train']['class_weights']:
            print('Using class weights')
            self.class_weights = torch.Tensor(CLASS_WEIGHTS)
        else: 
            self.class_weights = None

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

    def common_step(self, batch):
        x, y = batch['x'], batch['y']
        out = self(x)
        loss = F.cross_entropy(out, y, weight=self.class_weights.to(self.device),
                                ignore_index=self.target_padding)
        preds = out.argmax(dim=1)
        return preds, loss

    def training_step(self, batch, batch_idx):
        preds, loss = self.common_step(batch)
        self.log('loss/train', loss)

        # miou only for some batches - compute right now and log
        if random.random() > 0.7:
            ious, confmat = iou_confmat(preds, batch['y'], 
                                num_classes=self.num_classes, 
                                reduction='none', 
                                absent_score=-1, 
                                )
            self.log_ious(ious, 'train')
            self.log_confmat(confmat, 'train')
            accs = tmetricsF.accuracy(preds, batch['y'], average=None,
                                        num_classes=self.num_classes,
                                        )
            self.log_accs(accs, 'train')                                        

        return loss

    def log_accs(self, accs, split):
        for class_ndx, acc in enumerate(accs):
            tag = f'acc/{split}/{CLASS_NAMES[class_ndx]}'
            self.log(tag, acc)
        self.log(f'acc/{split}/mean', accs.mean())

    def log_ious(self, ious, split):
        for class_ndx, iou in enumerate(ious):
            if iou != -1:
                tag = f'iou/{split}/{CLASS_NAMES[class_ndx]}'
                self.log(tag, iou)
        valid_ious = list(filter(lambda i: i != -1, ious))
        if len(valid_ious) > 0:
            self.log(f'iou/{split}/mean', torch.Tensor(valid_ious).mean())

    def on_validation_epoch_start(self):
        self.iou = tmetrics.IoU(num_classes=self.num_classes, reduction='none', 
                                absent_score=-1, compute_on_step=False,
                                ignore_index=self.target_padding
                                ).to(self.device)
        self.acc = tmetrics.Accuracy(num_classes=self.num_classes, average=None,
                                compute_on_step=False,
                                ).to(self.device)                                

    def validation_step(self, batch, batch_idx):
        preds, loss = self.common_step(batch)
        # update the iou metric
        if random.random() > 0.7:
            self.iou(preds, batch['y'])
            self.acc(preds, batch['y'])

        return loss
    
    def log_confmat(self, mat, split):
        fig = confmat_to_fig(mat.cpu().numpy(), CLASS_NAMES)
        img = fig_to_arr(fig)
        plt.close()
        tag = f'confmat/{split}'
        self.logger.experiment.add_image(tag, img, global_step=self.global_step, 
                                        dataformats='HWC')


    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.Tensor(validation_step_outputs).mean()
        self.log('loss/val', loss)

        self.log_ious(self.iou.compute(), 'val')
        self.log_accs(self.acc.compute(), 'val')
        self.log_confmat(self.iou.confmat, 'val')

        self.log("hp_metric", loss)

        
