'''
3D fully conv network
'''
import random

import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF

import pytorch_lightning as pl
import torchmetrics as tmetrics 

from eval.vis import confmat_to_fig, fig_to_arr
from datasets.scannet.utils import CLASS_NAMES, CLASS_WEIGHTS
from datasets.scannet.sem_seg_3d import ScanNetGridTestSubvols, collate_func
from models.layers_3d import Down3D, Up3D

class SemSegNet(pl.LightningModule):
    '''
    Parent class for semantic segmentation on voxel grid
    '''
    def __init__(self, num_classes, cfg=None):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.target_padding = cfg['data']['target_padding']
        
        if cfg['train']['class_weights']:
            print('Using class weights')
            self.class_weights = torch.Tensor(CLASS_WEIGHTS)
        else: 
            self.class_weights = None

        # init the model layers
        self.init_model()

    def init_model(self):
        pass

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

    def on_fit_start(self):
        self.train_iou, self.train_acc = self.create_metrics()                               

    def training_step(self, batch, batch_idx):
        preds, loss = self.common_step(batch)
        self.log('loss/train', loss)

        if random.random() > 0.7:
            self.train_iou(preds, batch['y'])
            self.train_acc(preds, batch['y'])

            self.log_ious(self.train_iou.compute(), 'train')
            self.log_accs(self.train_acc.compute(), 'train')                                        
            self.log_confmat(self.train_iou.confmat, 'train')

        return loss

    def training_step_end(self, outputs):
        self.train_iou.reset()
        self.train_acc.reset()

        return outputs

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
        # exclude the none and padding classes while calculating miou
        valid_ious = list(filter(lambda i: i != -1, ious[1:-1]))
        if len(valid_ious) > 0:
            self.log(f'iou/{split}/mean', torch.Tensor(valid_ious).mean())

    def create_metrics(self):
        '''
        create iou and accuracy metrics objects
        '''
        iou = tmetrics.IoU(num_classes=self.num_classes + 1, reduction='none', 
                                absent_score=-1, compute_on_step=False,
                                ignore_index=self.target_padding
                                ).to(self.device)
        acc = tmetrics.Accuracy(num_classes=self.num_classes + 1, average=None,
                                compute_on_step=False,
                                ignore_index=self.target_padding,
                                ).to(self.device)
        return iou, acc                                

    def on_validation_epoch_start(self):
        self.val_iou, self.val_acc = self.create_metrics()                               

    def validation_step(self, batch, batch_idx):
        preds, loss = self.common_step(batch)
        # update the iou metric
        if random.random() > 0.7:
            self.val_iou(preds, batch['y'])
            self.val_acc(preds, batch['y'])

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

        self.log_ious(self.val_iou.compute(), 'val')
        self.log_accs(self.val_acc.compute(), 'val')
        self.log_confmat(self.val_iou.confmat, 'val')

        self.log("hp_metric", loss)
    
    def _get_test_metrics(self, dataset, test_cfg, transform):
        iou, acc = self.create_metrics()
        
        # init metrics
        for scene in tqdm(dataset, desc='scene'):
            subvols = ScanNetGridTestSubvols(scene, self.hparams['cfg']['data']['subvol_size'], 
                                target_padding=self.target_padding, 
                                transform=transform)
            test_loader = DataLoader(subvols, batch_size=test_cfg['test']['batch_size'],
                                    shuffle=False, num_workers=8, collate_fn=collate_func,
                                    pin_memory=True) 

            for batch in tqdm(test_loader, desc='batch', leave=False):
                preds, _ = self.common_step(batch)
                iou(preds, batch['y'])
                acc(preds, batch['y'])

        return iou, acc

    def test_scenes(self, dataset, test_cfg, transform):
        '''
        scene_dataset: list of scannet scenes
        '''
        iou, acc = self._get_test_metrics(dataset, test_cfg, transform)

        ious = iou.compute()
        accs = acc.compute()

        print('Accuracy:', accs)
        print('Mean:', accs[1:-1].mean())
        print('IoUs:', ious)
        print('Mean:', ious[1:-1].mean())

class SparseNet3D(SemSegNet):
    '''
    Sparse convs on 3D grid using Minkowski Engine
    '''
    def __init__(self, in_channels, num_classes, cfg=None):
        '''
        in_channels: number of channels in input

        '''
        self.in_channels = in_channels
        super().__init__(num_classes, cfg)

    def common_step(self, batch):
        coords, feats, y = batch['coords'], batch['feats'], batch['y']
        inputs = ME.SparseTensor(feats, coords)

        out = self(inputs)
        out_arr = out.F.squeeze()
        loss = F.cross_entropy(out_arr, y, weight=self.class_weights.to(self.device),
                                ignore_index=self.target_padding)
        preds = out_arr.argmax(dim=1)
        return preds, loss
        
    def _get_test_metrics(self, dataset, test_cfg, transform):
        # init metrics
        iou, acc = self.create_metrics()

        # set the transform on the full scene
        dataset.transform = transform

        test_loader = DataLoader(dataset, batch_size=test_cfg['test']['batch_size'],
                                shuffle=False, num_workers=8, collate_fn=self.collation_fn,
                                pin_memory=True) 

        for batch in tqdm(test_loader, desc='batch', leave=False):
            preds, _ = self.common_step(batch)
            iou(preds, batch['y'])
            acc(preds, batch['y'])

        return iou, acc

    @staticmethod
    def collation_fn(sample_list):
        '''
        Collate sparse inputs into ME tensors
        '''
        # Generate batched coordinates
        coords_batch = ME.utils.batched_coordinates([s['coords'] for s in sample_list])

        # Concatenate all lists
        feats_batch = torch.cat([torch.Tensor(s['feats']) for s in sample_list])
        labels_batch = torch.cat([torch.LongTensor(s['labels']) for s in sample_list])

        return {'coords': coords_batch, 'feats': feats_batch, 'y': labels_batch}

    def init_model(self):
        # dimension of the space
        D = 3

        self.block1 = torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=self.in_channels,
                out_channels=8,
                kernel_size=3,
                stride=1,
                dimension=D),
            ME.MinkowskiBatchNorm(8))

        self.block2 = torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=2,
                dimension=D),
            ME.MinkowskiBatchNorm(16),
        )

        self.block3 = torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                dimension=D),
            ME.MinkowskiBatchNorm(32))

        self.block3_tr = torch.nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=2,
                dimension=D),
            ME.MinkowskiBatchNorm(16))

        self.block2_tr = torch.nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=2,
                dimension=D),
            ME.MinkowskiBatchNorm(16))

        self.conv1_tr = ME.MinkowskiConvolution(
            in_channels=24,
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            dimension=D)

    def forward(self, x):
        out_s1 = self.block1(x)
        out = MF.relu(out_s1)

        out_s2 = self.block2(out)
        out = MF.relu(out_s2)

        out_s4 = self.block3(out)
        out = MF.relu(out_s4)

        out = MF.relu(self.block3_tr(out))
        out = ME.cat(out, out_s2)

        out = MF.relu(self.block2_tr(out))
        out = ME.cat(out, out_s1)

        return self.conv1_tr(out)

class FCN3D(SemSegNet):
    '''
    Dense 3d convs on a volume grid
    '''
    def __init__(self, in_channels, num_classes, cfg=None):
        '''
        in_channels: number of channels in input

        '''
        super().__init__(num_classes, cfg)

        self.layers = nn.ModuleList([
            # args: inchannels, outchannels, kernel, stride, padding
            # 1->1/2
            nn.Conv3d(in_channels, 32, 3, 2, 1),
            # same
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.ReLU(),

            # 1/2->1/4
            nn.Conv3d(32, 64, 3, 2, 1),
            # same
            nn.Conv3d(64, 64, 3, 1, 1),
            nn.ReLU(),
            
            # 1.4->1/8
            nn.Conv3d(64, 128, 3, 2, 1),
            # same
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.ReLU(),
            
            # inchannels, outchannels, kernel, stride, padding, output_padding
            # 1/8->1/4
            nn.ConvTranspose3d(128, 64, 4, 2, 1),
            nn.ReLU(),
            # 1/4->1/2
            nn.ConvTranspose3d(64, 64, 4, 2, 1),
            nn.ReLU(),
            # 1/2->original shape
            nn.ConvTranspose3d(64, num_classes, 4, 2, 1),
        ])

class UNet3D(SemSegNet):
    '''
    Dense 3d convs on a volume grid
    '''
    def __init__(self, in_channels, num_classes, cfg=None):
        '''
        in_channels: number of channels in input

        '''
        super().__init__(num_classes, cfg)

        self.layers = nn.ModuleList([
            # 1->1/2
            Down3D(in_channels, 32),
            # 1/2->1/4
            Down3D(32, 64),
            # 1/4->1/8
            Down3D(64, 128),
            
            # 1/8->1/4
            Up3D(128, 64),
            # 1/4->1/2
            Up3D(64*2, 32),
            # 1/2->original shape
            Up3D(32*2, num_classes, dropout=False),
        ])

    def forward(self, x):
        # length of the down/up path
        L = len(self.layers)//2
        outs = []

        # down layers
        # store the outputs of all but the last one
        for layer in self.layers[:L]:
            x = layer(x)
            outs.append(x)

        # remove the last output and reverse
        outs = list(reversed(outs[:-1]))
        
        # lowest connection in the "U"
        x = self.layers[L](x)

        # up layers
        for ndx, layer in enumerate(self.layers[L+1:]):
            x = torch.cat([x, outs[ndx]], dim=1)
            x = layer(x)
            
        return x

        

        

