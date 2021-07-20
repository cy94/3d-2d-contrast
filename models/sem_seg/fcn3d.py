'''
3D fully conv network
'''
from eval.common import ConfMat
import random

import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF

import pytorch_lightning as pl
import torchmetrics as tmetrics 

from eval.vis import confmat_to_fig, fig_to_arr
from datasets.scannet.common import CLASS_NAMES, CLASS_NAMES_ALL, CLASS_WEIGHTS, CLASS_WEIGHTS_ALL, VALID_CLASSES
from models.layers_3d import Down3D, Up3D

class SemSegNet(pl.LightningModule):
    '''
    Parent class for semantic segmentation on voxel grid
    '''
    def __init__(self, num_classes, cfg=None, log_all_classes=False):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.target_padding = cfg['data']['target_padding']

        self.init_class_weights(cfg)
        self.init_class_names()
        # subset of classes of interest to log separately
        self.init_class_subset()

        # init the model layers
        self.init_model()

        self.log_all_classes = log_all_classes

    def init_class_subset(self):
        self.class_subset = None
        if self.num_classes == 40:
            # subtract 1 because the array contains the raw class indices
            # starting at 1
            self.class_subset = np.array(VALID_CLASSES) - 1

    def init_class_names(self):
        names = {
            20: CLASS_NAMES,
            40: CLASS_NAMES_ALL
        }
        if self.num_classes in names:
            self.class_names = names[self.num_classes]
        else:
            raise NotImplementedError(f'Add class names for {self.num_classes} classes')

    def init_class_weights(self, cfg):
        if cfg['train']['class_weights']:
            print('Using class weights')
            weights = {
                20: CLASS_WEIGHTS,
                40: CLASS_WEIGHTS_ALL
            }
            if self.num_classes in weights:
                self.class_weights = torch.Tensor(weights[self.num_classes])
            else:
                raise NotImplementedError(f'Add class weights for {self.num_classes} classes')
        else: 
            self.class_weights = None
    def init_model(self):
        pass

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def display_metric(self, vals, name='metric'):
        print(f'mean {name}: {np.nanmean(name):.3f}')
        if self.class_subset is not None:
            print(f'mean {name} on subset: {np.nanmean(vals[self.class_subset]):.3f}')

        print('\nClasses: ' + ' '.join(CLASS_NAMES) + '\n')
        print(f'{name}: ' + ' '.join('{:.03f}'.format(i) for i in vals) + '\n')

    def configure_optimizers(self):
        cfg = self.hparams['cfg']['train']['opt']
        if cfg['name'] == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=cfg['lr'], 
                weight_decay=cfg['l2'],
                momentum=cfg['momentum'],
                dampening=cfg['dampening'])
        elif cfg['name'] == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=cfg['lr'], 
                weight_decay=cfg['l2'])
                            
        print('Using optimizer:', optimizer)
        self.optim = optimizer

        # use scheduler?
        if 'schedule' in self.hparams['cfg']['train']:
            cfg = self.hparams['cfg']['train']['schedule']
            if cfg['name'] == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                cfg['step_size'], cfg['gamma'])
            elif cfg['name'] == 'exp':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                cfg['gamma'])

            print('Using scheduler:', scheduler)
            # optimizer and scheduler - use lists
            return [optimizer], [scheduler]
        # only optimizer
        return optimizer

    def get_class_weights(self):
        if self.class_weights is not None:
            weight = self.class_weights.to(self.device)
        else:
            weight = None
        return weight

    def common_step(self, batch, mode=None):
        '''
        mode: train/val/None - can be used in subclasses for differing behaviour
        '''
        x, y = batch['x'], batch['y']
        out = self(x)
        
        loss = F.cross_entropy(out, y, weight=self.get_class_weights(),
                                ignore_index=self.target_padding)
        preds = out.argmax(dim=1)
        return preds, loss

    def on_fit_start(self):
        self.train_confmat = self.create_metrics()

    def training_step(self, batch, batch_idx):
        preds, loss = self.common_step(batch, 'train')
        self.log('loss/train', loss)

        self.train_confmat.update(preds, batch['y'])
        self.log_everything(self.train_confmat, 'train')

        # log LR
        self.log('lr', self.optim.param_groups[0]['lr'])

        return loss

    def training_step_end(self, outputs):
        self.train_confmat.reset()

        return outputs

    def log_accs(self, accs, split):
        if self.log_all_classes:
            for class_ndx, acc in enumerate(accs):
                tag = f'acc/{split}/{self.class_names[class_ndx]}'
                self.log(tag, acc)
        self.log(f'acc/{split}/mean', accs.mean())

        # using all classes -> log subset of 20 classes separately
        if self.class_subset is not None:
            self.log(f'acc/{split}/mean_subset', accs[self.class_subset].mean())

    def log_everything(self, confmat, split):
        self.log_ious(confmat.ious, split)
        self.log_accs(confmat.accs, split)                                        
        self.log_confmat(confmat.mat, split)

    def log_ious(self, ious, split):
        if self.log_all_classes:
            for class_ndx, iou in enumerate(ious):
                tag = f'iou/{split}/{self.class_names[class_ndx]}'
                self.log(tag, iou)

        self.log(f'iou/{split}/mean', torch.Tensor(ious).mean())

        # using all classes -> log subset of 20 classes separately
        if self.class_subset is not None:
            self.log(f'iou/{split}/mean_subset', ious[self.class_subset].mean())

    def create_metrics(self):
        return ConfMat(self.num_classes)                               

    def on_validation_epoch_start(self):
        self.val_confmat = self.create_metrics()

    def validation_step(self, batch, batch_idx):
        preds, loss = self.common_step(batch, 'val')

        self.val_confmat.update(preds, batch['y'])

        return loss
    
    def log_confmat(self, mat, split):
        '''
        mat: np array
        '''
        fig = confmat_to_fig(mat, CLASS_NAMES)
        img = fig_to_arr(fig)
        plt.close()
        tag = f'confmat/{split}'
        self.logger.experiment.add_image(tag, img, global_step=self.global_step, 
                                        dataformats='HWC')


    def validation_epoch_end(self, val_step_outputs):
        loss = torch.Tensor(val_step_outputs).mean()
        self.log('loss/val', loss)

        self.log_everything(self.val_confmat, 'val')

        self.log("hp_metric", loss)    

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

    def test_scenes(self, test_loader):
        confmat = self.create_metrics()

        with torch.no_grad():
            for batch in tqdm(test_loader):
                coords, feats, y = batch['coords'], batch['feats'], batch['y']
                # normalize colors
                feats[:, :3] = feats[:, :3] / 255. - 0.5
                sinput = ME.SparseTensor(feats, coords)
                # sparse output
                sout = self(sinput)
                # regular output
                out = sout.F
                pred = self.get_prediction(out).int()

                # update counts
                confmat.update(pred, y)

        # get IOUs              
        ious = confmat.ious
        accs = confmat.accs
        
        print(f'mIOU {np.nanmean(ious):.3f}')
        print(f'mAcc {np.nanmean(accs):.3f}')

        print('\nClasses: ' + ' '.join(CLASS_NAMES) + '\n')
        print('IOU: ' + ' '.join('{:.03f}'.format(i) for i in ious) + '\n')
        print('Acc: ' + ' '.join('{:.03f}'.format(i) for i in accs) + '\n')

    def get_prediction(self, output):
        return output.max(1)[1]

    def common_step(self, batch, mode):
        '''
        the inference function that is unique to the sparse model
        mode: train or val
        '''
        coords, feats, y = batch['coords'], batch['feats'], batch['y']
        
        if mode == 'train':
            # For some networks, making the network invariant to even, odd coords is important. Random translation
            coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)

        # Preprocess input
        feats[:, :3] = feats[:, :3] / 255. - 0.5
        sinput = ME.SparseTensor(feats, coords)

        out = self(sinput)
        out_arr = out.F.squeeze()
        loss = F.cross_entropy(out_arr, y, weight=self.get_class_weights(),
                                ignore_index=self.target_padding)
        preds = out_arr.argmax(dim=1)
        return preds, loss

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

    def init_model(self):
        self.layers = nn.ModuleList([
            # 1->1/2
            Down3D(self.in_channels, 32),
            # 1/2->1/4
            Down3D(32, 64),
            # 1/4->1/8
            Down3D(64, 128),
            
            # 1/8->1/4
            Up3D(128, 64),
            # 1/4->1/2
            Up3D(64*2, 32),
            # 1/2->original shape
            Up3D(32*2, self.num_classes, dropout=False),
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

    def test_scenes(self, test_loader):
        confmat = self.create_metrics()

        with torch.no_grad():
            for batch in tqdm(test_loader):
                preds, loss = self.common_step(batch, 'test')

                # update counts
                confmat.update(preds, batch['y'])

        self.display_metric(confmat.ious, 'iou')
        self.display_metric(confmat.accs, 'acc')

        
        

        

