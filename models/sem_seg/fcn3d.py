'''
3D fully conv network
'''
from datasets.scannet.utils_3d import ProjectionHelper, project_2d_3d
from eval.common import ConfMat

import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
from tqdm import tqdm

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF

import pytorch_lightning as pl

from eval.vis import confmat_to_fig, fig_to_arr
from datasets.scannet.common import CLASS_NAMES, CLASS_NAMES_ALL, CLASS_WEIGHTS, CLASS_WEIGHTS_ALL, VALID_CLASSES
from models.layers_3d import Down3D, Down3D_Big, Up3D, Up3D_Big

class SemSegNet(pl.LightningModule):
    '''
    Parent class for semantic segmentation on voxel grid
    '''
    def __init__(self, num_classes, cfg=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.should_log_confmat = kwargs.get('should_log_confmat', False)
        self.log_all_classes = kwargs.get('log_all_classes', False)

        self.target_padding = cfg['data']['target_padding']

        self.init_class_weights(cfg)
        self.init_class_names()
        # subset of classes of interest to log separately
        self.init_class_subset()

        # init the model layers
        self.init_model()


    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

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
                momentum=cfg.get('momentum', 0.9),
                dampening=cfg.get('dampening', 0))
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
        # init the best val loss with inf value
        self.best_val_loss = torch.Tensor([float('inf')])

    def log_losses(self, loss, split):
        # log each loss
        if isinstance(loss, dict):
            self.log(f'loss/{split}', loss['crossent'])
            for key in loss:
                if key != 'crossent':
                    self.log(f'loss/{split}/{key}', loss[key])
        else:
            self.log(f'loss/{split}', loss)

    def training_step(self, batch, batch_idx):
        out = self.common_step(batch, 'train')

        # skip this batch
        if out is None:
            return None
        else:
            preds, loss = out
        self.log_losses(loss, 'train') 

        self.train_confmat.update(preds, batch['y'])
        self.log_everything(self.train_confmat, 'train')

        # log LR
        self.log('lr', self.optim.param_groups[0]['lr'])

        if isinstance(loss, dict):
            # pick only these losses
            if 'losses' in self.hparams['cfg']['model']:
                loss_keys = self.hparams['cfg']['model']['losses']
            # all losses
            else:
                loss_keys = loss.keys()
            losses = [loss[key] for key in loss_keys]
            return sum(losses)
        else:
            return loss

    def training_step_end(self, outputs):
        self.train_confmat.reset()

        return outputs

    def log_accs(self, accs, split):
        if self.log_all_classes:
            for class_ndx, acc in enumerate(accs):
                tag = f'acc/{split}/{self.class_names[class_ndx]}'
                self.log(tag, acc)
        self.log(f'acc/{split}/mean', np.nanmean(accs))

        # using all classes -> log subset of 20 classes separately
        if self.class_subset is not None:
            self.log(f'acc/{split}/mean_subset', np.nanmean(accs[self.class_subset]))

    def log_everything(self, confmat, split):
        self.log_ious(confmat.ious, split)
        self.log_accs(confmat.accs, split)                                        
        
        if self.should_log_confmat:
            self.log_confmat(confmat.mat, split)
        
    def log_ious(self, ious, split):
        if self.log_all_classes:
            for class_ndx, iou in enumerate(ious):
                tag = f'iou/{split}/{self.class_names[class_ndx]}'
                self.log(tag, iou)

        self.log(f'iou/{split}/mean', np.nanmean(ious))

        # using all classes -> log subset of 20 classes separately
        if self.class_subset is not None:
            self.log(f'iou/{split}/mean_subset', np.nanmean(ious[self.class_subset]))

    def create_metrics(self):
        return ConfMat(self.num_classes)                               

    def on_validation_epoch_start(self):
        self.val_confmat = self.create_metrics()

    def validation_step(self, batch, batch_idx):
        out = self.common_step(batch, 'val')

        # skip this batch
        if out is None:
            return None
        else:
            preds, loss = out

        self.val_confmat.update(preds, batch['y'])

        return loss
    
    def log_confmat(self, mat, split):
        '''
        mat: np array
        '''
        fig = confmat_to_fig(mat, self.class_names)
        img = fig_to_arr(fig)
        plt.close()
        tag = f'confmat/{split}'

        # wandb                               
        self.logger.experiment.log({tag: wandb.Image(img), 
                                        'global_step': self.global_step}) 

    def validation_epoch_end(self, outputs):
        # hack because pytorch lightning gives empty outputs initially?
        if len(outputs) == 0:
            outputs = [0]

        if isinstance(outputs[0], dict):
            loss_mean = {
                key: np.nanmean(torch.Tensor([output[key] for output in outputs]))
                        for key in outputs[0]
                        }
            self.log("hp_metric", loss_mean['crossent'])    
        else:
            loss_mean = np.nanmean(torch.Tensor(outputs))
            self.log("hp_metric", loss_mean)    

        self.log_losses(loss_mean, 'val')

        self.log_everything(self.val_confmat, 'val')

        self.update_summaries(loss_mean)

        

    def update_summaries(self, val_loss):
        '''
        loss: single tensor or dict
        # update summary metrics - if val loss decreased, set these in wandb
        # best val loss
        # best val iou, acc metrics
        # corresponding step
        '''
        # check if we have a logger
        if self.logger is None:
            return

        current_val_loss = val_loss['crossent'] if isinstance(val_loss, dict) else val_loss

        if current_val_loss <= self.best_val_loss:
            self.best_val_loss = current_val_loss
            expt = self.logger.experiment
            expt.summary["best_val_loss"] = current_val_loss

            # get iou and acc            
            ious, accs = self.val_confmat.ious, self.val_confmat.accs

            # log for all classes
            expt.summary["best_iou"] = np.nanmean(ious)
            expt.summary["best_acc"] = np.nanmean(accs)

            expt.summary['best_step'] = self.global_step

            # log for subset of classes
            if self.class_subset is not None:
                expt.summary["best_iou_subset"] = np.nanmean(ious[self.class_subset])
                expt.summary["best_acc_subset"] = np.nanmean(accs[self.class_subset])

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
                # normalize 
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
        super().__init__(in_channels, num_classes, cfg)

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


class UNet3D_3DMV(SemSegNet):
    '''
    Dense 3d convs on a volume grid
    '''
    def __init__(self, in_channels, num_classes, cfg=None):
        '''
        in_channels: number of channels in input

        '''
        self.in_channels = in_channels
        super().__init__(num_classes, cfg)

    def init_model(self):
        # number of features
        self.nf0 = 32
        self.nf1 = 64 
        self.nf2 = 128 

        # 3d conv on subvols
        # use 1x1 convs, fewer params
        self.down1 = Down3D_Big(self.in_channels, self.nf0) 
        self.down2 = Down3D_Big(self.nf0, self.nf1) 
        self.down3 = Down3D_Big(self.nf1, self.nf2) 
        self.up1 = Up3D_Big(self.nf2, self.nf2) 
        self.up2 = Up3D_Big(self.nf2+self.nf1, self.nf2) 
        self.up3 = Up3D_Big(self.nf2+self.nf0, self.nf2)

        self.pred_layer = nn.Conv3d(self.nf2, self.num_classes, 3, 1, 1)

    def forward(self, x):
        x16 = self.down1(x)
        x8 = self.down2(x16)
        x4 = self.down3(x8)

        # skip connections
        xup8 = self.up1(x4)
        xup16 = self.up2(torch.cat((xup8, x8), 1))
        xup32 = self.up3(torch.cat((xup16, x16), 1))
            
        out = self.pred_layer(xup32)

        return out


class UNet3D(SemSegNet):
    '''
    Dense 3d convs on a volume grid
    '''
    def __init__(self, in_channels, num_classes, cfg=None, *args, **kwargs):
        '''
        in_channels: number of channels in input

        '''
        self.in_channels = in_channels
        super().__init__(num_classes, cfg, *args, **kwargs)

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
            Up3D(32*2, self.num_classes, dropout=False, relu=False),
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


class UNet2D3D(UNet3D):
    '''
    Dense 3d convs on a volume grid
    Uses 2D features from nearby images using a pretrained ENet
    '''
    def __init__(self, in_channels, num_classes, cfg, features_2d, intrinsic, *args, **kwargs):
        '''
        in_channels: number of channels in input

        '''
        # use 2d feats by default, else it becomes a 3D-only model
        self.use_2dfeat = cfg['model'].get('use_2dfeat', True)

        super().__init__(in_channels, num_classes, cfg, *args, **kwargs)
        self.in_channels = in_channels

        # 2D pretrained model
        self.features_2d = features_2d
        # train the 2d model on 2d labels?
        self.train_2d = cfg['model'].get('train_2d', False)

        if self.features_2d is not None:
            print('Using 2d feats in 2d3d model?: ', self.use_2dfeat)

            print(f'Train the 2D model on 2D labels?: {self.train_2d}')

            finetune_2d = cfg['model'].get('finetune_2d', False)
            if finetune_2d:
                print('Finetune 2D weights')
            else:
                # freeze the 2d weights
                print('Freeze 2D weights')
                for param in self.features_2d.parameters():
                    param.requires_grad = False

        self.subvol_size = cfg['data']['subvol_size']
        self.proj_img_dims = cfg['data']['proj_img_size']
        self.data_dir = cfg['data']['root']
        
        self.projection = ProjectionHelper(
            intrinsic, 
            cfg['data']['depth_min'], cfg['data']['depth_max'],
            cfg['data']['proj_img_size'],
            cfg['data']['subvol_size'], cfg['data']['voxel_size']
        )

        self.contrastive = 'contrastive' in cfg['model']
        print(f'Use contrastive loss? {self.contrastive}')
        if self.contrastive:
            self.contr_cfg = cfg['model']['contrastive']

    def on_fit_start(self):
        super().on_fit_start()
        self.change_device()

    def change_device(self):
        '''
        Change device for everything that ptL/torch doesn't handle
        '''
        if self.features_2d is not None:
            self.features_2d.to(self.device)
        self.projection.to(self.device)

    def init_model(self):
        self.pooling = nn.MaxPool1d(kernel_size=self.hparams['cfg']['data']['num_nearest_images'])
        
        self.feat2d_same = nn.ModuleList([
            nn.Identity()
        ])

        # conv on feats projected from 2d
        self.layers_2d = nn.ModuleList([
            nn.Identity(),
            # 1->1/2
            Down3D(128, 64),
            # 1/2->1/4
            Down3D(64, 64),
            # 1/4->1/8
            Down3D(64, 128),
        ]) 
        
        # initial same conv layers, if any - need to get high res features
        # for contrasting
        self.layers3d_same = nn.ModuleList([
            nn.Identity()
        ])

        self.layers3d_down = nn.ModuleList([
            # 1->1/2
            Down3D(self.in_channels, 32),
            # 1/2->1/4
            Down3D(32, 64),
            # 1/4->1/8
            Down3D(64, 128),
        ])

        self.layers3d_up = nn.ModuleList([
            # 1/8->1/4
            # twice the channels - half of them come from 2D features
            Up3D(128*2, 64),
            # 1/4->1/2, skip connection
            Up3D(64*2, 32),
            # 1/2->original shape
            Up3D(32*2, self.num_classes, dropout=False, relu=False),
        ])

        contrastive = 'contrastive' in self.hparams['cfg']['model']

        if contrastive:
            # parallel to the last up layer, project to 128 dim instead of num classes
            # and then contrast the features
            self.up3d_contr = Up3D(32*2, 128, dropout=False, relu=False)

    def common_step(self, batch, mode=None):
        '''
        mode: train/val/None - can be used in subclasses for differing behaviour
        '''
        x = batch['x']

        # dont need these in 3d-only model
        rgbs, depths, poses, transforms, frames = None, None, None, None, None

        if self.use_2dfeat or self.contrastive:
            # x is NCDHW
            bsize = x.shape[0]

            world_to_grid, frames = batch['world_to_grid'], batch['frames']

            # dataset should have atleast num_nearest_images frames per chunk
            # use only the number that is specified in the cfg
            num_nearest_imgs = self.hparams['cfg']['data']['num_nearest_images']
            # total number of frames (num chunks * num frames per chunk)
            num_imgs = bsize * num_nearest_imgs

            depths, poses, rgbs = batch['depths'], batch['poses'], batch['rgbs']

            # keep only the number of frames required
            depths = depths[:, :num_nearest_imgs, :, :]
            poses = poses[:, :num_nearest_imgs, :, :]
            rgbs = rgbs[:, :num_nearest_imgs, :, :, :]
            frames = frames[:, :num_nearest_imgs]

            if self.train_2d:
                labels2d = batch['labels2d'][:, :num_nearest_imgs, :, :]
                labels2d = labels2d.reshape(num_imgs, labels2d.shape[2], labels2d.shape[3])

            # collapse batch size and "num nearest img" dims
            # N, H, W (30, 40)
            depths = depths.reshape(num_imgs, depths.shape[2], depths.shape[3])
            # N, 4, 4
            poses = poses.reshape(num_imgs, poses.shape[2], poses.shape[3])
            # N, H, W (240, 320)
            rgbs = rgbs.reshape(num_imgs, 3, rgbs.shape[3], rgbs.shape[4])
            # N, 1
            frames = frames.reshape(num_imgs, 1)

            # repeat the w2g transform for each image
            # add an extra dimension 
            transforms = world_to_grid.unsqueeze(1)
            transforms = transforms.expand(bsize, num_nearest_imgs, 4, 4).contiguous().view(-1, 4, 4).to(self.device)
        
        # model forward pass 
        out = self(x, rgbs, depths, poses, transforms, frames, return_features=self.contrastive)
        
        # skip this batch
        if out is None:
            print('skip batch')
            return None
        
        losses = {}
        # main cross entropy loss
        losses['crossent'] = F.cross_entropy(out['logits3d'], batch['y'], 
                                weight=self.get_class_weights(),
                                ignore_index=self.target_padding)
        
        preds3d = out['logits3d'].argmax(dim=1)

        # 2d cross ent loss if needed
        if self.train_2d:
            losses['crossent2d'] = F.cross_entropy(out['logits2d'], labels2d, 
                                weight=self.features_2d.get_class_weights(),
                                ignore_index=self.target_padding)

        if self.contrastive:
            n_points = self.contr_cfg['n_points']

            # get N,C vectors for 2d and 3d
            feat2d_all, feat3d_all = out['feat2d'], out['feat3d']
            # sample N of these
            n_feats = len(feat2d_all)
            # actual number of points to compute loss over
            n_points_actual = min(n_points, n_feats)
            # shuffle all the feats, then pick the required number
            inds = torch.randperm(n_feats)[:n_points_actual]

            feat2d, feat3d = feat2d_all[inds], feat3d_all[inds]

            ct_loss = pointinfoNCE_loss(feat2d, feat3d, self.contr_cfg)
            losses['contrastive'] = ct_loss

        return preds3d, losses


    def rgb_to_feat3d(self, rgbs, depths, poses, transforms, frames):
        '''
        rgbs: N, C, H, W
        depths: N, H, W
        poses: N, 4, 4
        transforms: N, 4, 4
        N = batch_size of subvols * num_nearest_images

        output: (N, 128,) + subvol_size
        2d-3d projection indices (batch size, 1+32*32*32) 
            = indices into the flattened 3d volume with 2d features

        NOTE: during inference, if there are no corresponding RGBs, 
             0 features are created. no need to do projection
        '''
        # compute projection mapping b/w 2d and 3d
        # get 2d features from images
        proj_mapping = []
        # number of proj indices for each sample
        num_inds = torch.prod(torch.Tensor(self.subvol_size)).long().item()

        for d, c, t, f in zip(depths, poses, transforms, frames):
            # if sample has frames, pose is invertible
            proj = None
            # check if the pose is valid and invertible
            det_c = torch.det(c)
            if (-1 not in f) and (det_c != 0) and (not torch.isnan(det_c)):
                proj = self.projection.compute_projection(d, c, t)
            # no frame or no projection -> 
            # set projection indices to zero, use 0 features
            if (proj is None) or (-1 in f):
                # first element is the number of inds, zero -> no mapping
                ind3d_zero = torch.zeros(num_inds + 1, dtype=int).to(self.device)
                proj = (ind3d_zero, ind3d_zero.clone())
            proj_mapping.append(proj)

        proj_mapping = list(zip(*proj_mapping))
        proj_ind_3d = torch.stack(proj_mapping[0])
        proj_ind_2d = torch.stack(proj_mapping[1])

        out = self.features_2d(rgbs, return_features=True, return_preds=self.train_2d)
        if self.train_2d:
            feat2d, logits2d = out
        else:

            feat2d, logits2d = out, None

        # get C,D,H,W for each feature map
        # pass empty ind3d -> get zero features
        feat2d_proj = [project_2d_3d(ft, ind3d, ind2d, self.subvol_size) \
                            for ft, ind3d, ind2d in \
                            zip(feat2d, proj_ind_3d, proj_ind_2d)]
        N = rgbs.shape[0]
        # N x (C, D, H, W) -> C, D, H, W, N     
        # N = #volumes x #imgs/vol 
        # keep the features from different images separate                      
        feat2d_proj = torch.stack(feat2d_proj, dim=4)   
        
        # C, D, H, W, N
        sz = feat2d_proj.shape

        # keep the feats from each view separately as well, can be used later
        num_nearest_imgs = self.hparams['cfg']['data']['num_nearest_images']
        # N, num_nearest_images, C, D, H, W
        feat2d_all = feat2d_proj.permute(4, 0, 1, 2, 3).reshape(-1, num_nearest_imgs, 
                                                    sz[0], sz[1], sz[2], sz[3])
        
        # reshape to max pool over features
        # C, D*H*W, N
        feat2d_proj = feat2d_proj.view(sz[0], -1, N)
        # pool features from images
        # N, C, Lin -> N, C, Lout
        # kernel_size = #imgs/vol -> pool over all the images associated with a voxel

        # pool over all the images that contributed to a volume
        feat2d_proj = self.pooling(feat2d_proj)

        # back to C, D, H, W, #vols
        feat2d_proj = feat2d_proj.view(sz[0], sz[1], sz[2], sz[3], -1)
        # back to #vols, C, D, H, W
        feat2d_proj = feat2d_proj.permute(4, 0, 1, 2, 3)   

        return feat2d_proj, proj_ind_3d, logits2d, feat2d_all

    def forward(self, x, rgbs, depths, poses, transforms, frames, return_features=False):
        '''
        All the differentiable ops here
        return_features: return the intermediate 2d and 3d features
        '''
        input_x = x
        # fwd pass on rgb, then project to 3d volume and get features
        out = self.rgb_to_feat3d(rgbs, depths, poses, transforms, frames)
        # skip this batch
        if out is None:
            return None

        feat2d_proj, feat2d_ind3d, logits2d = out 
        # reduce 2d feat dim with same conv layers, then contrast
        for layer in self.feat2d_same:
            feat2d_proj = layer(feat2d_proj)

        feat2d = feat2d_proj.clone()
        # fwd pass projected features through convs
        for layer in self.layers_2d:
            feat2d = layer(feat2d)

        # same layers of the 3d branch
        for layer in self.layers3d_same:
            x = layer(x)
        
        # store the original res 3d features

        outs = []
        # down layers
        # store the outputs of all but the last one
        for layer in self.layers3d_down:
            x = layer(x)
            outs.append(x)

        # reverse the outputs, remove the last (lowest one)
        outs = outs[::-1][1:]

        # concat features from 2d with 3d along the channel dim
        x = torch.cat([x, feat2d], dim=1)

        # first up layer, input is 2d+3d
        x = self.layers3d_up[0](x)

        # up layers
        up_inputs = []
        for ndx, layer in enumerate(self.layers3d_up[1:]):
            x = torch.cat([x, outs[ndx]], dim=1)
            up_inputs.append(x)
            x = layer(x)

        ret_dict = {'logits3d': x, 'logits2d': logits2d}

        if return_features and self.contrastive:
            feat3d = self.up3d_contr(up_inputs[1])
            feat_dim = feat3d.shape[1]

            feat2d_vecs, feat3d_vecs = [], []  

            for ndx in range(input_x.shape[0]):
                # proj inds for this sample
                proj3d = feat2d_ind3d[ndx]
                # number of projected voxels
                num_inds = proj3d[0]
                # the  indices into the CDHW volume
                ind3d = proj3d[1:1+num_inds]
                # out of all projected locations, which locations are occupied
                # in the input?
                occupied_mask = (input_x[ndx].squeeze().view(-1)[ind3d] == 1)
                overlap_inds = ind3d[occupied_mask]
                # pick 3d feats at these locations
                feat3d_vecs.append(feat3d[ndx].view(feat_dim, -1)[:, overlap_inds])
                # pick 2d feats at these locations
                feat2d_vecs.append(feat2d_proj[ndx].view(feat_dim, -1)[:, overlap_inds])
                
            ret_dict['feat2d'] = torch.cat(feat2d_vecs, -1).T
            ret_dict['feat3d'] = torch.cat(feat3d_vecs, -1).T

        return ret_dict

class UNet2D3D_3DMV(UNet2D3D):
    def init_model(self):
        self.pooling = nn.MaxPool1d(kernel_size=self.hparams['cfg']['data']['num_nearest_images'])

        self.nf0 = 32 
        self.nf1 = 64 
        self.nf2 = 128 

        if self.use_2dfeat:
            # 32->16
            self.down1_2d = Down3D_Big(self.nf2, self.nf1) 
            self.down2_2d = Down3D_Big(self.nf1, self.nf0) 

        # 3d conv on subvols
        # 2 down blocks
        # 32->16
        self.down1 = Down3D_Big(self.in_channels, self.nf0) 
        self.down2 = Down3D_Big(self.nf0, self.nf1)
        self.down3 = Down3D_Big(self.nf1, self.nf2)
        # layers on top of combined features
        # one down block, 3 up blocks
        self.up1 = Up3D_Big(self.nf2, self.nf2)
        # previous layer + 2D features + skip connection to 3D 
        feat2d_dim = self.nf0 if self.use_2dfeat else 0
        self.up2 = Up3D_Big(self.nf2+feat2d_dim+self.nf1, self.nf2)
        self.up3 = Up3D_Big(self.nf2+self.nf0, self.nf2) 

        self.pred_layer = nn.Conv3d(self.nf2, self.num_classes, 3, 1, 1)
    
    def forward(self, x, rgbs, depths, poses, transforms, frames, return_features=False):
        '''
        All the differentiable ops here
        return_features: return the intermediate 2d and 3d features
        '''
        logits2d = None
        if self.use_2dfeat or self.contrastive:
            # fwd pass on rgb, then project to 3d volume and get features
            out = self.rgb_to_feat3d(rgbs, depths, poses, transforms, frames)
            # skip this batch
            if out is None:
                return None

            feat2d_proj, feat2d_ind3d, logits2d, feat2d_all = out 

        if self.use_2dfeat:
            # conv on 2d feats, down
            x2d_16 = self.down1_2d(feat2d_proj)
            x2d_8 = self.down2_2d(x2d_16)

        # conv on 3d, down and 1 up
        x16 = self.down1(x)
        x8 = self.down2(x16)
        x4 = self.down3(x8)

        xup8 = self.up1(x4)

        # if using 2d feats: upconvs+skip connection+2d feats
        # else plain 3d network
        inputs = (xup8, x8) + ((x2d_8,) if self.use_2dfeat else ())
        xup16 = self.up2(torch.cat(inputs, 1))
        xup32 = self.up3(torch.cat((xup16, x16), 1))
            
        out = self.pred_layer(xup32)

        ret_dict = {'logits3d': out, 'logits2d': logits2d}

        if return_features and self.contrastive:
            feat3d = xup32
            feat_dim = feat3d.shape[1]

            feat2d_vecs, feat3d_vecs = [], []  

            # N, num_nearest_images, C, D, H, W
            num_nearest_images = feat2d_all.shape[1]

            for ndx in range(x.shape[0]):
                # proj inds for this sample
                proj3d = feat2d_ind3d[ndx]
                # number of projected voxels
                num_inds = proj3d[0]
                # the  indices into the CDHW volume
                ind3d = proj3d[1:1+num_inds]
                # out of all projected locations, which locations are occupied
                # in the input?
                occupied_mask = (x[ndx].squeeze().view(-1)[ind3d] == 1)
                overlap_inds = ind3d[occupied_mask]

                # contrast with 2d feats that have not been pooled
                if self.contr_cfg.get('contrast_unpooled', False):
                    for view_ndx in range(num_nearest_images):
                        # pick the same 3d feats at these locations repeatedly
                        feat3d_flat = feat3d[ndx].view(feat_dim, -1)
                        feat3d_vecs.append(feat3d_flat[:, overlap_inds])

                        # pick 2d feats at these locations from the n-th view
                        feat2d_flat = feat2d_all[ndx, view_ndx].view(feat_dim, -1)
                        feat2d_vecs.append(feat2d_flat[:, overlap_inds])
                else:
                    # pick the 3d feats
                    feat3d_flat = feat3d[ndx].view(feat_dim, -1)
                    feat3d_vecs.append(feat3d_flat[:, overlap_inds])

                    # pick 2d feats
                    feat2d_flat = feat2d_proj[ndx].view(feat_dim, -1)
                    feat2d_vecs.append(feat2d_flat[:, overlap_inds])

            ret_dict['feat2d'] = torch.cat(feat2d_vecs, -1).T
            ret_dict['feat3d'] = torch.cat(feat3d_vecs, -1).T

        return ret_dict

def l2_norm_vecs(vecs, eps=1e-6):
    vecs_norm = vecs / (torch.norm(vecs, p=2, dim=1, keepdim=True) + eps)
    return vecs_norm

def hardest_contrastive_loss(feat1, feat2, margin_pos, margin_neg):
    # L2 normalize
    feat1_norm = l2_norm_vecs(feat1).unsqueeze(0)
    feat2_norm = l2_norm_vecs(feat2).unsqueeze(0)
    dists = torch.cdist(feat1_norm, feat2_norm).squeeze()
    # loss from positive pairs
    loss_pos = ((dists.diagonal() - margin_pos).clamp(min=0)**2).mean()

    # set diagonal to inf, then find the closest negatives
    ind = torch.arange(dists.shape[0])  
    dists_tmp = dists.clone()
    dists_tmp[ind, ind] = float('inf')

    loss_neg = 0.5*(
          (margin_neg - dists_tmp.min(axis=1)[0]).clamp(min=0)**2 \
        + (margin_neg - dists_tmp.min(axis=0)[0]).clamp(min=0)**2).mean()

    loss = loss_pos + loss_neg

    return loss

def NCE_loss(feat1, feat2, temp, negatives=None):
    '''
    contrasts feat2 with feat1 (feat2 in numerator, rest in denominator)

    feat1, feat2: N, C
    temp: temperature
    '''
    feat1_norm = l2_norm_vecs(feat1)
    feat2_norm = l2_norm_vecs(feat2)

    n_points = len(feat1)

    # find all pair feature distances
    # multiply (N,C) and (C,N), get (N,N)
    scores = torch.matmul(feat2_norm, feat1_norm.T)

    # use extra negatives in the denominator
    if n_points > 0 and negatives is not None:
        negatives_norm = l2_norm_vecs(negatives)
        # scores for negatives - dot product of corresponding pairs 
        neg_scores = torch.matmul(feat2_norm, negatives_norm.T)
        # remove the diagonal elements
        mask = torch.ones_like(neg_scores, dtype=bool)
        mask[torch.arange(n_points), torch.arange(n_points)] = False
        neg_scores_new = torch.masked_select(neg_scores, mask).reshape(n_points, -1)
        # append to the scores matrix
        # NxN + NxN-1 -> Nx2N-1
        scores = torch.cat((scores, neg_scores_new), -1)

    labels = torch.arange(len(feat1)).to(feat1.device)
    # get contrastive loss
    ct_loss = F.cross_entropy(scores/temp, labels)

    return ct_loss

def pointinfoNCE_loss(feat2d, feat3d, contr_cfg):
    '''
    feat2d: 2d features
    feat3d: 3d features
    '''
    temp = contr_cfg['temperature']

    if 'extra_pairs' in contr_cfg:
        if '3d' in contr_cfg['extra_pairs']:
            ct_loss = NCE_loss(feat2d, feat3d, temp, negatives=feat3d)
    else:
        ct_loss = NCE_loss(feat2d, feat3d, temp)

    return ct_loss

def map_indices(inds_list, in_dims, out_dims):
    '''
    inds_list: 1 + 32*32*32 indices of projected features, first elem = num inds
    in_dims: dims of the original volume (32^3)
    out_dims: dims of the target volume (4^3)
    '''
    num_inds = inds_list[0]
    inds = inds_list[1:1+num_inds]
    coords = inds_list.new_empty(4, num_inds)
    coords = ProjectionHelper.lin_ind_to_coords_static(inds, coords, in_dims)
    
    new_coords = ProjectionHelper.downsample_coords(coords, in_dims, out_dims)
    max_num_inds = len(inds_list) - 1
    new_inds = ProjectionHelper.coords_to_lin_inds(new_coords, max_num_inds, out_dims)

    return new_inds

def pick_features(vol, inds_list):
    '''
    vol: CDHW
    inds_list: 1 + 32*32*32 indices of projected features, first elem = num inds

    pick the features from vol using the indices in inds_list
    indices are according to WHD dims
    '''
    num_inds = inds_list[0]
    n_channels = vol.shape[0]

    if num_inds > 0:
        inds = inds_list[1:1+num_inds]
        
        # change CDHW -> WHDC and then pick features
        vecs = vol.permute(3, 2, 1, 0).reshape(-1, n_channels)[inds]
    else:
        vecs = torch.empty(0, n_channels)

    return vecs 
        

        

