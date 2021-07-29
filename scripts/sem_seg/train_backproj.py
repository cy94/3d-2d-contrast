import argparse
from datasets.scannet.utils_3d import adjust_intrinsic, make_intrinsic
from models.sem_seg.enet import ENet2
from models.sem_seg.fcn3d import UNet2D3D

from lib.misc import read_config
from models.sem_seg.utils import count_parameters

from torchvision.transforms import Compose
from torch.utils.data import Subset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from datasets.scannet.sem_seg_3d import ScanNet2D3DH5
from transforms.grid_3d import AddChannelDim, TransposeDims, LoadDepths, LoadPoses,\
                                LoadRGBs

def main(args):
    cfg = read_config(args.cfg_path)

    t = Compose([
        AddChannelDim(),
        TransposeDims(),
        LoadDepths(cfg),
        LoadPoses(cfg),
        LoadRGBs(cfg)
    ])

    train_set = ScanNet2D3DH5(cfg['data'], 'train', transform=t)
    val_set = ScanNet2D3DH5(cfg['data'], 'val', transform=t)
    print(f'Train set: {len(train_set)}')
    print(f'Val set: {len(val_set)}')

    if args.subset:
        print('Select a subset of data for quick run')
        train_set = Subset(train_set, range(1024))
        val_set = Subset(val_set, range(1024))

    train_loader = DataLoader(train_set, batch_size=cfg['train']['train_batch_size'],
                            collate_fn=ScanNet2D3DH5.collate_func,
                            shuffle=True, num_workers=8,
                            pin_memory=True)  

    val_loader = DataLoader(val_set, batch_size=cfg['train']['val_batch_size'],
                            collate_fn=ScanNet2D3DH5.collate_func,
                            shuffle=False, num_workers=8,
                            pin_memory=True) 

    features_2d = ENet2.load_from_checkpoint(cfg['model']['ckpt_2d'])

    # intrinsic of the color camera from scene0001_00
    intrinsic = make_intrinsic(1170.187988, 1170.187988, 647.75, 483.75)
    # adjust for smaller image size
    intrinsic = adjust_intrinsic(intrinsic, [1296, 968], cfg['data']['proj_img_size'])

    model = UNet2D3D(in_channels=1, num_classes=cfg['data']['num_classes'], cfg=cfg, 
                    features_2d=features_2d, intrinsic=intrinsic)

    print(f'Num params: {count_parameters(model)}')                                                      

    callbacks = [LearningRateMonitor(logging_interval='step')]
    if not args.no_ckpt:
        print('Saving checkpoints')
        callbacks.append(ModelCheckpoint(save_last=True, save_top_k=5, 
                                        monitor='iou/val/mean',
                                        mode='max',
                                # put the miou in the filename
                                filename='epoch{epoch:02d}-step{step}-miou{iou/val/mean:.2f}',
                                auto_insert_metric_name=False))
    ckpt = cfg['train']['resume']                                             
    if ckpt is not None:
        print(f'Resuming from checkpoint: {ckpt}')

    trainer = pl.Trainer(resume_from_checkpoint=ckpt,
                        gpus=1 if not args.cpu else None, 
                        log_every_n_steps=10,
                        callbacks=callbacks,
                        max_epochs=cfg['train']['epochs'],
                        val_check_interval=cfg['train']['eval_intv'],
                        limit_val_batches=cfg['train']['limit_val_batches'],
                        fast_dev_run=args.fast_dev_run,
                        accumulate_grad_batches=cfg['train'].get('accum_grad', 1))

    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cfg_path', help='Path to cfg')
    p.add_argument('--no-ckpt', action='store_true', dest='no_ckpt', 
                    default=False, help='Dont store checkpoints (for debugging)')
    p.add_argument('--cpu', action='store_true', dest='cpu', 
                    default=False, help='Train on CPU')                    
    p.add_argument('--subset', action='store_true', dest='subset', 
                    default=False, help='Use a subset of dataset')

    parser = pl.Trainer.add_argparse_args(p)
    args = p.parse_args()

    main(args)