import os 
import argparse
from lib.misc import read_config
from datasets.scannet.sem_seg_2d import ScanNetSemSeg2D, collate_func
from transforms.image_2d import Normalize, TransposeChannels, Resize
from models.sem_seg.enet import ENet2
from models.sem_seg.utils import count_parameters

from torch.utils.data import Subset, DataLoader
from torchinfo import summary

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from torchvision.transforms import Compose


def main(args):
    cfg = read_config(args.cfg_path)

    # create transforms list
    transforms = []
    if cfg['data']['img_size'] is not None:
        transforms.append(Resize(cfg['data']['img_size']))
    transforms.append(Normalize())
    # image is already H, W, C
    # only move channels to the front -> C, H, W
    transforms.append(TransposeChannels())

    t = Compose(transforms)

    train_set = ScanNetSemSeg2D(cfg, transform=t, split='train')
    val_set = ScanNetSemSeg2D(cfg, transform=t, split='val')        

    if args.subset:
        print('Select a subset of data for quick run')
        train_set = Subset(train_set, range(128))
        val_set = Subset(val_set, range(128))                        

    print(f'Train set: {len(train_set)}')
    print(f'Val set: {len(val_set)}')

    train_loader = DataLoader(train_set, batch_size=cfg['train']['train_batch_size'],
                            shuffle=True, num_workers=4, collate_fn=collate_func)  

    val_loader = DataLoader(val_set, batch_size=cfg['train']['val_batch_size'],
                            shuffle=False, num_workers=4, collate_fn=collate_func)      

    # ignored label in 2D is called target padding in 3D - use a common name 
    cfg['data']['target_padding'] = cfg['data']['ignore_label']
    model = ENet2(num_classes=cfg['data'].get('num_classes', 20), cfg=cfg)

    input_size = (cfg['train']['train_batch_size'], 3,) + tuple(cfg['data']['img_size'][::-1])
    summary(model, input_size=input_size)

    callbacks = [LearningRateMonitor(logging_interval='step')]
    if not args.no_ckpt:
        print('Saving checkpoints')
        callbacks.append(ModelCheckpoint(save_last=True, save_top_k=5, 
                                        monitor='iou/val/mean',
                                        mode='max',
                                # put the miou in the filename
                                filename='epoch{epoch:02d}-step{step}-miou{iou/val/mean:.2f}',
                                auto_insert_metric_name=False))
    ckpt = cfg['train'].get('resume', None)
    if ckpt is not None:
        print(f'Resuming from checkpoint: {ckpt}')

    trainer = pl.Trainer(resume_from_checkpoint=ckpt,
                        gpus=1, 
                        log_every_n_steps=10,
                        callbacks=callbacks,
                        max_epochs=cfg['train']['epochs'],
                        val_check_interval=cfg['train']['eval_intv'],
                        fast_dev_run=args.fast_dev_run,
                        accumulate_grad_batches=cfg['train'].get('accum_grad', 1))
    trainer.fit(model, train_loader, val_loader)

  
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cfg_path', help='Path to cfg')
    p.add_argument('--no-ckpt', action='store_true', dest='no_ckpt', 
                    default=False, help='Dont store checkpoints (for debugging)')
    p.add_argument('--subset', action='store_true', dest='subset', 
                    default=False, help='Use a subset of dataset')
    parser = pl.Trainer.add_argparse_args(p)
    args = p.parse_args()

    main(args)