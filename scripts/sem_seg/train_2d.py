import argparse
from lib.misc import get_args, get_logger_and_callbacks, read_config
from datasets.scannet.sem_seg_2d import ScanNetSemSeg2D
from models.sem_seg.utils import MODEL_MAP_2D
from transforms.image_2d import ColorJitter, GaussNoise, GaussianBlur, HueSaturationValue, LRFlip, Normalize

from torchsummary import summary

from torch.utils.data import Subset, DataLoader

import pytorch_lightning as pl

import numpy as np

from torchvision.transforms import Compose

from copy import deepcopy

def main(args):
    cfg = read_config(args.cfg_path)

    train_t = Compose([
        LRFlip(),
        ColorJitter(),
        HueSaturationValue(),
        GaussianBlur(),
        GaussNoise(),
        Normalize(),
    ])
    val_t = Normalize()

    train_set = ScanNetSemSeg2D(cfg, transform=train_t, split='train')
    print('Set val frame skip to 30')
    val_cfg = deepcopy(cfg)
    val_cfg['data']['frame_skip'] = 30
    val_set = ScanNetSemSeg2D(val_cfg, transform=val_t, split='val')        
    
    collate_func = train_set.collate_func
    
    if args.subset:
        print('Select a subset of data for quick run')
        train_set = Subset(train_set, range(2))
        val_set = Subset(val_set, range(16))                        

    print(f'Train set: {len(train_set)}')
    print(f'Val set: {len(val_set)}')

    train_loader = DataLoader(train_set, batch_size=cfg['train']['train_batch_size'],
                            shuffle=True, num_workers=4, collate_fn=collate_func)  

    val_loader = DataLoader(val_set, batch_size=cfg['train']['val_batch_size'],
                            shuffle=False, num_workers=4, collate_fn=collate_func)      

    # ignored label in 2D is called target padding in 3D - use a common name 
    cfg['data']['target_padding'] = cfg['data']['ignore_label']

    # pick the 2d model
    model = MODEL_MAP_2D[cfg['model']['name']](num_classes=cfg['data']['num_classes'], cfg=cfg)
    
    model = model.cuda()
    input_dims = (1, 3) + tuple(cfg['data']['img_size'][::-1])
    print(input_dims)
    # summary(model, input_dims)

    wblogger, callbacks = get_logger_and_callbacks(args, cfg)

    ckpt = cfg['train']['resume']

    trainer = pl.Trainer(resume_from_checkpoint=ckpt,
                        logger=wblogger,
                        num_sanity_val_steps=0,
                        gpus=1 if not args.cpu else 0, 
                        log_every_n_steps=10,
                        callbacks=callbacks,
                        max_epochs=cfg['train']['epochs'],
                        val_check_interval=cfg['train']['eval_intv'],
                        limit_val_batches=cfg['train']['limit_val_batches'],
                        fast_dev_run=args.fast_dev_run,
                        accumulate_grad_batches=cfg['train'].get('accum_grad', 1))

    trainer.fit(model, train_loader, val_loader)

  
if __name__ == '__main__':
    args = get_args()
    main(args)