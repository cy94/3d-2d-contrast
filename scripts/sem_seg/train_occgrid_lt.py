
import argparse

from lib.misc import read_config
from datasets.scannet.sem_seg_3d import ScanNetSemSegOccGrid, collate_func
from transforms.grid_3d import AddChannelDim, TransposeDims, MapClasses
from models.sem_seg.utils import count_parameters
from models.sem_seg.fcn3d import FCN3D, UNet3D

from torchinfo import summary
import torch
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import Compose

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


def main(args):
    cfg = read_config(args.cfg_path)

    # create transforms list
    transforms = []
    transforms.append(AddChannelDim())
    transforms.append(TransposeDims())
    # map none class to padding, no loss on this class
    transforms.append(MapClasses({0: cfg['data']['target_padding']}))
    t = Compose(transforms)

    if cfg['data']['train_list'] and cfg['data']['val_list']:
        train_set = ScanNetSemSegOccGrid(cfg['data'], transform=t, split='train')
        val_set = ScanNetSemSegOccGrid(cfg['data'], transform=t, split='val')
    else:
        dataset = ScanNetSemSegOccGrid(cfg['data'], transform=t)
        print(f'Full dataset size: {len(dataset)}')
        if cfg['train']['train_split']:
            train_size = int(cfg['train']['train_split'] * len(dataset))
            train_set = Subset(dataset, range(train_size))
            val_set = Subset(dataset, range(train_size, len(dataset)))
        elif cfg['train']['train_size'] and cfg['train']['val_size']:
            train_set = Subset(dataset, range(cfg['train']['train_size']))
            val_set = Subset(dataset, range(cfg['train']['train_size'], 
                                cfg['train']['train_size']+cfg['train']['val_size']))
        else:
            raise ValueError('Train val split not specified')

    print(f'Train set: {len(train_set)}')
    print(f'Val set: {len(val_set)}')
    
    print(f'Prepare a fixed val set')
    val_set = [s for s in val_set]


    train_loader = DataLoader(train_set, batch_size=cfg['train']['train_batch_size'],
                            shuffle=True, num_workers=8, collate_fn=collate_func,
                            pin_memory=True)  

    val_loader = DataLoader(val_set, batch_size=cfg['train']['val_batch_size'],
                            shuffle=False, num_workers=8, collate_fn=collate_func,
                            pin_memory=True) 

    models = {
        'FCN3D': FCN3D,
        'UNet3D': UNet3D,
    }
    model = models[cfg['model']['name']](in_channels=1, num_classes=21, cfg=cfg)
    print(f'Num params: {count_parameters(model)}')

    input_size = (cfg['train']['train_batch_size'], 1,) + tuple(cfg['data']['subvol_size'])
    summary(model, input_size=input_size)

    if cfg['train']['resume']:
        pass
    
    checkpoint_callback = ModelCheckpoint(save_last=True, save_top_k=5, 
                                        monitor='loss/val')

    trainer = pl.Trainer(gpus=1, 
                        auto_scale_batch_size='binsearch',
                        log_every_n_steps=10,
                        callbacks=[checkpoint_callback],
                        max_epochs=cfg['train']['epochs'],
                        val_check_interval=cfg['train']['eval_intv'],
                        fast_dev_run=args.fast_dev_run)
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cfg_path', help='Path to cfg')
    parser = pl.Trainer.add_argparse_args(p)
    args = p.parse_args()

    main(args)