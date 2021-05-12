import os 
import argparse
from datetime import datetime as dt
import random 

import numpy as np

from tqdm import tqdm

from lib.misc import read_config
from datasets.scannet.sem_seg_3d import ScanNetSemSegOccGrid, collate_func
from transforms.grid_3d import AddChannelDim, Pad, TransposeDims
from models.sem_seg.utils import count_parameters

import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import Compose
from torch.utils.tensorboard import SummaryWriter


def main(args):
    cfg = read_config(args.cfg_path)

    # create transforms list
    transforms = []
    transforms.append(Pad(cfg['data']['grid_size']))
    transforms.append(AddChannelDim())
    transforms.append(TransposeDims())
    t = Compose(transforms)

    dataset = ScanNetSemSegOccGrid(cfg['data']['root'],
                                cfg['data']['limit_scans'],
                                transform=t)

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

    train_loader = DataLoader(train_set, batch_size=cfg['train']['train_batch_size'],
                            shuffle=True, num_workers=4, collate_fn=collate_func)  

    val_loader = DataLoader(val_set, batch_size=cfg['train']['val_batch_size'],
                            shuffle=False, num_workers=4, collate_fn=collate_func) 

    for batch in train_loader:
        x, y = batch['x'], batch['y']
        print(x.shape, y.shape)     
  
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cfg_path', help='Path to cfg')
    p.add_argument('--quick', dest='quick_run', action='store_true', help='Quick run?')
    args = p.parse_args()

    main(args)