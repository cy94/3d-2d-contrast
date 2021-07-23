from datasets.scannet.sem_seg_3d import ScanNet2D3DH5
import random
import argparse

from lib.misc import read_config
from models.sem_seg.utils import count_parameters

from torchinfo import summary
from torch.utils.data import Subset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def main(args):
    cfg = read_config(args.cfg_path)

    train_set = ScanNet2D3DH5(cfg['data'], 'train')
    val_set = ScanNet2D3DH5(cfg['data'], 'val')
    print(f'Train set: {len(train_set)}')
    print(f'Val set: {len(val_set)}')

    if args.subset:
        print('Select a subset of data for quick run')
        train_set = Subset(train_set, range(1024))
        val_set = Subset(val_set, range(1024))

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