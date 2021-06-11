
import argparse

import torch

from lib.misc import read_config
from datasets.scannet.sem_seg_3d import ScanNetSemSegOccGrid

from models.sem_seg.utils import count_parameters, MODEL_MAP, get_transform

import pytorch_lightning as pl

def main(args):
    cfg = read_config(args.cfg_path)
    train_cfg = torch.load(cfg['test']['ckpt'])['hyper_parameters']['cfg']

    # create transforms list
    t = get_transform(train_cfg, 'val')

    model = MODEL_MAP[train_cfg['model']['name']].load_from_checkpoint(cfg['test']['ckpt'],
                                                in_channels=1, num_classes=21)
    model.eval()
    print(f'Num params: {count_parameters(model)}')

    test_set = ScanNetSemSegOccGrid(cfg['data'], split='test', full_scene=True)
    print(f'Test set: {len(test_set)}')

    model.test_scenes(test_set, cfg, transform=t)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cfg_path', help='Path to cfg')
    parser = pl.Trainer.add_argparse_args(p)
    args = p.parse_args()

    main(args)