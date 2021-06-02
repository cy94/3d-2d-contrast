
import argparse


from lib.misc import read_config
from datasets.scannet.sem_seg_3d import ScanNetSemSegOccGrid, ScanNetGridTestSubvols, collate_func
from transforms.grid_3d import AddChannelDim, TransposeDims
from models.sem_seg.utils import count_parameters
from models.sem_seg.fcn3d import FCN3D

from torchinfo import summary
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

import pytorch_lightning as pl

def main(args):
    cfg = read_config(args.cfg_path)

    model = FCN3D.load_from_checkpoint(cfg['test']['ckpt'])
    model.eval()
    print(f'Num params: {count_parameters(model)}')

    ckpt = torch.load(cfg['test']['ckpt'], map_location='cpu')
    train_cfg = ckpt['hyper_parameters']['cfg']

    # create transforms list
    transforms = []
    transforms.append(AddChannelDim())
    transforms.append(TransposeDims())
    t = Compose(transforms)

    test_set = ScanNetSemSegOccGrid(cfg['data'], split='test', full_scene=True)
    print(f'Test set: {len(test_set)}')

    for scene in test_set:
        subvols = ScanNetGridTestSubvols(scene, train_cfg['data']['subvol_size'], 
                                        target_padding=train_cfg['data']['target_padding'], 
                                        transform=t)
        test_loader = DataLoader(subvols, batch_size=cfg['test']['batch_size'],
                                shuffle=False, num_workers=8, collate_fn=collate_func,
                                pin_memory=True) 

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cfg_path', help='Path to cfg')
    parser = pl.Trainer.add_argparse_args(p)
    args = p.parse_args()

    main(args)