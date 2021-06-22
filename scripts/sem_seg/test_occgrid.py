
import argparse
from datasets.scannet.utils import get_trainval_loaders, get_trainval_sets

import torch

from lib.misc import read_config

from models.sem_seg.utils import MODEL_MAP

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    cfg = read_config(args.cfg_path)
    train_cfg = torch.load(cfg['test']['ckpt'], map_location=device)['hyper_parameters']['cfg']

    _, test_set = get_trainval_sets(train_cfg)
    print(f'Val set: {len(test_set)}')
    _, test_loader = get_trainval_loaders([], test_set, train_cfg)

    model = MODEL_MAP[train_cfg['model']['name']].load_from_checkpoint(cfg['test']['ckpt'],
                                                map_location=device,
                                                in_channels=3, num_classes=20)
    model.eval()

    model.test_scenes(test_loader)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cfg_path', help='Path to cfg')
    args = p.parse_args()

    main(args)