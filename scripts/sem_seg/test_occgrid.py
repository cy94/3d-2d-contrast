
import argparse
from datasets.scannet.sem_seg_3d import ScanNetGridTestSubvols
from datasets.scannet.utils import get_dataset, get_loader, get_transform_dense
import torch

from torch.utils.data import ConcatDataset
from lib.misc import read_config

from models.sem_seg.utils import MODEL_MAP, SPARSE_MODELS

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    cfg = read_config(args.cfg_path)
    train_cfg = torch.load(cfg['test']['ckpt'], map_location=device)['hyper_parameters']['cfg']

    # add the test_path to the train cfg
    # data params + path now in a single cfg
    train_cfg['data']['test_list'] = cfg['data']['test_list']
    # change 
    test_set = get_dataset(train_cfg, split='test')
    print(f'Test set: {len(test_set)} scenes')
    
    # dense dataset of full scenes -
    # get sliding window of chunks from each scene and combine into a single dataset
    is_sparse = train_cfg['model']['name'] in SPARSE_MODELS
    if not is_sparse:
        subvol_transform = get_transform_dense(train_cfg, 'test')

        subvol_datasets = [ScanNetGridTestSubvols(scene,
                                        # subvolume size from full scenes 
                                        train_cfg['data']['subvol_size'], 
                                        # value used to pad incomplete subvols
                                        train_cfg['data']['target_padding'],
                                        # full scene doesnt get transformed
                                        # transform subvols as usual
                                        transform=subvol_transform)
                            for scene in test_set
                        ]
        test_set = ConcatDataset(subvol_datasets)
        print(f'Test set: {len(test_set)} chunks')
        in_channels = 1
    else:
        in_channels = 3

    test_loader = get_loader(test_set, train_cfg, split='test',
                            batch_size=cfg['test']['batch_size'])

    num_classes = train_cfg['data'].get('num_classes', 20)

    model = MODEL_MAP[train_cfg['model']['name']].load_from_checkpoint(cfg['test']['ckpt'],
                                                map_location=device,
                                                in_channels=in_channels, num_classes=num_classes)
    model.eval()

    model.test_scenes(test_loader)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cfg_path', help='Path to cfg')
    args = p.parse_args()

    main(args)