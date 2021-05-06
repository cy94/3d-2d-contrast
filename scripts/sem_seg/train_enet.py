from os import read
import argparse

from lib.misc import read_config
from datasets.scannet.sem_seg_2d import ScanNetSemSeg2D, collate_func

from torch.utils.data import Subset, DataLoader


def main(args):
    cfg = read_config(args.cfg_path)
    dataset = ScanNetSemSeg2D(cfg['data']['root'], cfg['data']['label_file'],
                                cfg['data']['limit_scans'])
    if cfg['train']['train_split']:
        train_size = int(cfg['train']['train_split'] * len(dataset))
        train_set = Subset(dataset, range(train_size))
        val_set = Subset(dataset, range(train_size, len(dataset)))
    elif cfg['train']['train_size'] and cfg['train']['val_size']:
        train_set = Subset(dataset, range(cfg['train']['train_size']))
        val_set = Subset(dataset, range(cfg['train']['train_size'], cfg['train']['val_size']))
    else:
        raise ValueError('Train val split not specified')

    train_loader = DataLoader(train_set, batch_size=cfg['train']['train_batch_size'],
                            shuffle=True, num_workers=4, collate_fn=collate_func)        
    val_loader = DataLoader(train_set, batch_size=cfg['train']['train_batch_size'],
                            shuffle=False, num_workers=4, collate_fn=collate_func)      

    for batch in train_loader:
        img, label = batch['img'], batch['label']
        print(img.min(), img.max())
        print(label.min(), label.max())
        print(img.shape, label.shape)  

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cfg_path', help='Path to cfg')
    args = p.parse_args()

    main(args)