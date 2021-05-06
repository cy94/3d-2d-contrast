from os import read
import argparse

from tqdm import tqdm

from lib.misc import read_config
from datasets.scannet.sem_seg_2d import ScanNetSemSeg2D, collate_func
from transforms.image_2d import Normalize, TransposeChannels
from models.sem_seg.enet import ENet2
from models.sem_seg.utils import count_parameters

import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import Compose
from torch.utils.tensorboard import SummaryWriter


def main(args):
    cfg = read_config(args.cfg_path)
    t =  Compose([
        Normalize(),
        TransposeChannels(),
    ])

    dataset = ScanNetSemSeg2D(cfg['data']['root'], cfg['data']['label_file'],
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ENet2(num_classes=21).to(device)
    print(f'Num params: {count_parameters(model)}')
    optimizer = Adam(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['l2'])
    
    writer = SummaryWriter()

    iter = 0
    for _ in tqdm(range(cfg['train']['epochs'])):
        model.train()
        for batch in train_loader:
            iter += 1
            img, label = batch['img'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = F.cross_entropy(out, label)
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss/train', loss / len(batch), iter)            

        model.eval()
        val_loss = 0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                n_batches += 1
                img, label = batch['img'].to(device), batch['label'].to(device)
                out = model(img)
                val_loss += (F.cross_entropy(out, label) / len(batch))
            
            writer.add_scalar('loss/val', val_loss / n_batches, iter)            

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cfg_path', help='Path to cfg')
    args = p.parse_args()

    main(args)