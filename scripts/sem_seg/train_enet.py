import os 
import argparse
from datetime import datetime as dt
import random 

import numpy as np

from tqdm import tqdm

from lib.misc import read_config
from datasets.scannet.sem_seg_2d import ScanNetSemSeg2D, collate_func
from transforms.image_2d import Normalize, TransposeChannels, Resize
from models.sem_seg.enet import ENet2
from models.sem_seg.utils import count_parameters
from eval.sem_seg_2d import miou

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
    if cfg['data']['img_size'] is not None:
        transforms.append(Resize(cfg['data']['img_size']))
    transforms.append(Normalize())
    transforms.append(TransposeChannels())

    t = Compose(transforms)

    dataset = ScanNetSemSeg2D(cfg['data']['root'], cfg['data']['label_file'],
                                cfg['data']['limit_scans'],
                                transform=t, frame_skip=cfg['data']['frame_skip'])

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
    val_loader_shuffle = DataLoader(val_set, batch_size=cfg['train']['val_batch_size'],
                        shuffle=True, num_workers=4, collate_fn=collate_func)                                  


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ENet2(num_classes=21).to(device)
    print(f'Num params: {count_parameters(model)}')
    optimizer = Adam(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['l2'])
    
    save_dir = 'runs/' + dt.now().strftime('%d-%b-%H.%M.%S')
    ckpt_dir = f'{save_dir}/ckpt'
    
    if args.quick_run:
        print('Quick run')
        cfg['train']['epochs'] = 1
    else:
        os.makedirs(ckpt_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=f'{save_dir}')
    
    print(f'Save dir: {save_dir}')
    
    step = 0
    start_epoch = 0
    
    if cfg['train']['resume']:
        print(f'Resuming from ckpt: ', cfg['train']['resume'])
        ckpt = torch.load(cfg['train']['resume'])
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        step = (start_epoch + 1) * (len(train_set) // ckpt['cfg']['train']['train_batch_size'])

    for epoch in tqdm(range(start_epoch, cfg['train']['epochs']), desc='epoch'):
        for batch in tqdm(train_loader, desc='train', leave=False):
            model.train()
            step += 1
            img, label = batch['img'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = F.cross_entropy(out, label)
            loss.backward()
            optimizer.step()

            # log train loss
            if not args.quick_run:
                writer.add_scalar('loss/train', loss / len(batch), step)

            # compute train miou for a few random batches
            if random.random() > 0.8:  
                preds = out.argmax(dim=1)
                miou_train = miou(preds, label, 21)
                if not np.isnan(miou_train).any() and not args.quick_run:
                    writer.add_scalar('miou/train', miou_train.mean(), step)            

            # evaluate at intervals?
            if step % cfg['train']['eval_intv'] == 0:
                model.eval()
                val_loss = 0
                n_batches = 0

                with torch.no_grad():
                    for batch_ndx, batch in enumerate(tqdm(val_loader, desc='val', leave=False)):
                        n_batches += 1
                        img, label = batch['img'].to(device), batch['label'].to(device)
                        out = model(img)
                        val_loss += (F.cross_entropy(out, label) / len(batch))

                if not args.quick_run:
                    writer.add_scalar('loss/val', val_loss / n_batches, step)     
                
                # pick a random batch to compute val miou
                batch = next(iter(val_loader_shuffle))
                img, label = batch['img'].to(device), batch['label'].to(device)
                preds = model(img).argmax(dim=1)
                miou_val = miou(preds, label, 21)

                if not np.isnan(miou_val).any() and not args.quick_run:
                    writer.add_scalar('miou/val', miou_val.mean(), step)   

        if (not args.no_ckpt) \
            and (not args.quick_run) \
            and (epoch % cfg['train']['ckpt_intv'] == 0):
            torch.save({
                'cfg': cfg,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f'{ckpt_dir}/{epoch}.pt')    
  
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cfg_path', help='Path to cfg')
    p.add_argument('--quick', dest='quick_run', action='store_true', help='Quick run?')
    p.add_argument('--no-ckpt', dest='no_ckpt', action='store_true', help='Dont store checkpoints')
    args = p.parse_args()

    main(args)