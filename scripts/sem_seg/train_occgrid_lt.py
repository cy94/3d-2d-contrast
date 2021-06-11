
import argparse

from lib.misc import read_config
from datasets.scannet.sem_seg_3d import ScanNetSemSegOccGrid
from models.sem_seg.utils import count_parameters, get_transform, SPARSE_MODELS, \
                                MODEL_MAP, get_collate_func

from torchinfo import summary
from torch.utils.data import Subset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


def main(args):
    cfg = read_config(args.cfg_path)
    model_name = cfg['model']['name']
    is_sparse = model_name in SPARSE_MODELS

    # basic transforms + augmentation
    train_t = get_transform(cfg, 'train')
    # basic transforms, no augmentation
    val_t = get_transform(cfg, 'val')

    if cfg['data']['train_list'] and cfg['data']['val_list']:
        train_set = ScanNetSemSegOccGrid(cfg['data'], transform=train_t, split='train', 
                                        full_scene=is_sparse)
        val_set = ScanNetSemSegOccGrid(cfg['data'], transform=val_t, split='val', 
                                        full_scene=is_sparse)
    else:
        dataset = ScanNetSemSegOccGrid(cfg['data'], transform=None, full_scene=is_sparse)
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
        train_set.transform = train_t
        val_set.transform = val_t

    print(f'Train set: {len(train_set)}')
    print(f'Val set: {len(val_set)}')
    
    print(f'Prepare a fixed val set')
    val_set = [s for s in val_set]

    cfunc = get_collate_func(cfg)
    

    train_loader = DataLoader(train_set, batch_size=cfg['train']['train_batch_size'],
                            shuffle=True, num_workers=8, collate_fn=cfunc,
                            pin_memory=True)  

    val_loader = DataLoader(val_set, batch_size=cfg['train']['val_batch_size'],
                            shuffle=False, num_workers=8, collate_fn=cfunc,
                            pin_memory=True) 

    model = MODEL_MAP[model_name](in_channels=1, num_classes=21, cfg=cfg)
    print(f'Num params: {count_parameters(model)}')

    input_size = (cfg['train']['train_batch_size'], 1,) + tuple(cfg['data']['subvol_size'])
    # doesn't work with sparse tensors
    try:
        summary(model, input_size=input_size)
    except:
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