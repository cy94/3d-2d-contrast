import random
import argparse
from datasets.scannet.utils import get_dataset, get_loader

from lib.misc import read_config
from models.sem_seg.utils import count_parameters
from models.sem_seg.utils import SPARSE_MODELS, MODEL_MAP

from torchinfo import summary
from torch.utils.data import Subset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def main(args):
    cfg = read_config(args.cfg_path)
    model_name = cfg['model']['name']
    is_sparse = model_name in SPARSE_MODELS

    train_set = get_dataset(cfg, 'train')
    val_set = get_dataset(cfg, 'val')
    print(f'Train set: {len(train_set)}')
    print(f'Val set: {len(val_set)}')

    # train set gets shuffled by the dataloader
    # shuffle the val set once, then we can use a subset of it later
    indices = list(range(len(val_set)))
    random.shuffle(indices)
    val_set = Subset(val_set, indices)

    if args.subset:
        print('Select a subset of data for quick run')
        train_set = Subset(train_set, range(1024))
        val_set = Subset(val_set, range(1024))

    if not is_sparse:
        # training on chunks with binary feature
        in_channels = 1
    else:
        # sparse model always has 3 channels, with real or dummy RGB values
        in_channels = 3

    train_loader = get_loader(train_set, cfg, 'train', cfg['train']['train_batch_size'])
    val_loader = get_loader(val_set, cfg, 'val', cfg['train']['val_batch_size'])

    model = MODEL_MAP[model_name](in_channels=in_channels, num_classes=cfg['data']['num_classes'], cfg=cfg)
    print(f'Num params: {count_parameters(model)}')

    try:
        input_size = (cfg['train']['train_batch_size'], 1,) + tuple(cfg['data']['subvol_size'])
        summary(model, input_size=input_size)
    except:
        # doesn't work with sparse tensors
        pass

    callbacks = [LearningRateMonitor(logging_interval='step')]
    if not args.no_ckpt:
        print('Saving checkpoints')
        callbacks.append(ModelCheckpoint(save_last=True, save_top_k=5, 
                                        monitor='iou/val/mean',
                                        mode='max',
                                # put the miou in the filename
                                filename='epoch{epoch:02d}-step{step}-miou{iou/val/mean:.2f}',
                                auto_insert_metric_name=False))
    ckpt = cfg['train']['resume']                                             
    if ckpt is not None:
        print(f'Resuming from checkpoint: {ckpt}')

    trainer = pl.Trainer(resume_from_checkpoint=ckpt,
                        gpus=1 if not args.cpu else None, 
                        log_every_n_steps=10,
                        callbacks=callbacks,
                        max_epochs=cfg['train']['epochs'],
                        val_check_interval=cfg['train']['eval_intv'],
                        limit_val_batches=cfg['train']['limit_val_batches'],
                        fast_dev_run=args.fast_dev_run,
                        accumulate_grad_batches=cfg['train'].get('accum_grad', 1))

    trainer.fit(model, train_loader, val_loader)

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