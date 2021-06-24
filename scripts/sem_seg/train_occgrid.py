
import argparse
from datasets.scannet.utils import get_trainval_loaders, get_trainval_sets

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

    train_set, val_set = get_trainval_sets(cfg)
    print(f'Train set: {len(train_set)}')
    print(f'Val set: {len(val_set)}')

    if args.subset:
        print('Select a subset of data for quick run')
        train_set = Subset(train_set, range(128))
        val_set = Subset(val_set, range(128))

    # training on chunks
    # only train set random, val_set not random
    if not is_sparse:
        print(f'Prepare a fixed val set')
        val_set = [s for s in val_set]

    train_loader, val_loader = get_trainval_loaders(train_set, val_set, cfg)

    model = MODEL_MAP[model_name](in_channels=3, num_classes=20, cfg=cfg)
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
                        gpus=1, 
                        auto_scale_batch_size='binsearch',
                        log_every_n_steps=10,
                        callbacks=callbacks,
                        max_epochs=cfg['train']['epochs'],
                        val_check_interval=cfg['train']['eval_intv'],
                        fast_dev_run=args.fast_dev_run,
                        accumulate_grad_batches=cfg['train'].get('accum_grad', 1))
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cfg_path', help='Path to cfg')
    p.add_argument('--no-ckpt', action='store_true', dest='no_ckpt', 
                    default=False, help='Dont store checkpoints (for debugging)')
    p.add_argument('--subset', action='store_true', dest='subset', 
                    default=False, help='Use a subset of dataset')
    parser = pl.Trainer.add_argparse_args(p)
    args = p.parse_args()

    main(args)