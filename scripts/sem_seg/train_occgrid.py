from pathlib import Path
import argparse
from datasets.scannet.utils import get_dataset, get_loader

from lib.misc import read_config
from models.sem_seg.utils import count_parameters
from models.sem_seg.utils import SPARSE_MODELS, MODEL_MAP

from torchinfo import summary
from torch.utils.data import Subset

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def main(args):
    cfg = read_config(args.cfg_path)
    model_name = cfg['model']['name']
    is_sparse = model_name in SPARSE_MODELS

    train_set = get_dataset(cfg, 'train')
    val_set = get_dataset(cfg, 'val')
    print(f'Train set: {len(train_set)}')
    print(f'Val set: {len(val_set)}')

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

        # log LR with schedulers
    # without scheduler - done in model
    callbacks = [LearningRateMonitor(logging_interval='step')]

    # get the next version number from this
    ckpt = cfg['train']['resume']                                             
    resume = ckpt is not None
    if resume:
        print(f'Resuming from checkpoint: {ckpt}, reuse version')
        ckpt_version = Path(ckpt).parent.parent.stem.split('_')[1]
        name = f'version_{ckpt_version}'
    else:
        print('Create a new experiment version')
        tblogger = pl_loggers.TensorBoardLogger('lightning_logs', '')
        name = f'version_{tblogger.version}'

    # use for checkpoint
    if not args.no_ckpt:
        print('Saving checkpoints')
        ckpt_dir = f'lightning_logs/{name}/checkpoints'
        # resuming -> ok if exists
        # new expt -> dir should not exist
        Path(ckpt_dir).mkdir(parents=True, exist_ok=resume)

        # create the dir, version num doesn't get reused next time
        callbacks.append(ModelCheckpoint(ckpt_dir,
                                        save_last=True, save_top_k=5, 
                                        monitor='iou/val/mean',
                                        mode='max',
                                # put the miou in the filename
                                filename='epoch{epoch:02d}-step{step}-miou{iou/val/mean:.2f}',
                                auto_insert_metric_name=False))

    else:
        print('Log to a temp version of WandB')                                
    
    # create a temp version for WB if not checkpointing
    name += 'b'
    wbname = (name + 'tmp') if args.no_ckpt else name

    wblogger = pl_loggers.WandbLogger(name=wbname,
                                    project='thesis', 
                                    id=wbname,
                                    save_dir='lightning_logs',
                                    version=wbname,
                                    log_model=False)
    wblogger.log_hyperparams(cfg)


    trainer = pl.Trainer(resume_from_checkpoint=ckpt,
                        logger=wblogger,
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