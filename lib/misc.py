from pathlib import Path
import yaml
import argparse

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def read_config(path):
    """
    path: path to config yaml file
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    return cfg

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('cfg_path', help='Path to cfg')
    p.add_argument('--no-ckpt', action='store_true', dest='no_ckpt', 
                    default=False, help='Dont store checkpoints (for debugging)')
    p.add_argument('--cpu', action='store_true', dest='cpu', 
                    default=False, help='Train on CPU')                    
    p.add_argument('--subset', action='store_true', dest='subset', 
                    default=False, help='Use a subset of dataset')
    p.add_argument('--no-log', action='store_true', dest='no_log', 
                    default=False, help='Dont log to Weights and Biases')
    p.add_argument('--debug', action='store_true', dest='debug', 
        default=False, help='No checkpoint, no log')
    p.add_argument('--b', action='store_true', dest='b', 
                    default=False, help='Add b to wandb name')      
    
    p = pl.Trainer.add_argparse_args(p)
    args = p.parse_args()

    if args.debug:
        print('Debug: no checkpoint, no log')
        args.no_log = True
        args.no_ckpt = True

    return args

def get_logger_and_callbacks(args, cfg):
    '''
    get the wandb logger and other callbacks 
    by looking at args and cfg
    '''
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

    if (not args.no_log) or (not args.no_ckpt): 
    # create a version folder if 
    # we are logging -> so that even tmp versions can get upgraded
    # or if we are checkpointing
    # resuming -> ok if exists
    # new expt -> dir should not exist
        ckpt_dir = f'lightning_logs/{name}/checkpoints'
        Path(ckpt_dir).mkdir(parents=True, exist_ok=resume)

    if args.b:
        name += 'b'
    # create a temp version for WB if not checkpointing
    wbname = (name + 'tmp') if args.no_ckpt else name
    
    if args.no_log:
        wblogger = None
        print('Logging disabled -> Checkpoint and LR logging disabled as well')
        # cant log LR
        callbacks = []
    else:
        wblogger = pl_loggers.WandbLogger(name=wbname,
                                        project='thesis', 
                                        id=wbname,
                                        save_dir='lightning_logs',
                                        version=wbname,
                                        log_model=False)
        wblogger.log_hyperparams(cfg)

    # use for checkpoint
    if not args.no_ckpt:
        print('Saving checkpoints')
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
    
    # loss will be logged, can do early stopping
    # dont early stop when using a subset of data to overfit
    if not args.no_log and not args.subset:
        # monitor this value
        monitor_key = 'loss/val'

        # using contr loss?
        if 'contrastive' in cfg['model']:
            # only contr loss?
            if 'losses' in cfg['model'] and cfg['model']['losses'] == ['contrastive']:
                monitor_key = 'loss/val/contrastive'

        print(f'Add early stopping callback on {monitor_key}')
        callbacks.append(
            # loss ~ 3, need to improve atleast 0.01
            EarlyStopping(monitor=monitor_key, min_delta=0.005, 
            patience=5, verbose=True, mode="min", strict=True,
            check_finite=True,)
        )

    return wblogger, callbacks