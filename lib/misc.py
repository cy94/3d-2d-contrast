from pathlib import Path
import yaml
from pytorch_lightning import loggers as pl_loggers

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

def read_config(path):
    """
    path: path to config yaml file
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    return cfg

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
    
    return wblogger, callbacks