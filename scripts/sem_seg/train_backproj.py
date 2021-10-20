# from pytorch_lightning.utilities.seed import seed_everything
# seed_everything(42)

from pathlib import Path
import argparse
from datasets.scannet.utils_3d import adjust_intrinsic, make_intrinsic
from models.sem_seg.enet import ENet2
from models.sem_seg.fcn3d import UNet2D3D

from lib.misc import read_config
from models.sem_seg.utils import count_parameters

from torchvision.transforms import Compose
from torch.utils.data import Subset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers


from datasets.scannet.sem_seg_3d import ScanNet2D3DH5
from transforms.grid_3d import AddChannelDim, TransposeDims, LoadDepths, LoadPoses,\
                                LoadRGBs

def main(args):
    cfg = read_config(args.cfg_path)

    t = Compose([
        AddChannelDim(),
        TransposeDims(),
        LoadDepths(cfg),
        LoadPoses(cfg),
        LoadRGBs(cfg)
    ])

    train_set = ScanNet2D3DH5(cfg['data'], 'train', transform=t)
    val_set = ScanNet2D3DH5(cfg['data'], 'val', transform=t)
    print(f'Train set: {len(train_set)}')
    print(f'Val set: {len(val_set)}')

    if args.subset:
        print('Select a subset of data for quick run')
        train_set = Subset(train_set, range(1))
        val_set = Subset(val_set, range(16))
        print(f'Train set: {len(train_set)}')
        print(f'Val set: {len(val_set)}')

    train_loader = DataLoader(train_set, batch_size=cfg['train']['train_batch_size'],
                            collate_fn=ScanNet2D3DH5.collate_func,
                            shuffle=True, num_workers=8,
                            pin_memory=True)  

    val_loader = DataLoader(val_set, batch_size=cfg['train']['val_batch_size'],
                            collate_fn=ScanNet2D3DH5.collate_func,
                            shuffle=False, num_workers=8,
                            pin_memory=True) 

    features_2d = ENet2.load_from_checkpoint(cfg['model']['ckpt_2d'])

    # intrinsic of the color camera from scene0001_00
    intrinsic = make_intrinsic(1170.187988, 1170.187988, 647.75, 483.75)
    # adjust for smaller image size
    intrinsic = adjust_intrinsic(intrinsic, [1296, 968], cfg['data']['proj_img_size'])

    model = UNet2D3D(in_channels=1, num_classes=cfg['data']['num_classes'], cfg=cfg, 
                    features_2d=features_2d, intrinsic=intrinsic)

    print(f'Num params: {count_parameters(model)}')                                                      

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
    


    trainer = pl.Trainer(resume_from_checkpoint=ckpt,
                        logger=wblogger,
                        # num_sanity_val_steps=0,
                        gpus=1 if not args.cpu else 0, 
                        log_every_n_steps=10,
                        callbacks=callbacks,
                        max_epochs=cfg['train']['epochs'],
                        val_check_interval=cfg['train']['eval_intv'],
                        limit_val_batches=cfg['train']['limit_val_batches'],
                        fast_dev_run=args.fast_dev_run,
                        accumulate_grad_batches=cfg['train'].get('accum_grad', 1),)

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
    p.add_argument('--no-log', action='store_true', dest='no_log', 
                    default=False, help='Dont log to Weights and Biases')

    parser = pl.Trainer.add_argparse_args(p)
    args = p.parse_args()

    main(args)