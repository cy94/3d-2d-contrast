from datasets.scannet.utils_3d import adjust_intrinsic, make_intrinsic
from models.sem_seg.enet import ENet2
from models.sem_seg.fcn3d import UNet2D3D, UNet2D3D_3DMV

from lib.misc import get_args, get_logger_and_callbacks, read_config
from models.sem_seg.utils import MODEL_MAP_2D3D, count_parameters

from torchvision.transforms import Compose
from torch.utils.data import Subset, DataLoader

import pytorch_lightning as pl

from datasets.scannet.sem_seg_3d import ScanNet2D3DH5
from transforms.grid_3d import AddChannelDim, JitterOccupancy, RandomRotate, TransposeDims, LoadDepths, LoadPoses,\
                                LoadRGBs


def main(args):
    cfg = read_config(args.cfg_path)

    train_t = Compose([
        RandomRotate(aug_w2g=True),
        JitterOccupancy(0.01),
        AddChannelDim(),
        TransposeDims(),
        LoadDepths(cfg),
        LoadPoses(cfg),
        LoadRGBs(cfg)
    ])
    val_t = Compose([
        AddChannelDim(),
        TransposeDims(),
        LoadDepths(cfg),
        LoadPoses(cfg),
        LoadRGBs(cfg)
    ])

    train_set = ScanNet2D3DH5(cfg['data'], 'train', transform=train_t)
    val_set = ScanNet2D3DH5(cfg['data'], 'val', transform=val_t)
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

    model = MODEL_MAP_2D3D[cfg['model']['name']](in_channels=1, 
                    num_classes=cfg['data']['num_classes'], cfg=cfg, 
                    features_2d=features_2d, intrinsic=intrinsic)

    print(f'Num params: {count_parameters(model)}')                                                      

    wblogger, callbacks = get_logger_and_callbacks(args, cfg)
    ckpt = cfg['train']['resume']

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
    args = get_args()
    main(args)