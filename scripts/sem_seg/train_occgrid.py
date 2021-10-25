from datasets.scannet.sem_seg_3d import ScanNetOccGridH5
from datasets.scannet.utils import get_loader

from lib.misc import get_args, get_logger_and_callbacks, read_config
from models.sem_seg.utils import count_parameters
from models.sem_seg.utils import MODEL_MAP

from torch.utils.data import Subset

import pytorch_lightning as pl

from torchsummary import summary
from transforms.common import Compose
from transforms.grid_3d import AddChannelDim, JitterOccupancy, RandomRotate, TransposeDims


def main(args):
    cfg = read_config(args.cfg_path)

    train_t = Compose([
        RandomRotate(),
        JitterOccupancy(),
        AddChannelDim(),
        TransposeDims(),
    ])
    val_t = Compose([
        AddChannelDim(),
        TransposeDims()
    ])

    train_set = ScanNetOccGridH5(cfg['data'], transform=train_t, split='train')
    val_set = ScanNetOccGridH5(cfg['data'], transform=val_t, split='val')

    print(f'Train set: {len(train_set)}')
    print(f'Val set: {len(val_set)}')

    if args.subset:
        print('Select a subset of data for quick run')
        train_set = Subset(train_set, range(1024))
        val_set = Subset(val_set, range(1024))

    train_loader = get_loader(train_set, cfg, 'train', cfg['train']['train_batch_size'])
    val_loader = get_loader(val_set, cfg, 'val', cfg['train']['val_batch_size'])

    # training on chunks with binary feature
    in_channels = 1
    model = MODEL_MAP[cfg['model']['name']](in_channels=in_channels, num_classes=cfg['data']['num_classes'], cfg=cfg)
    print(f'Num params: {count_parameters(model)}')

    model = model.cuda()
    summary(model, (1, 32, 32, 32))

    wblogger, callbacks = get_logger_and_callbacks(args, cfg)
    ckpt = cfg['train']['resume']

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
    args = get_args()
    main(args)