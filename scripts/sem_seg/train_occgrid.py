from datasets.scannet.utils import get_dataset, get_loader

from lib.misc import get_args, get_logger_and_callbacks, read_config
from models.sem_seg.utils import count_parameters
from models.sem_seg.utils import SPARSE_MODELS, MODEL_MAP

from torch.utils.data import Subset

import pytorch_lightning as pl


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