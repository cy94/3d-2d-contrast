from pathlib import Path

from torchvision.transforms import Compose
from torch.utils.data import Subset, DataLoader
import torch

import pytorch_lightning as pl

from datasets.scannet.utils import BalancedUpSampler, get_scan_name
from eval.sem_seg_3d import gen_predictions, get_dataset_split_scans, save_preds
from datasets.scannet.utils_3d import adjust_intrinsic, make_intrinsic
from lib.misc import display_results, get_args, get_logger_and_callbacks, read_config
from models.sem_seg.utils import MODEL_MAP_2D, MODEL_MAP_2D3D, count_parameters
from datasets.scannet.sem_seg_3d import ScanNet2D3DH5
from transforms.grid_3d import AddChannelDim, LoadLabels2D, TransposeDims, LoadDepths, LoadPoses,\
                                LoadRGBs
from transforms.image_2d import Normalize

from transforms.image_2d import ColorJitter, GaussNoise, GaussianBlur, HueSaturationValue, Normalize


def main(args):
    cfg = read_config(args.cfg_path)

    train_2d = [Normalize()]
    if cfg['model'].get('augment_2d', False):
        print('Augment 2d images')
        train_2d = [ColorJitter(),
        HueSaturationValue(),
        GaussianBlur(),
        GaussNoise()] + train_2d
    train_t2d = Compose(train_2d)
    val_t2d = Normalize()

    train_t = [
        AddChannelDim(),
        TransposeDims()
    ]
    val_t = [
        AddChannelDim(),
        TransposeDims()
    ]
    # load only if using 2d features or contrastive
    if cfg['model']['use_2dfeat'] or 'contrastive' in cfg['model']:
        train_t += [
            LoadDepths(cfg),
            LoadPoses(cfg),
            LoadRGBs(cfg, transform=train_t2d)
        ]
        val_t += [
            LoadDepths(cfg),
            LoadPoses(cfg),
            LoadRGBs(cfg, transform=val_t2d)
        ]

    if cfg['model'].get('train_2d', False):
        print('Train the 2d network on 2d labels')
        train_t.append(LoadLabels2D(cfg))
        val_t.append(LoadLabels2D(cfg))
        
    train_t = Compose(train_t)
    val_t = Compose(val_t)

    train_set = ScanNet2D3DH5(cfg['data'], 'train', transform=train_t)
    val_set = ScanNet2D3DH5(cfg['data'], 'val', transform=val_t)
    print(f'Train set: {len(train_set)}')
    print(f'Val set: {len(val_set)}')

    if 'filter_train_label' in cfg['data']:
        print('Balanced sampler on train set, no shuffle')
        has_label = torch.zeros(len(train_set), dtype=int)
        # samples that have labels
        has_label_indices = train_set.labeled_samples_ndx
        has_label[has_label_indices] = 1
        # samples that dont have labels
        no_label_indices = torch.nonzero(has_label == 0)
        # upsample the minority and shuffle
        train_sampler = BalancedUpSampler(no_label_indices, has_label_indices, has_label)
        print(f'Train sampler size:{len(train_sampler)}')
        train_shuffle = False
    else:
        print('Default sampler on train set, shuffle')
        train_sampler = None
        train_shuffle = True

    if args.subset:
        print('Select a subset of data for quick run')
        train_set = Subset(train_set, range(1))
        val_set = Subset(val_set, range(16))
        print(f'Train set: {len(train_set)}')
        print(f'Val set: {len(val_set)}')


    train_loader = DataLoader(train_set, batch_size=cfg['train']['train_batch_size'],
                            collate_fn=ScanNet2D3DH5.collate_func,
                            sampler=train_sampler,
                            shuffle=train_shuffle, num_workers=8)
                            # pin_memory=True)  

    val_loader = DataLoader(val_set, batch_size=cfg['train']['val_batch_size'],
                            collate_fn=ScanNet2D3DH5.collate_func,
                            shuffle=False, num_workers=8)
                            # pin_memory=True) 

    model_name_2d = cfg['model'].get('name_2d', None)
    features_2d = None
    # do we have a 2d model?
    if model_name_2d:
        # create 2d have a pretrained full model? then
        if 'ckpt_2d' in cfg['model']:
            print('Use pretrained 2D model')
            features_2d = MODEL_MAP_2D[model_name_2d].load_from_checkpoint(cfg['model']['ckpt_2d'])
        else:
            print('Create a new 2D model')
            features_2d = MODEL_MAP_2D[model_name_2d](num_classes=cfg['data']['num_classes'], cfg=cfg)
            if 'pretrained' not in cfg['model']:
                print('Set finetune2d to True')
                cfg['model']['finetune_2d'] = True

    # intrinsic of the color camera from scene0001_00
    intrinsic = make_intrinsic(1170.187988, 1170.187988, 647.75, 483.75)
    # adjust for smaller image size
    intrinsic = adjust_intrinsic(intrinsic, [1296, 968], cfg['data']['proj_img_size'])

    model = MODEL_MAP_2D3D[cfg['model']['name']](in_channels=1, 
                    num_classes=cfg['data']['num_classes'], cfg=cfg, 
                    features_2d=features_2d, intrinsic=intrinsic,
                    log_all_classes=True)
    ckpt = cfg['train']['resume']
    # not resuming, have pretrained model? then load weights
    if not ckpt and 'pretrained' in cfg['model']:
        print(f'Use 2d+3d pretrained model: ', cfg['model']['pretrained'])
        model.load_state_dict(torch.load(cfg['model']['pretrained'])['state_dict'])
    
    print(f'Num params: {count_parameters(model)}')                                                      

    wblogger, callbacks = get_logger_and_callbacks(args, cfg)

    
    if args.eval or args.pred:
        assert (ckpt is not None), 'Evaluate but checkpoint not provided'
        trainer = pl.Trainer(logger=None, 
                            gpus=1 if not args.cpu else 0)
        if not args.cpu:
            # force change to GPU if required
            model = model.cuda()
            model.change_device()
        if args.eval:                    
            print('Evaluate with a checkpoint')
            model.log_all_classes = True
            results = trainer.validate(model, val_loader, ckpt, verbose=False)
            display_results(results[0])
        else:
            print('Generate predictions with a checkpoint')
            # predictions only on this scan
            scene_scan = (568, 0)
            # split the val set into scenes
            # see get_dataset_split_scans to create this dict, save as pth file
            splits = torch.load(cfg['data']['val_splits_file'])
            # pick only the required scene and do inference
            val_set_single = Subset(val_set, splits[scene_scan])
            val_loader = DataLoader(val_set_single, batch_size=cfg['train']['val_batch_size'],
                            collate_fn=ScanNet2D3DH5.collate_func,
                            shuffle=False, num_workers=8)
            preds = gen_predictions(model, val_loader, ckpt)
            # combine preds over all batches into a single tensor
            preds_combined = torch.cat(preds, dim=0)
            scan_name = get_scan_name(*scene_scan)
            out_path = f'{Path(ckpt).parent.parent.stem}_{scan_name}.ply'
            save_preds(preds_combined, val_set_single, out_path)

            
    else:
        print('Start training')
        trainer = pl.Trainer(resume_from_checkpoint=ckpt,
                        logger=wblogger,
                        num_sanity_val_steps=0,
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