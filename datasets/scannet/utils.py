import torch
from torchvision.transforms import Compose
from torch.utils.data import Subset, DataLoader
import MinkowskiEngine as ME

from transforms.grid_3d import DenseToSparse, RandomTranslate, RandomRotate, \
    MapClasses, AddChannelDim, TransposeDims
from transforms.common import ComposeCustom
from models.sem_seg.utils import SPARSE_MODELS
from transforms.sparse_3d import ChromaticAutoContrast, ChromaticJitter, ChromaticTranslation, ElasticDistortion, RandomDropout, RandomHorizontalFlip
from datasets.scannet.sem_seg_3d import ScanNetSemSegOccGrid, collate_func
from datasets.scannet.sparse_3d import ScannetVoxelization2cmDataset, ScannetVoxelizationDataset


class cfl_collate_fn_factory:
  """Generates collate function for coords, feats, labels.

    Args:
      limit_numpoints: If 0 or False, does not alter batch size. If positive integer, limits batch
                       size so that the number of input coordinates is below limit_numpoints.
  """

  def __init__(self, limit_numpoints):
    self.limit_numpoints = limit_numpoints

  def __call__(self, list_data):
    coords, feats, labels = list(zip(*list_data))
    coords_batch, feats_batch, labels_batch = [], [], []

    batch_id = 0
    batch_num_points = 0
    for batch_id, _ in enumerate(coords):
      num_points = coords[batch_id].shape[0]
      batch_num_points += num_points
      if self.limit_numpoints and batch_num_points > self.limit_numpoints:
        num_full_points = sum(len(c) for c in coords)
        num_full_batch_size = len(coords)
        print(
            f'\t\tCannot fit {num_full_points} points into {self.limit_numpoints} points '
            f'limit. Truncating batch size at {batch_id} out of {num_full_batch_size} with {batch_num_points - num_points}.'
        )
        break
      coords_batch.append(torch.from_numpy(coords[batch_id]).int())
      feats_batch.append(torch.from_numpy(feats[batch_id]))
      labels_batch.append(torch.from_numpy(labels[batch_id]).int())

      batch_id += 1

    # Concatenate all lists
    coords_batch, feats_batch, labels_batch = ME.utils.sparse_collate(coords_batch, feats_batch, labels_batch)

    return {
        'coords': coords_batch,
        'feats': feats_batch.float(),
        'y': labels_batch.long()
    } 

def get_collate_func(cfg):
    if cfg['model']['name'] in SPARSE_MODELS:
        return cfl_collate_fn_factory(0)
    else:
        return collate_func

def get_transform(cfg, mode):
    '''
    cfg: the full train cfg
    mode: train or val
    '''
    train = (mode == 'train')
    model_name = cfg['model']['name']
    # create transforms list
    # map none class to padding, no loss on this class
    transforms = [
        MapClasses({0: cfg['data']['target_padding']}),
        ]
    if train: transforms.append(RandomRotate())
    # augmentations for sparse conv
    if model_name in SPARSE_MODELS:
        transforms.append(DenseToSparse())
        if train: transforms.append(RandomTranslate())
    # augmentations for dense conv
    else:
        transforms.append(AddChannelDim())
        transforms.append(TransposeDims())
    t = Compose(transforms)

    return t

def get_trainval_loaders(train_set, val_set, cfg):
    cfunc = get_collate_func(cfg)
    
    train_loader = DataLoader(train_set, batch_size=cfg['train']['train_batch_size'],
                            shuffle=True, num_workers=8, collate_fn=cfunc,
                            pin_memory=True)  

    val_loader = DataLoader(val_set, batch_size=cfg['train']['val_batch_size'],
                            shuffle=False, num_workers=8, collate_fn=cfunc,
                            pin_memory=True) 
                            
    return train_loader, val_loader

def get_trainval_sparse(cfg):
    # augment the whole PC
    prevoxel_transform = ComposeCustom([
       ElasticDistortion(ScannetVoxelizationDataset.ELASTIC_DISTORT_PARAMS) 
    ]) 
    # augment coords
    input_transform = [
        RandomDropout(0.2),
        RandomHorizontalFlip(ScannetVoxelizationDataset.ROTATION_AXIS, \
                                ScannetVoxelizationDataset.IS_TEMPORAL),
    ]
    # augment the colors?
    use_rgb = cfg['data'].get('use_rgb', False)
    if use_rgb:
        input_transform += [
            ChromaticAutoContrast(),
            ChromaticTranslation(0.1),
            ChromaticJitter(0.05),
        ]

    input_transform = ComposeCustom(input_transform)

    train_set = ScannetVoxelization2cmDataset(
                    cfg,
                    prevoxel_transform=prevoxel_transform,
                    input_transform=input_transform,
                    target_transform=None,
                    cache=False,
                    augment_data=True,
                    phase='train',
                    use_rgb=use_rgb)

    val_set = ScannetVoxelization2cmDataset(
                    cfg,
                    prevoxel_transform=prevoxel_transform,
                    input_transform=input_transform,
                    target_transform=None,
                    cache=False,
                    augment_data=False,
                    phase='val',
                    use_rgb=use_rgb)

    return train_set, val_set

def get_trainval_dense(cfg):
    # basic transforms + augmentation
    train_t = get_transform(cfg, 'train')
    # basic transforms, no augmentation
    val_t = get_transform(cfg, 'val')

    if cfg['data']['train_list'] and cfg['data']['val_list']:
        train_set = ScanNetSemSegOccGrid(cfg['data'], transform=train_t, split='train')
        val_set = ScanNetSemSegOccGrid(cfg['data'], transform=val_t, split='val')
    else:
        dataset = ScanNetSemSegOccGrid(cfg['data'], transform=None)
        print(f'Full dataset size: {len(dataset)}')
        if cfg['train']['train_split']:
            train_size = int(cfg['train']['train_split'] * len(dataset))
            train_set = Subset(dataset, range(train_size))
            val_set = Subset(dataset, range(train_size, len(dataset)))
        elif cfg['train']['train_size'] and cfg['train']['val_size']:
            train_set = Subset(dataset, range(cfg['train']['train_size']))
            val_set = Subset(dataset, range(cfg['train']['train_size'], 
                                cfg['train']['train_size']+cfg['train']['val_size']))
        else:
            raise ValueError('Train val split not specified')
        train_set.transform = train_t
        val_set.transform = val_t
    
    return train_set, val_set

def get_trainval_sets(cfg):
    '''
    get train and val sets 
    cfg: full train cfg
    '''
    is_sparse = cfg['model']['name'] in SPARSE_MODELS

    if is_sparse:
        train_set, val_set = get_trainval_sparse(cfg)
    else:
        train_set, val_set = get_trainval_dense(cfg)
        
    return train_set, val_set