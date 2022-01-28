import torch

from torchvision.transforms import Compose
from torch.utils.data import DataLoader, Sampler
import MinkowskiEngine as ME


from transforms.grid_3d import AddChannelDim, TransposeDims
from transforms.common import ComposeCustom
from models.sem_seg.utils import SPARSE_MODELS
from transforms.sparse_3d import ChromaticAutoContrast, ChromaticJitter, ChromaticTranslation, ElasticDistortion, RandomDropout, RandomHorizontalFlip
from datasets.scannet.sem_seg_3d import ScanNetOccGridH5, ScanNetSemSegOccGrid, collate_func
from datasets.scannet.sparse_3d import ScannetVoxelizationDataset

def get_scan_name(scene_id, scan_id):
    return f'scene{str(scene_id).zfill(4)}_{str(scan_id).zfill(2)}'

class BalancedUpSampler(Sampler):
  '''
  Upsample the minority class so that it occurs the same number of times 
  as the majority class
  For binary classes - 0 and 1
  '''
  def __init__(self, majority_indices, minority_indices, has_label):
    majority_indices = list(majority_indices)
    minority_indices = list(minority_indices)
    ratio = len(majority_indices) // len(minority_indices)
    # maj class + repeat the min class ratio time
    indices = torch.LongTensor(majority_indices + minority_indices * ratio)
    shuffle = torch.randperm(len(indices))
    # shuffle the whole thing
    self.all_indices = indices[shuffle]

  def __iter__(self):
    return iter(self.all_indices)

  def __len__(self):
    return len(self.all_indices)


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

def get_transform_dense(cfg, mode):
    '''
    cfg: the full train cfg
    mode: train or val
    '''
    # create transforms list
    transforms = []

    transforms.append(AddChannelDim())
    transforms.append(TransposeDims())

    t = Compose(transforms)

    return t

def get_loader(dataset, cfg, split, batch_size):
    # could have diff collate funcs for train and val
    collate_func = get_collate_func(cfg)

    # infinite sampling for sparse models (from mink-nets repo)
    is_sparse = cfg['model']['name'] in SPARSE_MODELS
    # change for sparse
    if split == 'train':
      if is_sparse:
        # sampler does the shuffling, no need to set shuffle
        sampler = InfSampler(dataset, True) 
        shuffle = False
      else:
        # default for all models - no sampler, only shuffle
        sampler, shuffle = None, True
    else:
      # val and test - no sampler, no shuffle
      sampler, shuffle = None, False
    
    loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=8, collate_fn=collate_func,
                            sampler=sampler, pin_memory=True)  

    return loader

def get_trainval_loaders(train_set, val_set, cfg):
    # could have diff collate funcs for train and val
    train_cfunc = get_collate_func(cfg, 'train')
    val_cfunc = get_collate_func(cfg, 'val')

    # infinite sampling for sparse models (from mink-nets repo)
    is_sparse = cfg['model']['name'] in SPARSE_MODELS
    # change for sparse
    if is_sparse:
      train_sampler = InfSampler(train_set, True) 
      train_shuffle = False
    else:
      # default for all models  
      train_sampler, train_shuffle = None, True
    
    train_loader = DataLoader(train_set, batch_size=cfg['train']['train_batch_size'],
                            shuffle=train_shuffle, num_workers=8, collate_fn=train_cfunc,
                            sampler=train_sampler,
                            pin_memory=True)  

    val_loader = DataLoader(val_set, batch_size=cfg['train']['val_batch_size'],
                            shuffle=False, num_workers=8, collate_fn=val_cfunc,
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
    print('Sparse dataset, use RGB?:', use_rgb)
    if use_rgb:
        input_transform += [
            ChromaticAutoContrast(),
            ChromaticTranslation(0.1),
            ChromaticJitter(0.05),
        ]

    input_transform = ComposeCustom(input_transform)

    train_set = ScannetVoxelizationDataset(
                    cfg,
                    prevoxel_transform=prevoxel_transform,
                    input_transform=input_transform,
                    target_transform=None,
                    cache=False,
                    augment_data=True,
                    phase='train',
                    use_rgb=use_rgb)

    val_set = ScannetVoxelizationDataset(
                    cfg,
                    prevoxel_transform=None,
                    input_transform=None,
                    target_transform=None,
                    cache=False,
                    augment_data=False,
                    phase='val',
                    use_rgb=use_rgb)

    return train_set, val_set

def get_sparse_dataset(cfg, split):
    use_rgb = cfg['data'].get('use_rgb', False)
    print('Sparse dataset, use RGB?:', use_rgb)

    if split == 'train':
      # augmentations
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
      if use_rgb:
          input_transform += [
              ChromaticAutoContrast(),
              ChromaticTranslation(0.1),
              ChromaticJitter(0.05),
          ]
      input_transform = ComposeCustom(input_transform)
      augment_data = True
    elif split in ('val', 'test'):
      prevoxel_transform = None
      input_transform = None
      augment_data = False

    dataset = ScannetVoxelizationDataset(
                    cfg,
                    prevoxel_transform=prevoxel_transform,
                    input_transform=input_transform,
                    target_transform=None,
                    cache=False,
                    augment_data=augment_data,
                    phase=split,
                    use_rgb=use_rgb)

    return dataset

def get_dense_dataset(cfg, split):
    # dont transform full scenes, the chunks get transformed later
    transform = get_transform_dense(cfg, split) if split != 'test' else None

    # testing - get whole scenes, not random chunks
    # TBD: full scene dataset from original files, not H5
    # full_scene = (split == 'test')

    dataset = ScanNetOccGridH5(cfg['data'], transform=transform, split=split)

    return dataset

def get_trainval_dense(cfg):
    # basic transforms + augmentation
    train_t = get_transform_dense(cfg, 'train')
    # basic transforms, no augmentation
    val_t = get_transform_dense(cfg, 'val')

    train_set = ScanNetSemSegOccGrid(cfg['data'], transform=train_t, split='train')
    val_set = ScanNetSemSegOccGrid(cfg['data'], transform=val_t, split='val')

    return train_set, val_set

def get_dataset(cfg, split):
  '''
  cfg: has all params of the dataset
  split: train/val/test
  '''
  is_sparse = cfg['model']['name'] in SPARSE_MODELS

  if is_sparse:
      dataset = get_sparse_dataset(cfg, split)
  else:
      dataset = get_dense_dataset(cfg, split)
        
  return dataset

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

class InfSampler(Sampler):
  """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

  def __init__(self, data_source, shuffle=False):
    self.data_source = data_source
    self.shuffle = shuffle
    self.reset_permutation()

  def reset_permutation(self):
    perm = len(self.data_source)
    if self.shuffle:
      perm = torch.randperm(perm)
    self._perm = perm.tolist()

  def __iter__(self):
    return self

  def __next__(self):
    if len(self._perm) == 0:
      self.reset_permutation()
    return self._perm.pop()

  def __len__(self):
    return len(self.data_source)