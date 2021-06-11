from transforms.grid_3d import DenseToSparse, RandomTranslate, RandomRotate, \
    MapClasses, AddChannelDim, TransposeDims
from torchvision.transforms import Compose

from datasets.scannet.sem_seg_3d import collate_func
from models.sem_seg.fcn3d import FCN3D, SparseNet3D, UNet3D
from models.sem_seg.sparse.res16unet import SparseResUNet

MODEL_MAP = {
    'FCN3D': FCN3D,
    'UNet3D': UNet3D,
    'SparseNet3D': SparseNet3D,
    'SparseResUNet': SparseResUNet
}

SPARSE_MODELS = 'SparseNet3D', 'SparseResUNet'

def get_collate_func(cfg):
    model_name = cfg['model']['name']
    return SparseNet3D.collation_fn if model_name in SPARSE_MODELS \
        else collate_func

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

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)