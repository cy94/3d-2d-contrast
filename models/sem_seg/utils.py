from models.sem_seg.fcn3d import FCN3D, SparseNet3D, UNet3D, UNet3D_3DMV, \
    UNet2D3D, UNet2D3D_3DMV
from models.sem_seg.sparse.res16unet import SparseResUNet

from models.sem_seg.enet import ENet2
from models.sem_seg.deeplabv3 import DeepLabv3

MODEL_MAP = {
    'FCN3D': FCN3D,
    'UNet3D': UNet3D,
    'UNet3D_3DMV': UNet3D_3DMV,
    'SparseNet3D': SparseNet3D,
    'SparseResUNet': SparseResUNet
}

MODEL_MAP_2D3D = {
    'UNet2D3D': UNet2D3D,
    'UNet2D3D_3DMV': UNet2D3D_3DMV,
}

MODEL_MAP_2D = {
    'ENet': ENet2,
    'DeepLabv3': DeepLabv3
}

SPARSE_MODELS = 'SparseNet3D', 'SparseResUNet'

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)