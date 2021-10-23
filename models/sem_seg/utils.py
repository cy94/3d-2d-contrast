from models.sem_seg.fcn3d import FCN3D, SparseNet3D, UNet3D, UNet3D_3DMV, \
    UNet2D3D, UNet2D3D_3DMV
from models.sem_seg.sparse.res16unet import SparseResUNet

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

SPARSE_MODELS = 'SparseNet3D', 'SparseResUNet'

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)