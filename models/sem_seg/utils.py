from models.sem_seg.fcn3d import FCN3D, SparseNet3D, UNet3D
from models.sem_seg.sparse.res16unet import SparseResUNet

MODEL_MAP = {
    'FCN3D': FCN3D,
    'UNet3D': UNet3D,
    'SparseNet3D': SparseNet3D,
    'SparseResUNet': SparseResUNet
}

SPARSE_MODELS = 'SparseNet3D', 'SparseResUNet'

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)