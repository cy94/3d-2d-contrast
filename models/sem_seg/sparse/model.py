from models.sem_seg.fcn3d import SparseNet3D

class Model(SparseNet3D):
  """
  Base network for all sparse convnet

  By default, all networks are segmentation networks.
  """
  OUT_PIXEL_DIST = -1

  def __init__(self, in_channels, num_classes, cfg=None, D=3, **kwargs):
    super().__init__(in_channels, num_classes, cfg)
    self.D = D
    self.in_channels = in_channels
    self.num_classes = num_classes

  def init(self, x):
    """
    Initialize coordinates if it does not exist
    """
    nrows = self.get_nrows(1)
    if nrows < 0:
        if isinstance(x, SparseTensor):
            self.initialize_coords(x.coords_man)
        else:
            raise ValueError('Initialize input coordinates')
    elif nrows != x.F.size(0):
        raise ValueError('Input size does not match the coordinate size')
