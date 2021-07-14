import random

import numpy as np
import scipy
import scipy.ndimage, scipy.interpolate


class ChromaticTranslation(object):
  """Add random color to the image, input must be an array in [0,255] or a PIL image"""

  def __init__(self, trans_range_ratio=1e-1):
    """
    trans_range_ratio: ratio of translation i.e. 255 * 2 * ratio * rand(-0.5, 0.5)
    """
    self.trans_range_ratio = trans_range_ratio

  def __call__(self, coords, feats, labels):
    if random.random() < 0.95:
      tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.trans_range_ratio
      feats[:, :3] = np.clip(tr + feats[:, :3], 0, 255)
    return coords, feats, labels


class ChromaticAutoContrast(object):

  def __init__(self, randomize_blend_factor=True, blend_factor=0.5):
    self.randomize_blend_factor = randomize_blend_factor
    self.blend_factor = blend_factor

  def __call__(self, coords, feats, labels):
    if random.random() < 0.2:
      lo = feats[:, :3].min(0, keepdims=True)
      hi = feats[:, :3].max(0, keepdims=True)
      assert hi.max() > 1, f"invalid color value. Color is supposed to be [0-255]"

      scale = 255 / (hi - lo)

      contrast_feats = (feats[:, :3] - lo) * scale

      blend_factor = random.random() if self.randomize_blend_factor else self.blend_factor
      feats[:, :3] = (1 - blend_factor) * feats + blend_factor * contrast_feats
    return coords, feats, labels


class ChromaticJitter(object):

  def __init__(self, std=0.01):
    self.std = std

  def __call__(self, coords, feats, labels):
    if random.random() < 0.95:
      noise = np.random.randn(feats.shape[0], 3)
      noise *= self.std * 255
      feats[:, :3] = np.clip(noise + feats[:, :3], 0, 255)
    return coords, feats, labels

class RandomHorizontalFlip(object):

  def __init__(self, upright_axis, is_temporal):
    """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
    self.is_temporal = is_temporal
    self.D = 4 if is_temporal else 3
    self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
    # Use the rest of axes for flipping.
    self.horz_axes = set(range(self.D)) - set([self.upright_axis])

  def __call__(self, coords, feats, labels):
    if random.random() < 0.95:
      for curr_ax in self.horz_axes:
        if random.random() < 0.5:
          coord_max = np.max(coords[:, curr_ax])
          coords[:, curr_ax] = coord_max - coords[:, curr_ax]
    return coords, feats, labels

class RandomDropout(object):

  def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
    """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
    self.dropout_ratio = dropout_ratio
    self.dropout_application_ratio = dropout_application_ratio

  def __call__(self, coords, feats, labels):
    if random.random() < self.dropout_ratio:
      N = len(coords)
      inds = np.random.choice(N, int(N * (1 - self.dropout_ratio)), replace=False)
      return coords[inds], feats[inds], labels[inds]
    return coords, feats, labels

class ElasticDistortion:

  def __init__(self, distortion_params):
    self.distortion_params = distortion_params

  def elastic_distortion(self, coords, feats, labels, granularity, magnitude):
    """Apply elastic distortion on sparse coordinate space.

      pointcloud: numpy array of (number of points, at least 3 spatial dims)
      granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
      magnitude: noise multiplier
    """
    blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
    blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
    blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
    coords_min = coords.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
      noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
      noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
      noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity *
                                   (noise_dim - 2), noise_dim)
    ]
    interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
    coords += interp(coords) * magnitude
    return coords, feats, labels

  def __call__(self, coords, feats, labels):
    if self.distortion_params is not None:
      if random.random() < 0.95:
        for granularity, magnitude in self.distortion_params:
          coords, feats, labels = self.elastic_distortion(coords, feats, labels, granularity,
                                                          magnitude)
    return coords, feats, labels