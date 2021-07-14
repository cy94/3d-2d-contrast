import torch.nn as nn

import MinkowskiEngine as ME

from models.sem_seg.sparse.model import Model
from models.sem_seg.sparse.modules.common import ConvType, NormType, get_norm, conv, sum_pool


class ResNetBase(Model):
  BLOCK = None
  LAYERS = ()
  INIT_DIM = 64
  PLANES = (64, 128, 256, 512)
  OUT_PIXEL_DIST = 32
  HAS_LAST_BLOCK = False
  CONV_TYPE = ConvType.HYPERCUBE

  def __init__(self, in_channels, num_classes, cfg, D=3, **kwargs):
    assert self.BLOCK is not None
    assert self.OUT_PIXEL_DIST > 0

    super(ResNetBase, self).__init__(in_channels, num_classes, cfg, D, **kwargs)

    self.network_initialization(in_channels, num_classes, D)
    self.weight_initialization()

  def network_initialization(self, in_channels, num_classes, D):

    def space_n_time_m(n, m):
      return n if D == 3 else [n, n, n, m]

    if D == 4:
      self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

    dilations = [1,1,1,1]
    bn_momentum = 0.02
    self.inplanes = self.INIT_DIM
    self.conv1 = conv(
        in_channels,
        self.inplanes,
        kernel_size=space_n_time_m(3, 1),
        stride=1,
        D=D)

    self.bn1 = get_norm(NormType.BATCH_NORM, self.inplanes, D=self.D, bn_momentum=bn_momentum)
    self.relu = ME.MinkowskiReLU(inplace=True)
    self.pool = sum_pool(kernel_size=space_n_time_m(2, 1), stride=space_n_time_m(2, 1), D=D)

    self.layer1 = self._make_layer(
        self.BLOCK,
        self.PLANES[0],
        self.LAYERS[0],
        stride=space_n_time_m(2, 1),
        dilation=space_n_time_m(dilations[0], 1))
    self.layer2 = self._make_layer(
        self.BLOCK,
        self.PLANES[1],
        self.LAYERS[1],
        stride=space_n_time_m(2, 1),
        dilation=space_n_time_m(dilations[1], 1))
    self.layer3 = self._make_layer(
        self.BLOCK,
        self.PLANES[2],
        self.LAYERS[2],
        stride=space_n_time_m(2, 1),
        dilation=space_n_time_m(dilations[2], 1))
    self.layer4 = self._make_layer(
        self.BLOCK,
        self.PLANES[3],
        self.LAYERS[3],
        stride=space_n_time_m(2, 1),
        dilation=space_n_time_m(dilations[3], 1))

    self.final = conv(
        self.PLANES[3] * self.BLOCK.expansion, num_classes, kernel_size=1, bias=True, D=D)

  def weight_initialization(self):
    for m in self.modules():
      if isinstance(m, ME.MinkowskiBatchNorm):
        nn.init.constant_(m.bn.weight, 1)
        nn.init.constant_(m.bn.bias, 0)

  def _make_layer(self,
                  block,
                  planes,
                  blocks,
                  stride=1,
                  dilation=1,
                  norm_type=NormType.BATCH_NORM,
                  bn_momentum=0.1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          conv(
              self.inplanes,
              planes * block.expansion,
              kernel_size=1,
              stride=stride,
              bias=False,
              D=self.D),
          get_norm(norm_type, planes * block.expansion, D=self.D, bn_momentum=bn_momentum),
      )
    layers = []
    layers.append(
        block(
            self.inplanes,
            planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            conv_type=self.CONV_TYPE,
            D=self.D))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(
          block(
              self.inplanes,
              planes,
              stride=1,
              dilation=dilation,
              conv_type=self.CONV_TYPE,
              D=self.D))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.pool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.final(x)
    return x