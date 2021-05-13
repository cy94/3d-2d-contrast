'''
3D fully conv network
'''

import torch
import torch.nn as nn

class FCN3D(nn.Module):
    '''
    Dense 3d convs on a volume grid
    '''
    def __init__(self, in_channels, num_classes, grid_size):
        '''
        in_channels: number of channels in input

        '''
        super().__init__()
        self.model = nn.Sequential(
            # args: inchannels, outchannels, kernel, stride, padding
            nn.Conv3d(in_channels, 8, 3, 2),
            nn.Conv3d(8, 16, 3, 2),
            nn.Conv3d(16, 32, 3, 2),
            nn.Conv3d(32, 64, 3, 2),
            nn.Conv3d(64, num_classes, 3, 2),
            nn.Upsample(grid_size[::-1])
        )

    def forward(self, x):
        return self.model(x)