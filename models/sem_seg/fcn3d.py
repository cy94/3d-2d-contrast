'''
3D fully conv network
'''

import torch
import torch.nn as nn

class FCN3D(nn.Module):
    '''
    Dense 3d convs on a volume grid
    '''
    def __init__(self, in_channels, num_classes):
        '''
        in_channels: number of channels in input

        '''
        super().__init__()
        self.layers = nn.ModuleList([
            # args: inchannels, outchannels, kernel, stride, padding
            # 1->1/2
            nn.Conv3d(in_channels, 32, 3, 2, 1),
            # same
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.Conv3d(32, 64, 3, 1, 1),
            nn.ReLU(),

            # 1/2->1/4
            nn.Conv3d(64, 64, 3, 2, 1),
            # same
            nn.Conv3d(64, 64, 3, 1, 1),
            nn.Conv3d(64, 128, 3, 1, 1),
            nn.ReLU(),
            
            # 1.4->1/8
            nn.Conv3d(128, 128, 3, 2, 1),
            # same
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.ReLU(),
            
            # inchannels, outchannels, kernel, stride, padding, output_padding
            # 1/8->1/4
            nn.ConvTranspose3d(128, 128, 4, 2, 1),
            nn.ReLU(),
            # 1/4->1/2
            nn.ConvTranspose3d(128, 64, 4, 2, 1),
            nn.ReLU(),
            # 1/2->original shape
            nn.ConvTranspose3d(64, num_classes, 4, 2, 1),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x