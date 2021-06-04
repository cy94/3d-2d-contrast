import torch.nn as nn

class Down3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # x->x/2 dimensions
        self.layers = nn.Sequential(
            # args: inchannels, outchannels, kernel, stride, padding
            # x->x/2 dimensions
            nn.Conv3d(in_channels, out_channels, 3, 2, 1),
            nn.ReLU(),
            # same
            nn.Conv3d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout3d(0.1)
        )

    def forward(self, x):
        return self.layers(x)

class Up3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            # inchannels, outchannels, kernel, stride, padding, output_padding
            # x/2->x dimensions
            nn.ConvTranspose3d(in_channels, out_channels, 4, 2, 1),
            nn.ReLU(),
            # same
            nn.Conv3d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout3d(0.1)
        )

    def forward(self, x):
        return self.layers(x)


