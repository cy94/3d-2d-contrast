import torch.nn as nn

class SameConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=True, relu=True):
        super().__init__()
        self.layers = nn.Sequential(
            # args: inchannels, outchannels, kernel, stride, padding
            # same conv
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU() if relu else nn.Identity(),
            nn.Dropout3d(0.1) if dropout else nn.Identity()
        )

    def forward(self, x):
        return self.layers(x)

class Down3D_Big(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=True):
        super().__init__()
        # x->x/2
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm3d(out_channels),
        )
        # same conv
        self.block2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm3d(out_channels),
        )
        # same conv
        self.block3 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm3d(out_channels),
        )

        self.dropout = nn.Dropout3d(0.2) if dropout else nn.Identity()

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        # res connection
        x3 = self.block3(x2) + x1
        x4 = self.dropout(x3)

        return x4

class Up3D_Big(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=True):
        super().__init__()
        # x->2x
        self.block1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm3d(out_channels),
        )
        # same conv
        self.block2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm3d(out_channels),
        )
        # same conv
        self.block3 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm3d(out_channels),
        )

        self.dropout = nn.Dropout3d(0.2) if dropout else nn.Identity()

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        # res connection
        x3 = self.block3(x2) + x1
        x4 = self.dropout(x3)

        return x4


class Down3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=True, relu=True):
        super().__init__()
        # x->x/2 dimensions
        self.layers = nn.Sequential(
            # args: inchannels, outchannels, kernel, stride, padding
            # x->x/2 dimensions
            nn.Conv3d(in_channels, out_channels, 3, 2, 1),
            nn.ReLU(),
            # same
            nn.Conv3d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU() if relu else nn.Identity(),
            nn.Dropout3d(0.1) if dropout else nn.Identity()
        )

    def forward(self, x):
        return self.layers(x)

class Up3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=True, relu=True):
        super().__init__()
        self.layers = nn.Sequential(
            # inchannels, outchannels, kernel, stride, padding, output_padding
            # x/2->x dimensions
            nn.ConvTranspose3d(in_channels, out_channels, 4, 2, 1),
            nn.ReLU(),
            # same
            nn.Conv3d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU() if relu else nn.Identity(),
            nn.Dropout3d(0.1) if dropout else nn.Identity()
        )

    def forward(self, x):
        return self.layers(x)


