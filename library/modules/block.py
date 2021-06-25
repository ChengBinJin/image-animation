import torch
import torch.nn.functional as F
from torch import nn

from library.sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256, temporal=False):
        super(Encoder, self).__init__()

        down_blocks = []

        kernel_size = (3, 3, 3) if temporal else (1, 3, 3)
        padding = (1, 1, 1) if temporal else (0, 1, 1)
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock3DGen(in_features=in_features if i == 0 else min(max_features, block_expansion * (2**i)),
                               out_features=min(max_features, block_expansion * (2 ** (i+1))),
                               kernel_size=kernel_size, padding=padding))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, out_features, num_blocks=3, max_features=256, temporal=False,
                 additional_features_for_block=0, use_last_conv=True):
        super(Decoder, self).__init__()

        kernel_size = (3, 3, 3) if temporal else (1, 3, 3)
        padding = (1, 1, 1) if temporal else (0, 1, 1)

        up_blocks = []
        for i in range(num_blocks)[::-1]:
            up_blocks.append(UpBlock3D(
                in_features=(1 if i == num_blocks - 1 else 2) * min(
                    max_features, block_expansion * (2 ** (i + 1))) + additional_features_for_block,
                out_features=min(max_features, block_expansion * (2 ** i)),
                kernel_size=kernel_size, padding=padding))

        self.up_blocks = nn.ModuleList(up_blocks)
        if use_last_conv:
            self.conv = nn.Conv3d(in_channels=block_expansion + in_features + additional_features_for_block,
                                  out_channels=out_features, kernel_size=kernel_size, padding=padding)
        else:
            self.conv = None

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            out = torch.cat([out, x.pop()], dim=1)

        if self.conv is not None:
            return self.conv(out)
        else:
            return out


class Hourglass(nn.Module):
    """
    Hourglass architecture
    """

    def __init__(self, block_expansion, in_features, out_features, num_blocks=3, max_features=256, temporal=False):
        super(Hourglass, self).__init__()

        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features, temporal=temporal)
        self.decoder = Decoder(block_expansion, in_features, out_features, num_blocks, max_features, temporal=temporal)

    def forward(self, x):
        out = self.decoder(self.encoder(x))  # (N, num_kp, 1, H, W)
        return out


class DownBlock3DGen(nn.Module):
    """
    Simple block for processing video (encoder)
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(DownBlock3DGen, self).__init__()
        self.conv = nn.Conv3d(
            in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding)
        self.norm = BatchNorm3d(out_features, affine=True)
        self.activate = nn.ReLU()
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.activate(out)
        out = self.pool(out)
        return out


class DownBlock3DDis(nn.Module):
    """
    Simple block for processing video (encoder).
    """

    def __init__(self, in_features, out_features, norm=False, kernel_size=4):
        super(DownBlock3DDis, self).__init__()
        self.conv = nn.Conv3d(
            in_channels=in_features, out_channels=out_features, kernel_size=(1, kernel_size, kernel_size))
        if norm:
            self.norm = nn.InstanceNorm3d(out_features, affine=True)
        else:
            self.norm = None
        self.activate = nn.LeakyReLU(negative_slope=0.2)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        out = self.conv(x)
        if self.norm:
            out = self.norm(out)
        out = F.leaky_relu(out)
        out = self.pool(out)
        return out


class UpBlock3D(nn.Module):
    """
    Simple block for processing video (decoder).
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(UpBlock3D, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=(1, 2, 2), recompute_scale_factor=True)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class SameBlock3D(nn.Module):
    """
    Simple block with group convoluiton
    """

    def __init__(self, in_features, out_features, groups=None, kernel_size=3, padding=1):
        super(SameBlock3D, self).__init__()

        self.conv = nn.Conv3d(
            in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm3d(num_features=out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)

        return out


class ResBlock3D(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(
            in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, padding=padding)
        self.norm1 = BatchNorm3d(in_features, affine=True)
        self.norm2 = BatchNorm3d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x

        return out



