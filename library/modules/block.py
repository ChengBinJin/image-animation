import torch
import torch.nn.functional as F
from torch import nn

from library.third_partys.sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d
from library.third_partys.sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d


class Encoder2(nn.Module):
    """
    Hourglass2 Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder2, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i==0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))), kernel_size=3,
                                           padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder2(nn.Module):
    """
    Hourglass2 Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder2, self).__init__()

        up_blocks = []
        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out


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
                DownBlock3dGen(in_features=in_features if i == 0 else min(max_features, block_expansion * (2**i)),
                               out_features=min(max_features, block_expansion * (2 ** (i+1))),
                               kernel_size=kernel_size, padding=padding))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            out = down_block(outs[-1])
            outs.append(out)
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
            up_blocks.append(UpBlock3d(
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


class Hourglass2(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass2, self).__init__()
        self.encoder = Encoder2(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder2(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Hourglass(nn.Module):
    """
    Hourglass architecture。
    """

    def __init__(self, block_expansion, in_features, out_features, num_blocks=3, max_features=256, temporal=False):
        super(Hourglass, self).__init__()

        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features, temporal=temporal)
        self.decoder = Decoder(block_expansion, in_features, out_features, num_blocks, max_features, temporal=temporal)

    def forward(self, x):
        out = self.decoder(self.encoder(x))  # (N, num_kp, 1, H, W)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.activate = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.nom(out)
        out = self.activate(out)
        out = self.pool(out)
        return out


class DownBlock3dGen(nn.Module):
    """
    Simple block for processing video (encoder)
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(DownBlock3dGen, self).__init__()
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


class DownBlock3dDis(nn.Module):
    """
    Simple block for processing video (encoder).
    """

    def __init__(self, in_features, out_features, norm=False, kernel_size=4):
        super(DownBlock3dDis, self).__init__()
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


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.activate = nn.ReLU()

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = self.activate(out)
        return out


class UpBlock3d(nn.Module):
    """
    Simple block for processing video (decoder).
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(UpBlock3d, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=(1, 2, 2), recompute_scale_factor=True)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.nomr = BatchNorm2d(out_features, affine=True)
        self.activate = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.activate(out)
        return out


class SameBlock3d(nn.Module):
    """
    Simple block with group convoluiton
    """

    def __init__(self, in_features, out_features, groups=None, kernel_size=3, padding=1):
        super(SameBlock3d, self).__init__()

        self.conv = nn.Conv3d(
            in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm3d(num_features=out_features, affine=True)
        self.activate = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.activate(out)

        return out


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)
        self.activate1 = nn.ReLU()
        self.activate2 = nn.ReLU()

    def forward(self, x):
        out = self.norm1(x)
        out = self.activate1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.activate2(out)
        out = self.conv2(out)
        out += x
        return out


class ResBlock3d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock3d, self).__init__()
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


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """

    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernel = torch.tensor(1)
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])

        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equls 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, x):
        if self.scale == 1.0:
            return x

        out = F.pad(x, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out

