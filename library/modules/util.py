import torch
import torch.nn.functional as F
from torch import nn

from library.modules.sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d


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
                DownBlock3D(in_features=in_features if i == 0 else min(max_features, block_expansion * (2**i)),
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
        return self.decoder(self.encoder(x))


class DownBlock3D(nn.Module):
    """
    Simple block for processing video (encoder)
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(DownBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = BatchNorm3d(out_features, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
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
        out = F.interpolate(x, scale_factor=(1, 2, 2))
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


def make_coordinate_grid(spatial_size, dtype):
    """
    Create a meshgrid [-1, 1] x [-1, 1] of given spatial_size
    """
    h, w = spatial_size
    x = torch.arange(w).type(dtype)
    y = torch.arange(h).type(dtype)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(dim=2), yy.unsqueeze_(2)], dim=2)  # H, W, 2

    return meshed


def matrix_inverse(batch_of_matrix, eps=0):
    if eps != 0:
        init_shape = batch_of_matrix.shape
        a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
        b = batch_of_matrix[..., 0, 1].unsqueeze(-1)
        c = batch_of_matrix[..., 1, 0].unsqueeze(-1)
        d = batch_of_matrix[..., 1, 1].unsqueeze(-1)

        det = a * d - b * c
        out = torch.cat([d, -b, -c, a], dim=-1)
        eps = torch.tensor(eps).type(out.type())
        out /= det.max(eps)

        return out.view(init_shape)
    else:
        b_mat = batch_of_matrix
        eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
        b_inv, _ = torch.solve(eye, b_mat)
        return b_inv


def smallest_singular(batch_of_matrix):
    a = (batch_of_matrix[..., 0, 0]).unsqueeze(-1)
    b = (batch_of_matrix[..., 0, 1]).unsqueeze(-1)
    c = (batch_of_matrix[..., 1, 0]).unsqueeze(-1)
    d = (batch_of_matrix[..., 1, 1]).unsqueeze(-1)

    s1 = a ** 2 + b ** 2 + c ** 2 + d ** 2
    s2 = (a ** 2 + b ** 2 - c ** 2 - d ** 2) ** 2
    s2 = torch.sqrt(s2 + 4 * (a * c + b * d) ** 2)
    norm = torch.sqrt((s1 - s2) / 2)

    return norm
