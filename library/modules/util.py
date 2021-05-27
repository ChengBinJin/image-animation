import torch.nn.functional as F
from torch import nn

from sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d


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

