import torch
import torch.nn.functional as F

from torch import nn

from library.modules.dense_motion import MovementEmbeddingModule
from library.modules.block import DownBlock3DDis


class Discriminator(nn.Module):
    """
    Discriminator similar to Pix2Pix
    """

    def __init__(self, num_channels=3, num_kp=10, kp_variance=0.01, scale_factor=1, block_expansion=64, num_blocks=4,
                 max_features=512, kp_embedding_params=None):
        super(Discriminator, self).__init__()

        if kp_embedding_params is not None:
            self.kp_embedding = MovementEmbeddingModule(
                num_kp=num_kp, kp_variance=kp_variance, num_channels=num_channels, **kp_embedding_params)
            embedding_channels = self.kp_embedding.out_channels
        else:
            self.kp_embedding = None
            embedding_channels = 0

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock3DDis(
                    in_features=num_channels + embedding_channels if i == 0 else min(max_features, block_expansion * (2 ** i)),
                    out_features=min(max_features, block_expansion * (2 ** (i + 1))),
                    norm=(i != 0),
                    kernel_size=4))

        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv3d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
        self.scale_factor = scale_factor

    def forward(self, x, kp_driving, kp_source):
        out_maps = [x]
        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=(1, self.scale_factor, self.scale_factor))

        if self.kp_embedding:
            heatmap = self.kp_embedding(x, kp_driving, kp_source)
            out = torch.cat([x, heatmap], dim=1)
        else:
            out = x

        for down_block in self.down_blocks:
            out_maps.append(down_block(out))
            out = out_maps[-1]

        out = self.conv(out)
        out_maps.append(out)

        return out_maps


