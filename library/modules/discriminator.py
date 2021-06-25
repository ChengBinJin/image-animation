import torch
import torch.nn.functional as F

from torch import nn

from library.modules.dense_motion import MovementEmbeddingModule


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
                DownBlock3D(
                    num_channels + embedding_channels if i == 0 else min(max_features, block_expansion * (2 ** i)),
                    min(max_features, block_expansion * (2 ** (i + 1))),
                    norm=(i != 0),
                    kenerl_size=4))

        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv3d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
        self.scale_factor = scale_factor
