from torch import nn


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
                DownBlock3D(in_features if i == 0 else min(max_features, block_expansion * (2**i)),
                            min(max_features, block_expansion * (2 ** (i+1))),
                            kernel_size=kernel_size, padding=padding))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs
