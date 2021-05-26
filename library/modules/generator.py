# import torch
from torch import nn

from library.modules.util import Encoder


class MotionTransferGenerator(nn.Module):
    """
    Motion transfer generator. That given a keypoints and an appearance trying to reconstruct the target frame.
    Produce 2 versions of target frame, one warped with predicted optical flow and other refined.
    """
    def __init__(self, num_channels, num_kp, kp_variance, block_expansion, max_features, num_blocks,
                 num_refinement_blocks, dense_motion_params=None, kp_embedding_params=None,
                 interpolation_mode='neareset'):
        super(MotionTransferGenerator, self).__init__()
        print(" [!] Initialize MotionTransferGenerator!")

    def forward(self, x):
        return x
