# import torch
from torch import nn

from library.modules.util import Encoder
from library.modules.movement_embedding import MovementEmbeddingModule


class MotionTransferGenerator(nn.Module):
    """
    Motion transfer generator. That given a keypoints and an appearance trying to reconstruct the target frame.
    Produce 2 versions of target frame, one warped with predicted optical flow and other refined.
    """
    def __init__(self, num_channels, num_kp, kp_variance, block_expansion, max_features, num_blocks,
                 num_refinement_blocks, dense_motion_params=None, kp_embedding_params=None,
                 interpolation_mode='neareset'):
        super(MotionTransferGenerator, self).__init__()

        self.appearance_encoder = Encoder(block_expansion, in_features=num_channels, max_features=max_features,
                                          num_blocks=num_blocks)

        if kp_embedding_params is not None:
            self.kp_embedding_module = MovementEmbeddingModule(num_kp=num_kp, kp_variance=kp_variance,
                                                               num_channels=num_channels, **kp_embedding_params)
            embedding_features = self.kp_embedding_module.out_channels

        print(" [!] Initialize MotionTransferGenerator!")

    def forward(self, x):
        return x
