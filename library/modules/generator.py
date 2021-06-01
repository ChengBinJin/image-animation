import torch
from torch import nn

from library.modules.block import Encoder, Decoder, ResBlock3D
from library.modules.dense_motion import MovementEmbeddingModule, DenseMotionModule, IdentityDeformation


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
            self.kp_embedding_module = MovementEmbeddingModule(
                num_kp=num_kp, kp_variance=kp_variance, num_channels=num_channels, **kp_embedding_params)
            embedding_features = self.kp_embedding_module.out_channels
        else:
            self.kp_embedding_module = None
            embedding_features = 0

        if dense_motion_params is not None:
            self.dense_motion_module = DenseMotionModule(
                num_kp=num_kp, kp_variance=kp_variance, num_channels=num_channels, **dense_motion_params)
        else:
            self.dense_motion_module = IdentityDeformation()

        self.video_decoder = Decoder(
            block_expansion=block_expansion, in_features=num_channels, out_features=num_channels,
            max_features=max_features, num_blocks=num_blocks, additional_features_for_block=embedding_features,
            use_last_conv=False)

        in_features = block_expansion + num_channels + embedding_features
        self.refinement_module = torch.nn.Sequential()
        for i in range(num_refinement_blocks):
            self.refinement_module.add_module(
                'r' + str(i), ResBlock3D(in_features, kernel_sizes=(1, 3, 3), padding=(0, 1, 1)))

    def forward(self, x):
        return x
