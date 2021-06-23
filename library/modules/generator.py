import torch
import torch.nn.functional as F
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
                 interpolation_mode='nearest'):
        super(MotionTransferGenerator, self).__init__()

        self.appearance_encoder = Encoder(block_expansion, in_features=num_channels, max_features=max_features,
                                          num_blocks=num_blocks)

        if kp_embedding_params is not None:
            self.kp_embedding_module = MovementEmbeddingModule(
                num_kp=num_kp, kp_variance=kp_variance, num_channels=num_channels, **kp_embedding_params)
            #
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
                'r' + str(i), ResBlock3D(in_features, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
        self.refinement_module.add_module(
            'conv-last', nn.Conv3d(in_channels=in_features, out_channels=num_channels, kernel_size=1, padding=0))
        self.interpolation_mode = interpolation_mode

    def deform_input(self, inp, deformations_absolute):
        # inpu:                     (N, C, 1, H, W)
        # deformation_absolute:     (N, 1, H, W, 3)
        bs, d, h_old, w_old, _ = deformations_absolute.shape
        _, _, _, h, w = inp.shape
        deformations_absolute = deformations_absolute.permute(0, 4, 1, 2, 3)  # (N, 3, 1, H, W)
        deformation = F.interpolate(
            deformations_absolute, size=(d, h, w), mode=self.interpolation_mode, recompute_scale_factor=False)
        deformation = deformation.permute(0, 2, 3, 4, 1)  # (N, 1, H, W, 3)
        deformed_inp = F.grid_sample(inp, deformation, align_corners=True)  # (N, C, 1, H, W)
        return deformed_inp

    def forward(self, source_image, kp_driving, kp_source):
        # source_image: (1, 3, 1, H, W)
        # kp_driving    - mean: (1, 1, num_kp, 2)
        #               - var:  (1, 1, num_kp, 2, 2)
        # kp_source:    - mean: (1, 1, num_kp, 2)
        #               - var:  (1, 1, num_kp, 2, 2)
        appearance_skips = self.appearance_encoder(source_image)

        deformations_absolute = self.dense_motion_module(
            source_image=source_image, kp_driving=kp_driving, kp_source=kp_source)
        # deformations_absolute:    (1, 1, H, W, 3)

        deformed_skips = [self.deform_input(skip, deformations_absolute) for skip in appearance_skips]

        if self.kp_embedding_module is not None:
            d = kp_driving['mean'].shape[1]  # d = 1
            movement_embedding = self.kp_embedding_module(
                source_image=source_image, kp_driving=kp_driving, kp_source=kp_source)  # (1, num_kp, 1, H, W)
            kp_skips = [F.interpolate(
                movement_embedding, size=(d,) + skip.shape[3:], mode=self.interpolation_mode,
                recompute_scale_factor=False) for skip in appearance_skips]
            skips = [torch.cat([a, b], dim=1) for a, b in zip(deformed_skips, kp_skips)]
        else:
            skips = deformed_skips

        video_deformed = self.deform_input(source_image, deformations_absolute)  # (N, 3, 1, H, W)
        video_prediction = self.video_decoder(skips)  # (N, 3, 1, H, W)
        video_prediction = self.refinement_module(video_prediction)  # (N, 3, 1, H, W)
        video_prediction = torch.sigmoid(video_prediction)  # (N, 3, 1, H, W)

        return {"video_prediction": video_prediction, "video_deformed": video_deformed}
