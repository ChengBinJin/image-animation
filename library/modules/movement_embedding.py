# import torch
import torch.nn.functional as F
from torch import nn

from library.modules.keypoint_detector import kp2gaussian


class MovementEmbeddingModule(nn.Module):
    """
    Procude a keypoint representation that will be further used by other moudles
    """

    def __init__(self, num_kp, kp_variance, num_channels, use_deformed_source_image=False, use_difference=False,
                 use_heatmap=True, add_bg_feature_map=False, heatmap_type="gaussian", norm_const="sum", scale_factor=1):
        super(MovementEmbeddingModule, self).__init__()

        assert heatmap_type in ['guassian', 'difference']
        assert int(use_heatmap) + int(use_deformed_source_image) + int(use_difference) >= 1

        self.out_channels = (1 * use_heatmap + 2 * use_difference + num_channels * use_deformed_source_image) * (
                num_kp + add_bg_feature_map)

        self.kp_variance = kp_variance
        self.heatmap_type = heatmap_type
        self.use_difference = use_difference
        self.use_deformed_source_image = use_deformed_source_image
        self.use_heatmap = use_heatmap
        self.add_bg_feature_map = add_bg_feature_map
        self.norm_const = norm_const
        self.scale_factor = scale_factor

    def forward(self, source_image, kp_driving, kp_source):
        if self.scale_factor != 1:
            source_image = F.interpolate(source_image, scale_factor=(1, self.sacle_factor, self.scale_factor))

        spatial_size = source_image.shape[3:]

        bs, _, _, h, w = source_image.shape
        _, d, num_kp, _ = kp_driving['mean'].shape

        # inputs = []
        # if self.use_heatmap:
        #     heatmap = self.normalize_heatmap(
        #         kp2guassian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance))
