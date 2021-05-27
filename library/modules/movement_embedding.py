# import torch
from torch import nn


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

