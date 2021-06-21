import torch
import torch.nn.functional as F
from torch import nn

from library.modules.block import Hourglass
from library.utils.matrix import matrix_inverse, smallest_singular
from library.utils.flow import make_coordinate_grid
from library.utils.keypoint import gaussian2kp


class KPDetector(nn.Module):
    """
    Detecting a keypoint. Return keypoint position and variance.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features, num_blocks, temperature,
                 kp_variance, scale_factor=1, clip_variance=None):
        super(KPDetector, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels, out_features=num_kp,
                                   max_features=max_features, num_blocks=num_blocks)
        self.temperature = temperature
        self.kp_variance = kp_variance
        self.scale_factor = scale_factor
        self.clip_variance = clip_variance

    def forward(self, x):  # x shape: (N, 3, 1, H, W)
        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=(1, self.scale_factor, self.scale_factor), recompute_scale_factor=True)

        heatmap = self.predictor(x)     # [N, kp, 1, H/2, W/2]
        final_shape = heatmap.shape     # [N, kp, 1, H/2, W/2]
        heatmap = heatmap.view(final_shape[0], final_shape[1], final_shape[2], -1)  # [N, kp, 1, H/2*W/2]
        heatmap = F.softmax(heatmap / self.temperature, dim=3)                      # [N, kp, 1, H/2*W/2]
        heatmap = heatmap.view(*final_shape)                                        # [N, kp, 1, H/2, W/2]
        out = gaussian2kp(heatmap, self.kp_variance, self.clip_variance)
        # out   - mean:     (N, 1, mnum_kp, 2)
        #       - var:      (N, 1, num_kp, 2, 2)

        return out


