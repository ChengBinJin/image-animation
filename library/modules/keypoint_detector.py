import torch
import torch.nn.functional as F
from torch import nn

from library.modules.block import Hourglass, Hourglass2, AntiAliasInterpolation2d
# from library.utils.matrix import matrix_inverse, smallest_singular
# from library.utils.flow import make_coordinate_grid
from library.utils.keypoint import gaussian2kp, gaussian2kp2


class KPDetector2(nn.Module):
    """
    Detecting a keypoints. Retrun keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features, num_blocks, temperature,
                 estimate_jacobian=False, scale_factor=1, single_jacobian_map=False, pad=0):
        super(KPDetector2, self).__init__()

        self.predictor = Hourglass2(block_expansion, in_features=num_channels, max_features=max_features,
                                    num_blocks=num_blocks)

        self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
                            padding=pad)

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=4*self.num_jacobian_maps,
                                      kernel_size=(7, 7), padding=pad)
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float32))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)


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

    def forward(self, x):  # x shape: (N, 3, n, H, W)
        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=(1, self.scale_factor, self.scale_factor), recompute_scale_factor=False)

        heatmap = self.predictor(x)     # [N, kp, n, h, w]
        final_shape = heatmap.shape     # [N, kp, n, h, w]
        heatmap = heatmap.view(final_shape[0], final_shape[1], final_shape[2], -1)  # [N, kp, n, h*w]
        heatmap = F.softmax(heatmap / self.temperature, dim=3)                      # [N, kp, n, H/2*W/2]
        heatmap = heatmap.view(*final_shape)                                        # [N, kp, n, H/2, W/2]
        out = gaussian2kp(heatmap, self.kp_variance, self.clip_variance)
        # out   - mean:     (N, n, mnum_kp, 2)
        #       - var:      (N, n, num_kp, 2, 2)

        return out


