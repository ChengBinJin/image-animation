import torch
import torch.nn.functional as F
from torch import nn

from library.modules.util import Hourglass, make_coordinate_grid, matrix_inverse, smallest_singular


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

    def forward(self, x):  # x shape: (N, t, 3, H, W)
        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=(1, self.scale_factor, self.scale_factor))

        heatmap = self.predictor(x)     # [N, kp, 3, H/2, W/2]
        final_shape = heatmap.shape     # [N, kp, 3, H/2, W/2]
        heatmap = heatmap.view(final_shape[0], final_shape[1], final_shape[2], -1)  # [N, kp, 3, H/2*W/2]
        heatmap = F.softmax(heatmap / self.temperature, dim=3)                      # [N, kp, 3, H/2*W/2]
        heatmap = heatmap.view(*final_shape)                                        # [N, kp, 3, H/2, W/2]
        out = gaussian2kp(heatmap, self.kp_variance, self.clip_variance)

        return out


def kp2gaussian(kp, spatial_size, kp_variance='matrix'):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['mean']                                                       # (1, kp, 2)
    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())       # (H, W, 2)

    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape     # (1, 1, H, W, 2)

    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)         # (1, kp, 1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)                      # (1, kp, H, W, 2)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)           # (1, kp, 1, 1, 2)
    mean = mean.view(*shape)                                                # (1, kp, 1, 1, 2)

    mean_sub = (coordinate_grid - mean)
    if kp_variance == 'matrix':
        var = kp['var']
        inv_var = matrix_inverse(var)                                       # (1, 1, 2, 2)
        shape = inv_var.shape[:number_of_leading_dimensions] + (1, 1, 2, 2)  # (1, 1, 1, 1, 2, 2)
        inv_var = inv_var.view(*shape)
        under_exp = torch.matmul(torch.matmul(mean_sub.unsqueeze(-2), inv_var), mean_sub.unsqueeze(-1))
        under_exp = under_exp.squeeze(-1).squeeze(-1)
        out = torch.exp(-0.5 * under_exp)
    elif kp_variance == 'single':
        out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp['var'])
    else:
        out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


def gaussian2kp(heatmap, kp_variance='matrix', clip_variance=None):
    """
    Extract the mean and the variance from a heatmp
    """
    shape = heatmap.shape                   # [N, kp, 3, H/2, W/2]
    # adding small eps to avoid 'nan' in variance
    heatmap = heatmap.unsqueeze(-1) + 1e-7  # [N, kp, 3, H/2, W/2, 1]
    grid = make_coordinate_grid(shape[3:], heatmap.type()).unsqueeze_(0).unsqueeze_(0).unsqueeze_(0)  # [1, 1, 1, H/2, W/2, 2]

    mean = (heatmap * grid).sum(dim=(3, 4))  # [N, kp, 3, 2]
    kp = {'mean': mean.permute(0, 2, 1, 3)}  # [N, 3, kp, 2]

    if kp_variance == 'matrix':
        mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)  # [N, kp, 3, H/2, W/2, 2]
        var = torch.matmul(mean_sub.unsqueeze(-1), mean_sub.unsqueeze(-2))  # [N, kp, 3, H/2, W/2, 2, 2]
        var = var * heatmap.unsqueeze(-1)                   # [N, kp, 3, H/2, W/2, 2, 2]
        var = var.sum(dim=(3, 4))                           # [N, kp, 3, 2, 2]
        var = var.permute(0, 2, 1, 3, 4)                    # [N, 3, kp, 2, 2]
        if clip_variance:
            min_norm = torch.tensor(clip_variance).type(var.type())
            sg = smallest_singular(var).unsqueeze(-1)
            var = torch.max(min_norm, sg) * var / sg
        kp['var'] = var
    elif kp_variance == 'single':
        mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)  # [N, kp, 3, H/2, W/2, 2]
        var = mean_sub ** 2                                 # [N, kp, 3, H/2, W/2, 2]
        var = var * heatmap                                 # [N, kp, 3, H/2, W/2, 2]
        var = var.sum(dim=(3, 4))                           # [N, kp, 3, 2]
        var = var.mean(dim=-1, keepdim=True)                # [N, kp, 3, 1]
        var = var.unsqueeze(-1)                             # [N, kp, 3, 1, 1]
        var = var.permute(0, 2, 1, 3, 4)                    # [N, 3, kp, 1, 1]
        kp['var'] = var

    return kp
