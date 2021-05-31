import torch

from library.utils.matrix import matrix_inverse, smallest_singular
from library.utils.flow import make_coordinate_grid


def kp2gaussian(kp, spatial_size, kp_variance='matrix'):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['mean']                                                       # (1, 3, kp, 2)
    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())       # (H/2, W/2, 2)

    number_of_leading_dimensions = len(mean.shape) - 1                      # 3
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape     # (1, 1, 1, H/2, W/2, 2)

    coordinate_grid = coordinate_grid.view(*shape)                          # (1, 1, 1, H/2, W/2, 2)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)         # (1, 3, kp, 1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)                      # (1, 3, kp, H/2, W/2, 2)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)           # (1, 3, kp, 1, 1, 2)
    mean = mean.view(*shape)                                                # (1, 3, kp, 1, 1, 2)

    mean_sub = (coordinate_grid - mean)                                     # (1, 3, kp, H/2, W/2, 2)
    if kp_variance == 'matrix':
        var = kp['var']                                                     # (1, 3, kp, 2, 2)
        inv_var = matrix_inverse(var)                                       # (1, 3, kp, 2, 2)
        shape = inv_var.shape[:number_of_leading_dimensions] + (1, 1, 2, 2)  # (1, 3, 10, 1, 1, 2, 2)
        inv_var = inv_var.view(*shape)                                      # (1, 3, kp, 1, 1, 2, 2)
        under_exp = torch.matmul(torch.matmul(mean_sub.unsqueeze(-2), inv_var), mean_sub.unsqueeze(-1))     # (1, 3, kp, H/2, W/2, 1, 1)
        under_exp = under_exp.squeeze(-1).squeeze(-1)                       # (1, 3, kp, H/2, W/2)
        out = torch.exp(-0.5 * under_exp)                                   # (1, 3, kp, H/2, W/2)
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
