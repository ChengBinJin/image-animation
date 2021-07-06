import torch
import numpy as np

from scipy.spatial import ConvexHull
from library.utils.matrix import matrix_inverse, smallest_singular
from library.utils.flow import make_coordinate_grid
from library.utils.matrix import make_symetric_matrix


def kp2gaussian(kp, spatial_size, kp_variance='matrix'):
    """
    Transform a keypoint into gaussian like representation
    """
    # kp    - mean: (N, 1, num_kp, 2)
    #       - var:  (N, 1, num_kp, 2, 2)
    # spatial_size = (H, W)
    mean = kp['mean']                                                       # (N, 1, num_kp, 2)
    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())       # (H, W, 2)

    number_of_leading_dimensions = len(mean.shape) - 1                      # 3
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape     # (1, 1, 1, H, W, 2)

    coordinate_grid = coordinate_grid.view(*shape)                          # (1, 1, 1, H, W, 2)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)         # (N, 1, num_kp, 1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)                      # (N, 1, num_kp, H, W, 2)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)           # (N, 1, num_kp, 1, 1, 2)
    mean = mean.view(*shape)                                                # (N 1, num_kp, 1, 1, 2)

    mean_sub = (coordinate_grid - mean)                                     # (N, 1, num_kp, H, W, 2)
    if kp_variance == 'matrix':
        var = kp['var']                                                     # (N, 1, num_kp, 2, 2)
        inv_var = matrix_inverse(var)                                       # (N, 1, num_kp, 2, 2)
        shape = inv_var.shape[:number_of_leading_dimensions] + (1, 1, 2, 2)  # (N, 1, num_kp, 1, 1, 2, 2)
        inv_var = inv_var.view(*shape)                                      # (N, 1, num_kp, 1, 1, 2, 2)
        under_exp = torch.matmul(torch.matmul(mean_sub.unsqueeze(-2), inv_var), mean_sub.unsqueeze(-1))     # (N, 1, num_kp, H, W, 1, 1)
        under_exp = under_exp.squeeze(-1).squeeze(-1)                       # (N, 1, num_kp, H, W)
        out = torch.exp(-0.5 * under_exp)                                   # (N, 1, num_kp, H, W)
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


def normalize_kp(kp_video, kp_appearance, movement_mult=False, move_location=False, adapt_variance=False,
                 clip_mean=False):
    if movement_mult:
        appearance_area = ConvexHull(kp_appearance['mean'][0, 0].data.cpu().numpy()).volume
        video_area = ConvexHull(kp_video['mean'][0, 0].data.cpu().numpy()).volume
        movement_mult = np.sqrt(appearance_area) / np.sqrt(video_area)
    else:
        movement_mult = 1

    kp_video = {k: v for k, v in kp_video.items()}

    if move_location:
        kp_video_diff = (kp_video['mean'] - kp_video['mean'][:, 0:1])
        kp_video_diff *= movement_mult
        kp_video['mean'] = kp_video_diff + kp_appearance['mean']

    if move_location and clip_mean:
        one = torch.ones(1).type(kp_video_diff.type())
        kp_video['mean'] = torch.max(kp_video['mean'], -one)
        kp_video['mean'] = torch.min(kp_video['mean'], one)

    if ('var' in kp_video) and adapt_variance:
        var_first = kp_video['var'][:, 0:1].repeat(1, kp_video['var'].shape[1], 1, 1, 1)
        kp_var, _ = torch.solve(var_first, kp_video['var'])

        kp_var = torch.matmul(kp_video['var'], matrix_inverse(kp_video['var'][:, 0:1], eps=0))
        kp_var = torch.matmul(kp_var, kp_appearance['var'])

        kp_var = make_symetric_matrix(kp_var)
        kp_video['var'] = kp_var

    return kp_video


def split_kp(kp_joined, detach=False):
    if detach:
        kp_video = {k: v[:, 1:].detach() for k, v in kp_joined.items()}
        kp_appearance = {k: v[:, :1].detach() for k, v in kp_joined.items()}
    else:
        kp_video = {k: v[:, 1:] for k, v in kp_joined.items()}
        kp_appearance = {k: v[:, :1] for k, v in kp_joined.items()}

    out = dict()
    out['kp_driving'] = kp_video
    out['kp_source'] = kp_appearance

    return out

