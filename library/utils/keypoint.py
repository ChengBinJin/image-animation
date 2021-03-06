import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial import ConvexHull

from library.utils.matrix import matrix_inverse, smallest_singular
from library.utils.flow import make_coordinate_grid
from library.utils.matrix import make_symetric_matrix
import library.third_partys.face_alignment as face_alignment


def find_best_frame(source, driving, cpu=False):
    def local_normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks_from_image(255 * source)[0]
    kp_source = local_normalize_kp(kp_source)

    diff = float('inf')
    frame_num = 0
    for i, img in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * img)[0]
        kp_driving = local_normalize_kp(kp_driving)
        new_diff = (np.abs(kp_source - kp_driving)**2).sum()

        if new_diff < diff:
            diff = new_diff
            frame_num = i

    return frame_num


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


def cat_dict(data, dim):
    return {k: torch.cat([v[k] for v in data], dim=dim) for k in data[0]}


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


def kp2gaussian2(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representataion
    """
    mean = kp['value']

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


def gaussian2kp2(heatmap):
    """
    Extract the mean and from a heatmap
    """
    shape = heatmap.shape  # (N, num_kp, 58, 58)
    heatmap = heatmap.unsqueeze(-1)  # (N, num_kp, 58, 58, 1)
    grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)  # (1, 1, 58, 58, 2)
    value = (heatmap * grid).sum(dim=(2, 3))  # (N, num_kp, 2)
    kp = {'value': value}

    return kp


def gaussian2kp(heatmap, kp_variance='matrix', clip_variance=None):
    """
    Extract the mean and the variance from a heatmp
    """
    shape = heatmap.shape                   # [N, kp, 1, h, w]
    # adding small eps to avoid 'nan' in variance
    heatmap = heatmap.unsqueeze(-1) + 1e-7  # [N, kp, 2, h, w, 1]
    grid = make_coordinate_grid(shape[3:], heatmap.type()).unsqueeze_(0).unsqueeze_(0).unsqueeze_(0)  # [1, 1, 1, H/2, W/2, 2]

    mean = (heatmap * grid).sum(dim=(3, 4))  # [N, kp, 2, 1]
    kp = {'mean': mean.permute(0, 2, 1, 3)}  # [N, 2, kp, 1]

    if kp_variance == 'matrix':
        mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)  # [N, kp, 2, h, w, 2]
        var = torch.matmul(mean_sub.unsqueeze(-1), mean_sub.unsqueeze(-2))  # [N, kp, 2, h, w, 2, 2]
        var = var * heatmap.unsqueeze(-1)                   # [N, kp, 2, h, w, 2, 2]
        var = var.sum(dim=(3, 4))                           # [N, kp, 2, 2, 2]
        var = var.permute(0, 2, 1, 3, 4)                    # [N, 2, kp, 2, 2]
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


def normalize_kp(kp_source, kp_driving, movement_mult=False, move_location=False, adapt_variance=False,
                 clip_mean=False):
    # normliaze_kp used in MonKeyNet
    if movement_mult:
        appearance_area = ConvexHull(kp_source['mean'][0, 0].data.cpu().numpy()).volume
        video_area = ConvexHull(kp_driving['mean'][0, 0].data.cpu().numpy()).volume
        movement_mult = np.sqrt(appearance_area) / np.sqrt(video_area)
    else:
        movement_mult = 1

    kp_video = {k: v for k, v in kp_driving.items()}

    if move_location:
        kp_video_diff = (kp_video['mean'] - kp_video['mean'][:, 0:1])
        kp_video_diff *= movement_mult
        kp_video['mean'] = kp_video_diff + kp_source['mean']

    if clip_mean:
        one = torch.ones(1).type(kp_video_diff.type())
        kp_video['mean'] = torch.max(kp_video['mean'], -one)
        kp_video['mean'] = torch.min(kp_video['mean'], one)

    if ('var' in kp_video) and adapt_variance:
        var_first = kp_video['var'][:, 0:1].repeat(1, kp_video['var'].shape[1], 1, 1, 1)
        kp_var, _ = torch.solve(var_first, kp_video['var'])

        kp_var = torch.matmul(kp_video['var'], matrix_inverse(kp_video['var'][:, 0:1], eps=0))
        kp_var = torch.matmul(kp_var, kp_source['var'])

        kp_var = make_symetric_matrix(kp_var)
        kp_video['var'] = kp_var

    return kp_video


def normalize_kp2(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False, use_relative_movement=False,
                  use_relative_jacobian=False):
    # normalize_kp2 used in FOMM
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])  # (1, num_kp, 2)
        kp_value_diff *= adapt_movement_scale  # (1, num_kp, 2)
        kp_new['value'] = kp_value_diff + kp_source['value']  # (1, num_kp, 2)

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))  # (1, num_kp, 2, 2)
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new


def split_kp(kp_joined, detach=False):
    # kp_jointed    - mean:    (N, 2, kp, 2)
    #               - var:      (N, 2, kp, 2, 2)
    if detach:
        kp_appearance = {k: v[:, :1].detach() for k, v in kp_joined.items()}
        kp_video = {k: v[:, 1:].detach() for k, v in kp_joined.items()}
    else:
        kp_appearance = {k: v[:, :1] for k, v in kp_joined.items()}
        kp_video = {k: v[:, 1:] for k, v in kp_joined.items()}

    out = dict()
    out['kp_source'] = kp_appearance
    out['kp_driving'] = kp_video

    return out

