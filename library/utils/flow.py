import torch


def make_coordinate_grid(spatial_size, dtype):
    """
    Create a meshgrid [-1, 1] x [-1, 1] of given spatial_size
    """
    h, w = spatial_size
    x = torch.arange(w).type(dtype)
    y = torch.arange(h).type(dtype)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(dim=2), yy.unsqueeze_(2)], dim=2)  # H, W, 2

    return meshed
