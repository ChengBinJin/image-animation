import torch
import numpy as np


def matrix_inverse(batch_of_matrix, eps=0):
    if eps != 0:
        init_shape = batch_of_matrix.shape
        a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
        b = batch_of_matrix[..., 0, 1].unsqueeze(-1)
        c = batch_of_matrix[..., 1, 0].unsqueeze(-1)
        d = batch_of_matrix[..., 1, 1].unsqueeze(-1)

        det = a * d - b * c
        out = torch.cat([d, -b, -c, a], dim=-1)
        eps = torch.tensor(eps).type(out.type())
        out /= det.max(eps)

        return out.view(init_shape)
    else:
        b_mat = batch_of_matrix
        eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
        b_inv, _ = torch.solve(eye, b_mat)
        return b_inv


def smallest_singular(batch_of_matrix):
    a = (batch_of_matrix[..., 0, 0]).unsqueeze(-1)
    b = (batch_of_matrix[..., 0, 1]).unsqueeze(-1)
    c = (batch_of_matrix[..., 1, 0]).unsqueeze(-1)
    d = (batch_of_matrix[..., 1, 1]).unsqueeze(-1)

    s1 = a ** 2 + b ** 2 + c ** 2 + d ** 2
    s2 = (a ** 2 + b ** 2 - c ** 2 - d ** 2) ** 2
    s2 = torch.sqrt(s2 + 4 * (a * c + b * d) ** 2)
    norm = torch.sqrt((s1 - s2) / 2)

    return norm


def make_symetric_matrix(torch_matrix):
    a = torch_matrix.cpu().numpy()
    c = (a + np.transpose(a, (0, 1, 2, 4, 3))) / 2
    d, u = np.linalg.eig(c)
    d[d <= 0] = 1e-6
    d_matrix = np.zeros_like(a)
    d_matrix[..., 0, 0] = d[..., 0]
    d_matrix[..., 1, 1] = d[..., 1]
    res = np.matmul(np.matmul(u, d_matrix), np.transpose(u, (0, 1, 2, 4, 3)))
    res = torch.from_numpy(res).type(torch_matrix.type())

    return res

