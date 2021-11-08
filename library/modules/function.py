import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad
from torchvision import models
from library.modules.block import AntiAliasInterpolation2d
from library.utils.flow import make_coordinate_grid


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, model_path=None, requires_grad=False):
        super(Vgg19, self).__init__()

        if model_path is None:
            vgg_pretrained_features = models.vgg19(pretrained=True).features
        else:
            vgg_model = models.vgg19(pretrained=False)  # init model:
            vgg_model.load_state_dict(torch.load(model_path))
            vgg_pretrained_features = vgg_model.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = (x - self.mean) / self.std
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self). __init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = torch.nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform(object):
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sgima_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(
                mean=0, std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps']**2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        out = F.grid_sample(frame, grid, padding_mode="reflection")
        return out

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian
