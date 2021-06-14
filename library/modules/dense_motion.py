import torch
import torch.nn.functional as F
from torch import nn

from library.utils.keypoint import kp2gaussian
from library.utils.flow import make_coordinate_grid
from library.modules.block import SameBlock3D, Hourglass


class MovementEmbeddingModule(nn.Module):
    """
    Procude a keypoint representation that will be further used by other moudles
    """

    def __init__(self, num_kp, kp_variance, num_channels, use_deformed_source_image=False, use_difference=False,
                 use_heatmap=True, add_bg_feature_map=False, heatmap_type="gaussian", norm_const="sum", scale_factor=1):
        super(MovementEmbeddingModule, self).__init__()

        assert heatmap_type in ['gaussian', 'difference']
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

    def normalize_heatmap(self, heatmap):
        if self.norm_const == "sum":
            heatmap_shape = heatmap.shape   # (1, 3, kp, H/2, W/2)
            heatmap = heatmap.view(heatmap_shape[0], heatmap_shape[1], heatmap_shape[2], -1)    # (1, 3, kp, H/2*W/2)
            heatmap = heatmap / heatmap.sum(dim=3, keepdim=True)
            out = heatmap.view(*heatmap_shape)  # (1, 3, kp, H/2, W/2)
        else:
            out = heatmap / self.norm_const     # (1, 3, kp, H/2, W/2)

        return out

    def forward(self, source_image, kp_driving, kp_source):
        if self.scale_factor != 1:
            source_image = F.interpolate(
                source_image, scale_factor=(1, self.scale_factor, self.scale_factor), recompute_scale_factor=True)

        spatial_size = source_image.shape[3:]       # (H/2, W/2)

        bs, _, _, h, w = source_image.shape         # (N, 3, 3, H/2, W/2)
        _, d, num_kp, _ = kp_driving['mean'].shape  # (N, 3, kp, 2)

        inputs = []
        if self.use_heatmap:
            heatmap = self.normalize_heatmap(
                kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance))  # (1, 3, kp, H/2, W/2)

            if self.heatmap_type == 'difference':
                heatmap_appearance = self.normalize_heatmap(
                    kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance))
                heatmap = heatmap - heatmap_appearance

            if self.add_bg_feature_map:
                zeros = torch.zeros(bs, d, 1, h, w).type(heatmap.type())  # (N, 3, 1, H/2, W/2)
                heatmap = torch.cat([zeros, heatmap], dim=2)  # (N, 3, kp+1, H/2, W/2)

            heatmap = heatmap.unsqueeze(3)  # (N, 3, kp+1, 1, H/2, W/2)
            inputs.append(heatmap)

        num_kp += self.add_bg_feature_map
        kp_video_diff = None
        if self.use_difference or self.use_deformed_source_image:
            kp_video_diff = kp_source['mean'] - kp_driving['mean']              # (N, 3, kp, 2)

            if self.add_bg_feature_map:
                zeros = torch.zeros(bs, d, 1, 2).type(kp_video_diff.type())     # (N, 3, 1, 2)
                kp_video_diff = torch.cat([zeros, kp_video_diff], dim=2)       # (N, 3, kp+1, 2)
            kp_video_diff = kp_video_diff.view((bs, d, num_kp, 2, 1, 1)).repeat(1, 1, 1, 1, h, w)
            # (N, 3, kp+1, 2, H/2, W/2)

        if self.use_difference:
            inputs.append(kp_video_diff)

        if self.use_deformed_source_image:
            appearance_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, d, num_kp, 1, 1, 1, 1)
            # (N, 3, kp+1, 3, 3, H/2, W/2)
            appearance_repeat = appearance_repeat.view(bs * d * num_kp, -1, h, w)  # (N*3*(kp+1), 3*3, H/2, W/2)

            deformation_approx = kp_video_diff.view((bs * d * num_kp, -1, h, w)).permute(0, 2, 3, 1)
            # (N * 3 * (kp+1), H/2, W/2, 2)
            coordinate_grid = make_coordinate_grid((h, w), dtype=deformation_approx.type())  # (H/2, W/2, 2)
            coordinate_grid = coordinate_grid.view(1, h, w, 2)  # (1, H/2, W/2, 2)
            deformation_approx = coordinate_grid + deformation_approx  # (N * 3 * (kp+1), H/2, W/2, 2)

            appearance_approx_deform = F.grid_sample(appearance_repeat, deformation_approx, align_corners=True)
            appearance_approx_deform = appearance_approx_deform.view((bs, d, num_kp, -1, h, w))
            # (N, 3, (kp+1), 3*3, H/2, W/2)
            inputs.append(appearance_approx_deform)

        movement_encoding = torch.cat(inputs, dim=3)  # (N, 3, 11, 1+2+9, H/2, W/2)
        movement_encoding = movement_encoding.view(bs, d, -1, h, w)     # (N, 3, 11*(1+2+9), H/2, W/2)
        movement_encoding = movement_encoding.permute(0, 2, 1, 3, 4)    # (N, 11*(1+2+9), 3, H/2, W/2)

        return movement_encoding


class DenseMotionModule(nn.Module):
    """
    Module that predicting a dense optical flow only from the displacement of a keypoints and the appearance of the
    first frame
    """

    def __init__(self, block_expansion, num_blocks, max_features, mask_embedding_params, num_kp, num_channels,
                 kp_variance, use_correction, use_mask, bg_init=2, num_group_blocks=0, scale_factor=1):
        super(DenseMotionModule, self).__init__()

        self.mask_embedding = MovementEmbeddingModule(
            num_kp=num_kp, kp_variance=kp_variance, num_channels=num_channels, add_bg_feature_map=True,
            **mask_embedding_params)
        self.difference_embedding = MovementEmbeddingModule(
            num_kp=num_kp, kp_variance=kp_variance, num_channels=num_channels, add_bg_feature_map=True,
            use_difference=True, use_heatmap=False, use_deformed_source_image=False)

        group_blocks = []
        for i in range(num_group_blocks):
            # (self, in_features, out_features, groups=None, kernel_size=3, padding=1)
            group_blocks.append(
                SameBlock3D(in_features=self.mask_embedding.out_channels, out_features=self.mask_embedding.out_channels,
                            groups=num_kp+1, kernel_size=(1, 1, 1), padding=(0, 0, 0)))
        self.group_blocks = nn.ModuleList(group_blocks)

        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=self.mask_embedding.out_channels,
                                   out_features=(num_kp + 1) * use_mask + 2 * use_correction, max_features=max_features,
                                   num_blocks=num_blocks)
        self.hourglass.decoder.conv.weight.data.zero_()  # the last conv of decoder in hourglass

        bias_init = ([bg_init] + [0] * num_kp) * use_mask + [0, 0] * use_correction
        self.hourglass.decoder.conv.bias.data.copy_(torch.tensor(bias_init, dtype=torch.float))
        # the last conv of decoder in hourglass

        self.num_kp = num_kp
        self.use_correction = use_correction
        self.use_mask = use_mask
        self.scale_factor = scale_factor

    def forward(self, source_image, kp_driving, kp_source):
        if self.scale_factor != 1:
            source_image = F.interpolate(source_image, scale_factor=(1, self.scale_factor, self.scale_factor), recompute_scale_factor=True)

        prediction = self.mask_embedding(source_image, kp_driving, kp_source)  # (N, 11*(1+2+9), 3, H/2, W/2)
        for block in self.group_blocks:
            prediction = block(prediction)
            prediction = F.leaky_relu(prediction, 0.2)
        prediction = self.hourglass(prediction)

        bs, _, d, h, w = prediction.shape
        if self.use_mask:
            mask = prediction[:, :(self.num_kp + 1)]
            mask = F.softmax(mask, dim=1)
            mask = mask.unsqueeze(2)
            difference_embedding = self.difference_embedding(source_image, kp_driving, kp_source)
            difference_embedding = difference_embedding.view(bs, self.num_kp + 1, 2, d, h, w)
            deformations_relative = (difference_embedding * mask).sum(dim=1)
        else:
            deformations_relative = 0

        if self.use_correction:
            correction = prediction[:, -2:]
        else:
            correction = 0

        deformations_relative = deformations_relative + correction
        deformations_relative = deformations_relative.permute(0, 2, 3, 4, 1)

        coordinate_grid = make_coordinate_grid((h, w), dtype=deformations_relative.type())
        coordinate_grid = coordinate_grid.view(1, 1, h, w, 2)
        deformation = deformations_relative + coordinate_grid
        z_coordinate = torch.zeros(deformation.shape[:-1] + (1,)).type(deformation.type())
        out = torch.cat([deformation, z_coordinate], dim=-1)

        return out


class IdentityDeformation(nn.Module):
    @staticmethod
    def forward(appearance_frame, kp_video, _):
        bs, _, _, h, w = appearance_frame.shape  # (N, 3, 3, H, W)
        _, d, num_kp, _ = kp_video['mean'].shape    # (N, 3, kp, 2)
        coordinate_grid = make_coordinate_grid((h, w), dtype=appearance_frame.type())  # (H, W, 2)
        coordinate_grid = coordinate_grid.view(1, 1, h, w, 2).repeat(bs, d, 1, 1, 1)   # (N, 3, H, W, 2)

        z_coordinate = torch.zeros(coordinate_grid.shape[:-1] + (1,)).type(coordinate_grid.type())  # (N, 3, H, W, 1)
        out = torch.cat([coordinate_grid, z_coordinate], dim=-1)  # (N, 3, H, W, 3)

        return out
