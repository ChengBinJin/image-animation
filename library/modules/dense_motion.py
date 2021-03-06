import torch
import torch.nn.functional as F
from torch import nn

from library.utils.keypoint import kp2gaussian, kp2gaussian2
from library.utils.flow import make_coordinate_grid
from library.modules.block import SameBlock3d, Hourglass, Hourglass2, AntiAliasInterpolation2d


class DenseMotionNetwork(nn.Module):
    """
    Module that predicitn a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, num_channels, estimate_occlusion_map=False,
                 scale_factor=1, kp_variance=0.01):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass2(block_expansion=block_expansion, in_features=(num_kp + 1) * (num_channels + 1),
                                    max_features=max_features, num_blocks=num_blocks)

        self.mask = nn.Conv2d(self.hourglass.out_filters, num_kp + 1, kernel_size=(7, 7), padding=(3, 3))

        if estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
        else:
            self.occlusion = None

        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian2(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian2(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source

        # adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type())
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)
        return heatmap

    def create_sparse_motions(self, source_image, kp_driving, kp_source):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape  # (N, 3, h, w)
        identity_grid = make_coordinate_grid((h, w), dtype=kp_source['value'].type())  # (h, w, 2)
        identity_grid = identity_grid.view(1, 1, h, w, 2)  # (1, 1, h, w, 2)
        coordinate_grid = identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 2)  # (1, num_kp, h, w, 2)
        if 'jacobian' in kp_driving:
            jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian']))
            jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
            jacobian = jacobian.repeat(1, 1, h, w, 1, 1)
            coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)

        driving_to_source = coordinate_grid + kp_source['value'].view(bs, self.num_kp, 1, 1, 2)

        # adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
        return sparse_motions

    def create_deformed_source_image(self, source_image, sparse_motions):
        """
        Eq 7. in the paper hat{T}_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape  # (N, 3, h, w)
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_kp + 1), -1, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp + 1), h, w, -1))
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions, align_corners=True)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1, -1, h, w))
        return sparse_deformed

    def forward(self, source_image, kp_driving, kp_source):
        if self.scale_factor != 1:
            source_image = self.down(source_image)
        bs, _, h, w = source_image.shape
        # source_image: (N, 3, H/4, W/4)
        # kp_source:    - value:    (1, 10, 2)
        #               - jacobian: (1, 10, 2, 2)
        # kp_driving:   - value:    (1, 10, 2)
        #               - jacobian: (1, 10, 2, 2)

        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)
        sparse_motion = self.create_sparse_motions(source_image, kp_driving, kp_source)
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)
        out_dict['sparse_deformed'] = deformed_source

        in_feature = torch.cat([heatmap_representation, deformed_source], dim=2)
        in_feature = in_feature.view(bs, -1, h, w)

        prediction = self.hourglass(in_feature)  # (N, 64+44, h, w)

        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        out_dict['deformation'] = deformation

        # Sec. 3.2 in the paper
        if self.occlusion:
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map

        return out_dict


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
        # heatmap   (N, 1, num_kp, H, W)
        if self.norm_const == "sum":
            heatmap_shape = heatmap.shape
            heatmap = heatmap.view(heatmap_shape[0], heatmap_shape[1], heatmap_shape[2], -1)
            heatmap = heatmap / heatmap.sum(dim=3, keepdim=True)
            out = heatmap.view(*heatmap_shape)
        else:
            out = heatmap / self.norm_const     # (N, 1, num_kp, H, W)

        return out

    def forward(self, source_image, kp_driving, kp_source):
        # source_image:         (N, 3, 1, H, W)
        # kp_driving    - mean: (N, 1, num_kp, 2)
        #               - var:  (N, 1, num_kp, 2, 2)
        # kp_source     - mean: (N, 1, num_kp, 2)
        #               - var:  (N, 1, num_kp, 2, 2)
        if self.scale_factor != 1:
            source_image = F.interpolate(
                source_image, scale_factor=(1, self.scale_factor, self.scale_factor), recompute_scale_factor=True)

        spatial_size = source_image.shape[3:]       # (h, w)

        bs, _, _, h, w = source_image.shape         # (N, 3, 1, h, w)
        _, d, num_kp, _ = kp_driving['mean'].shape  # (N, 1, num_kp, 2)

        inputs = []
        if self.use_heatmap:
            heatmap = self.normalize_heatmap(
                kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance))
            # heatmap: (N, 1, num_kp, H, W)

            if self.heatmap_type == 'difference':
                heatmap_appearance = self.normalize_heatmap(
                    kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance))
                # heatmap_appearance:   (N, 1, num_kp, H, W)
                heatmap = heatmap - heatmap_appearance  # (N, 1, num_kp, H, W)

            if self.add_bg_feature_map:
                zeros = torch.zeros(bs, d, 1, h, w).type(heatmap.type())  # (N, 1, 1, H, W)
                heatmap = torch.cat([zeros, heatmap], dim=2)  # (N, 1, num_kp+1, H, W)

            heatmap = heatmap.unsqueeze(3)  # (N, 1, num_kp+1, 1, H, W)
            inputs.append(heatmap)

        num_kp += self.add_bg_feature_map   # 11
        kp_video_diff = None
        if self.use_difference or self.use_deformed_source_image:
            kp_video_diff = kp_source['mean'] - kp_driving['mean']              # (N, 1, num_kp, 2)

            if self.add_bg_feature_map:
                zeros = torch.zeros(bs, d, 1, 2).type(kp_video_diff.type())     # (N, 1, 1, 2)
                kp_video_diff = torch.cat([zeros, kp_video_diff], dim=2)       # (N, 1, num_kp+1, 2)
            kp_video_diff = kp_video_diff.view((bs, d, num_kp, 2, 1, 1)).repeat(1, 1, 1, 1, h, w)
            # kp_video_diff:     (N, 1, num_kp+1, 2, H, W)

        if self.use_difference:
            inputs.append(kp_video_diff)

        if self.use_deformed_source_image:
            appearance_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, d, num_kp, 1, 1, 1, 1)
            # (N, 1, num_kp+1, 3, 1, H, W)
            appearance_repeat = appearance_repeat.view(bs * d * num_kp, -1, h, w)  # (N*1*(kp+1), 3*1, H, W)

            deformation_approx = kp_video_diff.view((bs * d * num_kp, -1, h, w)).permute(0, 2, 3, 1)
            # (N * 1 * (num_kp+1), H, W, 2)
            coordinate_grid = make_coordinate_grid((h, w), dtype=deformation_approx.type())  # (H, W, 2)
            coordinate_grid = coordinate_grid.view(1, h, w, 2)  # (1, H, W, 2)
            deformation_approx = coordinate_grid + deformation_approx  # (N * 1 * (kp+1), H, W, 2)

            appearance_approx_deform = F.grid_sample(appearance_repeat, deformation_approx, align_corners=True)
            # appearance_approx_deform: (num_kp+1, 3, H, W)
            appearance_approx_deform = appearance_approx_deform.view((bs, d, num_kp, -1, h, w))
            # (N, 1, (num_kp+1), 3*1, H, W)
            inputs.append(appearance_approx_deform)

        movement_encoding = torch.cat(inputs, dim=3)  # (N, 1, num_kp+1, 1+3, H, W)
        movement_encoding = movement_encoding.view(bs, d, -1, h, w)     # (N, 1, 11*(1+2+3), H, W)
        movement_encoding = movement_encoding.permute(0, 2, 1, 3, 4)    # (N, 11*(1+2+3), 1, H, W)

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
                SameBlock3d(in_features=self.mask_embedding.out_channels, out_features=self.mask_embedding.out_channels,
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
        # source_image: (N, 3, 1, H, W)
        # kp_driving    - mean: (N, 1, num_kp, 2)
        #               - var:  (N, 1, num_kp, 2, 2)
        # kp_source     - mean: (N, 1, num_kp, 2)
        #               - var:  (N, 1, num_kp, 2, 2)
        if self.scale_factor != 1:
            source_image = F.interpolate(
                source_image, scale_factor=(1, self.scale_factor, self.scale_factor), recompute_scale_factor=True)

        prediction = self.mask_embedding(source_image, kp_driving, kp_source)
        # prediction:   (N, (num_kp+1)*(1+2+3), 1, H, W)
        for block in self.group_blocks:
            prediction = block(prediction)
            # prediction = F.leaky_relu(prediction, 0.2)  # block already includes relu
        # prediction:   (N, (num_kp+1)*(1+3), 1, H, W)
        prediction = self.hourglass(prediction)  # (N, (num_kp+1)+2, 1, H, W)

        bs, _, d, h, w = prediction.shape  # bs=N, d=1
        if self.use_mask:
            mask = prediction[:, :(self.num_kp + 1)]  # (N, num_kp+1, 1, H, W)
            mask = F.softmax(mask, dim=1)   # (N, num_kp+1, 1, H, W)
            mask = mask.unsqueeze(2)    # (N, num_kp, 1, 1, H, W)
            difference_embedding = self.difference_embedding(source_image, kp_driving, kp_source)  # (N, (num_kp+1)*2, 1, H, W)
            difference_embedding = difference_embedding.view(bs, self.num_kp + 1, 2, d, h, w)   # (N, num_kp+1, 2, 1, H, W)
            deformations_relative = (difference_embedding * mask).sum(dim=1)  # (N, 2, 1, H, W)
        else:
            deformations_relative = 0

        if self.use_correction:
            correction = prediction[:, -2:]  # (N, 2, 1, H, W)
        else:
            correction = 0

        deformations_relative = deformations_relative + correction  # (N, 2, 1, H, W)
        deformations_relative = deformations_relative.permute(0, 2, 3, 4, 1)  # (N, 1, H, W, 2)

        coordinate_grid = make_coordinate_grid((h, w), dtype=deformations_relative.type())  # (H, W, 2)
        coordinate_grid = coordinate_grid.view(1, 1, h, w, 2)  # (1, 1, H, W, 2)
        deformation = deformations_relative + coordinate_grid  # (N, 1, H, W, 2)
        z_coordinate = torch.zeros(deformation.shape[:-1] + (1,)).type(deformation.type())  # (N, 1, H, W, 1)
        out = torch.cat([deformation, z_coordinate], dim=-1)  # (N, 1, H, W, 3)

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
