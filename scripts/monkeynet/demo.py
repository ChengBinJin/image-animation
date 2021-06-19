import sys
import os
import yaml
import imageio
import torch
import numpy as np
from argparse import ArgumentParser

source_file_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
project_dir = os.path.split(source_file_dir)[0]
sys.path.append(project_dir)

from library.modules.generator import MotionTransferGenerator
from library.modules.keypoint_detector import KPDetector
from library.utils.logger.logger import Logger
from library.utils.process import transfer_one
from library.dataset.frames_dataset import read_video
from library.dataset.augmentation import VideoToTensor
from library.sync_batchnorm import DataParallelWithCallback
from library.utils.files import get_name


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--config", default='../../config/monkeynet/nemo.yaml', help="path to config")
    # parser.add_argument("--checkpoint", required=True, help="path to checkpoint")
    parser.add_argument("--checkpoint", default='../../checkpoints/monkeynet/nemo-ckp.pth.tar', help="path to checkpoint")
    parser.add_argument("--source_image", default="../../sup-mat/nemo-source.png", help="path to source image")
    parser.add_argument("--driving_video", default="../../sup-mat/nemo-driving.png", help="path to driving video")
    parser.add_argument("--out_file", default="demo.gif", help="path to out file")
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="use cpu")
    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        blocks_discrimiantor = config['model_params']['discriminator_params']['num_blocks']
        assert len(config['train_params']['loss_weights']['reconstruction']) == blocks_discrimiantor + 1

    kp_detector = KPDetector(**config['model_params']['common_params'],
                             **config['model_params']['kp_detector_params'])
    generator = MotionTransferGenerator(**config['model_params']['common_params'],
                                        **config['model_params']['generator_params'])

    if not opt.cpu:
        kp_detector = kp_detector.cuda()
        generator = generator.cuda()

    Logger.load_cpk(opt.checkpoint, generator=generator, kp_detector=kp_detector)

    kp_detector = DataParallelWithCallback(kp_detector)
    generator = DataParallelWithCallback(generator)

    kp_detector.eval()
    generator.eval()
    image_shape = tuple(config['dataset_params']['image_shape'])[:-1]

    with torch.no_grad():
        driving_video = VideoToTensor()(read_video(opt.driving_video, image_shape + (3,)))['video']  # (C, N, H, W)
        source_image = VideoToTensor()(read_video(opt.source_image, image_shape + (3,)))['video'][:, :1]  # (C, 1, H, W)

        driving_video = torch.from_numpy(driving_video).unsqueeze(0)    # (1, C, N, H, W)
        source_image = torch.from_numpy(source_image).unsqueeze(0)      # (1, C, 1, H, W)

        # video_prediciton: (1, C, N, H, W)
        # video_deformed:   (1, C, N, H, W)
        # kp_driving:   - mean: (1, N, num_kp, 2); - var: (1, N, num_kp, 2, 2)
        # kp_source:    - mean: (1, 1, num_kp, 2); - var: (1, 1, num_kp, 2, 2)
        # kp_norm:      - mean: (1, N, num_kp, 2); - var: (1, N, num_kp, 2, 2)
        out = transfer_one(kp_detector, generator, source_image, driving_video, config['transfer_params'])
        out_video_batch = out['video_prediction'].data.cpu().numpy()  # (1, C, N, H, W)
        out_video_batch = np.transpose(out_video_batch, [0, 2, 3, 4, 1])[0]  # (1, N, H, W, C) -> (N, H, W, C)
        imageio.mimsave(
            get_name(opt.config) + '_' + opt.out_file, (255 * out_video_batch).astype(np.uint8))
