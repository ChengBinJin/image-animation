import sys
import os
import yaml
# import matplotlib
# import imageio
import torch
# import numpy as np
from argparse import ArgumentParser

source_file_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
project_dir = os.path.split(source_file_dir)[0]
sys.path.append(project_dir)

# from transfer import transfer_one
# from logger import Logger
from library.modules.generator import MotionTransferGenerator
from library.modules.keypoint_detector import KPDetector
# from sync_batchnorm import DataParallelWithCallback
# from frames_dataset import read_video
# from augmentation import VideoToTensor


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--config", default='../../config/monkeynet/moving-gif.yaml', help="path to config")
    # parser.add_argument("--checkpoint", required=True, help="path to checkpoint")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint")
    parser.add_argument("--source_image", default="sup-mat/source.png", help="path to source image")
    parser.add_argument("--driving_video", default="sup-mat/driving.png", help="path to driving video")
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

    kp_detector.eval()

    with torch.no_grad():
        source_image_tensor = torch.rand((1, 3, 3, 256, 256), dtype=torch.float32).cuda()
        driving_image_tensor = torch.rand((1, 3, 3, 256, 256), dtype=torch.float32).cuda()
        kp_source_tensor = kp_detector(source_image_tensor)
        kp_driving_tensor = kp_detector(driving_image_tensor)

        for key, value in kp_source_tensor.items():
            print(f'key: {key}, value shape: {value.shape}')

        generator.kp_embedding_module(source_image_tensor, kp_source_tensor, kp_driving_tensor)

    print(f"Hello world!")

