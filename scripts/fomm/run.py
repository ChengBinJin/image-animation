import os
import sys

import torch.cuda
import yaml
from argparse import ArgumentParser
from datetime import datetime
from shutil import copy

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_dir)

from library.utils.files import get_name
from library.modules.keypoint_detector import KPDetector2
from library.modules.generator import OcclusionAwareGenerator
from library.modules.discriminator import MultiScaleDiscriminator
from library.pipeline.animate import animate2
from library.pipeline.train import train2
from library.pipeline.reconstruction import reconstruction2
from library.dataset.frames_dataset import FramesDataset2


if __name__ == "__main__":
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recomended version is Python 3.y")

    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--log_dir", default="logs/fomm", help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if opt.checkpoint is not None:
        log_dir = os.path.dirname(opt.checkpoint)
    else:
        log_dir = os.path.join(project_dir, opt.log_dir,
                               get_name(opt.config) + datetime.now().strftime("_%Y-%m-%d_%H:%M:%s"))

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    kp_detector = KPDetector2(**config['model_params']['kp_detector_params'],
                              **config['model_params']['common_params'])
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                            **config['model_params']['common_params'])

    if torch.cuda.is_available():
        kp_detector.to(opt.device_ids[0])
        generator.to(opt.device_ids[0])
        discriminator.to(opt.device_ids[0])

    if opt.verbose:
        print(kp_detector, "\n")
        print(generator, "\n")
        print(discriminator, "\n")

    dataset = FramesDataset2(is_train=(opt.mode == 'train'), **config['dataset_params'])

    if opt.mode == "train":
        print("Training...")
        train2(config, kp_detector, generator, discriminator, opt.checkpoint, log_dir, dataset, opt.device_ids)
    elif opt.mode == "reconstruction":
        print("Reconstruction...")
        reconstruction2(config, kp_detector, generator, opt.checkpoint, log_dir, dataset)
    elif opt.mode == "animate":
        print("Animate...")
        animate2(config, kp_detector, generator, opt.checkpoint, log_dir, dataset)
