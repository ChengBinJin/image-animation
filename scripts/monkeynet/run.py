import sys
import os

import torch.cuda
import yaml

from argparse import ArgumentParser
from datetime import datetime
from shutil import copy

source_file_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
project_dir = os.path.split(source_file_dir)[0]
sys.path.append(project_dir)

from library.modules.keypoint_detector import KPDetector
from library.modules.generator import MotionTransferGenerator
from library.modules.discriminator import Discriminator
from library.utils.files import get_name


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--config", default='config/monkeynet/moving-gif.yaml', help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "transfer", "prediction"])
    parser.add_argument("--log_dir", default="logs/monkeynet", help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        blocks_discriminator = config['model_params']['discriminator_params']['num_blocks']
        assert len(config['train_params']['loss_weights']['reconstruction']) == blocks_discriminator + 1

    if opt.checkpoint is not None:
        log_dir = os.path.join(project_dir, opt.log_dir, get_name(opt.checkpoint))
    else:
        log_dir = os.path.join(project_dir, opt.log_dir,
                               get_name(opt.config) + datetime.now().strftime("_%Y-%m-%d_%H:%M:%S"))

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    generator = MotionTransferGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    discriminator = Discriminator(**config['model_params']['discriminator_params'],
                                  **config['model_params']['common_params'])

    if torch.cuda.is_available():
        kp_detector.to(opt.device_ids[0])
        generator.to(opt.device_ids[0])
        discriminator.to(opt.device_ids[0])

    if opt.verbose:
        print(kp_detector, "\n")
        print(generator, "\n")
        print(discriminator, "\n")

    print("SUCCESS")
