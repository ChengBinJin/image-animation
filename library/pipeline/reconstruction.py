import os
from torch.utils.data import DataLoader

from library.utils.logger.logger import Logger


def reconstruction(config, generator, kp_detector, checkpoint, log_dir, dataset):
    png_dir = os.path.join(log_dir, 'reconstruction/png')
    log_dir = os.path.join(log_dir, 'reconstruction')

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError(" [!] Checkpoint should be specified for mode='test'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)