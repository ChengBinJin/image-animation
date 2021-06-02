import os
import torch
import numpy as np


class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=5, log_file_name="log.txt"):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    @staticmethod
    def load_cpk(checkpoint_path, generator=None, discriminator=None, kp_detector=None, optimizer_generator=None,
                 optimizer_discriminator=None, optimizer_kp_detector=None):
        checkpoint = torch.load(checkpoint_path)

        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if discriminator is not None:
            try:
                discriminator.load_state_dict(checkpoint['discriminator'])
            except:
                print("No discriminator in the state-dict. Discriminator will be randomly initialized")

        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            except RuntimeError as e:
                print("No discriminator optimizer in that state-dict. Optimizer will be not initialized")

        return checkpoint['epoch']

