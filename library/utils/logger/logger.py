import os
import imageio
import collections
import torch
import numpy as np


class Logger:
    def __init__(self, log_dir, log_file_name="log.txt", log_freq_iter=100, cpk_freq_epoch=100, zfill_num=5,
                 visualizer_params=None):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.log_freq = log_freq_iter
        self.checkpoint_freq = cpk_freq_epoch
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)

        self.epoch = 0
        self.it = 0
        self.names = None
        self.models = None

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        image = self.visualizer.visualize(inp['source'], inp['driving'], out)
        imageio.imsave(os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)

    def save_cpk(self):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk['it'] = self.it
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num))
        torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, generator=None, discriminator=None, kp_detector=None, optimizer_generator=None,
                 optimizer_discriminator=None, optimizer_kp_detector=None):
        checkpoint = torch.load(checkpoint_path)

        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if discriminator is not None:
            discriminator.load_state_dict(checkpoint['discriminator'])

        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_discriminator is not None:
            optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])

        return checkpoint['epoch'], checkpoint['it']

    def __enter__(self):
        return self

    def __exit__(self, exc_tye, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, it, names, values, inp, out):
        self.it = it
        self.names = names
        self.loss_list.append(values)
        if it % self.log_freq == 0:
            self.log_scores(self.names)
            self.visualize_rec(inp, out)

    def log_epoch(self, epoch, models):
        self.epoch = epoch
        self.models = models
        if self.epoch % self.checkpoint_freq == 0:
            self.save_cpk()

