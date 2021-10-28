import collections
import os
import imageio
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage.draw import disk


class Logger2:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=5, log_file_name='log.txt'):
        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer2(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
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
        image = self.visualizer.visualize(inp['driving'], inp['source'], out)
        imageio.imsave(
            os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)

    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, "%s-checkpoint.pth.tar" % str(self.epoch).zfill(self.zfill_num))
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, kp_detector=None, generator=None, discriminator=None, optimizer_kp_detector=None,
                 optimizer_generator=None, optimizer_discriminator=None):
        checkpoint = torch.load(checkpoint_path)
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if discriminator is not None:
            try:
                discriminator.load_state_dict(checkpoint['discrimiantor'])
            except RuntimeError:
                print(" [!] No discriminator in the state-dict. Discriminator will be randomly initialized!")

        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            except RuntimeError:
                print(" [!] No discriminator optimizer in the state-dict. "
                      "Optimizer of the discriminator will be not initialized!")

        return checkpoint['epoch']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models, inp, out):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)
        self.visualize_rec(inp, out)


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
        imgs = self.visualizer.visualize_reconstruction(inp, out)
        imageio.imsave(os.path.join(self.visualizations_dir, f"{str(self.it).zfill(self.zfill_num)}-rec.png"), imgs)

    def save_cpk(self):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk['it'] = self.it
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num))
        torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, kp_detector=None, generator=None, discriminator=None, optimizer_kp_detector=None,
                 optimizer_generator=None, optimizer_discriminator=None):
        checkpoint = torch.load(checkpoint_path)

        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
            print(f" [*] Load kp_detector SUCCESS!")
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
            print(f" [*] Load generator SUCCESS!")
        if discriminator is not None:
            discriminator.load_state_dict(checkpoint['discriminator'])
            print(f" [*] Load discriminator SUCCESS!")

        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])
            print(f" [*] Load optimizer_kp_detector SUCCESS!")
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
            print(f" [*] Load optimizer_generator_detector SUCCESS!")
        if optimizer_discriminator is not None:
            optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            print(f" [*] Load optimizer_discriminator_detector SUCCESS!")

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


class Visualizer2(object):
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]

        for kp_ind, kp in enumerate(kp_array):
            rr, cc = disk((kp[1], kp[0]), self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, driving, source, out):
        images = []

        # Source image with keypoints
        source = source.data.cpu()
        kp_source = out['kp_source']['value'].data.cpu().numpy()
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source, kp_source))

        # Equivariance visualization
        if 'transformed_frame' in out:
            transformed = out['transformed_frame'].data.cpu().numpy()
            transformed = np.transpose(transformed, [0, 2, 3, 1])
            transformed_kp = out['transformed_kp']['value'].data.cpu().numpy()
            images.append((transformed, transformed_kp))

        # Driving image with keypoints
        kp_driving = out['kp_driving']['value'].data.cpu().numpy()
        driving = driving.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))

        # Deformed image
        if 'deformed' in out:
            deformed = out['deformed'].data.cpu().numpy()
            deformed = np.transpose(deformed, [0, 2, 3, 1])
            images.append(deformed)

        # Result with and without keypoints
        prediction = out['prediction'].data.cpu().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        if 'kp_norm' in out:
            kp_norm = out['kp_norm']['value'].data.cpu().numpy()
            images.append((prediction, kp_norm))
        images.append(prediction)

        # Occlusion map
        if 'occlusion_map' in out:
            occlusion_map = out['occlusion_map'].data.cpu().repeat(1, 3, 1, 1)
            occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
            occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
            images.append(occlusion_map)

        # Deformed images according to each individual transform
        if 'sparse_deformed' in out:
            full_mask = []
            for i in range(out['sparse_deformed'].shape[1]):
                image = out['sparse_deformed'][:, i].data.cpu()
                image = F.interpolate(image, size=source.shape[1:3])
                mask = out['mask'][:, i:(i+1)].data.cpu().repeat(1, 3, 1, 1)
                mask = F.interpolate(mask, size=source.shape[1:3])
                image = np.transpose(image.numpy(), (0, 2, 3, 1))
                mask = np.transpose(mask.numpy(), (0, 2, 3, 1))

                if i != 0:
                    color = np.array(self.colormap((i - 1) / (out['sparse_deformed'].shape[1] - 1)))[:3]
                else:
                    color = np.array((0, 0, 0))
                color = color.reshape((1, 1, 1, 3))

                images.append(image)
                if i != 0:
                    images.append(mask * color)
                else:
                    images.append(mask)

                full_mask.append(mask * color)
            images.append(sum(full_mask))

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image


class Visualizer(object):
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_video_with_kp(self, video, kp_array):
        video_array = np.copy(video)
        spatial_size = np.array(video_array.shape[2:0:-1])[np.newaxis, np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[1]

        for i in range(len(video_array)):
            for kp_ind, kp in enumerate(kp_array[i]):
                rr, cc = disk((kp[1], kp[0]), self.kp_size, shape=video_array.shape[1:3])
                video_array[i][rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return video_array

    def create_video_column_with_kp(self, video, kp):
        video_array = np.array([self.draw_video_with_kp(v, k) for v, k in zip(video, kp)])
        return self.create_video_column(video_array)

    def create_video_column(self, videos):
        if self.draw_border:
            videos = np.copy(videos)
            videos[:, :, [0, -1]] = (1, 1, 1)
            videos[:, :, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(videos), axis=1)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_video_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_video_column(arg))
        return np.concatenate(out, axis=2)

    def visualize_animate(self, source_image, driving_video, out):
        out_video_batch = out['video_prediction'].data.cpu().numpy()  # (1, 3, n, H, W)
        appearance_deformed_batch = out['video_deformed'].data.cpu().numpy()  # (1, 3, n, H, W)
        motion_video_batch = driving_video.data.cpu().numpy()  # (1, 3, n, H, W)
        appearance_video_batch = source_image[:, :, 0:1].data.cpu().repeat(
            1, 1, out_video_batch.shape[2], 1, 1).numpy()  # (1, 3, n, H, W)
        # video_first_frame = driving_video[:, :, 0:1].data.cpu().repeat(
        #     1, 1, out_video_batch.shape[2], 1, 1).numpy()  # (1, 3, n, H, W)

        kp_video = out['kp_driving']['mean'].data.cpu().numpy()  # (1, n, num_kp, 2)
        kp_appearance = out['kp_source']['mean'].data.cpu().repeat(
            1, out_video_batch.shape[2], 1, 1).numpy()  # (1, n, num_kp, 2)
        kp_norm = out['kp_norm']['mean'].data.cpu().numpy()  # (1, n, num_kp, 2)
        # kp_video_first = out['kp_driving']['mean'][:, :1].data.cpu().repeat(
        #     1, out_video_batch.shape[2], 1, 1).numpy()  # (1, 12, num_kp, 2)

        # video_first_frame = np.transpose(video_first_frame, [0, 2, 3, 4, 1])  # (1, n, H, W, 3)
        out_video_batch = np.transpose(out_video_batch, [0, 2, 3, 4, 1])    # (1, n, H, W, 3)
        motion_video_batch = np.transpose(motion_video_batch, [0, 2, 3, 4, 1])  # (1, n, H, W, 3)
        appearance_video_batch = np.transpose(appearance_video_batch, [0, 2, 3, 4, 1])  # (1, num_kp, H, W, 3)
        appearance_deformed_batch = np.transpose(appearance_deformed_batch, [0, 2, 3, 4, 1])  # (1, num_kp, H, W, 3)

        image = self.create_image_grid((appearance_video_batch, kp_appearance),
                                       (motion_video_batch, kp_video),
                                       appearance_deformed_batch,
                                       (out_video_batch, kp_norm),
                                       out_video_batch)
        image = (255 * image).astype(np.uint8)

        return image

    def visualize_reconstruction(self, inp, out):
        out_video_batch = out['video_prediction'].data.cpu().numpy()
        if 'driving' in inp:
            gt_video_batch = inp['driving'].data.cpu().numpy()
        else:
            gt_video_batch = inp['video'].data.cpu().numpy()

        appearance_deformed_batch = out['video_deformed'].data.cpu().numpy()
        appearance_video_batch = inp['source'].data.cpu().repeat(
            1, 1, out_video_batch.shape[2], 1, 1).numpy()

        kp_video = out['kp_driving']['mean'].data.cpu().numpy()
        kp_appearance = out['kp_source']['mean'].data.cpu().repeat(
            1, out_video_batch.shape[2], 1, 1).numpy()

        out_video_batch = np.transpose(out_video_batch, [0, 2, 3, 4, 1])
        gt_video_batch = np.transpose(gt_video_batch, [0, 2, 3, 4, 1])
        appearance_video_batch = np.transpose(appearance_video_batch, [0, 2, 3, 4, 1])
        appearance_deformed_batch = np.transpose(appearance_deformed_batch, [0, 2, 3, 4, 1])

        image = self.create_image_grid((appearance_video_batch, kp_appearance),
                                       (gt_video_batch, kp_video),
                                       appearance_deformed_batch, out_video_batch, gt_video_batch)
        image = (255 * np.squeeze(image)).astype(np.uint8)

        return image

