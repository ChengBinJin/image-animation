import random

import numpy as np


class VideoToTensor(object):
    """ Convert video array to Tensor. """

    def __call__(self, driving):
        driving = np.array(driving, dtype='float32')
        out = {'video': driving.transpose((3, 0, 1, 2))}  # (C, N, H, W)
        return out


class SelectRandomFrames(object):
    def __init__(self, consequent=False, number_of_frames=2):
        self.consequent = consequent
        self.num_frames_to_select = number_of_frames

    def __call__(self, clip):
        """
        Args:
            img (PIL.Image or numpy.ndarray): List of images for selection in format (h, w, c) in numpy.ndarray
        Returns:
            PIL.Image or numpy.ndarray: List of number_of_frames images
        """
        frame_count = len(clip)

        if self.consequent:
            first_frame = np.random.choice(max(1, frame_count - self.num_frames_to_select + 1), size=1)[0]
            selected = clip[first_frame:first_frame + self.num_frames_to_select]
        else:
            selected_index = np.sort(np.random.choice(range(frame_count), replace=True, size=self.num_frames_to_select))
            selected = clip[selected_index]

        return selected


class RandomFlip(object):
    def __init__(self, time_flip=False, horizontal_flip=False):
        self.time_flip = time_flip
        self.horizontal_flip = horizontal_flip

    def __call__(self, clip):
        if random.random() > 0.5 and self.time_flip:
            clip = clip[::-1]
        if random.random() > 0.5 and self.horizontal_flip:
            clip = [np.fliplr(img) for img in clip]

        return clip


class AllAumgnetationTransform:
    def __init__(self, resize_param=None, rotation_param=None, flip_param=None, crop_param=None, jitter_param=None):
        self.transforms = []
        self.select = SelectRandomFrames()
        self.transforms.append(self.select)

        if flip_param is not None:
            self.transforms.append(RandomFlip(**flip_param))

        if rotation_param is not None:
            self.transforms.append(RnadomRotation(**rotation_param))

        if resize_param is not None:
            self.transforms.append(RandomResize(**resize_param))

        if crop_param is not None:
            self.transforms.append(RandomCrop(**crop_param))

        if jitter_param is not None:
            self.transforms.append(ColorJitter(**jitter_param))

        self.transforms.append(SplitSourceDriving())

    def __call__(self, clip):
        for transform in self.transforms:
            clip = transform(clip)
        return clip
