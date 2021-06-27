import random
import numbers
import PIL
import numpy as np

from skimage.transform import rotate, resize
from skimage.util import pad


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


class RandomRotation(object):
    """
    Rotate entire clip randomly by a random angle within given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from if degrees is a number instead of sequence like
    (min, max), the range of degrees, will be (-degree, +degree)
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number, must be positive')
            degrees = (-1*degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence, it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """

        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [rotate(image=img, angle=angle, preserve_range=True) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError(f'Expected numpy.ndarray or PIL.Image, but got list of {type(clip[0])}')

        return rotated


class RandomResize(object):
    """
    Resizes a list of (H x W x C) numpy.ndarray to the final size. The larger the original image is, the more times it
    takes to interpolate
    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear', the default is the nearest
    size (tuple): (width, height)
    """

    def __init__(self, ratio=(3./4., 4./3.), interpolation='nearest'):
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, clip):
        scalling_factor = random.uniform(self.ratio[0], self.ratio[1])

        if isinstance(clip[0], np.ndarray):
            img_h, img_w, _ = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            img_w, img_h = clip[0].size
        else:
            raise TypeError(f'Expected numpy.ndarray or PIL.Image, but got list of {type(clip[0])}')

        new_w = int(img_w * scalling_factor)
        new_h = int(img_h * scalling_factor)
        new_size = (new_w, new_h)
        resized = resize_clip(clip, new_size, interpolation=self.interpolation)

        return resized


class RandomCrop(object):
    """
    Extract random crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)
        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """

        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            img_h, img_w, img_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            img_w, img_h = clip[0].size
        else:
            raise TypeError(f'Expected numpy.ndarray or PIL.Image, but got list of {type(clip[0])}')

        clip = pad_clip(clip, h, w)
        img_h, img_w = clip.shape[1:3]
        ##############################################################
        # Need to check
        x1 = 0 if h == img_h else random.randint(0, img_w - w)
        y1 = 0 if w == img_w else random.randint(0, img_h - h)
        cropped = crop_clip(clip, y1, x1, h, w)
        ##############################################################

        return cropped


class AllAumgnetationTransform:
    def __init__(self, resize_param=None, rotation_param=None, flip_param=None, crop_param=None, jitter_param=None):
        self.transforms = []
        self.select = SelectRandomFrames()
        self.transforms.append(self.select)

        if flip_param is not None:
            self.transforms.append(RandomFlip(**flip_param))

        if rotation_param is not None:
            self.transforms.append(RandomRotation(**rotation_param))

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


def pad_clip(clip, h, w):
    img_h, img_w = clip[0].shape[:2]
    pad_h = (0, 0) if h < img_h else ((h - img_h) // 2, (h - img_h + 1) // 2)
    pad_w = (0, 0) if w < img_w else ((w - img_w) // 2, (w - img_w + 1) // 2)
    clip = pad(clip, ((0, 0), pad_h, pad_w, (0, 0)), mode='edge')

    return clip


def get_resize_sizes(img_h, img_w, size):
    if img_w < img_h:
        ow = size
        oh = int(size * img_h / img_w)
    else:
        oh = size
        ow = int(size * img_w / img_h)

    return oh, ow


def resize_clip(clip, size, interpolation='bilinear'):
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            img_h, img_w, img_c = clip[0].shape

            # Min spatial dim already matches minimal size
            if min(img_h, img_w) == size:
                return clip

            new_h, new_w = get_resize_sizes(img_h, img_w, size)
            size = (new_w, new_h)
        else:
            size = (size[1], size[0])

        order = 1 if interpolation == 'bilinear' else 0
        scaled = [resize(img, output_shape=size, order=order, preserve_range=True, mode='constant', anti_aliasing=True)
                  for img in clip]
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            img_w, img_h = clip[0].size

            # Min spatial dim already matches minimal size
            if min(img_h, img_w) == size:
                return clip

            new_h, new_w = get_resize_sizes(img_h, img_w, size)
            size = (new_w, new_h)
        else:
            size = (size[1], size[0])

        pil_inter = PIL.Image.NEAREST if interpolation == 'bilinear' else PIL.Image.BILINEAR
        scaled = [img.resize(size, interpolation=pil_inter) for img in clip]
    else:
        raise TypeError(f'Expected numpy.ndarray or PIL.Image, but got list of {type(clip[0])}')

    return scaled
