import numpy as np


class VideoToTensor(object):
    """ Convert video array to Tensor. """

    def __call__(self, driving):
        driving = np.array(driving, dtype='float32')
        out = {'video': driving.transpose((3, 0, 1, 2))}  # (N, C, H, W)
        return out
