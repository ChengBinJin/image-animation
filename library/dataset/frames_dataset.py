import os
import numpy as np

from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
from torch.utils.data import Dataset

from library.utils.files import all_files_under
from library.dataset.augmentation import AllAugmentationTransform, VideoToTensor


class FramesDataset(Dataset):
    """
    Dataset of videos, videos can be represented as an image of concatenated frames, or in '.mp4', '.gif' format
    """

    def __init__(self, root_dir, augmentation_params, img_shape=(256, 256, 3), is_train=True, random_seed=0,
                 pairs_list=None, transform=None):
        self.root_dir = root_dir
        self.img_shape = tuple(img_shape)
        self.is_train = is_train
        self.pairs_list = pairs_list

        if os.path.exists(os.path.join(self.root_dir, 'train')):
            assert os.path.exists(os.path.join(self.root_dir, 'test'))
            print(" [*] Use predefined train-test split!")

            train_img_names = all_files_under(os.path.join(self.root_dir, 'train'), extension=['.jpg', '.png', '.jpeg'],
                                              append_path=False, sort=False)
            test_img_names = all_files_under(os.path.join(self.root_dir, 'test'), extension=['.jpg', '.png', '.jpeg'],
                                             append_path=False, sort=True)
            self.root_dir = os.path.join(self.root_dir, 'train' if self.is_train else 'test')
        else:
            print(" [*] Use random train-test split!")

            img_names = all_files_under(
                self.root_dir, extension=['.jpg', '.png', '.jpeg'], append_path=False, sort=False)
            train_img_names, test_img_names = train_test_split(img_names, random_state=random_seed, test_size=0.2)
            test_img_names = sorted(test_img_names)

        self.train_img_names = train_img_names
        self.test_img_names = test_img_names

        if transform is None:
            if self.is_train:
                self.transform = AllAugmentationTransform(**augmentation_params)
            else:
                self.transform = VideoToTensor()
        else:
            self.transform = transform

    def __len__(self):
        num_train_imgs = len(self.train_img_names)
        num_test_imgs = len(self.test_img_names)
        return tuple([num_train_imgs, num_test_imgs])

    def __getitem__(self, idx):
        if self.is_train:
            img_name = os.path.join(self.root_dir, self.train_img_names[idx])
        else:
            img_name = os.path.join(self.root_dir, self.test_img_names[idx])

        video_array = read_video(img_name, img_shape=self.img_shape)
        out = self.transform(video_array)
        out['name'] = os.path.basename(img_name)

        return out


def read_video(name, img_shape):
    if name.lower().endswith('.png') or name.lower().endswith('.jpg') or name.lower().endswith('.jpeg'):
        img = io.imread(name)

        if len(img.shape) == 2 or img.shape[2] == 1:
            img = gray2rgb(img)

        if img.shape[2] == 4:
            img = img[..., :3]

        img = img_as_float32(img)   # (H, N*W, 3)
        video_array = np.moveaxis(img, 1, 0)  # (N*W, H, 3)
        video_array = video_array.reshape((-1,) + img_shape)  # (N, W, H, 3)
        video_array = np.moveaxis(video_array, 1, 2)  # (N, H, W, 3)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name, memtest=False))

        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception(f"Unknown file extensions {name}")

    return video_array

