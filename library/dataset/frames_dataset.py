import os
import time
import numpy as np
import pandas as pd
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
from torch.utils.data import Dataset

from library.utils.files import all_files_under
from library.dataset.augmentation import AllAugmentationTransform, VideoToTensor


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
        if self.is_train:
            num_imgs = len(self.train_img_names)
        else:
            num_imgs = len(self.test_img_names)
        return num_imgs

    def __getitem__(self, idx):
        if self.is_train:
            img_name = os.path.join(self.root_dir, self.train_img_names[idx])
        else:
            img_name = os.path.join(self.root_dir, self.test_img_names[idx])

        video_array = read_video(img_name, img_shape=self.img_shape)
        out = self.transform(video_array)
        out['name'] = os.path.basename(img_name)

        return out


class PairedDataset(Dataset):
    """
    Dataset of pairs for transfer.
    """

    def __init__(self, initial_dataset, number_of_pairs):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(int(time.time()))
        if pairs_list is None:
            max_id = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_id, max_id
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            images = self.initial_dataset.images
            name_to_index = {name: index for index, name in enumerate(images)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(images), pairs['driving'].isin(images))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append((name_to_index[pairs['source'].iloc[ind]], name_to_index[pairs['driving'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]
        first = {'source_' + key: value for key, value in first.items()}
        second = {'driving_' + key: value for key, value in second.items()}
        return {**first, **second}
