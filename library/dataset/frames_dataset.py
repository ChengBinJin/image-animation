import os
import time
import glob
import imageio
import numpy as np
import pandas as pd
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
from torch.utils.data import Dataset

from library.utils.files import all_files_under
from library.dataset.augmentation import AllAugmentationTransform, AllAugmentationTransform2, VideoToTensor


def read_video2(name):
    """
    Read video which can be:
        - an image of concatenated frames
        - '.mp4' and '.gif'
        - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array([img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg') or name.lower().endswith('.jpeg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        # video_array = np.moveaxis(image, 1, 0)
        # video_array = video_array.reshape((-1), + frame_shape)
        # video_array = np.moveaxis(video_array, 1, 2)
        video_array = np.expand_dims(image, axis=0)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name, memtest=False))

        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]

        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions %s" % name)

    return video_array


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


class FramesDataset2(Dataset):
    """
    Dataset of videos, each video can be represented as:
        - an image of concatenated frames
        - '.mp4' or '.gif'
        - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True, random_seed=0,
                 pairs_list=None, augmentation_params=None, img_dir=None):
        self.root_dir = root_dir
        # self.videos = os.listdir(root_dir)

        self.img_dir = img_dir
        if self.img_dir is not None:
            self.imgs = os.listdir(self.img_dir)

        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling

        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")

            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0]
                                for video in os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))

            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos
        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform2(**augmentation_params)
        else:
            self.transform = None

    def get_source_img(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img_array = read_video2(img_path)
        img_array = img_array[range(len(img_array))]

        out = dict()
        out['img'] = np.array(img_array, dtype='float32').transpose((3, 0, 1, 2))
        out['name'] = img_name
        return out

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)
        if self.is_train and os.path.isdir(path):
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
        else:
            video_array = read_video2(path)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) \
                if self.is_train else range(num_frames)
            video_array = video_array[frame_idx]

        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')

            out['source'] = source.transpose((2, 0, 1))
            out['driving'] = driving.transpose((2, 0, 1))
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))
            out['fps'] = imageio.get_reader(path).get_meta_data()['fps']

        out['name'] = video_name

        return out


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


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


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
            imgs = self.initial_dataset.imgs
            print(f'imgs shape: {len(imgs)}')
            videos = self.initial_dataset.videos
            print(f'videos: {len(videos)}')

            name_to_index_videos = {name: index for index, name in enumerate(videos)}
            name_to_index_images = {name: index for index, name in enumerate(imgs)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(imgs), pairs['driving'].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append((name_to_index_videos[pairs['driving'].iloc[ind]],
                                   name_to_index_images[pairs['source'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset.get_source_img(pair[1])

        first = {'driving_' + key: value for key, value in first.items()}
        second = {'source_' + key: value for key, value in second.items()}
        return {**first, **second}
