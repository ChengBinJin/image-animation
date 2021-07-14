import sys
import os
import cv2
import numpy as np
from imageio import mimread, imsave
from skimage import img_as_float32

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_dir)

from library.utils.files import all_files_under, get_name


def main(img_folder, save_folder, img_size=256):
    video_paths = all_files_under(img_folder, extension=".gif")
    for video_path in video_paths:
        frames = list()
        print(f'video_path: {video_path}')
        video = np.array(mimread(video_path, memtest=False))
        if video.shape[-1] == 4:
            video = video[..., :3]

        num_imgs = video.shape[0]
        for i in range(num_imgs):
            img = video[i, ...]
            if img.shape[0] != img_size:
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            frames.append(img)

        frames = np.hstack(frames)
        img_name = get_name(video_path)
        imsave(os.path.join(save_folder, img_name + '.png'), frames)


if __name__ == "__main__":
    img_folder = r"D:\Data\image_animation\monkey_net\MGif\moving-gif-processed\moving-gif\test"
    save_folder = r"D:\Data\image_animation\monkey_net\MGif\moving-gif-processed\moving-gif-256"
    save_folder = os.path.join(save_folder, os.path.basename(img_folder))
    os.makedirs(save_folder, exist_ok=True)

    main(img_folder, save_folder)
