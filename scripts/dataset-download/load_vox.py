import sys
import os
import pytube
import time
import numpy as np
from argparse import ArgumentParser

URL = 'https://www.youtube.com/watch?v='

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_dir)

parser = ArgumentParser()
parser.add_argument("--annotations_dir", default=r"scripts\dataset-download\VoxCeleb\vox1_test_txt")
parser.add_argument("--save_dir", default=r"D:\Data\image_animation\VoxCeleb\vox1_test")
args = parser.parse_args()


def random_wait():
    # waits a random amount of time so the spam protection doesn't trigger as easily
    wait_len = np.random.randint(3, 16)
    print("waiting {} secs".format(wait_len))
    time.sleep(wait_len)


def main():
    video_id_folders = sorted(os.path.join(args.annotations_dir, file) for file in os.listdir(args.annotations_dir))

    for video_id_folder in video_id_folders[:1]:
        for video_id in os.listdir(video_id_folder):
            url = URL + video_id

            random_wait()

            try:
                youtube = pytube.YouTube(url)
                video = youtube.streams.first()

                print(f' [*] Downloading {url}...')
                out_file = video.download(args.save_dir)
                video_name, ext = os.path.splitext(os.path.basename(out_file))
                os.rename(out_file, os.path.join(args.save_dir, f'{video_id}{ext}'))
            except:
                # os.remove(os.path.join(save_folder, video_name + ext))
                print(f' [!] Cannot find {url}...')
                continue


if __name__ == "__main__":
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    main()
