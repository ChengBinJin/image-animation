import numpy as np
from imageio import mimread


def diff_gif(path_1, path_2):
    print(f'path_1: {path_1}')
    print(f'path_2: {path_2}')

    gif_1 = np.array(mimread(path_1, memtest=False))
    gif_2 = np.array(mimread(path_2, memtest=False))
    assert gif_1.shape == gif_2.shape, print(f"The shape {gif_1.shape} and {gif_2.shape} is different")

    diff = np.sum((gif_1 - gif_2)**2) / gif_1.size
    print(f'Difference: {diff:.2f}')


if __name__ == "__main__":
    path_1 = '/workspace/nas-data/Codes/H0086_mokey-net/src/demo.gif'
    path_2 = '/workspace/nas-data/Codes/H0088_image-animation/demo.gif'
    diff_gif(path_1, path_2)
