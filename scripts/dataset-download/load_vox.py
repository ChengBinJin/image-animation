import sys
import os
from argparse import ArgumentParser

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_dir)

parser = ArgumentParser()
parser.add_argument("--annotations_dir", default=r"scripts\dataset-download\VoxCeleb\vox1_test_txt")
parser.add_argument("--save_dir", default=r"D:\Data\image_animation\VoxCeleb\vox1_test")
args = parser.parse_args()


def main():
    print("Hello world!")


if __name__ == "__main__":
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    main()
