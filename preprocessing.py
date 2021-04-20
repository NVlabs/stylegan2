import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np
import os
import random
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument("--resolution", type=int)
parser.add_argument("--mode", type=str)
parser.add_argument("--count", type=int)
parser.add_argument("--path", type=str)
args = parser.parse_args()


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def preprocessing(resolution=256, mode="gray", count=None, path=None):
    files = os.listdir(path)
    if not count:
        count = len(files)
    for i in tqdm(range(0, count)):
        if files[i][-3:] != 'png':
            continue
        try:
            img = plt.imread(os.path.join(path, files[i]))
            img = rgba2rgb(img)
            if mode == "gray":
                img = rgb2gray(img)
            img = cv2.resize(img, (resolution, resolution))
            cv2.imwrite("{}custom/{}.jpeg".format(path,files[i][:-4]), img)
        except:
            pass


if __name__ == "__main__":
    preprocessing(
        resolution=args.resolution,
        mode=args.mode,
        count=args.count,
        path=args.path
    )