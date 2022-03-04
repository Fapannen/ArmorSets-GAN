import cv2
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from hparams import hparams


"""
Load images from 'path' directory, convert them to RGB and return a list of them 
"""
def load_images(path, mode, only_zandalari=False):
    files = [f for f in listdir(path) if isfile(join(path + "/", f)) and f.endswith('png')]

    # Keep only zandalaris, dataset contains also other races. Defaults to True
    c = hparams.Config()
    if c.ONLY_ZANDALARI:
        files = [f for f in files if "_ZT" in f]

    ret = []
    for i in tqdm(range(len(files))):
        ret.append(cv2.imread(path + "/" + files[i], mode))

    return ret


def get_average_width(images):
    widths = [img.shape[1] for img in images]
    return int(sum(widths) / len(widths))


def get_average_height(images):
    heights = [img.shape[0] for img in images]
    return int(sum(heights) / len(heights))


def get_average_size(images):
    return (get_average_height(images), get_average_width(images))


def resize_to_average(images):
    avg = get_average_size(images)
    print("Resizing to average image size (HxW): ", avg)
    avg = (avg[1], avg[0]) # resize works with (width, height) format instead of common (height, width)
    return [cv2.resize(img, avg, cv2.INTER_CUBIC) for img in images]


def resize_to_dims(images, dims):
    return [cv2.resize(img, (dims[0], dims[1]), cv2.INTER_CUBIC) for img in images]


def normalize_image(image):
    # normalize images to [-1, 1] range
    return cv2.normalize(image , None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

"""
Use an image of wowhead background to remove the background from training images
"""
def define_background(path_to_background="../img/wowhead_background.png"):
    background = cv2.imread(path_to_background)
    ret = []
    rows, cols, _ = background.shape

    for r in range(rows):
        for c in range(cols):
            p = background[r, c]
            tup = (p[0], p[1], p[2])
            if tup not in ret:
                ret.append(tup)
    return ret


def subtract_background(img, uniques):
    rows, cols, _ = img.shape

    for i in range(rows):
        for j in range(cols):
            p = img[i,j]
            tup = (p[0], p[1], p[2])
            if tup in uniques:
                img[i,j] = [255, 255, 255]

    cv2.imwrite("../subtracted.png", img)
    return img


def apply_closing(img):
    kernel = np.ones((3,3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("../closed.png", img)
    return img

"""
Loads an image dataset from 'path' folder. Maps the size of images to the average of those found in the dataset.
Normalized image values to [0,1] range. Converts to RGB, opencv2 works with BGR by default.
"""
def load_and_preprocess(path, dims, mode=cv2.IMREAD_GRAYSCALE):
    images = load_images(path, mode)
    images = resize_to_dims(images, dims)
    images = [normalize_image(img) for img in images]
    return np.expand_dims(np.array(images), axis=-1)

im = cv2.imread("../dataset/278_ZT_recolor.png")
u = define_background()
wh = subtract_background(im, u)
apply_closing(wh)
