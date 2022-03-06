import cv2
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from hparams import hparams

preprocessing_output_path = "../preprocessing_steps/"

"""
Load images from 'path' directory, convert them to RGB and return a list of them 
"""
def load_images(path, mode):
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
Uses a 'unique' set of pixels to replace in the image by white color.
"""
def subtract_background(img, uniques=[(24,24,24), (0,0,0), (5,5,5), (12,12,12), (7,7,7)]):
    rows, cols, _ = img.shape

    for i in range(rows):
        for j in range(cols):
            p = img[i,j]
            tup = (p[0], p[1], p[2])
            if tup in uniques:
                img[i,j] = [255, 255, 255]

    cv2.imwrite(preprocessing_output_path + "subtracted.png", img)
    return img


def apply_closing(img):
    kernel = np.ones((3,3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(preprocessing_output_path + "closed.png", img)
    return img

def apply_opening(img):
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(preprocessing_output_path + "opened.png", img)
    return img


def histogram_equalization(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE()
    lab[...,0] = clahe.apply(lab[...,0])
    ret = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    cv2.imwrite(preprocessing_output_path + "histeq.png", ret)
    return ret


"""
Loads an image dataset from 'path' folder. Maps the size of images to the average of those found in the dataset.
Normalized image values to [0,1] range. Converts to RGB, opencv2 works with BGR by default.
"""
def load_and_prepare(path, dims, mode=cv2.IMREAD_GRAYSCALE):
    images = load_images(path, mode)
    images = resize_to_dims(images, dims)
    images = [normalize_image(img) for img in images]

    return np.expand_dims(np.array(images), axis=-1)


def preprocess_training_images(path, mode):
    images = load_images(path, mode)
    images = [preprocess_image(img) for img in images]

    for i in range(len(images)):
        cv2.imwrite(images[i], path + "/prepared/" + str(i) + "_prepared.png")


def preprocess_image(img):
    img = subtract_background(img)
    img = apply_closing(img)
    img = apply_opening(img)
    img = subtract_background(img, [(22,22,22), (16,16,16), (23, 23, 23), (21, 21, 21), (10,10,10), (14,14,14)])
    return img

