import os.path

import cv2
import numpy as np
import tensorflow as tf
from os import listdir, mkdir
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
def subtract_background(img, uniques):
    rows, cols, _ = img.shape

    for i in range(rows):
        for j in range(cols):
            p = img[i,j]
            tup = (p[0], p[1], p[2])
            if tup in uniques:
                img[i,j] = [255, 255, 255]

    return img


def apply_closing(img):
    kernel = np.ones((3,3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img

def apply_opening(img):
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img


def histogram_equalization(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE()
    lab[...,0] = clahe.apply(lab[...,0])
    ret = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return ret


def extract_largest_connected_component(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = 255 - thresh

    num_comps, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)

    max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_comps)], key=lambda x: x[1])

    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if labels[i][j] == max_label:
                gray[i][j] = True
            else:
                gray[i][j] = False

    return cv2.bitwise_and(img, img, mask=gray)


def merge_original_and_largest_region(orig, largest_reg, use_relative_size_threshold=False):
    # Extracting largest components succeeds at removing the wowhead logo in the background
    # But also removes some connected areas within the character. Mitigate this issue by the
    # Following code. Take the original image with its background subtracted, along with the
    # extracted image. If these two images disagree in the pixel value, keep the original one,
    # if it is not background color (x, x, x)
    for i in range(largest_reg.shape[0]):
        for j in range(largest_reg.shape[1]):
            img_tup = (largest_reg[i][j][0], largest_reg[i][j][1], largest_reg[i][j][2])
            orig_tup = (orig[i][j][0], orig[i][j][1], orig[i][j][2])
            if img_tup == (255, 255, 255) and (orig_tup != (255, 255, 255) and len(set(orig_tup)) != 1):
                largest_reg[i][j] = orig[i][j]
            if use_relative_size_threshold:
                img_height = largest_reg.shape[0]
                img_width  = largest_reg.shape[1]

                height_range = range(int((img_height / 100) * 20), int((img_height / 100) * 80))
                width_range = range(int((img_width / 100) * 33), int((img_width / 100) * 67))

                if img_tup == (255, 255, 255) and orig_tup != (255, 255, 255) and (i in height_range and j in width_range):
                    largest_reg[i][j] = orig[i][j]

    return largest_reg

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

    if not os.path.exists(path + "/prepared"):
        mkdir(path + "/prepared")

    for i in range(len(images)):
        cv2.imwrite(path + "/prepared/" + str(i) + "_prepared.png", images[i])


def preprocess_image(img):
    orig = subtract_background(img, [(24, 24, 24)])

    img = extract_largest_connected_component(orig)
    img = subtract_background(img, [(i,i,i) for i in range(24)])

    temp = apply_closing(orig)
    temp = apply_opening(temp)

    img = merge_original_and_largest_region(orig, img, use_relative_size_threshold=True)
    img = merge_original_and_largest_region(temp, img)

    return img

