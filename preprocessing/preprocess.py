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
    return cv2.normalize(image , None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

"""
Loads an image dataset from 'path' folder. Maps the size of images to the average of those found in the dataset.
Normalized image values to [0,1] range. Converts to RGB, opencv2 works with BGR by default.
"""
def load_and_preprocess(path, dims, mode=cv2.IMREAD_GRAYSCALE):
    images = load_images(path, mode)
    images = resize_to_dims(images, dims)
    images = [normalize_image(img) for img in images]
    return np.expand_dims(np.array(images), axis=-1)

def prepare_tensors(images, BATCH_SIZE = 16):
    num_samples = images.shape[0]
    num_channels = 1
    im_height = images[0].shape[0]
    im_width = images[0].shape[1]

    print("samples: ", num_samples, " H: ", im_height, " W:" , im_width)

    train_images = images.reshape(images.shape[0], images[0].shape[0], images[0].shape[1], num_channels).astype('float32')
    return tf.data.Dataset.from_tensor_slices(train_images).shuffle(num_samples).batch(BATCH_SIZE)
