import cv2
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

"""
Load images from 'path' directory, convert them to RGB and return a list of them 
"""
def load_images(path):
    files = [f for f in listdir(path) if isfile(join(path + "/", f)) and f.endswith('png')]
    ret = []
    for i in tqdm(range(len(files))):
        ret.append(cv2.cvtColor(cv2.imread(path + "/" + files[i]), cv2.COLOR_BGR2RGB))

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
    return [cv2.resize(img, avg, cv2.INTER_CUBIC) for img in images], avg

