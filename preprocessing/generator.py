import cv2
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

def generate_recolors(path):
    files = [f for f in listdir(path) if isfile(join(path + "/", f)) and f.endswith('png')]
    for i in tqdm(range(len(files))):
        image_bgr = cv2.imread(path + "/" + files[i]) # Image is now in BGR, read from the file in path
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # Image is now in RGB, but opencv will write it as if it were BGR

        cv2.imwrite(join(path + "/" + files[i].replace(".png", "") + "_recolor.png"), image_rgb)