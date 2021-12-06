import cv2
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import numpy as np

def generate_recolors(path):
    files = [f for f in listdir(path) if isfile(join(path + "/", f)) and f.endswith('png')]
    for i in tqdm(range(len(files))):
        image_bgr = cv2.imread(path + "/" + files[i]) # Image is now in BGR, read from the file in path
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # Image is now in RGB, but opencv will write it as if it were BGR

        cv2.imwrite(join(path + "/" + files[i].replace(".png", "") + "_recolor.png"), image_rgb)

def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = np.random.randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = np.ones((n_samples, 1))
	return X, y

def generate_fake_samples(n_samples, img_height, img_width):
	# generate uniform random numbers in [0,1]
	X = np.random.rand(img_height * img_width * n_samples)
	# reshape into a batch of grayscale images
	X = X.reshape((n_samples, img_height, img_width, 1))
	# generate 'fake' class labels (0)
	y = np.zeros((n_samples, 1))
	return X, y

def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = np.random.randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
