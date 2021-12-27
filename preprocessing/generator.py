import cv2
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import numpy as np
from hparams import hparams

def generate_recolors(path):
    files = [f for f in listdir(path) if isfile(join(path + "/", f)) and f.endswith('png')]
    for i in tqdm(range(len(files))):
        image_bgr = cv2.imread(path + "/" + files[i]) # Image is now in BGR, read from the file in path
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # Image is now in RGB, but opencv will write it as if it were BGR

        cv2.imwrite(join(path + "/" + files[i].replace(".png", "") + "_recolor.png"), image_rgb)

def generate_real_samples(dataset, n_samples):
	config = hparams.Config()
	# choose random instances
	ix = np.random.randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]

	# add noise to real images if enabled
	if config.DISC_NOISE:
		if config.IMG_CHANNELS == 3:
			noise = np.random.normal(0, config.LABEL_NOISE_VAR, (n_samples ,config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS, 1))
		if config.IMG_CHANNELS == 1:
			noise = np.random.normal(0, config.LABEL_NOISE_VAR, (n_samples, config.IMG_HEIGHT, config.IMG_WIDTH, 1))

		X += noise

	# generate 'real' class labels (1)
	if config.LABEL_SMOOTHING:
		y = np.full((n_samples, 1), 0.9)
		if config.LABEL_NOISE:
			label_noise = np.array(np.random.uniform(low=-config.LABEL_NOISE_VAR, high=config.LABEL_NOISE_VAR, size=n_samples))
			y = y[0] + label_noise
			y = np.expand_dims(y, axis=-1)
	else:
		y = np.ones((n_samples, 1))
	return X, y

def generate_fake_samples(n_samples, img_height, img_width, img_channels):
	# generate uniform random numbers in [-1,1]
	X = np.random.uniform(low=-1, high=1, size=(img_height * img_width * img_channels * n_samples,))
	# reshape into a batch of grayscale images
	if img_channels == 1:
		X = X.reshape((n_samples, img_height, img_width, img_channels))
	else:
		X = X.reshape((n_samples, img_height, img_width, img_channels, 1))
	# generate 'fake' class labels (0)
	y = np.zeros((n_samples, 1))
	return X, y

def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = np.random.randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input


