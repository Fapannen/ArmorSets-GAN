from keras.models import load_model
from matplotlib import pyplot as plt
from preprocessing import generator
from hparams import hparams
import cv2

config = hparams.Config()
# load model
model = load_model('checkpoints/generator_model_010.h5')
# generate images
latent_points = generator.generate_latent_points(100, 25)
# generate images
X = model.predict(latent_points)
# plot the result
for i in range(len(X)):
    if config.MODE == cv2.IMREAD_COLOR:
        plt.imsave("trained_and_generated" + str(i) + ".jpg", X[i])
    if config.MODE == "GRAYSCALE":
        plt.imsave("trained_and_generated" + str(i) + ".jpg", X[i, :, :, 0], cmap='gray')
