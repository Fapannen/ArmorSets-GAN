from keras.models import load_model
from preprocessing import generator
from hparams import hparams
import numpy as np
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
    cv2.imwrite("trained_and_generated" + str(i) + ".jpg", (np.array(X[i]) * 127.5) + 127.5)
