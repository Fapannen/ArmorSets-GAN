import numpy as np
import cv2
import argparse
from keras.models import load_model
from preprocessing import generator
from hparams import hparams

config = hparams.Config()

parser = argparse.ArgumentParser(description='Parser for generator results inspection')
parser.add_argument('--epochs', type=int, help='Which epoch to generate results from')

args = parser.parse_args()

model_name = "0" + str(args.epochs) if args.epochs < 100 else str(args.epochs)

# load model
model = load_model('checkpoints/generator_model_323_' + model_name + '.h5')
# generate images
latent_points = generator.generate_latent_points(100, 25)
# generate images
X = model.predict(latent_points)
# plot the result
for i in range(len(X)):
    cv2.imwrite("trained_and_generated" + str(i) + ".jpg", (np.array(X[i]) * 127.5) + 127.5)
