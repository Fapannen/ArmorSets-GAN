import numpy as np
import cv2
import argparse
from tensorflow.keras.models import load_model
from preprocessing import generator
from hparams import hparams

config = hparams.Config()

parser = argparse.ArgumentParser(description='Parser for generator results inspection')
parser.add_argument('--epochs', type=int, help='Which epoch to generate results from')
parser.add_argument('--num_samples', type=int, default=25, help="How many images you want to generate")
parser.add_argument('--inspect_index', type=int, default=None, help="Which dimension from latent space to inspect")

args = parser.parse_args()

model_name = "0" + str(args.epochs) if args.epochs < 100 else str(args.epochs)

# load model
model = load_model('checkpoints/generator_model_323_' + model_name + '.h5')
# generate images
latent_points = generator.generate_latent_points(config.LATENT_DIM, args.num_samples)

if args.inspect_index is not None:
    increment = 2 / float(args.num_samples)
    vals = [increment * i for i in range(args.num_samples)]

    for z in range(len(latent_points)):
        latent_points[z][args.inspect_index] = vals[z]

# generate images
X = model.predict(latent_points)
# plot the result
for i in range(len(X)):
    cv2.imwrite("trained_and_generated" + str(i) + ".jpg", (np.array(X[i]) * 127.5) + 127.5)
