import pandas as pd
import numpy as np
import cv2
import argparse
from os import listdir
from os.path import isfile, join
from tensorflow.keras.models import load_model

def vizualize_dataset(path_to_generator, path_to_csv, num_examples):

    # Load the dataset
    df = pd.read_csv(path_to_csv, header=0, delimiter=';')
    df_dim = len(df.columns)

    # Convert all non-numeric data to numeric so that we can feed it into the generator
    for col in df.columns:
        if df[col].dtype != "int64" or df[col].dtype != "float64":
            df[col] = df[col].astype("category")
            df[col] = df[col].cat.codes

    # Normalize all values to [-1, 1] range
    df.apply(lambda x: (2 * ((x - x.min()) / (x.max() - x.min()))) - 1, axis=0)

    # Load the generator model
    generator = load_model(path_to_generator)
    input_dim = generator.input_shape[1]

    if df_dim < input_dim:
        print("Dataset dimension is less than the input to the generator. Values from the dataset will be used and then padded with zeros.")
    if df_dim > input_dim:
        print("Dataset dimension is greater than the input to the generator. Some features will be discarded.")

    for i in range(num_examples):
        input_vector = []

        # Fill the vector with features from the dataset
        for col in df.columns:
            input_vector.append(df.iloc[i][col])
            if len(input_vector) == input_dim:
                break

        while len(input_vector) != input_dim:
            input_vector.append(0)

        input_vector = np.array(input_vector)
        input_vector = input_vector.reshape(1, input_dim)

        generated = generator.predict(input_vector)

        for j in range(len(generated)):
            cv2.imwrite("dataset_viz_" + str(i) + ".jpg", (np.array(generated[j]) * 127.5) + 127.5)

def create_training_timelapse(path, intermediate_steps = 10):
    def interpolate_images(path_img1, path_img2):
        im1 = cv2.imread(path_img1, cv2.IMREAD_COLOR)
        im2 = cv2.imread(path_img2, cv2.IMREAD_COLOR)

        delta = np.array(im2) - np.array(im1)
        increment = delta / intermediate_steps

        intermediate_images = []

        for i in range(intermediate_steps):
            tmp = i * np.round(increment)
            intermediate_images.append(np.array(im1) + tmp)

        return intermediate_images

    files = [f for f in listdir(path) if isfile(join(path + "/", f)) and f.endswith('jpg') and "intermediate" in f]
    nums = [int(filename.split("intermediate")[1].split(".jpg")[0]) for filename in files]

    random_image = cv2.imread(files[0], cv2.IMREAD_COLOR)
    h, w, _ = random_image.shape


    video_output = cv2.VideoWriter("training_interpolation.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60, (w, h))


    for i in range(min(nums), max(nums) - 1):
        interpol = interpolate_images("intermediate" + str(i) + ".jpg", "intermediate" + str(i+1) + ".jpg")
        for img in interpol:
            video_output.write(img.astype(np.uint8) - 128)

    video_output.release()


def main():

    parser = argparse.ArgumentParser(description='Parser for vizualization')
    parser.add_argument('--generator', type=str, help='Path to generator file')
    parser.add_argument('--csv', type=str, help="Path to the csv file you want to feed into generator")
    parser.add_argument('--num_examples', type=int, help="How many examples you want to visualize")

    args = parser.parse_args()

    vizualize_dataset(args.generator, args.csv, args.num_examples)


create_training_timelapse(".")