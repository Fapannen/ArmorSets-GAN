from preprocessing import preprocess
from preprocessing import generator
from gan import armorGAN
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2


def main():
    # Generate 2x more images for training (Use only on the first run, without recolors)
    # generator.generate_recolors("dataset")

    # And only them load them for use
    images = preprocess.load_and_preprocess("dataset")

    # Draw an example how recolor works
    cv2.imwrite('test1.png', np.array(images[58]) * 255)
    cv2.imwrite('test2.png', np.array(images[59]) * 255)

    print(images)

    print(preprocess.prepare_tensors(images))

def minor():
    gen = armorGAN.build_generator()
    print(gen)

    # Create a random noise and generate a sample
    noise = tf.random.normal([1, 100])
    generated_image = gen(noise, training=False)
    # Visualize the generated sample
    i = generated_image[0, :, :, 0]
    plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    plt.imsave("generated.jpg", i, cmap='gray')

main()
minor()