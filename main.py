from preprocessing import preprocess
from preprocessing import generator
from gan import armorGAN
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

IMG_HEIGHT = 60
IMG_WIDTH  = 40
IMG_CHANNELS = 1 # Grayscale 1, RGB 3
BATCH_SIZE = 16

def main():
    # Generate 2x more images for training (Use only on the first run, without recolors)
    # TBD when working with RGB images, forget this for now
    # generator.generate_recolors("dataset")

    # And only them load them for use
    images = preprocess.load_and_preprocess("dataset", (IMG_WIDTH, IMG_HEIGHT), cv2.IMREAD_GRAYSCALE)

    # train_dataset = preprocess.prepare_tensors(images) # Tf.Dataset instance

    # Draw an example images
    cv2.imwrite('test1.png', np.array(images[58]) * 255)
    cv2.imwrite('test2.png', np.array(images[59]) * 255)

    # define latent space dimension
    noise_length = 100

    # define the generator with target IMG height and IMG width
    gan_generator = armorGAN.define_generator(noise_length, IMG_HEIGHT, IMG_WIDTH)

    gan_discriminator = armorGAN.define_discriminator((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Create a random noise and generate a sample
    noise = tf.random.normal([1, noise_length])
    generated_image = gan_generator(noise, training=False)
    # Visualize the generated sample
    i = generated_image[0, :, :, 0]
    plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    plt.imsave("generated.jpg", i, cmap='gray')

    gan_generator.summary()
    gan_discriminator.summary()

    real_samples = generator.generate_real_samples(images, 16)
    print(real_samples[0].shape)

    fake_samples = generator.generate_fake_samples(16, IMG_HEIGHT, IMG_WIDTH)
    print(fake_samples[0].shape)

    gan = armorGAN.define_gan(gan_generator, gan_discriminator)

    armorGAN.train(gan_generator, gan_discriminator, gan, images, noise_length, 50, 16)

main()