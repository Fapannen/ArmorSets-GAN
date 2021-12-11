import tensorflow as tf
import numpy as np
import cv2
from preprocessing import preprocess
from preprocessing import generator
from hparams import hparams
from gan import armorGAN


def main():
    # Generate 2x more images for training (Use only on the first run, without recolors)
    # generator.generate_recolors("dataset")

    config = hparams.Config()

    # And only them load them for use
    images = preprocess.load_and_preprocess("dataset", (config.IMG_WIDTH, config.IMG_HEIGHT), config.MODE)

    # Draw an example images
    if config.IMG_CHANNELS == 3:
        cv2.imwrite('test1.png', (np.array(images[58, :, :, :, 0]) * 127.5) + 127.5)
        cv2.imwrite('test2.png', (np.array(images[59, :, :, :, 0]) * 127.5) + 127.5)
    else: # if grayscale
        cv2.imwrite('test1.png', (np.array(images[58]) * 127.5) + 127.5)
        cv2.imwrite('test2.png', (np.array(images[59]) * 127.5) + 127.5)

    # define the generator with target IMG height and IMG width
    gan_generator = armorGAN.define_generator(config.LATENT_DIM, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)

    gan_discriminator = armorGAN.define_discriminator((config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS))

    # Create a random noise and generate a sample
    noise = tf.random.normal([1, config.LATENT_DIM])
    generated_image = gan_generator(noise, training=False)
    # Visualize the generated sample
    print(generated_image.shape)
    cv2.imwrite('generated.jpg', np.array(generated_image[0]))

    gan_generator.summary()
    gan_discriminator.summary()

    real_samples = generator.generate_real_samples(images, config.BATCH_SIZE)
    print(real_samples[0].shape)

    fake_samples = generator.generate_fake_samples(config.BATCH_SIZE, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
    print(fake_samples[0].shape)

    gan = armorGAN.define_gan(gan_generator, gan_discriminator)

    armorGAN.train(gan_generator, gan_discriminator, gan, images, config.LATENT_DIM, config.EPOCHS, config.BATCH_SIZE, config.IMG_CHANNELS)

main()