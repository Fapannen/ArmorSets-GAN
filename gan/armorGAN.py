import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import (Dense,
                                     BatchNormalization,
                                     LeakyReLU,
                                     Reshape,
                                     Conv2DTranspose,
                                     Conv2D,
                                     Dropout,
                                     Flatten)
import matplotlib.pyplot as plt

def define_generator(latent_dim, target_img_height, target_img_width):
	begin_h = int(target_img_height / 4)
	begin_w = int(target_img_width / 4)

	model = Sequential()
	n_nodes = 128 * begin_h * begin_w # 128 neurons, quarter of the final image
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((begin_h, begin_w, 128)))
	# upsample to half the size of the final image
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to the final img dimensions
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	return model