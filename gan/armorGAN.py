import numpy as np
from preprocessing import generator
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from hparams import hparams
from tensorflow.keras.layers import (Dense,
                                     LeakyReLU,
                                     Reshape,
                                     Conv2DTranspose,
                                     Conv2D,
                                     Dropout,
                                     Flatten)


def define_generator(latent_dim, target_img_height, target_img_width, target_img_channels):
	begin_h = int(target_img_height / 4)
	begin_w = int(target_img_width / 4)
	c = hparams.Config()

	model = Sequential()
	n_nodes = 128 * begin_h * begin_w # 128 neurons, quarter of the final image
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((begin_h, begin_w, 128)))
	# upsample to half the size of the final image
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to the final img dimensions
	model.add(Conv2DTranspose(128, (6,6), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(target_img_channels, (9,9), activation=c.GEN_ACT, padding='same'))
	return model


# define the standalone discriminator model
def define_discriminator(in_shape=(600,400,1)):
	model = Sequential()
	model.add(Conv2D(128, (5,5), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(128, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model


def train_discriminator(model, dataset, n_iter=100, n_batch=16, img_height = 600, img_width=400):
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_iter):
		# get randomly selected 'real' samples
		X_real, y_real = generator.generate_real_samples(dataset, half_batch)
		# update discriminator on real samples
		_, real_acc = model.train_on_batch(X_real, y_real)
		# generate 'fake' examples
		X_fake, y_fake = generator.generate_fake_samples(half_batch, img_height, img_width)
		# update discriminator on fake samples
		_, fake_acc = model.train_on_batch(X_fake, y_fake)
		# summarize performance
		print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples, num_channels):
	# generate points in latent space
	x_input = generator.generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)
	if num_channels == 3:
		X = np.expand_dims(X, axis=-1)
	# create 'fake' class labels (0)
	y = np.zeros((n_samples, 1))
	return X, y


def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model


def train_gan(gan_model, latent_dim, n_epochs=100, n_batch=16):
	# manually enumerate epochs
	for i in range(n_epochs):
		# prepare points in latent space as input for the generator
		x_gan = generator.generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = np.ones((n_batch, 1))
		# update the generator via the discriminator's error
		gan_model.train_on_batch(x_gan, y_gan)


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=16, num_channels=1):
	c = hparams.Config()
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			X_real, y_real = generator.generate_real_samples(dataset, half_batch)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch, num_channels)
			# create training set for the discriminator
			X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
			# update discriminator model weights
			d_loss, _ = d_model.train_on_batch(X, y)
			# prepare points in latent space as input for the generator
			X_gan = generator.generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = np.ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))

		if (i + 1) % 10 == 0:
			summarize_performance(i, g_model, d_model, dataset, latent_dim, num_channels=num_channels)
			# save the generator model tile file
			filename = 'checkpoints/generator_model_' + str(int(c.IMG_HEIGHT / 100)) + str(int(c.IMG_WIDTH / 100)) + str(c.IMG_CHANNELS) +'_%03d.h5' % (i + 1)
			g_model.save(filename)

		if (i + 1) % 100 == 0:
			# Save also the discriminator model if needed to resume the training
			filename = 'checkpoints/discriminator_model_' + str(int(c.IMG_HEIGHT / 100)) + str(int(c.IMG_WIDTH / 100)) + str(c.IMG_CHANNELS) +'_%03d.h5' % (i + 1)
			d_model.save(filename)


def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100, num_channels=1):
	# prepare real samples
	X_real, y_real = generator.generate_real_samples(dataset, n_samples)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples, num_channels)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))