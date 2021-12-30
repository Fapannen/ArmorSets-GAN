from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GaussianNoise, Conv2DTranspose, LeakyReLU, Conv2D

def replace_layer(path_to_generator, path_to_output, index):
    old_model = load_model(path_to_generator)
    new_model = Sequential(name="sequential2times")  # Need to set a name, otherwise it will conflict

    print(old_model.layers)
    for layeridx in range(len(old_model.layers)):
        layer = old_model.layers[layeridx]
        if layeridx == index:
            new_model.add(layer)
            s = layer.get_output_shape_at(0)
            new_model.add(LeakyReLU(alpha=0.2, name="ler"))
            # upsample to the final img dimensions
            new_model.add(Conv2DTranspose(384, (7, 7),input_shape=(s[1], s[2], s[3]), strides=(2, 2), padding='same', name="modif"))
            new_model.add(LeakyReLU(alpha=0.2, name="ler3"))
            new_model.add(Conv2D(3, (11, 11), activation="tanh", padding='same', name="modif3"))
            break
        new_model.add(layer)

    for l in new_model.layers:
        print(l)
        print(l.name)
        print(l.get_input_shape_at(0))
        print()

    old_model.summary()
    new_model.summary()

    new_model.save(path_to_output)
    return new_model

# Adds gaussian noise layer before every conv2D operation
def add_gaussian_noise_disc(path_to_discriminator, path_to_output):
    old_model = load_model(path_to_discriminator)
    new_model = Sequential(name="gaussian_sequential2") # Need to set a name, otherwise it will conflict

    for layer in old_model.layers:
        if "conv2d" in layer.name:
            s = layer.get_input_shape_at(0)
            new_model.add(GaussianNoise(0.1, input_shape=(s[1], s[2], s[3])), seed=42)
        new_model.add(layer)
        new_model.layers[-1].set_weights(layer.get_weights())

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    new_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    for l in new_model.layers:
        print(l)
        print(l.name)
        print(l.get_input_shape_at(0))
        print()

    old_model.summary()
    new_model.summary()

    new_model.save(path_to_output)
    return new_model

n = replace_layer("../checkpoints/generator_model_323_2500.h5", "../checkpoints/generator_model_doubled_2500.h5", 5)
n.summary()
