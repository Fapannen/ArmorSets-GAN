from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GaussianNoise

def add_gaussian_noise_disc(path_to_discriminator, path_to_output):
    old_model = load_model(path_to_discriminator)
    new_model = Sequential(name="gaussian_sequential") # Need to set a name, otherwise it will conflict

    for layer in old_model.layers:
        if "conv2d" in layer.name:
            s = layer.get_input_shape_at(0)
            new_model.add(GaussianNoise(0.1, input_shape=(s[1], s[2], s[3])))
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


m = load_model("../checkpoints/discriminator_model_323_1900.h5")
n = add_gaussian_noise_disc("../checkpoints/discriminator_model_323_1900.h5", "../checkpoints/discriminator_model_gauss_1900.h5")

n.summary()
