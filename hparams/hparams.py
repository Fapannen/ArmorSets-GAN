import cv2

class Config:

    def __init__(self):
        # General hyperparameters
        self.IMG_HEIGHT = 300
        self.IMG_WIDTH = 200
        self.MODE = cv2.IMREAD_COLOR
        self.IMG_CHANNELS = 3 if self.MODE == cv2.IMREAD_COLOR else 1
        self.BATCH_SIZE = 8
        self.EPOCHS = 10000
        self.LATENT_DIM = 100
        self.GEN_CHECKPOINT = 10
        self.DIS_CHECKPOINT = 100
        self.SUMMARIZE_PERFORMANCE = False

        # Optimization hyperparameters
        self.LR_INITIAL = 0.000002
        self.LR_DECREASE = True
        self.LR_DECREASE_AFTER = 1000
        self.LR_DECREASE_CONST = 0.1

        # Select an activation function to use for last generator layer. Tanh is recommended.
        self.GEN_ACT = "tanh" # or "sigmoid", but tanh is preferred

        # Use label smoothing or not (set target label of real images to 0.9 instead of 1)
        # This will result to the discriminator accuracy to be 0% during summarize_performance()
        # enabling LABEL_NOISE will adjust the smoothened labels to be 'around' the smoothened value
        # ie [0.8, 1] for real samples and [0, 0.2] for fake samples
        self.LABEL_SMOOTHING = True
        self.LABEL_NOISE = True
        self.LABEL_NOISE_VAR = 0.15 # (The +- range from 0.1 and 0.9)

        # Adding noise to discriminator inputs
        self.DISC_NOISE_INPUT = True
        self.DISC_NOISE_VAR = 1 / 255.0

        # Adding noise to all discriminator layers (does not work only by enabling this)
        self.DISC_NOISE_ALL = True
        self.DISC_NOISE_LAYER_VAR = 0.1

        # Keep only Zandalari Trolls training images
        self.ONLY_ZANDALARI = True