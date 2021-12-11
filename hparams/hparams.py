import cv2

class Config:

    def __init__(self):
        # General hyperparameters
        self.IMG_HEIGHT = 300
        self.IMG_WIDTH = 200
        self.MODE = cv2.IMREAD_COLOR
        self.IMG_CHANNELS = 3 if self.MODE == cv2.IMREAD_COLOR else 1
        self.BATCH_SIZE = 16
        self.EPOCHS = 1000
        self.LATENT_DIM = 100

        # Optimization hyperparameters

        # Select an activation function to use for last generator layer. Tanh is recommended.
        self.GEN_ACT = "tanh" # or "sigmoid", but tanh is preferred

        # Use label smoothing or not (set target label of real images to 0.9 instead of 1)
        # This will result to the discriminator accuracy to be 0% during summarize_performance()
        self.LABEL_SMOOTHING = True

        # Keep only Zandalari Trolls training images
        self.ONLY_ZANDALARI = True