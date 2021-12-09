import cv2

class Config:

    def __init__(self):
        self.IMG_HEIGHT = 180
        self.IMG_WIDTH = 120
        self.MODE = cv2.IMREAD_COLOR
        self.IMG_CHANNELS = 3 if self.MODE == cv2.IMREAD_COLOR else 1
        self.BATCH_SIZE = 32
        self.EPOCHS = 1000