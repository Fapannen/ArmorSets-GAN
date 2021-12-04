from preprocessing import preprocess
import cv2

def main():
    images = preprocess.load_images("dataset")
    images, avg = preprocess.resize_to_average(images)

    cv2.imwrite('out.png', images[200])

main()