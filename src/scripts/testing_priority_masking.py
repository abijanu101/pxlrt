import numpy as np
import cv2

DIR = './resources/ashlord00/images'

def k_means_thing(img):
    data = img.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, 3, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

    palette_img = centers[labels.flatten()].reshape(img.shape)

    cv2.imshow("quantized palette", palette_img.astype(np.uint8))
    cv2.waitKey(0)

def canny_thing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 200, 250)

    cv2.imshow("edges", edges)
    cv2.waitKey(0)

def playingAround(path):
    img = cv2.imread(path)
    cv2.imshow('original', img)
    k_means_thing(img)
    canny_thing(img)

# playingAround(DIR + '/0b0611b4-fd91-48f3-a2b7-bb191aeb4e3d.png')
# playingAround(DIR + '/6cbd9c21-4976-442e-b25f-67cb080e9004.png')
# playingAround(DIR + '/2b569f4c-ec87-4cd1-8481-2e5db5903369.png')
playingAround(DIR + '/3ed3c7ce-e0ca-4625-b792-2cc86a6632fc.png')