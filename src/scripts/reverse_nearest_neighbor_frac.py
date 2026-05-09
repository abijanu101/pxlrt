import cv2
import numpy as np

DIR = './resources/ashlord00/images'

img = cv2.imread(DIR + '/3ed3c7ce-e0ca-4625-b792-2cc86a6632fc.png')
cv2.imshow('Original', img)

# print()

OG_SIZE = (img.shape[0], img.shape[1])
NEW_SIZE = (OG_SIZE[0] // 6, OG_SIZE[1] // 6)

img = cv2.resize(img, NEW_SIZE, interpolation=cv2.INTER_NEAREST)  
img = cv2.resize(img, OG_SIZE, interpolation=cv2.INTER_NEAREST)  
cv2.imshow('downscaled', img)

cv2.waitKey(0)