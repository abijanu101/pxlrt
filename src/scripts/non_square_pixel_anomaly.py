import cv2
import numpy as np

img = np.zeros((30,30,3))
print(img)
for i in range(30):
    img[i][i] = [255,255,255]
    img[i][29 - i] = [255,0,0]

# fractional scaling (factor of 13.33x)
img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_NEAREST)  

cv2.imshow('hi', img)
cv2.waitKey(0)

cv2.imwrite('test.png',img)