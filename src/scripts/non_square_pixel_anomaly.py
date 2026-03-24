import cv2
import matplotlib.pyplot as plt
import numpy as np

ORIGINAL_SIZE = 30
SCALING_FACTOR = 13.333

# image drawing
img = np.zeros((ORIGINAL_SIZE,ORIGINAL_SIZE,3), dtype=np.uint8)
for i in range(30):
    img[i][i] = [255,255,255]
    img[i][ORIGINAL_SIZE - 1 - i] = [255,0,0]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

print("Before Scaling: ", img.shape)
# plt.imshow(img)
# plt.show()


# scaling
NEW_SIZE = round(ORIGINAL_SIZE * SCALING_FACTOR)
img = cv2.resize(img, (NEW_SIZE, NEW_SIZE), interpolation=cv2.INTER_NEAREST)  


print('After Scaling: ', img.shape)
plt.imshow(img)
plt.show()