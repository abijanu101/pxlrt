import numpy as np
import cv2

def neighbor_contrast(img:np.ndarray) -> np.ndarray:
    'scale(sum(x - neighbor))'

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    r = np.roll(gray, 1, axis=0)
    l = np.roll(gray, -1, axis=0)
    u = np.roll(gray, 1, axis=1)
    d = np.roll(gray, -1, axis=1)

    contrast = np.abs(gray - r) + np.abs(gray - l) + np.abs(gray - u) + np.abs(gray - d)

    return contrast / (contrast.max() + 1e-6)

def difference_of_gaussians(img:np.ndarray, mean1:np.number=1, mean2:np.number=5) -> np.ndarray:
    'scale(blur(m1) - blur(m2))'

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    g1 = cv2.GaussianBlur(gray, (0, 0), mean1)
    g2 = cv2.GaussianBlur(gray, (0, 0), mean2)

    dog = np.abs(g1 - g2)
    return dog / (np.max(dog) + 1e-6)
