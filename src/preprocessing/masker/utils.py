import numpy as np
import cv2

def get_regional_variance_deviation(img: np.ndarray) -> np.ndarray:
    '''(H,W,C) input -> (k1,k2,1) variance map -> (H,W,1) Regional Variance Deviation'''

