import cv2
import numpy as np
from src.cropper.config import NATIVE_SCALING_FACTOR

def get_scaled(img: np.ndarray) -> np.ndarray:
    """
    Applies the native scaling factor to the image to restore its pixel-perfect resolution.
    """
    h_orig, w_orig = img.shape[:2]
    w_native = round(w_orig / NATIVE_SCALING_FACTOR)
    h_native = round(h_orig / NATIVE_SCALING_FACTOR)
    
    # Use Nearest Neighbor to preserve pixel art structure
    return cv2.resize(img, (w_native, h_native), interpolation=cv2.INTER_NEAREST)
