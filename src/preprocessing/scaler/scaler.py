import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def scale_image(input_path: str, scale_factor: float, save_path: str = None) -> np.ndarray:
    """
    Applies the scaling factor to the image to restore its pixel-perfect resolution.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {input_path}")

    h_orig, w_orig = img.shape[:2]
    
    if scale_factor <= 0:
        raise ValueError("scale_factor must be greater than 0")

    w_native = round(w_orig / scale_factor)
    h_native = round(h_orig / scale_factor)
    
    logging.info(f"Scaling image: original size ({w_orig}x{h_orig}) -> scaled size ({w_native}x{h_native})")
    
    # Use Nearest Neighbor to preserve pixel art structure
    scaled_img = cv2.resize(img, (w_native, h_native), interpolation=cv2.INTER_NEAREST)

    if save_path:
        cv2.imwrite(save_path, scaled_img)
        logging.info(f"Saved scaled image to: {save_path}")

    return scaled_img
