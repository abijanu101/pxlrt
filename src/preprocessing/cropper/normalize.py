import numpy as np
import cv2
from src.config.shared_config import GAN_INPUT_SIZE

def extract_crop(img: np.ndarray, bbox: tuple) -> np.ndarray:
    """Extracts a crop from the image using slicing."""
    x, y, w, h = bbox
    return img[y:y+h, x:x+w]

def handle_integer_upscaling(crop: np.ndarray, target_size: int) -> np.ndarray:
    """
    Upscales the crop using integer factors and Nearest Neighbor to preserve pixels.
    Only scales up if the crop is smaller than target_size.
    """
    h, w = crop.shape[:2]
    
    # Calculate max possible integer scale factor
    scale_x = target_size // w
    scale_y = target_size // h
    scale = min(scale_x, scale_y)
    
    if scale > 1:
        new_w, new_h = w * scale, h * scale
        return cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST), scale
    
    return crop, 1

def normalize_to_canvas(crop: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resizes the crop to exactly fill the target canvas using nearest-neighbor
    interpolation to preserve pixel-art sharpness. No black padding.
    """
    h, w = crop.shape[:2]
    
    # If already the right size, return as-is
    if h == target_size and w == target_size:
        return crop
    
    # Resize to exactly fill the canvas using nearest-neighbor (pixel-perfect)
    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

class RegionNormalizer:
    """
    Orchestrates the conversion of a raw crop to a standardized GAN input.
    """
    
    def __init__(self, target_size: int = GAN_INPUT_SIZE):
        self.target_size = target_size

    def process(self, img: np.ndarray, bbox: tuple) -> tuple:
        """
        Extracts, upscales (if needed), and pads a crop to target_size.
        Returns (normalized_img, scale_factor).
        """
        crop = extract_crop(img, bbox)
        
        # 1. Apply integer upscaling for small crops
        upscaled_crop, scale = handle_integer_upscaling(crop, self.target_size)
        
        # 2. Pad to fixed canvas
        normalized_img = normalize_to_canvas(upscaled_crop, self.target_size)
        
        return normalized_img, scale
