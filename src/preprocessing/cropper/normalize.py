import numpy as np
import cv2
from .config import GAN_INPUT_SIZE

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
    Centers the crop on a fixed-size black canvas.
    Handles both smaller-than and larger-than cases safely.
    """
    h, w = crop.shape[:2]
    
    # If crop is larger than canvas, center-crop it first
    if h > target_size or w > target_size:
        start_y = max(0, (h - target_size) // 2)
        start_x = max(0, (w - target_size) // 2)
        crop = crop[start_y:start_y+target_size, start_x:start_x+target_size]
        h, w = crop.shape[:2]

    # Create empty canvas
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # Calculate offsets for centering
    off_y = (target_size - h) // 2
    off_x = (target_size - w) // 2
    
    # Place crop on canvas
    canvas[off_y:off_y+h, off_x:off_x+w] = crop
    
    return canvas

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
