import numpy as np
import cv2

def calculate_integral_image(importance_map: np.ndarray) -> np.ndarray:
    return cv2.integral(importance_map)

def get_region_score(integral_image: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    region_sum = (integral_image[y2, x2] - 
                  integral_image[y1, x2] - 
                  integral_image[y2, x1] + 
                  integral_image[y1, x1])
    area = w * h
    return region_sum / area if area > 0 else 0
