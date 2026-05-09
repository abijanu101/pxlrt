import numpy as np
import cv2

def generate_priority_mask(img: np.ndarray) -> np.ndarray:
    """
    Returns a heatmap for areas of relative importance based on color novelty.
    Areas with colors far from the mean are considered more 'novel'.
    """
    mean_color = img.reshape(-1, 3).mean(axis=0)
    diff = img.astype(np.float32) - mean_color
    score = np.linalg.norm(diff, axis=2)
    
    # Normalize to 0-1
    max_val = score.max()
    if max_val > 1e-6:
        score /= max_val
        
    return score


# playground

if __name__ == '__main__':
    pass