import numpy as np
import cv2

def color_range(img: np.ndarray) -> np.ndarray:
    'Max(R,G,B) - Min(R,G,B)'

    maxc = np.max(img, axis=2)
    minc = np.min(img, axis=2)
    return maxc - minc

def color_novelty(img: np.ndarray) -> np.ndarray:
    'norm(x - Mean)'

    mean_color = img.reshape(-1, 3).mean(axis=0)
    diff = img.astype(np.float32) - mean_color
    score = np.linalg.norm(diff, axis=2)
    return score / (score.max() + 1e-6) 

def kmeans_palette_error(img:np.ndarray, k:int=4) -> np.ndarray:
    'norm(x - palette_quantized_reconstruction)'

    data = img.reshape(-1, 3).astype(np.float32)

    _, labels, centers = cv2.kmeans(
        data, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
        5, cv2.KMEANS_RANDOM_CENTERS
    )

    recon = centers[labels.flatten()].reshape(img.shape)
    error = np.linalg.norm(img.astype(np.float32) - recon.astype(np.float32), axis=2)

    return error / (error.max() + 1e-6)
