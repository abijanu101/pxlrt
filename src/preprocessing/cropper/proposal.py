from typing import List, Dict
import numpy as np
from .config import WINDOW_SIZES, WINDOW_STEP, MIN_IMPORTANCE_SCORE
from .utils import get_region_score

class RegionProposer:
    def __init__(self):
        self.window_sizes = WINDOW_SIZES
        self.window_step = WINDOW_STEP
        self.min_score = MIN_IMPORTANCE_SCORE

    def propose(self, integral_image: np.ndarray, img_shape: tuple) -> List[Dict]:
        h_img, w_img = img_shape[:2]
        proposals = []
        
        for size in self.window_sizes:
            for y in range(0, h_img - size + 1, self.window_step):
                for x in range(0, w_img - size + 1, self.window_step):
                    score = get_region_score(integral_image, x, y, size, size)
                    if score >= self.min_score:
                        proposals.append({
                            'bbox': (x, y, size, size),
                            'score': float(score)
                        })
        return proposals
