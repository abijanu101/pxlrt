import numpy as np
from typing import Dict
from .config import HILL_CLIMB_STEPS, HILL_CLIMB_STEP_SIZE, EDGE_PENALTY_WEIGHT
from .utils import get_region_score

class RegionOptimizer:
    def __init__(self):
        self.steps = HILL_CLIMB_STEPS
        self.step_size = HILL_CLIMB_STEP_SIZE
        self.penalty_weight = EDGE_PENALTY_WEIGHT

    def compute_boundary_penalty(self, bbox: tuple, importance_map: np.ndarray) -> float:
        x, y, w, h = bbox
        h_img, w_img = importance_map.shape
        top = importance_map[y, x:x+w] if y < h_img else []
        bottom = importance_map[y+h-1, x:x+w] if y+h-1 < h_img else []
        left = importance_map[y:y+h, x] if x < w_img else []
        right = importance_map[y:y+h, x+w-1] if x+w-1 < w_img else []
        boundary_vals = np.concatenate([top, bottom, left, right])
        return float(np.mean(boundary_vals)) * self.penalty_weight if boundary_vals.size > 0 else 0.0

    def get_refined_score(self, bbox: tuple, integral_img: np.ndarray, importance_map: np.ndarray) -> float:
        base_score = get_region_score(integral_img, bbox[0], bbox[1], bbox[2], bbox[3])
        penalty = self.compute_boundary_penalty(bbox, importance_map)
        return max(0.0, base_score - penalty)

    def optimize_region(self, proposal: Dict, integral_img: np.ndarray, importance_map: np.ndarray, img_shape: tuple) -> Dict:
        img_h, img_w = img_shape[:2]
        current_bbox = proposal['bbox']
        current_score = self.get_refined_score(current_bbox, integral_img, importance_map)
        
        for _ in range(self.steps):
            best_local_bbox = current_bbox
            best_local_score = current_score
            x, y, w, h = current_bbox
            candidates = [
                (x + self.step_size, y, w, h), (x - self.step_size, y, w, h),
                (x, y + self.step_size, w, h), (x, y - self.step_size, w, h),
                (x, y, w + self.step_size, h + self.step_size), (x, y, w - self.step_size, h - self.step_size),
            ]
            for cx, cy, cw, ch in candidates:
                if cx < 0 or cy < 0 or cx + cw > img_w or cy + ch > img_h or cw < 8 or ch < 8:
                    continue
                score = self.get_refined_score((cx, cy, cw, ch), integral_img, importance_map)
                if score > best_local_score:
                    best_local_score = score
                    best_local_bbox = (cx, cy, cw, ch)
            if best_local_score > current_score:
                current_score = best_local_score
                current_bbox = best_local_bbox
            else:
                break
        return {'bbox': current_bbox, 'score': float(current_score)}
