import numpy as np
from typing import List, Dict
from .config import MAX_OVERLAP_THRESHOLD, SEMANTIC_DIFF_THRESHOLD, DIVERSITY_BUCKETS, SAMPLES_PER_BUCKET

def calculate_iou(bbox1: tuple, bbox2: tuple) -> float:
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0

def calculate_importance_variance(bbox: tuple, importance_map: np.ndarray) -> float:
    x, y, w, h = bbox
    region = importance_map[y:y+h, x:x+w]
    return float(np.var(region)) if region.size > 0 else 0.0

class RegionFilter:
    def __init__(self):
        self.iou_threshold = MAX_OVERLAP_THRESHOLD
        self.semantic_threshold = SEMANTIC_DIFF_THRESHOLD
        self.buckets = DIVERSITY_BUCKETS
        self.samples_per_bucket = SAMPLES_PER_BUCKET

    def apply_nms(self, proposals: List[Dict], importance_map: np.ndarray) -> List[Dict]:
        if not proposals: return []
        proposals.sort(key=lambda p: p['score'], reverse=True)
        keep = []
        for p in proposals:
            discard = False
            p_var = calculate_importance_variance(p['bbox'], importance_map)
            for k in keep:
                if calculate_iou(p['bbox'], k['bbox']) > self.iou_threshold:
                    if abs(p_var - calculate_importance_variance(k['bbox'], importance_map)) < self.semantic_threshold:
                        discard = True
                        break
            if not discard: keep.append(p)
        return keep

    def apply_diversity_balancing(self, proposals: List[Dict]) -> List[Dict]:
        if not proposals: return []
        scores = [p['score'] for p in proposals]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s: return proposals[:self.samples_per_bucket * self.buckets]
        bucket_edges = np.linspace(min_s, max_s, self.buckets + 1)
        final_set = []
        for i in range(self.buckets):
            low, high = bucket_edges[i], bucket_edges[i+1]
            in_bucket = [p for p in proposals if low <= p['score'] <= (high if i == self.buckets - 1 else high)]
            in_bucket.sort(key=lambda p: p['score'], reverse=True)
            final_set.extend(in_bucket[:self.samples_per_bucket])
        final_set.sort(key=lambda p: p['score'], reverse=True)
        return final_set
