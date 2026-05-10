import cv2
import numpy as np
from src.preprocessing.cropper.utils import calculate_integral_image
from src.preprocessing.cropper.proposal import RegionProposer
from src.preprocessing.cropper.optimize import RegionOptimizer
from src.preprocessing.cropper.filter import RegionFilter
from src.preprocessing.cropper.normalize import RegionNormalizer
from src.preprocessing.cropper.config import GAN_INPUT_SIZE, MIN_VALID_SCORE_THRESHOLD, REJECT_EMPTY_THRESHOLD

def extract_priority_regions(img: np.ndarray, importance_map: np.ndarray) -> list:
    """
    Extracts and optimizes region proposals based on the priority mask (importance_map).
    """
    proposer = RegionProposer()
    optimizer = RegionOptimizer()
    filterer = RegionFilter()

    integral_img = calculate_integral_image(importance_map)
    
    # Part 1: Proposal
    initial_proposals = proposer.propose(integral_img, img.shape)
    
    # Part 2: Optimization & Filter
    optimized = [optimizer.optimize_region(p, integral_img, importance_map, img.shape) 
                 for p in initial_proposals]
    filtered = filterer.apply_nms(optimized, importance_map)
    final_regions = filterer.apply_diversity_balancing(filtered)
    
    return final_regions

import os
import uuid
import logging

def crop_regions(image: np.ndarray, regions: list, importance_map: np.ndarray = None, 
                 source_id: str = "unknown", save_dir: str = None) -> list:
    """
    Takes an explicit list of regions and crops/normalizes them from the scaled image.
    If importance_map is provided, quality checks are performed to reject empty/low-quality regions.
    Returns a list of dicts: [{'image': np.ndarray, 'metadata': dict}]
    """
    normalizer = RegionNormalizer(GAN_INPUT_SIZE)
    crops_output = []
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    for i, region in enumerate(regions):
        bbox = region['bbox']
        score = region['score']
        
        # Quality Checks if importance_map is provided
        if importance_map is not None:
            if score < MIN_VALID_SCORE_THRESHOLD:
                continue
            
            x, y, w, h = bbox
            region_map = importance_map[y:y+h, x:x+w]
            if region_map.size > 0 and np.max(region_map) < REJECT_EMPTY_THRESHOLD:
                continue

        normalized_img, scale_used = normalizer.process(image, bbox)
        
        x, y, w, h = bbox
        
        metadata = {
            "source_id": source_id,
            "crop_index": i,
            "coordinates": {"x": x, "y": y},
            "dimensions": {"w": w, "h": h},
            "scale_used": scale_used,
            "score": round(float(score), 4)
        }
        
        crop_data = {
            "image": normalized_img,
            "metadata": metadata
        }
        
        if save_dir:
            file_name = f"{source_id}_crop_{i}_{uuid.uuid4().hex[:6]}.png"
            file_path = os.path.join(save_dir, file_name)
            cv2.imwrite(file_path, normalized_img)
            metadata["saved_path"] = file_path
            
        crops_output.append(crop_data)

    logging.info(f"Cropper extracted {len(crops_output)} optimized crops from source {source_id}.")
    
    return crops_output
