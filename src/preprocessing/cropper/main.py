import os
import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path to allow relative imports
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cropper.config import (TEST_IMAGE_DIR, OUTPUT_DIR, DATASET_DIR, 
                               GAN_INPUT_SIZE, MIN_VALID_SCORE_THRESHOLD, 
                               REJECT_EMPTY_THRESHOLD, NATIVE_SCALING_FACTOR)
from src.cropper.utils import calculate_color_novelty, calculate_integral_image
from src.cropper.proposal import RegionProposer
from src.cropper.optimize import RegionOptimizer
from src.cropper.filter import RegionFilter
from src.cropper.normalize import RegionNormalizer
from src.cropper.export import DatasetExporter

def main():
    # Setup paths
    image_dir = Path(TEST_IMAGE_DIR)
    output_dir = Path(OUTPUT_DIR)
    dataset_dir = Path(DATASET_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test images
    all_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    test_files = all_files[:4]
    
    if not test_files:
        print(f"Error: No images found in {image_dir}")
        return

    # Initialize Modules
    proposer = RegionProposer()
    optimizer = RegionOptimizer()
    filterer = RegionFilter()
    normalizer = RegionNormalizer(GAN_INPUT_SIZE)
    exporter = DatasetExporter(str(dataset_dir))
    
    total_rejections = 0
    
    for filename in test_files:
        path = image_dir / filename
        img = cv2.imread(str(path))
        if img is None:
            continue
            
        print(f"\n" + "="*50)
        print(f"Processing {filename}...")
        
        # 0. Pre-process: Downscale to native resolution
        h_orig, w_orig = img.shape[:2]
        w_native = round(w_orig / NATIVE_SCALING_FACTOR)
        h_native = round(h_orig / NATIVE_SCALING_FACTOR)
        img = cv2.resize(img, (w_native, h_native), interpolation=cv2.INTER_NEAREST)
        print(f"Downscaled to native resolution: {w_native}x{h_native}")
        
        # 1. Pipeline
        importance_map = calculate_color_novelty(img)
        integral_img = calculate_integral_image(importance_map)
        
        # Part 1: Proposal
        initial_proposals = proposer.propose(integral_img, img.shape)
        
        # Part 2: Optimization & Filter
        optimized = [optimizer.optimize_region(p, integral_img, importance_map, img.shape) 
                     for p in initial_proposals]
        filtered = filterer.apply_nms(optimized, importance_map)
        final_regions = filterer.apply_diversity_balancing(filtered)
        
        print(f"Regions to process: {len(final_regions)}")
        
        # Part 3: Normalization & Export
        image_rejections = 0
        samples_for_vis = []
        
        for region in final_regions:
            bbox = region['bbox']
            score = region['score']
            
            # Quality Checks
            # 1. Score threshold
            if score < MIN_VALID_SCORE_THRESHOLD:
                image_rejections += 1
                continue
            
            # 2. Emptiness check (check max importance in bbox)
            x, y, w, h = bbox
            region_map = importance_map[y:y+h, x:x+w]
            if region_map.size > 0 and np.max(region_map) < REJECT_EMPTY_THRESHOLD:
                image_rejections += 1
                continue

            # Process
            normalized_img, scale = normalizer.process(img, bbox)
            
            # Metadata
            metadata = {
                "source_image": filename,
                "bbox": bbox,
                "score": round(score, 4),
                "scale_used": scale,
                "initial_size": (w, h),
                "importance_stats": {
                    "mean": round(float(np.mean(region_map)), 4),
                    "max": round(float(np.max(region_map)), 4)
                }
            }
            
            # Export
            sample_id = exporter.export_sample(normalized_img, metadata)
            samples_for_vis.append(normalized_img)
            
        total_rejections += image_rejections
        print(f"Exported {len(samples_for_vis)} samples. Rejected {image_rejections} due to quality.")
        
        # Visualization
        vis_display = img.copy()
        for p in final_regions:
            x, y, w, h = p['bbox']
            cv2.rectangle(vis_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        # Create a preview of normalized samples (up to 5)
        if samples_for_vis:
            previews = samples_for_vis[:5]
            # Ensure they are all the same size (they should be)
            preview_strip = np.hstack(previews)
            # Add padding to match original image height if needed for hstack
            h_orig = img.shape[0]
            h_pre = preview_strip.shape[0]
            if h_pre < h_orig:
                pad = np.zeros((h_orig - h_pre, preview_strip.shape[1], 3), dtype=np.uint8)
                preview_strip = np.vstack([preview_strip, pad])
            
            # Save debug view
            combined = np.hstack([vis_display, preview_strip[:, :vis_display.shape[1]]])
            
            # Upscale for better visibility (eyes) while keeping pixel art sharp
            h_vis, w_vis = combined.shape[:2]
            upscale_factor = int(round(NATIVE_SCALING_FACTOR))
            combined_up = cv2.resize(combined, (w_vis * upscale_factor, h_vis * upscale_factor), 
                                     interpolation=cv2.INTER_NEAREST)
            
            save_path = output_dir / f"final_{filename}"
            cv2.imwrite(str(save_path), combined_up)
            print(f"Saved upscaled final debug visualization to {save_path}")

    # Finalize
    exporter.save_metadata()
    print("\n" + "="*50)
    print(f"Normalization & Export Part 3 Complete.")
    print(f"Total Quality Rejections: {total_rejections}")

if __name__ == "__main__":
    main()
