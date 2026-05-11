import os
import cv2
import numpy as np
import sys
import logging
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessing.scaler import scale_image
from src.preprocessing.masker import generate_priority_mask
from src.preprocessing.cropper import extract_priority_regions, crop_regions
from src.preprocessing.cropper.export import DatasetExporter
from src.config.shared_config import (DATASET_DIR, TEST_IMAGE_DIR,
                                      VALIDATION_MODE, VALIDATION_SAMPLE_COUNT, VERBOSE_LOGGING, DEBUG_DIR)
from src.config.scaler import SCALING_FACTOR

# Set logging level dynamically based on validation settings
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    
if VALIDATION_MODE and VERBOSE_LOGGING:
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=logging.WARNING)

def run_pipeline():
    """
    Master pipeline orchestration:
    Load -> Scale -> Mask -> Extract Regions -> Crop -> Export
    """
    image_dir = Path(TEST_IMAGE_DIR)
    dataset_dir = Path(DATASET_DIR)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Conditional debug directory
    debug_dir = Path(DEBUG_DIR) if VALIDATION_MODE else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        
    exporter = DatasetExporter(str(dataset_dir))
    
    all_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    if not all_files:
        print(f"No images found in {image_dir}")
        return

    process_files = all_files[:VALIDATION_SAMPLE_COUNT] if VALIDATION_MODE else all_files
    total_files = len(process_files)

    print(f"\nStarting Preprocessing Pipeline (Validation Mode: {VALIDATION_MODE})\n" + "="*50)

    for i, filename in enumerate(process_files, 1):
        try:
            print(f"[{i}/{total_files}] Processing {filename}")
            path = image_dir / filename
            
            # Read original shape for logging
            orig_img = cv2.imread(str(path))
            if orig_img is None:
                raise ValueError("Image could not be read")
            h_orig, w_orig = orig_img.shape[:2]
            
            # 1. Scaler
            scaled_save_path = str(debug_dir / f"{filename.split('.')[0]}_scaled.png") if debug_dir else None
            native_img = scale_image(str(path), scale_factor=SCALING_FACTOR, save_path=scaled_save_path)
            h_new, w_new = native_img.shape[:2]
            
            if VALIDATION_MODE:
                print(f"Scaled: {w_orig}x{h_orig} -> {w_new}x{h_new}")
            
            # 2. Masker
            mask_save_path = str(debug_dir / f"{filename.split('.')[0]}_priority_mask.png") if debug_dir else None
            priority_mask = generate_priority_mask(native_img, save_path=mask_save_path)
            
            # 3. Extract Regions
            initial_proposals, priority_regions = extract_priority_regions(native_img, priority_mask)
            if VALIDATION_MODE:
                print(f"Detected {len(priority_regions)} priority regions")

            # 4. Cropper
            source_id = filename.split('.')[0]
            crops = crop_regions(image=native_img, regions=priority_regions, importance_map=priority_mask,
                                 source_id=source_id, save_dir=None) # We don't save raw crops to disk here
            if VALIDATION_MODE:
                print(f"Extracted {len(crops)} crops")
            
            # 5. Export and Debug Overlays
            samples_for_vis = []
            for crop_data in crops:
                crop_img = crop_data["image"]
                info = crop_data["metadata"]
                info["type"] = "native_crop"
                exporter.export_sample(crop_img, info)
                
                if VALIDATION_MODE:
                    samples_for_vis.append(crop_img)
                    
            if VALIDATION_MODE and debug_dir:
                # Debug a) Initial randomly placed boxes
                vis_initial = native_img.copy()
                for p in initial_proposals:
                    x, y, w, h = p['bbox']
                    cv2.rectangle(vis_initial, (x, y), (x+w, y+h), (0, 0, 255), 1)
                
                # Debug b) Final optimized regions
                vis_final = native_img.copy()
                for p in priority_regions:
                    x, y, w, h = p['bbox']
                    cv2.rectangle(vis_final, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                # Upscale for better visibility (eyes) while keeping pixel art sharp
                upscale_factor = int(round(SCALING_FACTOR))
                h_vis, w_vis = native_img.shape[:2]
                vis_initial_up = cv2.resize(vis_initial, (w_vis * upscale_factor, h_vis * upscale_factor), interpolation=cv2.INTER_NEAREST)
                vis_final_up = cv2.resize(vis_final, (w_vis * upscale_factor, h_vis * upscale_factor), interpolation=cv2.INTER_NEAREST)
                
                cv2.imwrite(str(debug_dir / f"{filename.split('.')[0]}_initial_boxes.png"), vis_initial_up)
                cv2.imwrite(str(debug_dir / f"{filename.split('.')[0]}_final_regions.png"), vis_final_up)
                
                # Debug c) Cutouts combined
                if samples_for_vis:
                    previews = samples_for_vis[:5]
                    preview_strip = np.hstack(previews)
                    cv2.imwrite(str(debug_dir / f"{filename.split('.')[0]}_cutouts.png"), preview_strip)
                
                print("Saved successfully\n")
            
        except Exception as e:
            print(f"ERROR processing {filename}: {str(e)}")
            if VALIDATION_MODE and VERBOSE_LOGGING:
                import traceback
                print(traceback.format_exc())
            print("Skipping to next image...\n")
            continue
            
    exporter.save_metadata()
    print("="*50 + f"\nPipeline Execution Complete. Processed {total_files} files.")

if __name__ == '__main__':
    run_pipeline()
