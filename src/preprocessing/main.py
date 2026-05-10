import os
import cv2
import numpy as np
import sys
import logging
import traceback
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Set logging level to WARNING to suppress module-level logs and print purely from orchestrator
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.WARNING)

from src.preprocessing.scaler import scale_image
from src.preprocessing.masker import generate_priority_mask
from src.preprocessing.cropper import extract_priority_regions, crop_regions
from src.preprocessing.cropper.export import DatasetExporter
from src.config.config import DATASET_DIR, TEST_IMAGE_DIR, SCALE_FACTOR

VALIDATION_MODE = True
VALIDATION_LIMIT = 5

def run_pipeline():
    """
    Master pipeline orchestration:
    Load -> Scale -> Mask -> Extract Regions -> Crop -> Export
    """
    image_dir = Path(TEST_IMAGE_DIR)
    dataset_dir = Path(DATASET_DIR)
    scaled_dir = Path("resources/scaled")
    crop_save_dir = scaled_dir / "crops"
    
    scaled_dir.mkdir(parents=True, exist_ok=True)
    crop_save_dir.mkdir(parents=True, exist_ok=True)
    
    exporter = DatasetExporter(str(dataset_dir))
    
    all_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    if not all_files:
        print(f"No images found in {image_dir}")
        return

    process_files = all_files[:VALIDATION_LIMIT] if VALIDATION_MODE else all_files
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
            scaled_save_path = scaled_dir / f"{filename.split('.')[0]}_scaled.png"
            native_img = scale_image(str(path), scale_factor=SCALE_FACTOR, save_path=str(scaled_save_path))
            h_new, w_new = native_img.shape[:2]
            
            print(f"Scaled: {w_orig}x{h_orig} -> {w_new}x{h_new}")
            
            # 2. Masker
            mask_save_path = scaled_dir / f"{filename.split('.')[0]}_priority_mask.png"
            priority_mask = generate_priority_mask(native_img, save_path=str(mask_save_path))
            
            # 3. Extract Regions
            priority_regions = extract_priority_regions(native_img, priority_mask)
            print(f"Detected {len(priority_regions)} priority regions")

            # 4. Cropper
            source_id = filename.split('.')[0]
            crops = crop_regions(image=native_img, regions=priority_regions, importance_map=priority_mask,
                                 source_id=source_id, save_dir=str(crop_save_dir))
            print(f"Extracted {len(crops)} crops")
            
            # 5. Export
            for crop_data in crops:
                crop_img = crop_data["image"]
                info = crop_data["metadata"]
                info["type"] = "native_crop"
                exporter.export_sample(crop_img, info)
                
            print("Saved successfully\n")
            
        except Exception as e:
            print(f"ERROR processing {filename}: {str(e)}")
            # print(traceback.format_exc()) # uncomment for deep debugging
            print("Skipping to next image...\n")
            continue
            
    exporter.save_metadata()
    print("="*50 + f"\nPipeline Execution Complete. Processed {total_files} files.")

if __name__ == '__main__':
    run_pipeline()
