import os
import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessing.scaler import get_scaled
from src.preprocessing.masker import generate_priority_mask
from src.preprocessing.cropper import get_cropped
from src.cropper.export import DatasetExporter
from src.cropper.config import DATASET_DIR, TEST_IMAGE_DIR

def run_pipeline():
    """
    Pipeline orchestration for all files:
    Image -> Scaler -> Priority Mask -> Cropper -> Export
    """
    image_dir = Path(TEST_IMAGE_DIR)
    dataset_dir = Path(DATASET_DIR)
    
    # Initialize Exporter
    exporter = DatasetExporter(str(dataset_dir))
    
    # Process first few images for validation
    all_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    test_files = all_files[:5] # Process 5 images
    
    if not test_files:
        print(f"No images found in {image_dir}")
        return

    for filename in test_files:
        print(f"\nProcessing {filename}...")
        path = image_dir / filename
        img = cv2.imread(str(path))
        if img is None:
            continue
            
        # 1. Scaler (Restore native resolution)
        native_img = get_scaled(img)
        print(f"  Scaled to native: {native_img.shape[1]}x{native_img.shape[0]}")
        
        # 2. Masker (Identify important areas)
        priority_mask = generate_priority_mask(native_img)
        
        # 3. Cropper (Extract optimized patches)
        crops = get_cropped(native_img, priority_mask)
        print(f"  Generated {len(crops)} crops.")
        
        # 4. Export
        for i, crop_img in enumerate(crops):
            # Minimal info for export sample
            info = {
                "source": filename,
                "type": "native_crop"
            }
            exporter.export_sample(crop_img, info)
            
    # Save final metadata
    exporter.save_metadata()
    print("\nPreprocessing Pipeline Complete.")

if __name__ == '__main__':
    run_pipeline()
