import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict

class DatasetExporter:
    """
    Handles the structured export of GAN training samples and metadata.
    """
    
    def __init__(self, dataset_dir: str):
        self.base_dir = Path(dataset_dir)
        self.images_dir = self.base_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = {}
        self.counter = 0

    def export_sample(self, img: np.ndarray, info: Dict):
        """
        Saves a single image sample and registers its metadata.
        """
        self.counter += 1
        sample_id = f"sample_{self.counter:04d}"
        filename = f"{sample_id}.png"
        
        # Save image
        save_path = self.images_dir / filename
        cv2.imwrite(str(save_path), img)
        
        # Store metadata
        self.metadata[sample_id] = info
        self.metadata[sample_id]["filename"] = filename
        
        return sample_id

    def save_metadata(self):
        """Writes the accumulated metadata to a JSON file."""
        metadata_path = self.base_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        print(f"Dataset metadata saved to {metadata_path}")
        print(f"Total samples exported: {self.counter}")
