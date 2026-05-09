import os
import cv2
import numpy as np
import sys

# Add the current directory to sys.path to import evolution
sys.path.append(os.path.dirname(__file__))
from evolution import EvolutionaryOptimizer

def main():
    # Adjust path to resources relative to project root
    # Since we are in src/preprocessing/scaler, root is ../../../
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    image_dir = os.path.join(root_dir, 'resources', 'ashlord00')
    
    if not os.path.exists(image_dir):
        print(f"Error: Could not find image directory at {image_dir}")
        return

    # Get a batch of images
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    if not image_files:
        print("No images found in the directory.")
        return
        
    # Load a large sample of images (stochastic EA will sample from these)
    sample_size = 500
    image_files = image_files[:sample_size]
    
    images = []
    for f in image_files:
        path = os.path.join(image_dir, f)
        img = cv2.imread(path)
        if img is not None:
            # Crop center 256x256 to speed up optimization significantly
            h, w = img.shape[:2]
            ch, cw = 256, 256
            if h >= ch and w >= cw:
                start_h = (h - ch) // 2
                start_w = (w - cw) // 2
                img = img[start_h:start_h+ch, start_w:start_w+cw]
            images.append(img)
            
    print(f"Loaded {len(images)} image crops for optimization.", flush=True)
    
    optimizer = EvolutionaryOptimizer(
        population_size=40,
        generations=200,
        mutation_rate=0.5,
        mutation_strength=0.3,
        min_s=5.0,
        max_s=6.0
    )
    
    print(f"Starting evolutionary optimization on {len(images)} images (batch_size=20)...", flush=True)
    best_s = optimizer.evolve(images, initial_guess=5.21, batch_size=20)
    
    print("\n" + "="*40, flush=True)
    print(f"Final Estimated Scaling Factor: {best_s:.6f}", flush=True)
    print("="*40, flush=True)
    
    # Verification: Show loss for the found S
    loss = optimizer.compute_loss(best_s, images)
    print(f"Average Loss: {loss:.6f}", flush=True)

if __name__ == "__main__":
    main()
