import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from src.gan import GAN
from src.config.shared_config import DATASET_DIR
from src.config.gan import (
    GAN_INPUT_SIZE,
    GAN_CHANNELS,
    GAN_BATCH_SIZE,
    GAN_NUM_EPOCHS,
    GAN_LEARNING_RATE
)

def main():
    print("Initializing GAN Training...")
    print(f"Dataset Directory: {DATASET_DIR}")
    print(f"Input Size: {GAN_INPUT_SIZE}x{GAN_INPUT_SIZE}")
    
    if not os.path.exists(DATASET_DIR):
        print(f"Warning: Dataset directory {DATASET_DIR} does not exist.")
        print("Please ensure preprocessed data is available before training.")
        return
        
    models_dir = "models"
    samples_dir = "samples"
    
    # Check if a trained model already exists
    resume = False
    if os.path.exists(os.path.join(models_dir, 'generator.pth')):
        print("Warning: A trained model already exists in models/.")
        user_input = input("Do you want to resume training (r), overwrite (o), or cancel (c)? [r/o/c]: ")
        if user_input.lower() == 'r' or user_input == '':
            resume = True
        elif user_input.lower() == 'o':
            resume = False
        else:
            print("Training cancelled.")
            return

    # Using config constants
    gan = GAN(nc=GAN_CHANNELS)
    
    print("Starting training...")
    print("You can safely press Ctrl+C to stop training at any time. Models are saved after each epoch.")
    # You are leaving this overnight, so num_epochs is set to 200 by default. 
    # Feel free to adjust as needed.
    gan.fit(
        dataset_dir=DATASET_DIR,
        image_size=GAN_INPUT_SIZE,
        batch_size=GAN_BATCH_SIZE,
        num_epochs=GAN_NUM_EPOCHS, 
        lr=GAN_LEARNING_RATE,
        models_dir=models_dir,
        samples_dir=samples_dir,
        resume=resume
    )
    
    print("Training complete.")

if __name__ == '__main__':
    main()