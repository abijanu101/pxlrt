import logging

# Paths
TEST_IMAGE_DIR = "resources/ashlord00"
DEBUG_DIR = "debug"

# GAN and Window Configuration
GAN_INPUT_SIZE = 128
NATIVE_SCALING_FACTOR = 5.306
WINDOW_SIZES = [GAN_INPUT_SIZE // 2, GAN_INPUT_SIZE]  # Dynamic window size based on GAN_INPUT_SIZE
WINDOW_STEP = 8

DATASET_DIR = f"resources/{GAN_INPUT_SIZE}x{GAN_INPUT_SIZE}"

# Shared Preprocessing Constants
SCALE_FACTOR = 5.21
USE_INTEGER_SCALING = True
PADDING_MODE = "center"

# Validation & Debugging
VALIDATION_MODE = False
VALIDATION_SAMPLE_COUNT = 5
SAVE_DEBUG_OVERLAYS = True
SAVE_INTERMEDIATE_OUTPUTS = True
VERBOSE_LOGGING = True
