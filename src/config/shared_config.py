import logging
from src.config.gan import GAN_INPUT_SIZE

# Paths
TEST_IMAGE_DIR = "resources/ashlord00"
DEBUG_DIR = "debug"

DATASET_DIR = f"resources/{GAN_INPUT_SIZE}x{GAN_INPUT_SIZE}"

# Shared Preprocessing Constants
USE_INTEGER_SCALING = True
PADDING_MODE = "center"

# Validation & Debugging
VALIDATION_MODE = False
VALIDATION_SAMPLE_COUNT = 5
SAVE_DEBUG_OVERLAYS = True
SAVE_INTERMEDIATE_OUTPUTS = True
VERBOSE_LOGGING = True
