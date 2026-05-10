

MIN_WINDOW_SIZE = 24
MAX_WINDOW_SIZE = 96

WINDOW_STEP = 8

MAX_OVERLAP = 0.5

MIN_REGION_SCORE = 0.35

USE_INTEGER_SCALING = True

PADDING_MODE = "center"

MAX_REGIONS_PER_IMAGE = 10

# Scaler config
INITIAL_GUESS = 5.21
BATCH_SIZE = 20

# GAN and Window Configuration
GAN_INPUT_SIZE = 64
NATIVE_SCALING_FACTOR = 5.306
WINDOW_SIZES = [32, 48, 64] # Adjusted for smaller native resolution
WINDOW_STEP = 8

# Proposal Filtering
MIN_IMPORTANCE_SCORE = 0.25
MAX_REGIONS_PER_IMAGE = 15  # Increased for Part 2 processing
MAX_OVERLAP_THRESHOLD = 0.5
SEMANTIC_DIFF_THRESHOLD = 0.15 # Variance difference to keep overlap

# Optimization
HILL_CLIMB_STEPS = 2
HILL_CLIMB_STEP_SIZE = 2
EDGE_PENALTY_WEIGHT = 0.3

# Diversity
DIVERSITY_BUCKETS = 3
SAMPLES_PER_BUCKET = 3

# Normalization & Quality
GAN_INPUT_SIZE = 64
MIN_VALID_SCORE_THRESHOLD = 0.20
REJECT_EMPTY_THRESHOLD = 0.05 # Reject if max importance is too low

# Paths
TEST_IMAGE_DIR = "resources/ashlord00"
OUTPUT_DIR = "output/optimized_regions"
DATASET_DIR = "output/dataset"

SCALE_FACTOR = 5.21
