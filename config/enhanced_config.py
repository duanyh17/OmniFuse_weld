# enhanced_config.py
# Enhanced configuration with current modality emphasis

import os

DATASET_ROOT = r"E:\11_weld_data\dataset"

# Three modality paths
IMAGE_ROOT   = DATASET_ROOT + r"\image"
SOUND_ROOT   = DATASET_ROOT + r"\sound"  
CURRENT_ROOT = DATASET_ROOT + r"\current"

# Six classes
CLASSES = [
    "burn_through",
    "lack_of_penetration", 
    "misalignment",
    "normal",
    "over_penetration",
    "stomata"
]

# Model architecture parameters
NUM_CLASSES = len(CLASSES)
IMAGE_DIM = 512
SOUND_DIM = 256
CURRENT_LEN = 100  # Will be auto-detected if needed
CURRENT_DIM = 128

# CRA parameters
K = 5  # CRA residual adapters

# Enhanced fusion parameters
BETA = 0.3  # Reduced beta for more balanced fusion
CURRENT_BOOST = 2.5  # Boost factor for current modality
ALPHA = 0.1  # DWFuse auxiliary coefficient

# Loss function weights - Enhanced for current emphasis
LAMBDA1 = 1.0   # backward loss
LAMBDA2 = 15.0  # DWFuse loss (increased)
LAMBDA3 = 0.5   # TLA loss
LAMBDA_CURRENT = 3.0  # Additional current loss weight

# Training parameters
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50  # Increased epochs for better current learning

# Current-specific training parameters
CURRENT_PRETRAIN_EPOCHS = 10  # Pretrain current encoder
CURRENT_LR_MULTIPLIER = 2.0   # Higher learning rate for current components
CURRENT_DROPOUT = 0.1         # Dropout for current encoder
CURRENT_L2_REG = 1e-5         # L2 regularization for current features

# Three-stage training schedule
STAGE1_EPOCHS = 15  # Individual modality training
STAGE2_EPOCHS = 20  # Fusion training with current emphasis
STAGE3_EPOCHS = 15  # Fine-tuning with balanced objectives

# Current pattern recognition parameters
USE_TEMPORAL_ATTENTION = True
USE_FREQUENCY_FEATURES = True
CURRENT_CONV_CHANNELS = [16, 32, 64]
CURRENT_KERNEL_SIZES = [7, 5, 3]

# Evaluation parameters  
EVAL_BATCH_SIZE = 32
SAVE_BEST_MODEL = True
PATIENCE = 10  # Early stopping patience

# Logging and checkpointing
LOG_INTERVAL = 10
SAVE_INTERVAL = 5
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)