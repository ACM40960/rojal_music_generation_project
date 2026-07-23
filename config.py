import os
import torch

# Random Seed for Reproducibility
SEED = 42

# Data & Sequence Settings
WINDOW_LENGTH = 32
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Model Architecture Dimensions
PITCH_EMBED_DIM = 64
DURATION_EMBED_DIM = 32
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5  # Early stopping patience
GRAD_CLIP_NORM = 1.0

# Special Tokens
PAD_TOKEN = "<PAD>"
REST_TOKEN = "<REST>"

# Paths
BASE_DIR = r"d:\JS Batch Chorales"
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MIDI_DIR = os.path.join(OUTPUT_DIR, "midi")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")

# Create Directories if not present
for d in [CHECKPOINT_DIR, OUTPUT_DIR, MIDI_DIR, PLOTS_DIR, METRICS_DIR]:
    os.makedirs(d, exist_ok=True)

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
