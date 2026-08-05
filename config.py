import os
import random
import numpy as np
import torch

SEED = 42

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
MIDI_DIR = os.path.join(OUTPUT_DIR, "midi")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

for d in (OUTPUT_DIR, PLOTS_DIR, MIDI_DIR, CKPT_DIR):
    os.makedirs(d, exist_ok=True)

VOICE_NAMES = ["Soprano", "Alto", "Tenor", "Bass"]
NUM_VOICES = 4

WINDOW_SIZE = 32
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remainder is test

PAD_TOKEN = "PAD"
REST_TOKEN = "REST"

EMBED_DIM = 32
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2

BATCH_SIZE = 64
MAX_EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 20
GRAD_CLIP_NORM = 1.0
LR_SCHEDULER_PATIENCE = 10
LR_SCHEDULER_FACTOR = 0.7

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

MODEL_NAMES = ["mlp", "rnn", "lstm", "gru"]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)