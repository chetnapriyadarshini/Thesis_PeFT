import os
import random
import numpy as np
import torch

# ── Reproducibility ──────────────────────────────────────────
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Paths ─────────────────────────────────────────────────────
# Automatically works on both local and Kaggle
if os.path.exists("/kaggle/working"):
    BASE_DIR = "/kaggle/working"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR        = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Model config ──────────────────────────────────────────────
DISTILBERT_MODEL = "distilbert-base-uncased"
LLAMA_MODEL      = "meta-llama/Llama-3.2-3B-Instruct"

# ── Training hyperparameters ──────────────────────────────────
LEARNING_RATE = 2e-4
BATCH_SIZE    = 16
NUM_EPOCHS    = 5
MAX_LENGTH    = 128

# ── Run on import ─────────────────────────────────────────────
set_seed()
print(f"Config loaded |  BASE_DIR: {BASE_DIR}")