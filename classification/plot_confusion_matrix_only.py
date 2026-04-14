# =============================================================================
# classification/plot_confusion_matrix_only.py
# Standalone — loads best training checkpoint and plots confusion matrix.
# Run instead of the full train_lora.py when you only need the plot.
# =============================================================================

import os, sys, json, glob, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "disabled"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

# =============================================================================
# ── PATHS ────────────────────────────────────────────────────────────────────
# =============================================================================
import config

BASE_DIR   = config.BASE_DIR
DATA_DIR   = os.path.join(BASE_DIR, "data", "classification")
CKPT_DIR   = os.path.join(BASE_DIR, "checkpoints", "lora_classification")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "lora_classification")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"BASE_DIR  : {BASE_DIR}")
print(f"CKPT_DIR  : {CKPT_DIR}")

# =============================================================================
# 1.  Find the best checkpoint
# =============================================================================
def find_best_checkpoint(ckpt_dir):
    """
    Returns the path to the best checkpoint.
    Prefers trainer_state.json best_model_checkpoint field.
    Falls back to the highest-numbered checkpoint folder.
    """
    # Check if trainer_state.json exists in any checkpoint
    state_files = glob.glob(os.path.join(ckpt_dir, "checkpoint-*", "trainer_state.json"))
    if state_files:
        # Use the most recent trainer_state to find best_model_checkpoint
        state_files.sort(key=os.path.getmtime)
        with open(state_files[-1]) as f:
            state = json.load(f)
        best = state.get("best_model_checkpoint")
        if best and os.path.isdir(best):
            print(f"Best checkpoint (from trainer_state): {best}")
            return best

    # Fallback: highest-numbered checkpoint
    checkpoints = glob.glob(os.path.join(ckpt_dir, "checkpoint-*"))
    if checkpoints:
        best = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        print(f"Best checkpoint (highest step): {best}")
        return best

    return None

ckpt_path = find_best_checkpoint(CKPT_DIR)
if ckpt_path is None:
    raise FileNotFoundError(
        f"No checkpoint found in {CKPT_DIR}.\n"
        "Make sure the training run completed and checkpoints were saved to Kaggle output."
    )

# =============================================================================
# 2.  Load dataset (test split only)
# =============================================================================
print("\n── Loading dataset ─────────────────────────────────────────────────────")
dataset = load_from_disk(DATA_DIR)
print(dataset)

label_df   = pd.read_csv(os.path.join(DATA_DIR, "label_names.csv"))
id2label   = dict(zip(label_df["id"].astype(int), label_df["name"]))
label2id   = {v: k for k, v in id2label.items()}
num_labels = len(id2label)
print(f"Labels: {id2label}")

# =============================================================================
# 3.  Tokenise test split
# =============================================================================
print("\n── Tokenising test split ───────────────────────────────────────────────")
tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256, padding=False)

test_tok = dataset["test"].map(
    tokenize, batched=True, remove_columns=["text"], num_proc=1
)
test_tok.set_format("torch")
print(f"Test samples: {len(test_tok)}")

# =============================================================================
# 4.  Load model from checkpoint (full weights — includes classifier head)
# =============================================================================
print(f"\n── Loading model from checkpoint ───────────────────────────────────────")
print(f"   {ckpt_path}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Device: {device}")

model = AutoModelForSequenceClassification.from_pretrained(
    ckpt_path,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
model.to(device)
model.eval()
print("Model loaded — all weights including classifier head restored from checkpoint.")

# =============================================================================
# 5.  Run inference (no Trainer — avoids version-compatibility issues)
# =============================================================================
print("\n── Running inference ───────────────────────────────────────────────────")
collator = DataCollatorWithPadding(tokenizer)
loader = DataLoader(test_tok, batch_size=64, collate_fn=collator)

all_preds, all_labels = [], []

with torch.no_grad():
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch   = batch["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds   = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels_batch.numpy())

preds  = np.array(all_preds)
labels = np.array(all_labels)
print(f"Inference complete. Samples: {len(preds)}")

# =============================================================================
# 6.  Plot confusion matrix
# =============================================================================
print("\n── Plotting confusion matrix ────────────────────────────────────────────")
label_names = [id2label[i] for i in range(num_labels)]
cm      = confusion_matrix(labels, preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

for ax, data, title, fmt in zip(
    axes,
    [cm, cm_norm],
    ["Confusion Matrix — Raw Counts", "Confusion Matrix — Normalised (Recall)"],
    ["d", ".2f"],
):
    sns.heatmap(
        data,
        annot=False,
        fmt=fmt,
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        linewidths=0.5,
        ax=ax,
    )

    # Text-colour fix: 0.7 threshold (was 0.5 — caused white-on-dark-blue unreadable cells)
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text_colour = "white" if data_norm[i, j] > 0.7 else "black"
            ax.text(j + 0.5, i + 0.5, f"{data[i, j]:{fmt}}",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_colour)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

plt.suptitle("DistilBERT + LoRA — Test Set", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()

fig_path = os.path.join(OUTPUT_DIR, "confusion_matrix_test.png")
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\nConfusion matrix saved → {fig_path}")
plt.show()

# =============================================================================
# 7.  Print per-class recall — copy these numbers into the thesis
# =============================================================================
print("\n── Per-class recall ────────────────────────────────────────────────────")
for i, name in enumerate(label_names):
    recall = cm_norm[i, i]
    count  = cm[i].sum()
    print(f"  {name:<15} recall = {recall:.4f}   (n={count})")

print("\nDone.")
