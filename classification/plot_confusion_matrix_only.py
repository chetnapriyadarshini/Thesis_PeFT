# =============================================================================
# classification/plot_confusion_matrix_only.py
# Standalone — loads best checkpoint and plots confusion matrix.
# Manually merges LoRA weights to avoid PEFT version mismatches.
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

import config

# =============================================================================
# ── PATHS
# =============================================================================
BASE_DIR   = config.BASE_DIR
DATA_DIR   = os.path.join(BASE_DIR, "data", "classification")
CKPT_DIR   = os.path.join(BASE_DIR, "checkpoints", "lora_classification")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "lora_classification")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LoRA hyper-parameters (must match what was used during training)
LORA_R     = 8
LORA_ALPHA = 16

print(f"BASE_DIR  : {BASE_DIR}")
print(f"CKPT_DIR  : {CKPT_DIR}")

# =============================================================================
# 1.  Find best checkpoint
# =============================================================================
def find_best_checkpoint(ckpt_dir):
    state_files = glob.glob(os.path.join(ckpt_dir, "checkpoint-*", "trainer_state.json"))
    if state_files:
        state_files.sort(key=os.path.getmtime)
        with open(state_files[-1]) as f:
            state = json.load(f)
        best = state.get("best_model_checkpoint")
        if best and os.path.isdir(best):
            print(f"Best checkpoint (trainer_state): {best}")
            return best
    checkpoints = glob.glob(os.path.join(ckpt_dir, "checkpoint-*"))
    if checkpoints:
        best = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        print(f"Best checkpoint (highest step): {best}")
        return best
    return None

ckpt_path = find_best_checkpoint(CKPT_DIR)
if ckpt_path is None:
    raise FileNotFoundError(f"No checkpoint found in {CKPT_DIR}")

# =============================================================================
# 2.  Load dataset
# =============================================================================
print("\n── Loading dataset ─────────────────────────────────────────────────────")
dataset  = load_from_disk(DATA_DIR)
label_df = pd.read_csv(os.path.join(DATA_DIR, "label_names.csv"))
id2label = dict(zip(label_df["id"].astype(int), label_df["name"]))
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)
print(f"Labels: {id2label}")

# =============================================================================
# 3.  Tokenise test split
# =============================================================================
print("\n── Tokenising test split ───────────────────────────────────────────────")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256, padding=False)
test_tok = dataset["test"].map(tokenize, batched=True, remove_columns=["text"], num_proc=1)
test_tok.set_format("torch")
print(f"Test samples: {len(test_tok)}")

# =============================================================================
# 4.  Load checkpoint weights and manually merge LoRA into base model
#     (avoids PEFT version mismatches entirely)
# =============================================================================
def load_checkpoint_weights(ckpt_path):
    """Load raw state dict from a checkpoint directory."""
    for fname in ["adapter_model.safetensors", "adapter_model.bin",
                  "model.safetensors", "pytorch_model.bin"]:
        fpath = os.path.join(ckpt_path, fname)
        if os.path.exists(fpath):
            print(f"  Loading weights from: {fname}")
            if fpath.endswith(".safetensors"):
                from safetensors.torch import load_file
                return load_file(fpath)
            else:
                return torch.load(fpath, map_location="cpu")
    raise FileNotFoundError(f"No weights file found in {ckpt_path}\nFiles: {os.listdir(ckpt_path)}")


def merge_lora_into_model(model, ckpt_weights, lora_r, lora_alpha):
    """
    Manually apply LoRA delta weights and overwrite classifier head.
    Handles both old PEFT key format (base_model.model.* prefix)
    and new format (no prefix).
    """
    scaling = lora_alpha / lora_r  # 2.0

    base_sd = model.state_dict()
    updated = set()

    # ── Normalise checkpoint keys: strip 'base_model.model.' prefix if present
    norm = {}
    for k, v in ckpt_weights.items():
        new_k = k.replace("base_model.model.", "")
        # Also handle modules_to_save wrapper (new PEFT format)
        new_k = new_k.replace(".modules_to_save.default", "")
        norm[new_k] = v

    # ── Apply classifier / pre_classifier weights directly
    for key in ["pre_classifier.weight", "pre_classifier.bias",
                "classifier.weight",     "classifier.bias"]:
        if key in norm:
            base_sd[key] = norm[key].to(base_sd[key].dtype)
            updated.add(key)
            print(f"  Applied: {key}")
        else:
            print(f"  WARNING: {key} not found in checkpoint — classifier head may be wrong")

    # ── Merge LoRA deltas: W_merged = W_base + (lora_B @ lora_A) * scaling
    lora_a_keys = [k for k in norm if "lora_A" in k]
    for a_key in lora_a_keys:
        b_key = a_key.replace("lora_A", "lora_B")
        if b_key not in norm:
            continue

        # Build base weight key
        # e.g. distilbert.transformer.layer.0.attention.q_lin.lora_A.default.weight
        #   →  distilbert.transformer.layer.0.attention.q_lin.weight
        base_key = a_key
        for suffix in [".lora_A.default.weight", ".lora_A.weight"]:
            base_key = base_key.replace(suffix, ".weight")

        if base_key not in base_sd:
            print(f"  WARNING: base key {base_key} not found, skipping")
            continue

        lora_A = norm[a_key].float()
        lora_B = norm[b_key].float()
        delta  = (lora_B @ lora_A) * scaling

        base_sd[base_key] = (base_sd[base_key].float() + delta).to(base_sd[base_key].dtype)
        updated.add(base_key)

    print(f"  LoRA layers merged: {len(lora_a_keys)}")
    model.load_state_dict(base_sd)
    return model


print(f"\n── Merging LoRA weights from checkpoint ────────────────────────────────")
print(f"   {ckpt_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Device: {device}")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

ckpt_weights = load_checkpoint_weights(ckpt_path)
model = merge_lora_into_model(model, ckpt_weights, LORA_R, LORA_ALPHA)
model.to(device)
model.eval()
print("Model ready.")

# =============================================================================
# 5.  Run inference
# =============================================================================
print("\n── Running inference ───────────────────────────────────────────────────")
collator = DataCollatorWithPadding(tokenizer)
loader   = DataLoader(test_tok, batch_size=64, collate_fn=collator)

all_preds, all_labels = [], []
with torch.no_grad():
    for batch in loader:
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["labels"].numpy())

preds  = np.array(all_preds)
labels = np.array(all_labels)

# Quick sanity check — print prediction distribution
unique, counts = np.unique(preds, return_counts=True)
print("Prediction distribution:")
for u, c in zip(unique, counts):
    print(f"  {id2label[u]:<15} {c} ({100*c/len(preds):.1f}%)")

# =============================================================================
# 6.  Plot confusion matrix
# =============================================================================
print("\n── Plotting ────────────────────────────────────────────────────────────")
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
    sns.heatmap(data, annot=False, cmap="Blues",
                xticklabels=label_names, yticklabels=label_names,
                linewidths=0.5, ax=ax)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Use actual heatmap colour luminance to pick readable text colour
            norm_val = (data[i, j] - data.min()) / (data.max() - data.min() + 1e-9)
            rgba = plt.cm.Blues(norm_val)
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            colour = "white" if luminance < 0.5 else "black"
            ax.text(j+0.5, i+0.5, f"{data[i,j]:{fmt}}",
                    ha="center", va="center", fontsize=11, fontweight="bold", color=colour)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.tick_params(axis="x", rotation=45)

plt.suptitle("DistilBERT + LoRA — Test Set", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()

fig_path = os.path.join(OUTPUT_DIR, "confusion_matrix_test.png")
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Saved → {fig_path}")
plt.show()

# =============================================================================
# 7.  Per-class recall
# =============================================================================
print("\n── Per-class recall ────────────────────────────────────────────────────")
for i, name in enumerate(label_names):
    print(f"  {name:<15} recall = {cm_norm[i,i]:.4f}   (n={cm[i].sum()})")

print("\nDone.")
