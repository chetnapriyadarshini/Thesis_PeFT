# =============================================================================
# classification/plot_confusion_matrix_only.py
# Standalone script — loads saved LoRA adapter and replots confusion matrix.
# Run this instead of the full train_lora.py when you only need the plot.
# =============================================================================

import os, sys, json, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "disabled"   # no W&B needed for plotting

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import PeftModel
from sklearn.metrics import confusion_matrix

# =============================================================================
# ── PATHS — adjust if your Kaggle dataset is mounted elsewhere ───────────────
# =============================================================================
import config

BASE_DIR   = config.BASE_DIR
DATA_DIR   = os.path.join(BASE_DIR, "data", "classification")
ADAPTER_DIR = os.path.join(BASE_DIR, "results", "lora_classification", "lora_adapter")
OUTPUT_DIR  = os.path.join(BASE_DIR, "results", "lora_classification")

print(f"BASE_DIR    : {BASE_DIR}")
print(f"DATA_DIR    : {DATA_DIR}")
print(f"ADAPTER_DIR : {ADAPTER_DIR}")

# =============================================================================
# 1.  Load dataset
# =============================================================================
print("\n── Loading dataset ─────────────────────────────────────────────────────")
dataset = load_from_disk(DATA_DIR)
print(dataset)

label_df  = __import__("pandas").read_csv(os.path.join(DATA_DIR, "label_names.csv"))
id2label  = dict(zip(label_df["id"].astype(int), label_df["name"]))
label2id  = {v: k for k, v in id2label.items()}
num_labels = len(id2label)
print(f"Labels: {id2label}")

# =============================================================================
# 2.  Tokenise test split only
# =============================================================================
print("\n── Tokenising test split ───────────────────────────────────────────────")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256, padding=False)

test_tok = dataset["test"].map(
    tokenize, batched=True, remove_columns=["text"], num_proc=1
)
test_tok.set_format("torch")
print(f"Test samples: {len(test_tok)}")

# =============================================================================
# 3.  Load base model + LoRA adapter (no training needed)
# =============================================================================
print("\n── Loading LoRA adapter ────────────────────────────────────────────────")
base_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()
print("Model loaded.")

# =============================================================================
# 4.  Run inference with a minimal Trainer (eval only, no logging)
# =============================================================================
print("\n── Running inference on test set ───────────────────────────────────────")
eval_args = TrainingArguments(
    output_dir="/tmp/cm_only",
    per_device_eval_batch_size=64,
    fp16=torch.cuda.is_available(),
    report_to="none",
    disable_tqdm=False,
)

trainer = Trainer(
    model=model,
    args=eval_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
)

preds_out = trainer.predict(test_tok)
preds  = np.argmax(preds_out.predictions, axis=-1)
labels = preds_out.label_ids

# =============================================================================
# 5.  Plot confusion matrix (with text-colour fix applied)
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

    # ── Text colour fix: threshold raised to 0.7 (was 0.5 — caused white-on-dark bug)
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

plt.suptitle(
    "DistilBERT + LoRA — Test Set",
    fontsize=15, fontweight="bold", y=1.01,
)
plt.tight_layout()

fig_path = os.path.join(OUTPUT_DIR, "confusion_matrix_test.png")
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\nConfusion matrix saved → {fig_path}")
plt.show()

# =============================================================================
# 6.  Print per-class recall for easy copying into thesis
# =============================================================================
print("\n── Per-class recall ────────────────────────────────────────────────────")
for i, name in enumerate(label_names):
    recall = cm_norm[i, i]
    print(f"  {name:<15} recall = {recall:.4f}")
