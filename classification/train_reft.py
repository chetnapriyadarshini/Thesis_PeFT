# =============================================================================
# classification/train_reft.py
# DistilBERT + ReFT (LoReFT) — Mental Health Stress Classification
#
# Key difference from train_lora.py:
#   LoRA  → modifies weight matrices (W = W₀ + AB)
#   ReFT  → modifies hidden representations at specific layers/positions
#           Φ(h) = h + Rᵀ(Wh + b − Rh)  where R is a low-rank projection
#
# This script is kept as close as possible to train_lora.py so that
# any performance difference is attributable to the fine-tuning method only.
# =============================================================================

import os, sys, json, random, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Allow running from repo root ──────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)
from pyreft import (
    ReftConfig,
    LoreftIntervention,
    get_reft_model,
    ReftTrainerForSequenceClassification,
)
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score,
)

import config   # repo-level config.py

# =============================================================================
# 0.  Reproducibility
# =============================================================================
set_seed(config.SEED)
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)

# =============================================================================
# 1.  Paths & hyper-parameters
# =============================================================================
DATA_DIR   = os.path.join(config.BASE_DIR, "data", "classification")
CKPT_DIR   = os.path.join(config.BASE_DIR, "checkpoints", "reft_classification")
OUTPUT_DIR = os.path.join(config.BASE_DIR, "results", "reft_classification")

os.makedirs(CKPT_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"

# ReFT hyper-parameters
# DistilBERT has 6 transformer layers (0–5) — intervene on all
REFT_LAYERS       = list(range(6))
REFT_RANK         = 4       # low-rank dimension of the intervention subspace
                            # LoRA used r=8; ReFT r=4 still gives ~10x fewer params
REFT_POSITIONS    = "f1+l1" # intervene at first token (CLS) + last token
                            # "f1" = first token only is also valid for classification

# Training hyper-parameters — identical to LoRA for fair comparison
LEARNING_RATE  = 2e-4
BATCH_SIZE     = 32
NUM_EPOCHS     = 10
WEIGHT_DECAY   = 0.01
WARMUP_RATIO   = 0.1

# =============================================================================
# 2.  Load dataset & class weights
# =============================================================================
print("\n── Loading classification dataset ──────────────────────────────────────")
dataset = load_from_disk(DATA_DIR)
print(dataset)

label_df    = pd.read_csv(os.path.join(DATA_DIR, "label_names.csv"))
id2label    = dict(zip(label_df["id"].astype(int), label_df["name"]))
label2id    = {v: k for k, v in id2label.items()}
num_labels  = len(id2label)
print(f"Labels ({num_labels}): {id2label}")

with open(os.path.join(DATA_DIR, "class_weights.json")) as f:
    cw_dict = json.load(f)

class_weights = torch.tensor(
    [cw_dict[str(i)] for i in range(num_labels)],
    dtype=torch.float,
)
print(f"Class weights: {class_weights.tolist()}")

# =============================================================================
# 3.  Tokeniser & tokenisation
# =============================================================================
print("\n── Tokenising ──────────────────────────────────────────────────────────")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=256,
        padding=False,
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized.set_format("torch")
print(tokenized)

# =============================================================================
# 4.  Build DistilBERT + ReFT model
# =============================================================================
print("\n── Building DistilBERT + ReFT (LoReFT) ────────────────────────────────")

base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

# ── ReftConfig ───────────────────────────────────────────────────────────────
# Each entry in representations defines ONE intervention:
#   layer     → which transformer layer to intervene on
#   component → "block_output" maps to the layer's residual stream output
#   intervention → LoReFT: Φ(h) = h + Rᵀ(Wh + b − Rh)
#
# We apply interventions to all 6 DistilBERT layers.
# Positions "f1+l1" means first token (CLS) and last token — standard for
# classification tasks in the ReFT paper.
# =============================================================================
reft_config = ReftConfig(
    representations=[
        {
            "layer"             : l,
            "component"         : "block_output",
            "low_rank_dimension": REFT_RANK,
            "intervention"      : LoreftIntervention(
                embed_dim        = base_model.config.dim,  # DistilBERT hidden size = 768
                low_rank_dimension = REFT_RANK,
            ),
        }
        for l in REFT_LAYERS
    ]
)

reft_model = get_reft_model(base_model, reft_config)
reft_model.set_device("cuda" if torch.cuda.is_available() else "cpu")
reft_model.print_trainable_parameters()

# =============================================================================
# 5.  Custom ReFT Trainer with weighted cross-entropy loss
# =============================================================================
class WeightedReftTrainer(ReftTrainerForSequenceClassification):
    """ReftTrainerForSequenceClassification with class-weighted CE loss."""

    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss    = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# =============================================================================
# 6.  Evaluation metrics  (identical to LoRA for fair comparison)
# =============================================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    probs = torch.softmax(torch.tensor(logits, dtype=torch.float), dim=-1).numpy()

    f1        = f1_score(labels, preds, average="weighted", zero_division=0)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall    = recall_score(labels, preds, average="weighted", zero_division=0)
    accuracy  = accuracy_score(labels, preds)

    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="weighted")
    except ValueError:
        auc = float("nan")

    return {
        "f1":        round(f1, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "accuracy":  round(accuracy, 4),
        "auc":       round(auc, 4),
    }

# =============================================================================
# 7.  Confusion matrix helper  (identical to LoRA)
# =============================================================================
def plot_confusion_matrix(trainer, dataset, id2label, output_dir, split="test"):
    from sklearn.metrics import confusion_matrix

    predictions = trainer.predict(dataset)
    preds  = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    label_names = [id2label[i] for i in range(len(id2label))]

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
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-9)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                text_colour = "white" if data_norm[i, j] > 0.5 else "black"
                ax.text(
                    j + 0.5, i + 0.5,
                    f"{data[i, j]:{fmt}}",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color=text_colour,
                )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.suptitle(
        f"DistilBERT + ReFT — {split.capitalize()} Set",
        fontsize=15, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    fig_path = os.path.join(output_dir, f"confusion_matrix_{split}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix saved → {fig_path}")

    wandb.log({f"confusion_matrix_{split}": wandb.Image(fig)})
    plt.close(fig)

    return preds, labels

# =============================================================================
# 8.  W&B initialisation
# =============================================================================
wandb.init(
    project="peft-mental-health",
    name="distilbert-reft-classification",
    config={
        "model":          MODEL_NAME,
        "method":         "ReFT (LoReFT)",
        "reft_rank":      REFT_RANK,
        "reft_layers":    REFT_LAYERS,
        "reft_positions": REFT_POSITIONS,
        "lr":             LEARNING_RATE,
        "batch_size":     BATCH_SIZE,
        "epochs":         NUM_EPOCHS,
        "weight_decay":   WEIGHT_DECAY,
        "warmup_ratio":   WARMUP_RATIO,
        "num_labels":     num_labels,
        "seed":           config.SEED,
    },
    tags=["reft", "loreft", "distilbert", "classification", "mental-health"],
)

# =============================================================================
# 9.  TrainingArguments  (identical to LoRA for fair comparison)
# =============================================================================
training_args = TrainingArguments(
    output_dir=CKPT_DIR,

    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",

    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    fp16=torch.cuda.is_available(),

    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=50,
    report_to="wandb",

    seed=config.SEED,
    data_seed=config.SEED,
)

# =============================================================================
# 10.  Train
# =============================================================================
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = WeightedReftTrainer(
    class_weights=class_weights,
    model=reft_model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["val"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("\n── Starting ReFT training ───────────────────────────────────────────────")
trainer.train()

# =============================================================================
# 11.  Final evaluation on held-out test set
# =============================================================================
print("\n── Evaluating on test set ──────────────────────────────────────────────")
test_results = trainer.evaluate(eval_dataset=tokenized["test"])
print(test_results)

results_path = os.path.join(OUTPUT_DIR, "test_results.json")
with open(results_path, "w") as f:
    json.dump(test_results, f, indent=2)
print(f"Test results saved → {results_path}")

# =============================================================================
# 12.  Confusion matrix on test set
# =============================================================================
print("\n── Generating confusion matrix ─────────────────────────────────────────")
plot_confusion_matrix(
    trainer=trainer,
    dataset=tokenized["test"],
    id2label=id2label,
    output_dir=OUTPUT_DIR,
    split="test",
)

# =============================================================================
# 13.  Save final ReFT model
# =============================================================================
reft_model_path = os.path.join(OUTPUT_DIR, "reft_model")
reft_model.save(reft_model_path)
tokenizer.save_pretrained(reft_model_path)
print(f"ReFT model saved → {reft_model_path}")

wandb.finish()
print("\n✅  ReFT classification training complete.")