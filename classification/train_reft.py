# =============================================================================
# classification/train_reft.py
# DistilBERT + LoReFT (manual) — Mental Health Stress Classification
#
# ReFT is implemented manually using PyTorch forward hooks — no pyvene/pyreft
# dependency. This avoids version conflicts and makes the implementation fully
# transparent and explainable.
#
# LoReFT formula per layer, per position:
#   h_new = h + Rᵀ(Wh + b − Rh)
#   where:
#     h  = hidden representation at the intervened position (CLS + last token)
#     R  = low-rank orthonormal projection matrix (embed_dim × rank)
#     W  = learned linear transform (embed_dim → rank)
#     b  = learned bias
#
# This script is kept as close as possible to train_lora.py so that any
# performance difference is attributable to the fine-tuning method only.
# =============================================================================

import os, sys, json, random, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]    = "3"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
os.environ["CUDA_VISIBLE_DEVICES"]    = "0"   # prevent DataParallel issues

# ── Allow running from repo root ──────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput
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
REFT_LAYERS    = list(range(6))   # all 6 DistilBERT layers
REFT_RANK      = 4                # low-rank dimension — ~10x fewer params than LoRA r=8
REFT_POSITIONS = [0, -1]          # CLS token + last token (f1+l1 in pyreft notation)

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
        padding="max_length",   # static padding required for default_data_collator
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized.set_format("torch")
print(tokenized)

# =============================================================================
# 4.  LoReFT intervention module
# =============================================================================
class LoReFTIntervention(nn.Module):
    """
    Manual implementation of LoReFT (Low-Rank Linear Subspace ReFT).

    For each intervened position (CLS + last token) in each layer:
        h_new = h + Rᵀ(Wh + b − Rh)

    Parameters
    ----------
    embed_dim : int
        Hidden size of the base model (768 for DistilBERT)
    rank : int
        Rank of the intervention subspace (REFT_RANK)
    positions : list[int]
        Token positions to intervene on. [0, -1] = CLS + last token.
    """

    def __init__(self, embed_dim: int, rank: int, positions: list):
        super().__init__()
        self.positions = positions

        # R: orthonormal projection matrix (rank × embed_dim)
        # Projects representations down to the low-rank subspace
        self.R = nn.Linear(embed_dim, rank, bias=False)
        torch.nn.init.orthogonal_(self.R.weight)  # orthonormal initialisation

        # W: learned transform (embed_dim → rank) with bias
        # Learns where the representation should move in the subspace
        self.W = nn.Linear(embed_dim, rank, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Apply LoReFT intervention to hidden states.

        Parameters
        ----------
        h : torch.Tensor  shape (batch, seq_len, embed_dim)

        Returns
        -------
        h : torch.Tensor  shape (batch, seq_len, embed_dim)
        """
        h = h.clone()   # avoid in-place modification of the computation graph
        for pos in self.positions:
            h_pos = h[:, pos, :]                    # (batch, embed_dim)

            # ── Core LoReFT formula ─────────────────────────────────────────
            # W(h_pos) → learned target in subspace  (batch, rank)
            # R(h_pos) → current position in subspace (batch, rank)
            # difference = edit needed in subspace    (batch, rank)
            # @ R.weight → project edit back to full space (batch, embed_dim)
            difference   = self.W(h_pos) - self.R(h_pos)   # (batch, rank)
            intervention = difference @ self.R.weight       # (batch, embed_dim)

            h[:, pos, :] = h_pos + intervention
            # ────────────────────────────────────────────────────────────────

        return h


# =============================================================================
# 5.  DistilBERT + LoReFT model
# =============================================================================
print("\n── Building DistilBERT + LoReFT ────────────────────────────────────────")

class DistilBertWithLoReFT(nn.Module):
    """
    DistilBERT with LoReFT interventions applied at the output of each
    transformer layer's output_layer_norm.

    All base model parameters are FROZEN. Only the LoReFT intervention
    parameters (R, W, b) are trained.

    Architecture
    ------------
    For each of DistilBERT's 6 layers:
        [transformer block] → output_layer_norm → [LoReFT hook] → next layer

    The hook intercepts the output of output_layer_norm and applies the
    LoReFT intervention before passing activations to the next layer.
    """

    def __init__(self, model_name, num_labels, id2label, label2id,
                 reft_layers, reft_rank, reft_positions):
        super().__init__()

        # ── Load base model ──────────────────────────────────────────────────
        self.base = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        # ── Freeze ALL base model parameters ─────────────────────────────────
        for param in self.base.parameters():
            param.requires_grad = False

        # ── Create one LoReFT intervention per layer ─────────────────────────
        embed_dim = self.base.config.dim   # 768 for DistilBERT
        self.interventions = nn.ModuleList([
            LoReFTIntervention(embed_dim, reft_rank, reft_positions)
            for _ in reft_layers
        ])

        # ── Register forward hooks on output_layer_norm of each layer ────────
        # The hook fires after output_layer_norm computes its output,
        # intercepts the tensor, and passes it through LoReFTIntervention
        self._hooks = []
        for idx, layer_idx in enumerate(reft_layers):
            layer_norm  = self.base.distilbert.transformer.layer[layer_idx].output_layer_norm
            intervention = self.interventions[idx]

            def make_hook(iv):
                def hook(module, input, output):
                    return iv(output)
                return hook

            handle = layer_norm.register_forward_hook(make_hook(intervention))
            self._hooks.append(handle)

        self.config = self.base.config

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Standard HuggingFace forward — hooks fire automatically inside
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return SequenceClassifierOutput(logits=outputs.logits)

    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.base.parameters())
        print(f"trainable params: {trainable:,} || "
              f"all params: {total:,} || "
              f"trainable%: {100 * trainable / total:.4f}")

    def save_interventions(self, path):
        """Save only the intervention weights — equivalent to LoRA adapter save."""
        os.makedirs(path, exist_ok=True)
        torch.save(
            {f"intervention_{i}": iv.state_dict()
             for i, iv in enumerate(self.interventions)},
            os.path.join(path, "reft_interventions.pt")
        )
        print(f"ReFT interventions saved → {path}/reft_interventions.pt")


# ── Instantiate ───────────────────────────────────────────────────────────────
model = DistilBertWithLoReFT(
    model_name     = MODEL_NAME,
    num_labels     = num_labels,
    id2label       = id2label,
    label2id       = label2id,
    reft_layers    = REFT_LAYERS,
    reft_rank      = REFT_RANK,
    reft_positions = REFT_POSITIONS,
)
model.print_trainable_parameters()

# =============================================================================
# 6.  Weighted Trainer
# =============================================================================
class WeightedTrainer(Trainer):
    """Standard HuggingFace Trainer with class-weighted CE loss."""

    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss    = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss

# =============================================================================
# 7.  Evaluation metrics  (identical to LoRA)
# =============================================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds  = np.argmax(logits, axis=-1)
    probs  = torch.softmax(torch.tensor(logits, dtype=torch.float), dim=-1).numpy()

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
# 8.  Confusion matrix helper  (identical to LoRA)
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
            data, annot=False, fmt=fmt, cmap="Blues",
            xticklabels=label_names, yticklabels=label_names,
            linewidths=0.5, ax=ax,
        )
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                # Use actual heatmap colour luminance to pick readable text colour
                norm_val = (data[i, j] - data.min()) / (data.max() - data.min() + 1e-9)
                rgba = plt.cm.Blues(norm_val)
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_colour = "white" if luminance < 0.5 else "black"
                ax.text(j + 0.5, i + 0.5, f"{data[i, j]:{fmt}}",
                        ha="center", va="center",
                        fontsize=11, fontweight="bold", color=text_colour)

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.suptitle(f"DistilBERT + LoReFT (manual) — {split.capitalize()} Set",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, f"confusion_matrix_{split}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix saved → {fig_path}")

    wandb.log({f"confusion_matrix_{split}": wandb.Image(fig)})
    plt.close(fig)
    return preds, labels

# =============================================================================
# 9.  W&B initialisation
# =============================================================================
wandb.init(
    project="peft-mental-health",
    name="distilbert-loreft-manual-classification",
    config={
        "model":          MODEL_NAME,
        "method":         "LoReFT (manual)",
        "reft_rank":      REFT_RANK,
        "reft_layers":    REFT_LAYERS,
        "reft_positions": str(REFT_POSITIONS),
        "lr":             LEARNING_RATE,
        "batch_size":     BATCH_SIZE,
        "epochs":         NUM_EPOCHS,
        "weight_decay":   WEIGHT_DECAY,
        "warmup_ratio":   WARMUP_RATIO,
        "num_labels":     num_labels,
        "seed":           config.SEED,
    },
    tags=["reft", "loreft", "manual", "distilbert", "classification", "mental-health"],
)

# =============================================================================
# 10.  TrainingArguments  (identical to LoRA)
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
# 11.  Train
# =============================================================================
trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["val"],
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("\n── Starting LoReFT training ─────────────────────────────────────────────")
trainer.train()

# =============================================================================
# 12.  Final evaluation on held-out test set
# =============================================================================
print("\n── Evaluating on test set ──────────────────────────────────────────────")
test_results = trainer.evaluate(eval_dataset=tokenized["test"])
print(test_results)

results_path = os.path.join(OUTPUT_DIR, "test_results.json")
with open(results_path, "w") as f:
    json.dump(test_results, f, indent=2)
print(f"Test results saved → {results_path}")

# =============================================================================
# 13.  Confusion matrix on test set
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
# 14.  Save intervention weights
# =============================================================================
reft_model_path = os.path.join(OUTPUT_DIR, "reft_model")
model.save_interventions(reft_model_path)
tokenizer.save_pretrained(reft_model_path)

wandb.finish()
print("\n LoReFT (manual) classification training complete.")