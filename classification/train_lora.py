# =============================================================================
# classification/train_lora.py
# DistilBERT + LoRA — Mental Health Stress Classification
# =============================================================================

import os, sys, json, random, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Allow running from repo root ──────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import wandb

from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)
from peft import get_peft_model, LoraConfig, TaskType
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
CKPT_DIR   = os.path.join(config.BASE_DIR, "checkpoints", "lora_classification")
OUTPUT_DIR = os.path.join(config.BASE_DIR, "results", "lora_classification")

os.makedirs(CKPT_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"

# LoRA hyper-parameters
LORA_R        = 8       # rank — keep low for DistilBERT
LORA_ALPHA    = 16      # scaling factor (2 × r is a common starting point)
LORA_DROPOUT  = 0.1

# Training hyper-parameters
LEARNING_RATE  = 2e-4
BATCH_SIZE     = 32
NUM_EPOCHS     = 10      # EarlyStopping will cut this short if needed
WEIGHT_DECAY   = 0.01
WARMUP_RATIO   = 0.1

# =============================================================================
# 2.  Load dataset & class weights
# =============================================================================
print("\n── Loading classification dataset ──────────────────────────────────────")
dataset = load_from_disk(DATA_DIR)
print(dataset)

# label names
label_df    = pd.read_csv(os.path.join(DATA_DIR, "label_names.csv"))
id2label    = dict(zip(label_df["id"].astype(int), label_df["label"]))
label2id    = {v: k for k, v in id2label.items()}
num_labels  = len(id2label)
print(f"Labels ({num_labels}): {id2label}")

# class weights for imbalanced dataset
with open(os.path.join(DATA_DIR, "class_weights.json")) as f:
    cw_dict = json.load(f)

# cw_dict keys may be strings — align with label order
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
        max_length=256,         # reddit posts are rarely longer
        padding=False,          # DataCollatorWithPadding handles dynamic padding
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized.set_format("torch")
print(tokenized)

# =============================================================================
# 4.  Build DistilBERT + LoRA model
# =============================================================================
print("\n── Building DistilBERT + LoRA ──────────────────────────────────────────")

base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

# DistilBERT attention projection names differ from BERT
# q_lin / v_lin are the query and value projections in each layer
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_lin", "v_lin"],  # DistilBERT-specific
    bias="none",
    inference_mode=False,
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# =============================================================================
# 5.  Custom Trainer with weighted cross-entropy loss
# =============================================================================
class WeightedTrainer(Trainer):
    """Standard HuggingFace Trainer with class-weighted CE loss."""

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
# 6.  Evaluation metrics
# =============================================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # Softmax probabilities for AUC
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float), dim=-1).numpy()

    f1        = f1_score(labels, preds, average="weighted", zero_division=0)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall    = recall_score(labels, preds, average="weighted", zero_division=0)
    accuracy  = accuracy_score(labels, preds)

    # AUC — one-vs-rest for multi-class
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="weighted")
    except ValueError:
        auc = float("nan")  # can happen if a class is missing in a small batch

    return {
        "f1":        round(f1, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "accuracy":  round(accuracy, 4),
        "auc":       round(auc, 4),
    }

# =============================================================================
# 7.  W&B initialisation
# =============================================================================
wandb.init(
    project="peft-mental-health",
    name="distilbert-lora-classification",
    config={
        "model":         MODEL_NAME,
        "method":        "LoRA",
        "lora_r":        LORA_R,
        "lora_alpha":    LORA_ALPHA,
        "lora_dropout":  LORA_DROPOUT,
        "lr":            LEARNING_RATE,
        "batch_size":    BATCH_SIZE,
        "epochs":        NUM_EPOCHS,
        "weight_decay":  WEIGHT_DECAY,
        "warmup_ratio":  WARMUP_RATIO,
        "num_labels":    num_labels,
        "seed":          config.SEED,
    },
    tags=["lora", "distilbert", "classification", "mental-health"],
)

# =============================================================================
# 8.  TrainingArguments
# =============================================================================
training_args = TrainingArguments(
    output_dir=CKPT_DIR,

    # ── Epochs & steps ──────────────────────────────────────────────────────
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    # ── Optimiser ───────────────────────────────────────────────────────────
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",

    # ── Evaluation & saving ─────────────────────────────────────────────────
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    # ── Precision ───────────────────────────────────────────────────────────
    fp16=torch.cuda.is_available(),   # use mixed precision on GPU

    # ── Logging ─────────────────────────────────────────────────────────────
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=50,
    report_to="wandb",

    # ── Reproducibility ─────────────────────────────────────────────────────
    seed=config.SEED,
    data_seed=config.SEED,
)

# =============================================================================
# 9.  Train
# =============================================================================
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["val"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("\n── Starting LoRA training ───────────────────────────────────────────────")
trainer.train()

# =============================================================================
# 10.  Final evaluation on held-out test set
# =============================================================================
print("\n── Evaluating on test set ──────────────────────────────────────────────")
test_results = trainer.evaluate(eval_dataset=tokenized["test"])
print(test_results)

# Save results to disk
results_path = os.path.join(OUTPUT_DIR, "test_results.json")
with open(results_path, "w") as f:
    json.dump(test_results, f, indent=2)
print(f"Test results saved → {results_path}")

# =============================================================================
# 11.  Save final model adapter
# =============================================================================
adapter_path = os.path.join(OUTPUT_DIR, "lora_adapter")
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"LoRA adapter saved → {adapter_path}")

wandb.finish()
print("\n  LoRA classification training complete.")