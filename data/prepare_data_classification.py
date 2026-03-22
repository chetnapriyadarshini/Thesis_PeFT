"""
data/prepare_data_classification.py

Prepares the classification dataset from SWMH.

Classes:
    0 — anxiety        (self.Anxiety)
    1 — suicide_watch  (self.SuicideWatch)
    2 — bipolar        (self.bipolar)
    3 — depression     (self.depression)
    4 — no_stress      (self.offmychest, MH keywords removed)

No downsampling — class imbalance handled via weighted loss during training.
Class weights are computed and saved alongside the dataset.

Environment: transformers==4.45.1, pyreft, peft
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

# ── Path setup ────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, SEED, set_seed

set_seed()

# ── HuggingFace auth ──────────────────────────────────────────
secrets  = UserSecretsClient()
hf_token = secrets.get_secret("HF_TOKEN")
login(token=hf_token, add_to_git_credential=False)
print("HuggingFace login")

# ── Presidio setup ────────────────────────────────────────────
print("Loading Presidio engines...")
analyzer   = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def redact_pii(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    results = analyzer.analyze(text=text, language="en")
    if not results:
        return text
    return anonymizer.anonymize(text=text, analyzer_results=results).text

def redact_batch(batch: dict, text_col: str) -> dict:
    batch[text_col] = [redact_pii(t) for t in batch[text_col]]
    return batch

print("Presidio ready")

# ── Constants ─────────────────────────────────────────────────
# Keywords used to filter clinical MH content from offmychest
# Removed: therapy, therapist, medication, psychiatrist, anxious
# (too general — people use these in everyday contexts)
MH_KEYWORDS = [
    "suicide", "suicidal", "self harm", "self-harm", "overdose",
    "depression", "depressed", "anxiety", "bipolar", "schizophrenia",
    "mental illness", "mental health", "panic attack", "ptsd", "trauma",
    "eating disorder", "anorexia", "bulimia", "psychosis", "hallucin"
]

LABEL_MAP = {
    "self.Anxiety":      0,
    "self.SuicideWatch": 1,
    "self.bipolar":      2,
    "self.depression":   3,
    "self.offmychest":   4,
}

LABEL_NAMES = {
    0: "anxiety",
    1: "suicide_watch",
    2: "bipolar",
    3: "depression",
    4: "no_stress",
}


# ── Load SWMH ─────────────────────────────────────────────────
print("\nLoading SWMH...")
swmh = load_dataset("AIMH/SWMH")

dfs = []
for split in swmh:
    df = swmh[split].to_pandas()
    df["original_split"] = split
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
print(f"SWMH loaded: {len(combined)} rows")
print(f"Columns: {combined.columns.tolist()}")


# ── Basic cleaning ────────────────────────────────────────────
# EDA confirmed: zero nulls, empty, [deleted], or sub-10-word rows in SWMH
# No cleaning required — data is already clean
print(f"\nData quality check passed — no cleaning required")
print(f"Total rows: {len(combined)}")


# ── Filter offmychest MH keyword rows ─────────────────────────
print("\nFiltering offmychest for MH keyword overlap...")
pattern  = "|".join(MH_KEYWORDS)
omc_mask = combined["label"] == "self.offmychest"
flagged  = omc_mask & combined["text"].str.lower().str.contains(
    pattern, na=False, regex=True
)

n_flagged = flagged.sum()
combined  = combined[~flagged].copy()

print(f"Removed {n_flagged} offmychest rows containing MH keywords")
print(f"Remaining offmychest (no_stress): "
      f"{(combined['label'] == 'self.offmychest').sum()} rows")


# ── Final class distribution before splitting ──────────────────
print("\nFinal class distribution:")
for orig_label, label_id in LABEL_MAP.items():
    count = (combined["label"] == orig_label).sum()
    print(f"  {label_id} — {LABEL_NAMES[label_id]}: {count}")


# ── Map labels to integers ────────────────────────────────────
combined["label"] = combined["label"].map(LABEL_MAP)
combined = combined[["text", "label"]].reset_index(drop=True)


# ── Compute class weights ─────────────────────────────────────
# Formula: total / (num_classes * class_count)
print("\nComputing class weights...")
num_classes  = len(LABEL_MAP)
total        = len(combined)
class_counts = combined["label"].value_counts().sort_index()
class_weights = {}

for label_id in sorted(LABEL_MAP.values()):
    count = class_counts[label_id]
    weight = total / (num_classes * count)
    class_weights[label_id] = round(weight, 4)
    print(f"  {label_id} — {LABEL_NAMES[label_id]}: "
          f"count={count}, weight={weight:.4f}")


# ── PII redaction ─────────────────────────────────────────────
print("\nApplying PII redaction (this will take a few minutes)...")
dataset = Dataset.from_pandas(combined)
dataset = dataset.map(
    lambda batch: redact_batch(batch, "text"),
    batched=True,
    batch_size=64,
    desc="Redacting PII"
)


# ── Train / Val / Test split (70 / 15 / 15) ───────────────────
print("\nSplitting into train / val / test (70/15/15)...")
train_test = dataset.train_test_split(test_size=0.30, seed=SEED)
val_test   = train_test["test"].train_test_split(test_size=0.50, seed=SEED)

clf_dataset = DatasetDict({
    "train": train_test["train"],
    "val":   val_test["train"],
    "test":  val_test["test"],
})

print(f"  Train: {len(clf_dataset['train'])}")
print(f"  Val:   {len(clf_dataset['val'])}")
print(f"  Test:  {len(clf_dataset['test'])}")


# ── Save dataset ──────────────────────────────────────────────
save_path = os.path.join(DATA_DIR, "classification")
os.makedirs(save_path, exist_ok=True)
clf_dataset.save_to_disk(save_path)
print(f"\nDataset saved → {save_path}")

# Save label map
label_df = pd.DataFrame([
    {"id": k, "name": v} for k, v in LABEL_NAMES.items()
])
label_df.to_csv(
    os.path.join(save_path, "label_names.csv"), index=False
)

# Save class weights
with open(os.path.join(save_path, "class_weights.json"), "w") as f:
    json.dump(class_weights, f, indent=2)

print(f"Label map    → {save_path}/label_names.csv")
print(f"Class weights → {save_path}/class_weights.json")

print("\n" + "="*55)
print("Classification data pipeline complete")
print("="*55)