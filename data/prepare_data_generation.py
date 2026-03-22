"""
data/prepare_data_generation.py

Prepares the generation dataset from Empathetic Dialogues (Kaggle version).

Prompt format:
    "[{emotion}] {Situation} | {customer_utterance}"
Response:
    Agent's empathetic reply (labels column)

Steps:
    - Drop corrupted emotion rows and nulls
    - Parse Customer utterance from empathetic_dialogues column
    - Format prompt with emotion + situation + utterance
    - Apply PII redaction
    - Split 70 / 15 / 15

Environment: transformers==4.45.1, pyreft, peft
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
from datasets import Dataset, DatasetDict

# ── Path setup ────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, SEED, set_seed

set_seed()

# ── Presidio setup ────────────────────────────────────────────
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

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

print("Presidio ready ✅")

# ── Constants ─────────────────────────────────────────────────
DATA_PATH = "/kaggle/input/datasets/atharvjairath/empathetic-dialogues-facebook-ai/empathetic-dialogues-facebook-ai.csv"

# Valid emotion labels (32 known emotions)
VALID_EMOTIONS = [
    "surprised", "excited", "angry", "proud", "annoyed", "sad",
    "lonely", "afraid", "grateful", "terrified", "guilty", "furious",
    "disgusted", "confident", "anxious", "anticipating", "hopeful",
    "impressed", "nostalgic", "disappointed", "jealous", "joyful",
    "prepared", "content", "devastated", "embarrassed", "sentimental",
    "caring", "trusting", "ashamed", "apprehensive", "faithful"
]

# ── Load data ─────────────────────────────────────────────────
print("\nLoading Empathetic Dialogues...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")

# ── Drop junk columns ─────────────────────────────────────────
df = df.drop(columns=["Unnamed: 0", "Unnamed: 5", "Unnamed: 6"],
             errors="ignore")
print(f"\nAfter dropping junk columns: {df.columns.tolist()}")

# ── Drop nulls and corrupted emotion rows ─────────────────────
before = len(df)
df = df[df["emotion"].notna()]
df = df[df["emotion"].isin(VALID_EMOTIONS)]
print(f"\nDropped {before - len(df)} corrupted/null emotion rows")
print(f"Remaining: {len(df)} rows")

# ── Drop null prompts or responses ───────────────────────────
df = df[df["empathetic_dialogues"].notna()]
df = df[df["labels"].notna()]
df = df[df["Situation"].notna()]
print(f"After dropping null text rows: {len(df)} rows")

# ── Parse Customer utterance ──────────────────────────────────
def parse_customer_text(dialogue: str) -> str:
    """Extract Customer utterance from dialogue string."""
    if not isinstance(dialogue, str):
        return ""
    text = dialogue.replace("Customer :", "").split("\nAgent :")[0].strip()
    return text

df["customer_utterance"] = df["empathetic_dialogues"].apply(parse_customer_text)

# Drop rows where parsing yields empty string
df = df[df["customer_utterance"].str.strip() != ""]
print(f"After parsing utterances: {len(df)} rows")

# ── Build prompt/response pairs ───────────────────────────────
# Format: "[{emotion}] {Situation} | {customer_utterance}"
df["prompt"] = (
    "[" + df["emotion"] + "] " +
    df["Situation"].str.strip() + " | " +
    df["customer_utterance"].str.strip()
)
df["response"] = df["labels"].str.strip()

# ── Keep only what we need ────────────────────────────────────
df = df[["prompt", "response", "emotion"]].reset_index(drop=True)

print(f"\nFinal dataset: {len(df)} rows")
print(f"\nEmotion distribution:")
print(df["emotion"].value_counts())

print("\nSample prompt/response pairs:")
for i, row in df.head(3).iterrows():
    print(f"\n[{i}] Prompt:   {row['prompt'][:120]}")
    print(f"     Response: {row['response']}")

# ── Train / Val / Test split (70 / 15 / 15) ───────────────────
print("\nSplitting 70 / 15 / 15...")
dataset    = Dataset.from_pandas(df)
train_test = dataset.train_test_split(test_size=0.30, seed=SEED)
val_test   = train_test["test"].train_test_split(test_size=0.50, seed=SEED)

gen_dataset = DatasetDict({
    "train": train_test["train"],
    "val":   val_test["train"],
    "test":  val_test["test"],
})

print(f"  Train: {len(gen_dataset['train'])}")
print(f"  Val:   {len(gen_dataset['val'])}")
print(f"  Test:  {len(gen_dataset['test'])}")

# ── PII redaction ─────────────
print("\nApplying PII redaction (this may take a while)...")
for split in gen_dataset:
    gen_dataset[split] = gen_dataset[split].map(
        lambda batch: redact_batch(batch, "prompt"),
        batched=True, batch_size=64,
        desc=f"Redacting prompts ({split})"
    )
    gen_dataset[split] = gen_dataset[split].map(
        lambda batch: redact_batch(batch, "response"),
        batched=True, batch_size=64,
        desc=f"Redacting responses ({split})"
    )

# ── Save ──────────────────────────────────────────────────────
save_path = os.path.join(DATA_DIR, "generation")
os.makedirs(save_path, exist_ok=True)
gen_dataset.save_to_disk(save_path)
print(f"\nGeneration data saved → {save_path} ✅")

# Save emotion list for reference
emotion_df = pd.DataFrame({"emotion": VALID_EMOTIONS})
emotion_df.to_csv(os.path.join(save_path, "emotions.csv"), index=False)
print(f"Emotion list saved → {save_path}/emotions.csv ✅")

print("\n" + "="*55)
print("Generation data pipeline complete ✅")
print("="*55)