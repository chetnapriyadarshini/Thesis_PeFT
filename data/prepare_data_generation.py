"""
data/prepare_data_generation.py

Prepares the generation dataset from Estwld/empathetic_dialogues_llm (HuggingFace).

This version replaces the original Kaggle CSV approach which had role confusion
issues — agent turns were being parsed as user turns in multi-turn conversations.

The HuggingFace dataset has correct user/assistant role assignments and stores
each conversation as a complete list of turns.

Dataset structure:
    - conv_id:       unique conversation ID
    - situation:     the emotional context/situation
    - emotion:       one of 32 emotion labels
    - conversations: list of {"role": "user"/"assistant", "content": "..."}

Prompt format:
    System: empathetic assistant prompt
    [emotion] situation
    User: turn 1
    Assistant: turn 2
    User: turn 3
    ...
    (final user turn — model generates the next assistant response)

Response:
    Final assistant turn in the conversation

Steps:
    - Load from HuggingFace (no Kaggle CSV needed)
    - Filter conversations ending in user turn (ensures response is assistant)
    - Build prompt with full conversation history up to last user turn
    - Extract final assistant response as label
    - Apply PII redaction
    - Split 70 / 15 / 15
    - Save to disk

Environment: transformers==4.45.1
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict

# ── Path setup ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, SEED, set_seed

set_seed()

# ── Presidio setup ────────────────────────────────────────────
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

print("Loading Presidio engines...")
analyzer   = AnalyzerEngine()
anonymizer = AnonymizerEngine()
print("Presidio ready ✅")

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

# ── Load dataset from HuggingFace ─────────────────────────────
print("\nLoading Estwld/empathetic_dialogues_llm from HuggingFace...")
raw = load_dataset("Estwld/empathetic_dialogues_llm")
print(raw)

# Combine all splits — we'll do our own 70/15/15 split
all_data = []
for split in raw:
    for row in raw[split]:
        all_data.append(row)

print(f"\nTotal conversations loaded: {len(all_data)}")

# ── Parse conversations into prompt/response pairs ────────────
print("\nParsing conversations into prompt/response pairs...")

def parse_conversation(row):
    """
    Convert a conversation row into a prompt/response pair.

    Strategy:
    - Only keep conversations where the last turn is 'assistant'
      (this guarantees the response is always an agent/empathetic reply)
    - Prompt = [emotion] situation + full conversation history EXCEPT last assistant turn
    - Response = last assistant turn

    This ensures:
    - Prompt always ends with a user turn
    - Response is always an empathetic assistant reply
    - Multi-turn context is preserved
    """
    conversations = row["conversations"]
    situation     = row["situation"].strip()
    emotion       = row["emotion"]

    # Must have at least one turn and last turn must be assistant
    if not conversations or conversations[-1]["role"] != "assistant":
        return None

    # Response = last assistant turn
    response = conversations[-1]["content"].strip()

    # History = all turns except the last assistant turn
    history_turns = conversations[:-1]

    # Build prompt
    prompt = f"[{emotion}] {situation}"
    for turn in history_turns:
        if turn["role"] == "user":
            prompt += f"\nUser: {turn['content'].strip()}"
        else:
            prompt += f"\nAssistant: {turn['content'].strip()}"

    return {"prompt": prompt, "response": response, "emotion": emotion}

parsed = []
skipped = 0
for row in all_data:
    result = parse_conversation(row)
    if result is not None:
        parsed.append(result)
    else:
        skipped += 1

print(f"Valid pairs: {len(parsed)}")
print(f"Skipped (last turn not assistant): {skipped}")

# ── Sample check ──────────────────────────────────────────────
print("\nSample prompt/response pairs:")
for i, row in enumerate(parsed[:3]):
    print(f"\n[{i}] Prompt:\n{row['prompt']}")
    print(f"     Response: {row['response']}")

# ── Build HuggingFace Dataset ─────────────────────────────────
df = pd.DataFrame(parsed)
print(f"\nEmotion distribution:")
print(df["emotion"].value_counts().head(10))

dataset = Dataset.from_pandas(df, preserve_index=False)

# ── Train / Val / Test split (70 / 15 / 15) ───────────────────
print("\nSplitting 70 / 15 / 15...")
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

# ── PII redaction ─────────────────────────────────────────────
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
emotion_df = pd.DataFrame({"emotion": df["emotion"].unique().tolist()})
emotion_df.to_csv(os.path.join(save_path, "emotions.csv"), index=False)
print(f"Emotion list saved → {save_path}/emotions.csv ✅")

print("\n" + "="*55)
print("Generation data pipeline complete ✅")
print("="*55)