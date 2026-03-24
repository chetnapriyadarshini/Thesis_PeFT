# =============================================================================
# generation/train_lora_llama.py
# Llama 3.2 3B Instruct + LoRA — Empathetic Dialogue Generation
#
# Key differences from classification:
#   - Causal LM task (generation) vs sequence classification
#   - 4-bit quantization (QLoRA) to fit 3B model on T4 16GB
#   - SFTTrainer with instruction template + loss masking on prompt tokens
#   - LoRA targets attention + MLP projections (Llama architecture)
#   - Evaluation: BERTScore (semantic similarity) — EPIC/LLM-as-Judge run separately
# =============================================================================

import os, sys, json, random, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"]   = "0"

# ── Allow running from repo root ──────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import wandb

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer, SFTConfig
import evaluate

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
DATA_DIR   = os.path.join(config.BASE_DIR, "data", "generation")
CKPT_DIR   = os.path.join(config.BASE_DIR, "checkpoints", "lora_generation")
OUTPUT_DIR = os.path.join(config.BASE_DIR, "results", "lora_generation")

os.makedirs(CKPT_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# LoRA hyper-parameters
# Llama uses q_proj/k_proj/v_proj/o_proj for attention
# gate_proj/up_proj/down_proj for the MLP (SwiGLU FFN)
LORA_R               = 16
LORA_ALPHA           = 32     # 2 × r standard
LORA_DROPOUT         = 0.05   # lower than classification — generation benefits from less dropout
LORA_TARGET_MODULES  = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "gate_proj", "up_proj", "down_proj",        # MLP
]

# Training hyper-parameters
# Batch size kept small (2) due to 4-bit + 3B model on T4
# Gradient accumulation (8) gives effective batch size of 16
LEARNING_RATE              = 2e-4
BATCH_SIZE                 = 2
GRADIENT_ACCUMULATION      = 8    # effective batch = BATCH_SIZE × GRADIENT_ACCUMULATION = 16
NUM_EPOCHS                 = 3    # generation models converge faster than classification
WEIGHT_DECAY               = 0.01
WARMUP_RATIO               = 0.1
MAX_SEQ_LENGTH             = 512  # prompt + response combined

# =============================================================================
# 2.  Load dataset
# =============================================================================
print("\n── Loading generation dataset ──────────────────────────────────────────")
dataset = load_from_disk(DATA_DIR)
print(dataset)

# Quick sanity check on data format
print("\nSample prompt:")
print(dataset["train"][0]["prompt"])
print("\nSample response:")
print(dataset["train"][0]["response"])

# =============================================================================
# 3.  Format dataset as instruction template
# =============================================================================
# Llama 3.2 Instruct uses the following chat template:
#   <|begin_of_text|>
#   <|start_header_id|>system<|end_header_id|>
#   {system_prompt}<|eot_id|>
#   <|start_header_id|>user<|end_header_id|>
#   {user_message}<|eot_id|>
#   <|start_header_id|>assistant<|end_header_id|>
#   {response}<|eot_id|>
#
# SFTTrainer with this "messages" format automatically:
#   1. Applies the tokenizer's built-in chat template
#   2. Masks prompt tokens from the loss (only response tokens contribute)
# =============================================================================

SYSTEM_PROMPT = (
    "You are an empathetic mental health support assistant. "
    "A person has shared their situation and emotional state with you. "
    "Respond with warmth, understanding, and supportive intent. "
    "Acknowledge their feelings and provide gentle, non-clinical support."
)

def format_as_messages(example):
    """Convert prompt/response pairs into the messages format for SFTTrainer."""
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
    }

print("\n── Formatting dataset as instruction template ───────────────────────────")
formatted = dataset.map(format_as_messages, remove_columns=["prompt", "response"])
print(formatted)
print("\nSample formatted message:")
print(formatted["train"][0]["messages"])

# =============================================================================
# 4.  Load tokeniser
# =============================================================================
print("\n── Loading tokeniser ───────────────────────────────────────────────────")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Llama 3.2 doesn't have a dedicated pad token — use eos_token
# This is standard practice for causal LM fine-tuning
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Right padding for training (causal LM reads left to right)
tokenizer.padding_side = "right"

print(f"Vocab size:    {tokenizer.vocab_size:,}")
print(f"Pad token:     {tokenizer.pad_token}")
print(f"EOS token:     {tokenizer.eos_token}")

# =============================================================================
# 5.  Load model with 4-bit quantization (QLoRA)
# =============================================================================
print("\n── Loading Llama 3.2 3B Instruct (4-bit) ───────────────────────────────")

# NF4 (Normal Float 4) is the best quantization type for LLMs
# double_quant further reduces memory by quantizing the quantization constants
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NF4 is better than FP4 for LLMs
    bnb_4bit_compute_dtype=torch.float16, # T4 doesn't support bfloat16
    bnb_4bit_use_double_quant=True,     # quantize the quantization constants too
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",                  # auto places layers across available GPU/CPU
    torch_dtype=torch.float16,
)

# Required for QLoRA — enables gradient checkpointing compatibility
# with 4-bit quantized models
model.config.use_cache = False              # disable KV cache during training
model.config.pretraining_tp = 1            # tensor parallelism = 1 (single GPU)

print(f"Model loaded. Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# =============================================================================
# 6.  Apply LoRA
# =============================================================================
print("\n── Applying LoRA ────────────────────────────────────────────────────────")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    inference_mode=False,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =============================================================================
# 7.  W&B initialisation
# =============================================================================
wandb.init(
    project="peft-mental-health",
    name="llama3.2-lora-generation",
    config={
        "model":                  MODEL_NAME,
        "method":                 "LoRA (QLoRA 4-bit)",
        "lora_r":                 LORA_R,
        "lora_alpha":             LORA_ALPHA,
        "lora_dropout":           LORA_DROPOUT,
        "target_modules":         LORA_TARGET_MODULES,
        "lr":                     LEARNING_RATE,
        "batch_size":             BATCH_SIZE,
        "gradient_accumulation":  GRADIENT_ACCUMULATION,
        "effective_batch_size":   BATCH_SIZE * GRADIENT_ACCUMULATION,
        "epochs":                 NUM_EPOCHS,
        "max_seq_length":         MAX_SEQ_LENGTH,
        "quantization":           "4-bit NF4",
        "seed":                   config.SEED,
    },
    tags=["lora", "llama3.2", "generation", "mental-health", "qlora"],
)

# =============================================================================
# 8.  SFTConfig (replaces TrainingArguments for SFTTrainer)
# =============================================================================
sft_config = SFTConfig(
    output_dir=CKPT_DIR,

    # ── Epochs & steps ──────────────────────────────────────────────────────
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,

    # ── Optimiser ───────────────────────────────────────────────────────────
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",       # memory-efficient optimiser for QLoRA

    # ── Sequence length ─────────────────────────────────────────────────────
    max_seq_length=MAX_SEQ_LENGTH,

    # ── SFT-specific ─────────────────────────────────────────────────────────
    # dataset_text_field not needed when using messages format
    packing=False,                  # don't pack sequences — empathetic dialogues vary in length

    # ── Evaluation & saving ─────────────────────────────────────────────────
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # ── Precision ───────────────────────────────────────────────────────────
    fp16=True,                      # T4 uses fp16 (no bfloat16 support)
    bf16=False,

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
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=sft_config,
    train_dataset=formatted["train"],
    eval_dataset=formatted["val"],
    peft_config=lora_config,
)

print("\n── Starting LoRA generation training ───────────────────────────────────")
trainer.train()

# =============================================================================
# 10.  BERTScore evaluation on test set
# =============================================================================
print("\n── Evaluating BERTScore on test set ────────────────────────────────────")

# Set model to eval mode and generate responses for test set
model.eval()
bertscore = evaluate.load("bertscore")

# Generate responses for a subset of test samples
# Full test set (9,689) would take too long — use 500 samples for evaluation
TEST_EVAL_SAMPLES = 500
test_subset = formatted["test"].select(range(TEST_EVAL_SAMPLES))

predictions = []
references  = []

print(f"Generating responses for {TEST_EVAL_SAMPLES} test samples...")
for i, example in enumerate(test_subset):
    if i % 50 == 0:
        print(f"  {i}/{TEST_EVAL_SAMPLES}...")

    # Extract user message and reference response
    user_msg  = example["messages"][1]["content"]   # user turn
    reference = example["messages"][2]["content"]   # assistant turn (ground truth)

    # Format as prompt for generation (no assistant response)
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,   # adds <|start_header_id|>assistant<|end_header_id|>
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,           # greedy decoding for reproducibility
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (not the prompt)
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    predictions.append(generated)
    references.append(reference)

# Compute BERTScore
print("\nComputing BERTScore...")
bert_results = bertscore.compute(
    predictions=predictions,
    references=references,
    lang="en",
    model_type="distilbert-base-uncased",  # lightweight scorer
)

bert_f1_mean = np.mean(bert_results["f1"])
bert_p_mean  = np.mean(bert_results["precision"])
bert_r_mean  = np.mean(bert_results["recall"])

print(f"BERTScore — Precision: {bert_p_mean:.4f} | Recall: {bert_r_mean:.4f} | F1: {bert_f1_mean:.4f}")

# Save results
results = {
    "bertscore_f1":        round(float(bert_f1_mean), 4),
    "bertscore_precision": round(float(bert_p_mean),  4),
    "bertscore_recall":    round(float(bert_r_mean),  4),
    "eval_samples":        TEST_EVAL_SAMPLES,
}
results_path = os.path.join(OUTPUT_DIR, "test_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved → {results_path}")

wandb.log({
    "test/bertscore_f1":        bert_f1_mean,
    "test/bertscore_precision": bert_p_mean,
    "test/bertscore_recall":    bert_r_mean,
})

# Save a few example generations for qualitative review
examples_path = os.path.join(OUTPUT_DIR, "generation_examples.json")
with open(examples_path, "w") as f:
    json.dump([
        {
            "prompt":    test_subset[i]["messages"][1]["content"],
            "reference": references[i],
            "generated": predictions[i],
        }
        for i in range(min(20, TEST_EVAL_SAMPLES))
    ], f, indent=2)
print(f"Generation examples saved → {examples_path}")

# =============================================================================
# 11.  Save LoRA adapter
# =============================================================================
adapter_path = os.path.join(OUTPUT_DIR, "lora_adapter")
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"LoRA adapter saved → {adapter_path}")

wandb.finish()
print("\n LoRA generation training complete.")