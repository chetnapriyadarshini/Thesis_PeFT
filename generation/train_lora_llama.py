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

#Package.                        Used by
#transformersAutoTokenizer,      AutoModelForCausalLM, set_seed
#datasets.                       load_from_diskpeftLoraConfig, TaskType, get_peft_modelaccelerateRequired internally by transformers Trainer for GPU trainingtrlSFTTrainer, SFTConfigbitsandbytespaged_adamw_8bit optimiserwandbexperiment tracking
#peft                            LoraConfig, TaskType, get_peft_model
#accelerate                      Required internally by transformers Trainer for GPU training
#trl                             SFTTrainer, SFTConfig
#bitsandbytes                    paged_adamw_8bit optimiser
#wandb                           experiment tracking

import os, sys, json, random, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"]   = "0"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
# ── Allow running from repo root ──────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import wandb

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

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

MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"   # 4-bit quantized version of Llama 3.2 3B Instruct

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
WEIGHT_DECAY               = 0.01
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
# 3.  Load tokeniser
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
# 4.  Format dataset as instruction template
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

def formatting_func(example):
    """Format messages into a single text string for trl 0.11.4."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return text

print("\n── Formatting dataset as instruction template ───────────────────────────")
formatted = dataset.map(format_as_messages, remove_columns=["prompt", "response", "emotion"])
print(formatted)
print("\nSample formatted message:")
print(formatted["train"][0]["messages"])

# =============================================================================
# 5.  Load model with 4-bit quantization (QLoRA)
# =============================================================================
print("\n── Loading Llama 3.2 3B Instruct (4-bit via unsloth mirror) ────────────")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

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
        "max_steps":              500,
        "max_seq_length":         MAX_SEQ_LENGTH,
        "quantization":           "4-bit NF4 (unsloth pre-quantized)",
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
    #num_train_epochs=NUM_EPOCHS,
    max_steps = 500,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,

    # ── Optimiser ───────────────────────────────────────────────────────────
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=50,               # # 10% of max_steps=500
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",       # memory-efficient optimiser for QLoRA

    # ── Sequence length ─────────────────────────────────────────────────────
    max_seq_length=MAX_SEQ_LENGTH,

    # ── SFT-specific ─────────────────────────────────────────────────────────
    # dataset_text_field not needed when using messages format
    packing=False,                  # don't pack sequences — empathetic dialogues vary in length

    # ── Evaluation & saving ─────────────────────────────────────────────────
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # ── Precision ───────────────────────────────────────────────────────────
    fp16=False,
    bf16=True, #works with Unsloth's 4-bit models on T4 (16GB) — faster than fp16 and compatible with older GPUs

    # ── Logging ─────────────────────────────────────────────────────────────
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
    formatting_func=formatting_func,
)

# Print trainable params — SFTTrainer applies LoRA internally
trainer.model.print_trainable_parameters()
print("\n── Starting LoRA generation training ───────────────────────────────────")
trainer.train()

# =============================================================================
# 10.  Save LoRA adapter
# =============================================================================
adapter_path = os.path.join(OUTPUT_DIR, "lora_adapter")
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"LoRA adapter saved → {adapter_path}")

wandb.finish()
print("\n LoRA generation training complete.")