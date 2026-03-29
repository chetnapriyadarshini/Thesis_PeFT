# =============================================================================
# generation/train_reft_llama.py
# Llama 3.2 3B Instruct + LoReFT (manual) — Empathetic Dialogue Generation
#
# ReFT is implemented manually using PyTorch forward hooks — no pyvene/pyreft
# dependency. Same approach as classification/train_reft.py.
#
# LoReFT formula per layer, per position:
#   h_new = h + Rᵀ(Wh + b − Rh)
#
# Key differences from train_lora_llama.py:
#   - No LoraConfig/peft — ReFT interventions via forward hooks
#   - Interventions on all 28 Llama transformer layers
#   - Base model frozen, only R and W matrices trained
#   - SFTTrainer used but without peft_config
# =============================================================================

#Package.                        Used by
#transformers                    AutoTokenizer, AutoModelForCausalLM, set_seed
#datasets                        load_from_disk
#accelerate                      Required internally by transformers Trainer
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
import torch.nn as nn
import wandb

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTTrainer, SFTConfig

import config   # repo-level config.py
import transformers.trainer as trainer_module

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
CKPT_DIR   = os.path.join(config.BASE_DIR, "checkpoints", "reft_generation")
OUTPUT_DIR = os.path.join(config.BASE_DIR, "results", "reft_generation")

os.makedirs(CKPT_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# ReFT hyper-parameters
# Llama 3.2 3B has 28 transformer layers (0-27)
REFT_LAYERS    = list(range(28))
REFT_RANK      = 4          # low-rank dimension — same as classification ReFT
REFT_POSITIONS = [0, -1]    # first token + last token

# Training hyper-parameters — identical to LoRA for fair comparison
LEARNING_RATE        = 2e-4
BATCH_SIZE           = 2
GRADIENT_ACCUMULATION = 8
WEIGHT_DECAY         = 0.01
MAX_SEQ_LENGTH       = 512

# =============================================================================
# 2.  Load dataset
# =============================================================================
print("\n── Loading generation dataset ──────────────────────────────────────────")
dataset = load_from_disk(DATA_DIR)
print(dataset)

print("\nSample prompt:")
print(dataset["train"][0]["prompt"])
print("\nSample response:")
print(dataset["train"][0]["response"])

# =============================================================================
# 3.  Load tokeniser
# =============================================================================
print("\n── Loading tokeniser ───────────────────────────────────────────────────")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.padding_side = "right"

print(f"Vocab size:    {tokenizer.vocab_size:,}")
print(f"Pad token:     {tokenizer.pad_token}")
print(f"EOS token:     {tokenizer.eos_token}")

# =============================================================================
# 4.  Pre-tokenize dataset (bypasses SFTTrainer tokenization issues)
# =============================================================================
from transformers import default_data_collator

SYSTEM_PROMPT = (
    "You are an empathetic mental health support assistant. "
    "A person has shared their situation and emotional state with you. "
    "Respond with warmth, understanding, and supportive intent. "
    "Acknowledge their feelings and provide gentle, non-clinical support."
)

def tokenize_example(dataset):
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": dataset["prompt"]},
        {"role": "assistant", "content": dataset["response"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )
    # Set labels to -100 for padding tokens — CrossEntropyLoss ignores -100
    labels = tokenized["input_ids"].copy()
    labels = [
        token_id if token_id != tokenizer.pad_token_id else -100
        for token_id in labels
    ]
    tokenized["labels"] = labels
    return tokenized

print("\n── Tokenising dataset ──────────────────────────────────────────────────")
tokenized_dataset = dataset.map(
    tokenize_example,
    remove_columns=["prompt", "response", "emotion"],
    batched=False,
)
tokenized_dataset.set_format("torch")
print(tokenized_dataset)

# =============================================================================
# 5.  Load model with 4-bit quantization (QLoRA)
# =============================================================================
print("\n── Loading Llama 3.2 3B Instruct ────────────")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

print(f"Model loaded. Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# =============================================================================
# 6.  LoReFT intervention module
# =============================================================================
class LoReFTIntervention(nn.Module):
    """
    Manual LoReFT intervention — no pyvene dependency.

    Formula: h_new = h + Rᵀ(Wh + b − Rh)
      R = low-rank orthonormal projection (rank × embed_dim)
      W = learned transform (embed_dim → rank) with bias
    """

    def __init__(self, embed_dim: int, rank: int, positions: list):
        super().__init__()
        self.positions = positions

        self.R = nn.Linear(embed_dim, rank, bias=False)
        torch.nn.init.orthogonal_(self.R.weight)

        self.W = nn.Linear(embed_dim, rank, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h.clone()
        for pos in self.positions:
            h_pos = h[:, pos, :]
            difference   = self.W(h_pos) - self.R(h_pos)
            intervention = difference @ self.R.weight
            h[:, pos, :] = h_pos + intervention
        return h

# =============================================================================
# 7.  Apply LoReFT interventions via forward hooks
# =============================================================================
print("\n── Applying LoReFT interventions ───────────────────────────────────────")

# Freeze ALL base model parameters
for param in model.parameters():
    param.requires_grad = False

# Llama 3.2 hidden size is 3072 for 3B model
embed_dim = model.config.hidden_size
print(f"Hidden size: {embed_dim}")

# Create interventions and register as a module list on the model
# so their parameters are visible to the optimizer
model.reft_interventions = nn.ModuleList([
    LoReFTIntervention(embed_dim, REFT_RANK, REFT_POSITIONS)
    for _ in REFT_LAYERS
])

# Move interventions to same device as model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.reft_interventions.to(device)

# Register forward hooks on the last layer norm of each transformer block
# Llama uses model.layers[i] for transformer blocks
hooks = []
for idx, layer_idx in enumerate(REFT_LAYERS):
    layer_norm   = model.model.layers[layer_idx].post_attention_layernorm
    intervention = model.reft_interventions[idx]

    def make_hook(iv):
        def hook(module, input, output):
            # output may be a tuple in some transformer versions
            if isinstance(output, tuple):
                hidden = iv(output[0])
                return (hidden,) + output[1:]
            return iv(output)
        return hook

    handle = layer_norm.register_forward_hook(make_hook(intervention))
    hooks.append(handle)

# Tell the Trainer this model has trainable parameters
# by requiring gradients on intervention params
for param in model.reft_interventions.parameters():
    param.requires_grad = True

# Print trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"trainable params: {trainable:,} || all params: {total:,} || "
      f"trainable%: {100 * trainable / total:.4f}")

# =============================================================================
# 8.  W&B initialisation
# =============================================================================
wandb.init(
    project="peft-mental-health",
    name="llama3.2-reft-generation",
    config={
        "model":                  MODEL_NAME,
        "method":                 "LoReFT (manual)",
        "reft_rank":              REFT_RANK,
        "reft_layers":            len(REFT_LAYERS),
        "reft_positions":         str(REFT_POSITIONS),
        "lr":                     LEARNING_RATE,
        "batch_size":             BATCH_SIZE,
        "gradient_accumulation":  GRADIENT_ACCUMULATION,
        "effective_batch_size":   BATCH_SIZE * GRADIENT_ACCUMULATION,
        "max_steps":              500,
        "max_seq_length":         MAX_SEQ_LENGTH,
        "quantization":           "4-bit NF4 (official meta-llama weights)",
        "seed":                   config.SEED,
    },
    tags=["reft", "loreft", "manual", "llama3.2", "generation", "mental-health"],
)

# =============================================================================
# 9.  SFTConfig — identical to LoRA for fair comparison
# =============================================================================
sft_config = SFTConfig(
    output_dir=CKPT_DIR,

    max_steps=500,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    # Allow training quantized model with custom adapters
    gradient_checkpointing=False,
    remove_unused_columns=False,  # ReFT model doesn't follow PEFT base_model pattern

    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",

     # ── SFT-specific ─────────────────────────────────────────────────────────
    # dataset_text_field not needed when using messages format
    packing=False,                  # don't pack sequences — empathetic dialogues vary in length

    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    fp16=False,
    bf16=True,

    logging_steps=50,
    report_to="wandb",

    seed=config.SEED,
    data_seed=config.SEED,
)

# =============================================================================
# 10.  Train
# =============================================================================
"""
Disable the quantized model check in SFTTrainer 
since we're handling ReFT manually and not using a peft_config. 
This allows the trainer to recognise that there are trainable parameters (the ReFT intervention weights) 
even though the base model is frozen and quantized.

"""
trainer_module._is_peft_model = lambda model: True 

from transformers import TrainerCallback

class SaveReFTCallback(TrainerCallback):
    """Save ReFT intervention weights at each checkpoint."""
    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        torch.save(
            {f"intervention_{i}": iv.state_dict()
             for i, iv in enumerate(model.reft_interventions)},
            os.path.join(ckpt_dir, "reft_interventions.pt")
        )
        print(f"ReFT interventions saved at step {state.global_step}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=sft_config,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
    data_collator=default_data_collator,
    callbacks=[SaveReFTCallback()],
    # no peft_config — ReFT is applied via hooks
)

print("\n── Starting LoReFT generation training ─────────────────────────────────")
try:
    trainer.train()
except TypeError as e:
    print(f"Warning: post-training error (expected): {e}")
    print("Continuing to save intervention weights...")

# ── Manually load best checkpoint for ReFT ───────────────────────────────
import glob

# Find checkpoint with lowest eval loss from saved checkpoints
checkpoints = glob.glob(os.path.join(CKPT_DIR, "checkpoint-*"))
if checkpoints:
    # Read trainer_state.json from each checkpoint to find best
    best_ckpt = None
    best_loss = float("inf")
    for ckpt in checkpoints:
        state_file = os.path.join(ckpt, "trainer_state.json")
        if os.path.exists(state_file):
            with open(state_file) as f:
                state = json.load(f)
            loss = state.get("best_metric", float("inf"))
            if loss < best_loss:
                best_loss = loss
                best_ckpt = ckpt
    
    if best_ckpt:
        print(f"Loading best checkpoint: {best_ckpt} (eval_loss={best_loss:.4f})")
        # Load intervention weights from best checkpoint
        interventions_path = os.path.join(best_ckpt, "reft_interventions.pt")
        if os.path.exists(interventions_path):
            state_dicts = torch.load(interventions_path)
            for i, iv in enumerate(model.reft_interventions):
                iv.load_state_dict(state_dicts[f"intervention_{i}"])
            print("Best intervention weights loaded ✅")

# =============================================================================
# 11.  Save ReFT intervention weights
# =============================================================================
reft_model_path = os.path.join(OUTPUT_DIR, "reft_model")
os.makedirs(reft_model_path, exist_ok=True)

torch.save(
    {f"intervention_{i}": iv.state_dict()
     for i, iv in enumerate(model.reft_interventions)},
    os.path.join(reft_model_path, "reft_interventions.pt")
)
tokenizer.save_pretrained(reft_model_path)
print(f"ReFT interventions saved → {reft_model_path}")

wandb.finish()
print("\n  LoReFT generation training complete.")