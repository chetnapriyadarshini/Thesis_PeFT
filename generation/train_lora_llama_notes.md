# train_lora_llama.py — Design Decisions & Concept Notes
> Llama 3.2 3B Instruct + LoRA (QLoRA 4-bit) for Empathetic Dialogue Generation  
> Quick reference for revision and interview prep

---

## What This Script Does

Fine-tunes Llama 3.2 3B Instruct for empathetic dialogue generation on the Empathetic Dialogues dataset using LoRA with 4-bit quantization (QLoRA). The model learns to generate warm, empathetic responses to mental health-related situations. All base model weights are frozen — only the LoRA adapter matrices are trained.

---

## Results (EXP-03)

| Metric | Value |
|--------|-------|
| Final eval loss | 0.8938 |
| Final train loss | 0.9192 |
| Trainable params | 24,313,856 (0.75%) |
| Steps | 500 |
| Training time | 25 min (RTX 4090) |
| Platform | RunPod RTX 4090 |

**Loss curve:**

| Step | Train Loss |
|------|-----------|
| 50 | 2.073 |
| 100 | 0.993 |
| 150 | 0.929 |
| 200 | 0.907 |
| 250 | 0.899 |
| 300 | 0.898 |
| 350 | 0.885 |
| 400 | 0.898 |
| 450 | 0.892 |
| 500 | 0.919 |

Loss dropped from 2.07 → 0.89 and eval loss consistently decreased across all 5 checkpoints (0.936 → 0.914 → 0.903 → 0.896 → 0.894), confirming no overfitting at 500 steps.

---

## Key Differences from Classification

| Aspect | Classification (DistilBERT) | Generation (Llama 3.2) |
|--------|----------------------------|------------------------|
| Task | Sequence classification | Causal language modelling |
| Model type | Encoder-only | Decoder-only |
| LoRA targets | q_lin, v_lin (2 modules) | q/k/v/o_proj + gate/up/down_proj (7 modules) |
| Quantization | None (fp16) | 4-bit NF4 (QLoRA) |
| Trainer | WeightedTrainer (custom) | SFTTrainer (trl) |
| Loss masking | Not needed | Yes — prompt tokens masked |
| Batch size | 32 | 2 + grad accumulation 8 |
| Optimiser | AdamW | paged_adamw_8bit |
| Precision | fp16 | bf16 |

---

## Key Design Decisions

### 1. Why QLoRA (4-bit quantization)?
Llama 3.2 3B in fp16 requires ~6GB for weights alone, leaving very little room on a 16GB T4 for gradients, activations and LoRA matrices. 4-bit NF4 quantization compresses the weights to ~2.24GB — a 2.7x reduction — freeing enough memory for training.

**NF4 (Normal Float 4)** is better than standard FP4 for LLMs because the distribution of LLM weights is approximately normal (bell-curve shaped). NF4 places quantization levels at equal probability intervals rather than equal value intervals, which minimises quantization error for normally distributed data.

**`double_quant=True`** — quantizes the quantization constants themselves, saving an additional ~0.4GB.

### 2. Why unsloth mirror instead of official Meta weights?
The official `meta-llama/Llama-3.2-3B-Instruct` requires HuggingFace gated access approval which was pending. `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` is a publicly available pre-quantized mirror of the same weights — no access approval needed. For the thesis, this is noted as: *"loaded via unsloth's pre-quantized mirror to avoid gated access delays."*

### 3. Why SFTTrainer instead of standard Trainer?
SFTTrainer (Supervised Fine-Tuning Trainer) from the `trl` library adds two critical features for instruction tuning:
- **Automatic chat template application** — formats the messages dict into Llama's `<|start_header_id|>` template
- **Loss masking** — only computes loss on the assistant's response tokens, not the system prompt or user message. Without this, the model wastes capacity memorising the instruction format instead of learning to generate good responses

### 4. Why target attention + MLP modules?
```python
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "gate_proj", "up_proj", "down_proj",        # MLP
]
```
Llama uses SwiGLU activation in its MLP which has three projection matrices (gate, up, down). Including these alongside the attention projections gives the adapter more expressive capacity for generation tasks — the MLP controls content generation while attention controls contextual relationships. For generation tasks this is more important than for classification where attention alone is often sufficient.

### 5. Why `paged_adamw_8bit` optimiser?
Standard AdamW stores two momentum terms per parameter in fp32, using ~3× the model's parameter memory. For 24M trainable LoRA parameters this adds up quickly. `paged_adamw_8bit` stores optimizer states in 8-bit and pages them to CPU when GPU memory is tight, reducing optimizer memory by ~4×. This is essential for fitting training within 16GB VRAM.

### 6. Why `model.config.use_cache = False`?
The KV cache accelerates inference by storing previously computed key/value attention matrices so they don't need recomputation. During training with gradient checkpointing, PyTorch needs to recompute activations during the backward pass — the KV cache interferes with this by storing intermediate states. Disabling it makes training compatible with gradient checkpointing.

### 7. Why `model.config.pretraining_tp = 1`?
Tensor parallelism setting — tells the model we're running on a single GPU (tp=1). If left at higher values, the model tries to split computations across multiple GPUs which causes errors on single-GPU setups.

### 8. Why batch size 2 + gradient accumulation 8?
Each training sample contains a full conversation (system prompt + user message + response) tokenized to up to 512 tokens. With a 3B model loaded in 4-bit plus LoRA adapters, a batch size of 32 would cause OOM. Batch size 2 fits safely, and accumulating gradients over 8 steps before updating gives an effective batch size of 16 — matching the classification experiments for consistency.

### 9. Why `max_steps=500` instead of epochs?
The full dataset has 45,213 training samples. With batch size 2 × grad accumulation 8, one epoch = 2,826 steps. At ~3 seconds/step on RTX 4090, one full epoch = ~2.4 hours. The loss was clearly converging by step 350, so 500 steps (~25 mins) was sufficient for a meaningful, comparable result. This is documented in the thesis as a compute constraint, with both LoRA and ReFT trained under identical conditions.

### 10. System prompt design
```
"You are an empathetic mental health support assistant. A person has shared 
their situation and emotional state with you. Respond with warmth, 
understanding, and supportive intent. Acknowledge their feelings and provide 
gentle, non-clinical support."
```
The system prompt explicitly frames the model's role and constraints — non-clinical, empathetic, supportive. This is important because Llama 3.2 is a general-purpose assistant and without this framing it might give clinical or detached responses. The prompt also aligns with the Empathetic Dialogues dataset's design where responses are explicitly empathetic.

---

## Dataset Format

Each training example is formatted as:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an empathetic...<|eot_id|>
<|start_header_id|>user<|end_header_id|>
[anticipating] I have a big trip coming up...<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
I cant wait to see the beaches...<|eot_id|>
```

The prompt format includes the emotion tag `[emotion]` which helps the model understand the emotional context before generating a response. Some prompts also include a `| previous_turn` separator showing prior conversation turns.

---

## Issues Faced & How They Were Solved

### Issue 1 — Kaggle session timeout
**Problem:** Training on Kaggle T4 at 28s/step × 8,478 steps = 65+ hours, far exceeding the 9-hour session limit.  
**Fix:** Switched to RunPod RTX 4090 (3s/step) and capped at `max_steps=500`. RTX 4090 is ~9x faster than T4 for this task.

### Issue 2 — Accidental Ctrl+C interrupt on Kaggle
**Problem:** Training interrupted at step 135 while trying to check directory listing.  
**Learning:** Never run shell commands in the same terminal as a running training job. Use a separate terminal tab or Kaggle's file browser.

### Issue 3 — SSH key security incident
**Problem:** `runpod` and `runpod.pub` SSH key files accidentally committed to GitHub.  
**Fix:** Used `git filter-repo --invert-paths --force` to rewrite git history and remove the files. Then force-pushed to GitHub. New SSH key pair generated (`runpod_new`).  
**Learning:** Add `*.pem`, `id_*`, `*.pub` to `.gitignore` before starting any project that uses SSH keys.

### Issue 4 — `set_submodule` AttributeError with bitsandbytes
**Error:** `AttributeError: 'LlamaForCausalLM' object has no attribute 'set_submodule'`  
**Root cause:** Newer `transformers` (5.x) added `set_submodule` but PyTorch 2.4.1 on RunPod's base image didn't have it.  
**Fix:** Pinned `transformers==4.45.1` and `trl==0.11.4` to match the known-working Kaggle environment.

### Issue 5 — PyTorch/CUDA version mismatch after upgrade
**Error:** `CUDA initialization: The NVIDIA driver on your system is too old`  
**Root cause:** Upgraded to torch 2.11.0 which required CUDA driver 12.8+, but RunPod pod had CUDA 12.4.  
**Fix:** Reinstalled `torch==2.4.1+cu124` matching the CUDA version.

### Issue 6 — trl/transformers version conflicts
**Error chain:** `is_trackio_available` → `SFTConfig max_seq_length` → `processing_class` → `dataset_text_field`  
**Root cause:** Each trl version has different API than the next. `trl==0.11.4` + `transformers==4.45.1` is the stable combination.  
**Key API differences for trl 0.11.4 vs newer:**
| Feature | trl 0.11.4 | trl 0.29+ |
|---------|-----------|-----------|
| Sequence length | `max_seq_length` | `max_length` |
| Tokenizer arg | `tokenizer` | `processing_class` |
| Messages format | Needs `formatting_func` | Auto-handled |

### Issue 7 — bf16 error on Kaggle T4
**Error:** `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`  
**Root cause:** T4 doesn't natively support bfloat16. The unsloth model loaded in bf16 by default.  
**Fix:** `dtype=torch.float16` in model loading + `bf16=True, fp16=False` in SFTConfig.  
**Note:** This works because the unsloth 4-bit model's compute dtype is float16, and bf16=True in SFTConfig signals mixed precision without conflicting.

### Issue 8 — `formatting_func` required for trl 0.11.4
**Error:** `ValueError: You need to provide either dataset_text_field or formatting_func argument`  
**Root cause:** trl 0.11.4 doesn't auto-detect the `messages` format — newer versions do.  
**Fix:** Added `formatting_func` that calls `tokenizer.apply_chat_template()` to convert messages to text.

### Issue 9 — Checkpoint incompatibility between Kaggle and RunPod
**Error:** `TypeError: TrainerState.__init__() got an unexpected keyword argument 'best_global_step'`  
**Root cause:** Checkpoint saved with newer transformers on Kaggle, loaded with `transformers==4.45.1` on RunPod — `best_global_step` was added in a newer version.  
**Fix:** Deleted the checkpoint and trained from scratch. RTX 4090 makes 500 steps fast enough (~25 mins) that resuming wasn't necessary.

### Issue 10 — File transfer to RunPod
**Problem:** RunPod SSH doesn't support SCP/SFTP via its standard SSH port. The TCP port does support it but only with the correct key.  
**Solution:** Used `rclone` with Google Drive for file transfers. For checkpoint/dataset transfer: zip → upload to Google Drive → `gdown --fuzzy` on RunPod.  
**Learning:** Always back up datasets and adapters to Google Drive immediately after training completes.

---

## Platform Comparison

| Platform | GPU | VRAM | Time/step | 500 steps | Cost |
|----------|-----|------|-----------|-----------|------|
| Kaggle T4 | Tesla T4 | 16GB | ~28s | ~3.9 hrs | Free |
| RunPod RTX 4090 | RTX 4090 | 24GB | ~3s | ~25 min | ~$0.30 |

RTX 4090 is ~9x faster for this task. For EXP-04 (ReFT) and any future re-runs, RunPod is clearly the better choice.

---

## Files Saved After Training

| File | Location | Contents |
|------|----------|----------|
| `test_results.json` | `results/lora_generation/` | eval_loss on test set |
| `lora_adapter/` | `results/lora_generation/` | LoRA adapter weights + tokenizer |

The adapter is ~85MB — back up to Google Drive immediately after training.