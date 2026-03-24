# train_reft.py — Design Decisions & Concept Notes
> DistilBERT + LoReFT (Manual Implementation) for Mental Health Stress Classification  
> Quick reference for revision and interview prep

---

## What This Script Does

Fine-tunes DistilBERT for multi-class stress classification on the SWMH dataset using a **manual implementation of LoReFT** (Low-Rank Linear Subspace ReFT). Instead of modifying weight matrices like LoRA, ReFT modifies the hidden representations at specific positions in each transformer layer. All base model weights remain completely frozen — only the intervention parameters are trained.

---

## Why Manual Implementation (No pyreft/pyvene)?

The original plan was to use the official `pyreft` library. After extensive debugging, it was abandoned due to:

| Issue | Root Cause |
|-------|-----------|
| `ImportError: cannot import name 'is_flax_available'` | pyvene pulled a newer transformers version incompatible with pinned 4.45.1 |
| `AttributeError: object has no attribute ''` | `"block_output"` shorthand not supported for DistilBERT in pyvene |
| `UnboundLocalError: hook_type not associated with value` | pyvene's path parser couldn't handle bracket notation `layer[0]` |
| `ValueError: invalid literal for int() with base 10: '%s'` | pyvene's `%s` template substitution failed for DistilBERT paths |
| `KeyError: Parameter containing: tensor(...)` | pyvene's `IntervenableModel` incompatible with PyTorch DataParallel |
| `TypeError: IntervenableModel.forward() missing 'base'` | pyvene expects `model(base=inputs)` not standard `model(**inputs)` |
| `UnboundLocalError: batch_size not associated with value` | pyvene's `get_batch_size()` couldn't parse dict-style inputs |

**Lesson:** pyvene is designed for research with LLaMA-family models. It has limited support for encoder-only models like DistilBERT and version conflicts with pinned transformers. The manual implementation is more transparent, fully controllable, and actually better for a thesis since every line is explainable.

---

## LoReFT — Core Concept

### What ReFT Does Differently from LoRA

| | LoRA | ReFT |
|--|------|------|
| What it modifies | Weight matrices | Hidden representations |
| Formula | W = W₀ + AB | h_new = h + Rᵀ(Wh + b − Rh) |
| Where it operates | Weight space | Representation space |
| Params vs LoRA | Baseline | ~10-15x fewer |

### The LoReFT Formula
```
h_new = h + Rᵀ(Wh + b − Rh)
```

Breaking down each term:

**`h`** — the hidden representation at the intervened position (CLS or last token), shape `(batch, 768)`

**`R`** — low-rank orthonormal projection matrix `(rank × embed_dim)`. Projects representations down to the low-rank subspace. Initialised with `torch.nn.init.orthogonal_()` to ensure the subspace is well-conditioned at the start of training.

**`W`** — learned linear transform `(embed_dim → rank)` with bias. Learns where the representation *should* move in the subspace.

**`Wh + b`** — the learned target position in the subspace

**`Rh`** — the current position of `h` projected into the subspace

**`Wh + b − Rh`** — the *difference* between where we want to go and where we currently are. This is the key design choice — by subtracting `Rh`, the intervention is a **residual correction** rather than an absolute offset. If the representation is already well-positioned, the intervention is small. If it needs adjustment, the intervention is large.

**`Rᵀ(...)` or `(...) @ R.weight`** — project the edit back up from rank-dimensional subspace to full 768-dimensional space

> **Interview tip:** Why subtract `Rh`? Because we're editing *relative to the current state* of the representation, not adding an absolute vector. This makes the intervention more stable and interpretable — you're asking "how far is h from where it should be in the subspace?" rather than "add this fixed vector to h".

### Shapes walkthrough (embed_dim=768, rank=4, batch=32)
```
h_pos           (32, 768)   — hidden state at CLS token
R(h_pos)        (32, 4)     — project down to subspace
W(h_pos)        (32, 4)     — learned target in subspace
difference      (32, 4)     — edit needed
@ R.weight      (32, 768)   — project edit back up (R.weight is (4, 768))
h_pos + edit    (32, 768)   — updated representation
```

---

## Key Design Decisions

### 1. Why forward hooks instead of subclassing?
Forward hooks attach to any module and fire automatically during the model's forward pass — no changes to DistilBERT's internal code needed. The hook intercepts the output of `output_layer_norm`, passes it through `LoReFTIntervention`, and returns the modified tensor. This is cleaner than subclassing because DistilBERT's forward method doesn't need to be touched at all.

### 2. Why `output_layer_norm`?
This is the final operation in each DistilBERT transformer block — after attention, FFN, and residual connections. Its output is the representation that flows to the next layer. Intervening here means:
- The representation is already normalised (stable scale)
- It's the most information-rich signal at that layer
- It's the conceptual equivalent of `block_output` in the ReFT paper

### 3. Why intervene on CLS + last token `[0, -1]`?
- **CLS token (pos 0)** — DistilBERT uses CLS for classification via mean pooling. It aggregates global sequence meaning.
- **Last token (pos -1)** — carries the final contextual state after attending to the full sequence. Provides complementary information, especially for longer posts.
- The ReFT paper found `f1+l1` (first + last) consistently outperformed CLS-only for classification.
- Zero extra parameters — the same R and W matrices are applied at both positions.

### 4. Why `h = h.clone()` in the intervention?
Without cloning, the in-place operation `h[:, pos, :] = ...` modifies the tensor that was passed in by reference. During fp16 training, PyTorch's autocast mechanism can't correctly backpropagate through in-place operations on shared tensors, causing gradient errors. `clone()` creates a new tensor so the modification is safe.

### 5. Why `make_hook(iv)` closure pattern?
```python
def make_hook(iv):
    def hook(module, input, output):
        return iv(output)
    return hook

handle = layer_norm.register_forward_hook(make_hook(intervention))
```
Without the closure, all 6 hooks would capture the same variable `intervention` from the loop — whichever value it has at the end of the loop (layer 5's intervention). The closure captures `iv` by value at the time `make_hook` is called, so each hook correctly points to its own layer's intervention. This is a classic Python loop capture bug.

### 6. Why freeze ALL base model parameters?
ReFT's core claim is that representations already encode rich task-relevant information — we just need to steer them, not retrain the weights. Freezing everything except intervention parameters:
- Preserves pretrained language understanding
- Prevents catastrophic forgetting
- Ensures any performance difference vs LoRA is due to the fine-tuning method, not different parameter counts in the base model

### 7. REFT_RANK = 4 vs LoRA r = 8
LoRA with r=8 adds 741,893 trainable parameters. ReFT with rank=4 across 6 layers and 2 positions adds far fewer. Per intervention: `R` has `rank × embed_dim = 4 × 768 = 3,072` params, `W` has `embed_dim × rank + rank = 3,076` params. For 6 layers: `6 × (3,072 + 3,076) = 36,888` trainable parameters — roughly **20x fewer than LoRA**. This is the core efficiency claim being tested in the thesis.

---

## Padding Strategy — Why Different from LoRA

| | train_lora.py | train_reft.py |
|--|---------------|---------------|
| Tokenization padding | `padding=False` | `padding="max_length"` |
| Collator | `DataCollatorWithPadding` | `default_data_collator` |

**Why the difference?** `DataCollatorWithPadding` calls `tokenizer.pad()` at batch time. When the model is a custom `nn.Module` (our `DistilBertWithLoReFT`), the Trainer's data pipeline processes inputs differently and `DataCollatorWithPadding` only received `['label']` instead of the full encoding — causing a `ValueError`.

The fix: pad all sequences to `max_length=256` during tokenisation so all tensors are already the same shape. `default_data_collator` then just stacks them — no padding logic needed, nothing to break.

**Tradeoff:** Every sequence is padded to 256 tokens regardless of actual length. `DataCollatorWithPadding` was smarter (pads to longest in batch). Slightly more memory, but correct.

**Fair comparison:** Both scripts truncate at `max_length=256` — the padding strategy doesn't affect what the model sees, only memory usage.

---

## Issues Faced & How They Were Solved

### Issue 1 — `sys.path.append` picking up wrong `config`
**Error:** `AttributeError: 'tuple' object has no attribute 'SEED'`  
**Cause:** Python's installed packages contain a module also named `config`. `sys.path.append` adds the repo to the END of the search path, so the installed one was found first.  
**Fix:** `sys.path.insert(0, ...)` — puts repo at the FRONT of the search path.

### Issue 2 — pyreft/pyvene version conflicts (entire chain)
**Error chain:** `is_flax_available` → `block_output` → `hook_type` → `%s` → DataParallel → `base` argument → `batch_size`  
**Root cause:** pyvene is built for LLaMA-family decoder models. DistilBERT's module naming, encoder-only architecture, and HuggingFace Trainer pipeline are all incompatible with pyvene's assumptions.  
**Fix:** Dropped pyreft entirely. Implemented LoReFT manually using PyTorch forward hooks.

### Issue 3 — `TOKENIZERS_PARALLELISM` deadlock
**Cause:** HuggingFace tokenizers use Rust-based multithreading. When the Trainer forks processes for data loading, forked processes inherit the threaded tokenizer state, causing deadlocks.  
**Fix:** `os.environ["TOKENIZERS_PARALLELISM"] = "false"` — disables tokenizer threading so forked processes are safe.

### Issue 4 — DataParallel KeyError on intervention parameters
**Error:** `KeyError: Parameter containing: tensor(...)`  
**Cause:** PyTorch DataParallel tries to replicate all model parameters across GPUs. ReFT's intervention parameters registered via `nn.ModuleList` were not visible to DataParallel's parameter index.  
**Fix:** `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` — forces single GPU, preventing DataParallel from triggering. No performance cost on Kaggle T4 (single GPU anyway).

### Issue 5 — In-place modification error during fp16 training
**Cause:** `h[:, pos, :] = h_pos + intervention` modifies the tensor in-place. fp16 autocast can't correctly backpropagate through in-place ops on tensors shared with the computation graph.  
**Fix:** `h = h.clone()` at the start of `LoReFTIntervention.forward()`.

### Issue 6 — All hooks pointing to the same intervention (loop capture bug)
**Cause:** Python closures capture variables by reference. Without `make_hook(iv)`, all 6 hooks captured the same `intervention` loop variable — which after the loop ended pointed to layer 5's intervention.  
**Fix:** `make_hook(iv)` closure — captures `iv` by value at the time each hook is created.

---

## Parameter Count Comparison

| Method | Trainable Params | % of Total | Notes |
|--------|-----------------|------------|-------|
| LoRA (r=8) | 741,893 | 1.09% | q_lin + v_lin across 6 layers |
| LoReFT (rank=4) | ~36,888 | ~0.05% | R + W across 6 layers, 2 positions |
| Full fine-tuning | ~67.7M | 100% | All parameters |

> **Interview tip:** ReFT's parameter efficiency claim is that representation space is more information-dense than weight space — a rank-4 intervention on representations can achieve similar task adaptation as a rank-8 weight update because the representations already encode the pretrained knowledge structure.

---

## Evaluation Metrics (identical to LoRA for fair comparison)

Same as LoRA — F1, Precision, Recall, Accuracy, AUC. See `train_lora_notes.md` for full explanation.

---

## Files Saved After Training

| File | Location | Contents |
|------|----------|----------|
| `test_results.json` | `results/reft_classification/` | F1, AUC, precision, recall on test set |
| `confusion_matrix_test.png` | `results/reft_classification/` | Raw + normalised confusion matrix |
| `reft_interventions.pt` | `results/reft_classification/reft_model/` | Intervention weights only (R, W, b per layer) |

The saved `.pt` file contains only the 6 intervention modules — not the base model. To use it later, recreate `DistilBertWithLoReFT` and load the state dicts back into `self.interventions`.

---

## What to Look For in Results

When comparing against LoRA:
- **If F1 is similar** — ReFT achieves comparable performance with ~20x fewer parameters. Strong thesis finding.
- **If F1 is lower by 1-2%** — consistent with Nwaiwu (2025) findings. Acceptable tradeoff for parameter efficiency.
- **If confusion patterns differ** — interesting finding about what each method learns. Does ReFT handle the anxiety/depression overlap differently?
- **AUC** — if AUC is similar but F1 differs, the model separates classes well but the decision boundary is different. Worth analysing.