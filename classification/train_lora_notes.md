# train_lora.py — Design Decisions & Concept Notes
> DistilBERT + LoRA for Mental Health Stress Classification  
> Quick reference for revision and interview prep

---

## What This Script Does

Fine-tunes DistilBERT for multi-class text classification on the SWMH mental health dataset using LoRA (Low-Rank Adaptation). Classifies Reddit posts into 5 categories: `anxiety`, `suicide_watch`, `bipolar`, `depression`, `no_stress`.

---

## Dataset

| Split | Rows |
|-------|------|
| Train | 36,875 |
| Val   | 7,902 |
| Test  | 7,902 |

**Labels:** `{0: anxiety, 1: suicide_watch, 2: bipolar, 3: depression, 4: no_stress}`

**Non-stress data source:** `r/offmychest` posts from within SWMH — not IMDB. Using IMDB was rejected because it risked the model learning formal vs informal language as a proxy for stress rather than actual stress signals.

---

## Key Design Decisions

### 1. Why DistilBERT?
- 40% smaller than BERT, 60% faster, retains 97% of BERT's performance
- Encoder-only — ideal for classification tasks (not generation)
- Stable, predictable training — no emergent behaviours unlike very large models
- Widely benchmarked, makes it easy to attribute performance differences to fine-tuning method rather than model architecture
- Same base model used in Nwaiwu (2025) comparative PEFT study — allows fair comparison

### 2. Why LoRA over full fine-tuning?
- Full fine-tuning updates all ~67M parameters — expensive and risks catastrophic forgetting
- LoRA freezes all pretrained weights and only trains two small low-rank matrices (A and B) per target layer
- Only 741,893 trainable parameters (~1.09% of total) — drastically reduces compute and memory
- No inference latency — at deployment, LoRA weights are merged back: `W = W₀ + AB`
- To switch tasks, subtract the delta to recover original weights

### 3. LoRA target modules: `["q_lin", "v_lin"]`
- **Critical DistilBERT-specific detail** — BERT uses `query`/`value`, but DistilBERT names them `q_lin` and `v_lin`
- Getting this wrong causes LoRA to silently train zero parameters on the wrong layers
- Query and value projections are targeted because they control what the model attends to and how it weighs information — the most task-relevant parts of the attention mechanism

### 4. Why class weights?
SWMH is imbalanced — depression has far more posts than other categories. Without correction, the model would learn to predict the majority class and still get decent accuracy while ignoring rare classes entirely.

**Formula:** `weight = total / (num_classes × class_count)`

| Class | Weight | Interpretation |
|-------|--------|----------------|
| anxiety | 1.10 | Slightly underrepresented |
| suicide_watch | 1.03 | Close to balanced |
| bipolar | 1.38 | Underrepresented |
| depression | 0.56 | Majority class — penalised less |
| no_stress | 1.61 | Most underrepresented — highest weight |

Weights are passed into `CrossEntropyLoss(weight=...)` so misclassifying a minority class produces a larger loss penalty, forcing the model to pay more attention to it.

**Why `cw_dict[str(i)]`?** JSON always serialises dictionary keys as strings, so integer index `i` must be converted to `str(i)` to match the stored keys.

### 5. max_length = 256
Reddit posts are rarely longer than 256 tokens. Using the full 512 would double memory usage with no meaningful benefit for this dataset.

### 6. WeightedTrainer
Subclasses HuggingFace `Trainer` and overrides `compute_loss()` to inject the class weights into the loss function. This is cleaner than patching the model itself and keeps the rest of the Trainer pipeline intact.

---

## Hyperparameters Explained

### LoRA Rank (r = 8)
Rank of the decomposition matrices A and B. Lower rank = fewer parameters, less expressive. Higher rank = more parameters, more expressive but risks overfitting. r=8 is a conservative, well-tested starting point for small models like DistilBERT. The effective weight update is: `ΔW = AB` where A ∈ R^(d×r) and B ∈ R^(r×d).

### LoRA Alpha (α = 16)
Controls how much the LoRA update influences the model. The actual scaling applied is `α / r = 16 / 8 = 2.0`. Think of it as a volume knob — higher alpha = LoRA has more influence over model behaviour, lower alpha = defers more to pretrained weights. Setting `alpha = 2 × r` is the standard starting point.

> **Interview tip:** The reason alpha and r are separate rather than just scaling r is to allow you to change the magnitude of the update independently from the rank. You might want a high-rank (expressive) update but at low scale, or a low-rank update at high scale.

### Weight Decay (0.01)
Regularisation that prevents overfitting by slightly shrinking weights toward zero after each update:

```
new_weight = old_weight - lr × (gradient + weight_decay × old_weight)
```

Discourages the model from over-relying on any single parameter. Light penalty of 0.01 is appropriate here since LoRA already has very few trainable parameters and doesn't need aggressive regularisation.

### Warmup Ratio (0.1)
At the start of training, LoRA matrices are essentially random and gradients are noisy. Starting with the full learning rate causes large noisy updates that destabilise early training.

Warmup ratio 0.1 means the learning rate starts near zero and linearly increases to `LEARNING_RATE` over the first 10% of training steps (~1,150 steps), then cosine decay takes over.

```
Step:  0%  →  10%      →      100%
LR:    0   →  2e-4     →      ~0
       ↑ warmup ↑   ↑ cosine decay ↑
```

### Cosine Decay
Learning rate schedule that follows a cosine curve from `lr_max` to near zero after warmup. Formula:

```
lr(t) = 0.5 × lr_max × (1 + cos(π × t/T))
```

Key characteristic — **slow start, fast middle, slow end:**
- Decreases slowly at first — stays high while model is making fast progress
- Decreases fastest in the middle — rapid weight refinement
- Decreases slowly at the end — fine-grained adjustments near convergence, avoids overshooting the optimum

**Why cosine over linear?** Transformer loss landscapes are not smooth — there are local minima and sharp curves. The gentle tapering at the end gives the model time to settle into a good minimum rather than bouncing around it with a still-too-large learning rate.

**Why cosine + warmup together?** Warmup handles unstable beginning, cosine handles smooth convergence at the end. They cover the full training lifecycle — this combination is now the de facto standard for fine-tuning transformers.

### Batch Size (32)
Standard for DistilBERT on a T4 GPU. Larger batches are more stable but use more VRAM. 32 fits comfortably within 16GB.

### Early Stopping (patience = 3)
Stops training if validation F1 doesn't improve for 3 consecutive epochs. Prevents wasted Kaggle GPU hours and overfitting. The best checkpoint is restored at the end via `load_best_model_at_end=True`.

---

## Evaluation Metrics

| Metric | Why it matters for this task |
|--------|------------------------------|
| F1 (weighted) | Balances precision and recall across imbalanced classes |
| Precision | Of posts predicted as stressed, how many actually are |
| Recall | Of all actually stressed posts, how many did we catch — **false negatives are costly** in mental health |
| AUC | How well the model separates classes across all thresholds, robust to imbalance |
| Accuracy | Baseline sanity check only — misleading on imbalanced data |

> **Interview tip:** Why not just accuracy? If 60% of samples are depression, a model that always predicts depression gets 60% accuracy but is useless. F1 and AUC don't have this problem.

---

## Confusion Matrix

Two plots generated:
1. **Raw counts** — absolute numbers, useful for spotting which classes are confused most often
2. **Normalised (row-normalised)** — each row sums to 1, showing per-class recall. Useful for comparing across classes of different sizes

Both are saved locally and logged to W&B automatically.

---

## sys.path.insert(0, ...) vs sys.path.append(...)

`append` adds the repo to the **end** of Python's search list — an installed package named `config` gets found first. `insert(0, ...)` puts the repo at the **front** so your local `config.py` always takes priority. This was a real bug encountered during training.

---

## W&B Integration

All hyperparameters, loss curves, eval metrics per epoch, and the confusion matrix are logged to Weights & Biases. This is important for the thesis because:
- Provides reproducible experiment tracking
- Easy side-by-side comparison with the ReFT run
- Loss curves show whether training was stable
- Confusion matrix directly in the dashboard without downloading from Kaggle

---

## Files Saved After Training

| File | Location | Contents |
|------|----------|----------|
| `test_results.json` | `results/lora_classification/` | F1, AUC, precision, recall on test set |
| `confusion_matrix_test.png` | `results/lora_classification/` | Raw + normalised confusion matrix |
| `lora_adapter/` | `results/lora_classification/` | LoRA adapter weights + tokenizer |

The adapter saved is just the LoRA delta weights — not the full model. To use it later, load DistilBERT base and attach the adapter.
