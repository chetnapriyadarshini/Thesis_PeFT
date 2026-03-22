# Thesis Research Notes — Phase 1: Environment Setup & Data Pipeline
**Project:** Comparing LoRA and ReFT for Mental Health NLP  
**Program:** MSc AI/ML — Liverpool John Moores University  
**Date:** March 2026  
**Status:** Data pipeline complete, training phase next

---

## 1. What This Research Is About

This thesis compares two Parameter-Efficient Fine-Tuning (PEFT) methods — **LoRA** (Low-Rank Adaptation) and **ReFT** (Representation Fine-Tuning) — on mental health NLP tasks.

The core research question: *Can ReFT match or outperform LoRA on sensitive NLP tasks while using fewer trainable parameters?*

### Two experimental tasks:
| Task | Model | Dataset |
|------|-------|---------|
| Mental health text classification (5 classes) | DistilBERT | SWMH |
| Empathetic response generation | Llama 3.2 3B | Empathetic Dialogues |

### Why PEFT matters:
Full fine-tuning of large language models requires updating billions of parameters — computationally expensive and prone to catastrophic forgetting. PEFT methods freeze most of the model and only train a small fraction of parameters, making fine-tuning accessible on consumer hardware.

- **LoRA** injects trainable low-rank matrices into the model's weight space
- **ReFT** intervenes on the model's internal representations (hidden states) rather than weights — a fundamentally different approach

---

## 2. Technical Stack Decisions

### Why these tools:
| Tool | Purpose | Why chosen |
|------|---------|------------|
| DistilBERT | Classification backbone | Lightweight, fast, well-benchmarked for text classification |
| Llama 3.2 3B | Generation backbone | Small enough for T4 GPU, strong baseline |
| HuggingFace PEFT | LoRA implementation | Industry standard, well-maintained |
| PyReFT | ReFT implementation | Official Stanford NLP library for ReFT |
| Unsloth | Faster Llama training | Optimised LoRA for Llama, separate env needed |
| Presidio | PII redaction | Microsoft's production-grade anonymisation |
| W&B | Experiment tracking | Reproducible logging of all runs |
| Kaggle | Training compute | Free T4 GPU (16GB), 30hrs/week |

### Key dependency conflict discovered and resolved:
PyReFT requires `transformers==4.45.1` while Unsloth requires `transformers>=4.51.3`. These cannot coexist — solved by maintaining **two separate Kaggle notebooks**:

- `peft_thesis_classification` → transformers==4.45.1, pyreft, peft
- `peft_thesis_generation` → transformers>=4.51.3, unsloth, trl

**Lesson learned:** Always check dependency trees before starting. When libraries share a core dependency but pin different versions, the cleanest solution is environment isolation rather than version negotiation.

### Local vs Cloud workflow:
```
Local (VS Code)          Cloud (Kaggle)
────────────────         ──────────────
Write all code     →     Run training
Git commits        →     Data pipelines  
EDA analysis       →     Heavy compute
Read results       →     Save outputs
```

**The golden rule:** Never write important code only on Kaggle. Always push to GitHub first.

---

## 3. Repository Structure

GitHub: `mental-health-nlp-peft` (Public, MIT License)

```
mental-health-nlp-peft/
├── config.py                        ← Global seeds, paths, hyperparameters
├── requirements-classification.txt  ← transformers==4.45.1, pyreft
├── requirements-generation.txt      ← transformers>=4.51.3, unsloth
├── data/
│   ├── prepare_data_classification.py
│   └── prepare_data_generation.py
├── classification/
│   ├── train_lora.py
│   ├── train_reft.py
│   └── evaluate.py
├── generation/
│   ├── train_lora_llama.py
│   ├── train_reft_llama.py
│   └── evaluate.py
├── notebooks/exploratory/
└── results/figures/
```

**Why public repo:** Portfolio visibility for job hunting. Mental health NLP + PEFT is a current, interesting intersection that stands out to recruiters. The README serves as a project one-pager even before results are in.

---

## 4. Dataset: Classification (SWMH)

### Source
SWMH (Social media for Mental Health) — gated dataset on HuggingFace (AIMH/SWMH). Requires access request approval.

### Raw structure
- 54,412 total rows across train/validation/test splits
- Columns: `text`, `label`
- 5 subreddits used as class sources

### Initial label distribution
| Subreddit | Count | Nature |
|-----------|-------|--------|
| self.depression | 18,746 | Clinical |
| self.SuicideWatch | 10,182 | Clinical |
| self.Anxiety | 9,555 | Clinical |
| self.offmychest | 8,284 | General venting |
| self.bipolar | 7,645 | Clinical |

### Key EDA finding: offmychest is not a mental health subreddit
Manual inspection of 10 random `offmychest` samples revealed it contains general life venting — relationship drama, work frustrations, financial regrets — not clinical mental health content. This made it a candidate for a "no stress" / neutral class, avoiding the need to import a completely different domain (e.g., IMDB movie reviews).

**Why this matters for thesis:** Using Reddit text as the neutral class (same domain, same writing style) is more defensible than cross-domain mixing. A model distinguishing depression from movie reviews is a much easier and less interesting task than distinguishing depression from general emotional venting.

### The keyword filtering decision
To use `offmychest` as a neutral class, rows containing clinical mental health keywords were removed.

**Initial keyword list:** `suicide, suicidal, self harm, self-harm, overdose, depression, depressed, anxiety, bipolar, schizophrenia, mental illness, mental health, cutting, panic attack, ptsd, trauma, eating disorder, anorexia, bulimia, psychosis, hallucin`

**Keywords removed after deliberation:**
- `cutting` — too general (cutting food, cutting costs)
- `therapy, therapist, medication, psychiatrist` — normalised in modern discourse; someone discussing their friend's psychiatrist is not necessarily in mental health distress
- `anxious` — used colloquially ("I'm anxious about my exam")

**Result:** 1,733 rows removed (21% of offmychest), leaving 6,551 clean neutral rows.

### Why not downsample depression?
Depression has 18,746 rows vs bipolar's 7,645 — a 2.45x imbalance. Two options considered:

**Downsampling:** Cap depression at ~8,000 rows.  
**Weighted loss:** Keep all data, adjust loss function to penalise misclassification of minority classes.

**Decision: Weighted loss.** Reasons:
- Downsampling discards ~10,000 real signal rows permanently
- Weighted loss is mathematically equivalent but keeps all training signal
- 2.45x is a mild imbalance — weighted loss handles this well
- More academically standard for this imbalance range
- Both LoRA and ReFT receive identical weights → fair comparison preserved

### Class weights computed
Formula: `weight = total / (num_classes × class_count)`

| Class | Count | Weight |
|-------|-------|--------|
| anxiety | 9,555 | 1.10 |
| suicide_watch | 10,182 | 1.03 |
| bipolar | 7,645 | 1.38 |
| depression | 18,746 | 0.56 |
| no_stress | 6,551 | 1.61 |

### Data quality
EDA confirmed zero nulls, empty texts, `[deleted]` rows, or sub-10-word texts across all classes. No cleaning required beyond the keyword filter.

### Final splits (70/15/15)
| Split | Rows |
|-------|------|
| Train | 36,875 |
| Val | 7,902 |
| Test | 7,902 |
| **Total** | **52,679** |

### PII Redaction
All text processed through Microsoft Presidio before saving. Took ~27 minutes on Kaggle T4. Saved as a permanent Kaggle output dataset to avoid re-running.

---

## 5. Dataset: Generation (Empathetic Dialogues)

### Source issues encountered
- `facebook/empathetic_dialogues` on HuggingFace uses a legacy dataset script → `RuntimeError: Dataset scripts are no longer supported`
- Alternative HuggingFace mirrors lacked dialogue response columns
- **Solution:** Kaggle dataset by atharvjairath — full conversations preserved

### Raw structure
- 64,636 rows, columns: `Situation`, `emotion`, `empathetic_dialogues`, `labels`
- `empathetic_dialogues`: contains `"Customer: [utterance]\nAgent: "` format
- `labels`: Agent's empathetic response

### EDA findings
- 42 corrupted rows where situation text leaked into the emotion column (e.g., `"t even like scary things"` as an emotion)
- 32 valid emotion categories: surprised, excited, angry, proud, sad, lonely, afraid, etc.
- `Unnamed: 5/6` columns: near-entirely null, dropped

### Prompt format decision
Two options considered:

**Option A — Utterance only:**
```
"This was a best friend. I miss her."
```

**Option B — Emotion + Situation + Utterance:**
```
"[sentimental] I remember going to the fireworks... | This was a best friend. I miss her."
```

**Decision: Option B.** Richer context allows the model to generate emotion-aware empathetic responses. More interesting for thesis — the model must learn to condition on emotional state, not just respond to surface text.

### Data pipeline
- Drop 42 corrupted emotion rows
- Parse Customer utterance from dialogue string
- Build prompt: `[{emotion}] {Situation} | {customer_utterance}`
- Response: `labels` column (Agent reply)
- PII redaction on both prompt and response
- 70/15/15 split

---

## 6. Decisions Log (For Thesis Methodology Section)

| Decision | Options Considered | Choice | Rationale |
|----------|-------------------|--------|-----------|
| Neutral class source | IMDB vs offmychest | offmychest | Same domain, more defensible |
| offmychest filtering | Keep all / keyword filter / drop entirely | Keyword filter | Removes clinical overlap while preserving neutral signal |
| Keywords removed | therapy, therapist, medication, psychiatrist, anxious, cutting | Removed | Too general for everyday usage |
| Class imbalance | Downsample / weighted loss | Weighted loss | Preserves all training signal |
| Prompt format | Utterance only / emotion+situation+utterance | Full context | Richer generation signal |
| Compute platform | AWS/Azure/Vertex/Kaggle | Kaggle (free tier) | Sufficient for DistilBERT; generation will use paid RunPod if needed |
| Environment split | Single env / two envs | Two notebooks | pyreft/unsloth dependency deadlock |

---

## 7. What's Next

- [ ] Run `prepare_data_generation.py` and verify output
- [ ] Write `classification/train_lora.py`
- [ ] Write `classification/train_reft.py`
- [ ] Write `classification/evaluate.py`
- [ ] Run first training experiments on Kaggle
- [ ] Fill in results table in README

---

## 8. Blog Post Outline (Draft)

**Title ideas:**
- *"Building a Mental Health NLP Classifier: The Decisions Nobody Talks About"*
- *"LoRA vs ReFT: Setting Up a Fair Comparison for Sensitive NLP"*
- *"What I Learned Preparing Data for a Mental Health AI Thesis"*

**Narrative arc:**
1. **Hook** — Why mental health NLP is hard (sensitive data, class ambiguity, ethical stakes)
2. **The setup problem** — pyreft vs unsloth dependency deadlock and how two environments solved it
3. **The data problem** — SWMH looks like a clean 5-class dataset until you inspect offmychest
4. **The interesting decision** — Why movie reviews are a bad neutral class for Reddit mental health data
5. **The keyword filtering dilemma** — Where do you draw the line? "therapy" removed vs "suicide" kept
6. **The imbalance question** — Downsample vs weighted loss: why the answer isn't obvious
7. **What's next** — LoRA and ReFT training, comparing methods fairly

**Key insight for blog:** The most interesting work in ML projects isn't the model training — it's the data decisions. Every decision in this pipeline (which keywords to filter, whether to downsample, what to use as a neutral class) will affect final metrics and needs to be defensible.

---

*Notes compiled: March 2026*  
*Next update: After first training run results*
