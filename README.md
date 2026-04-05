# Comparing LoRA and ReFT for Mental Health NLP

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers%204.45.1-FFD21E?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

Master's thesis research comparing **LoRA** (Low-Rank Adaptation) and **ReFT**
(Representation Fine-Tuning) — two parameter-efficient fine-tuning (PEFT) methods —
on mental health NLP tasks. Evaluated on text classification and empathetic
response generation using DistilBERT and Llama 3.2 3B Instruct.

---

## 🔍 Research Questions

> - Can ReFT match or outperform LoRA on sensitive NLP tasks while using fewer trainable parameters?
> - Does extreme parameter efficiency of ReFT cause it to lose task performance compared to LoRA?
> - Do LoRA and ReFT behave differently for classification versus generation tasks in the mental health domain?

---

## 🧠 Tasks & Models

| Task | Model | Dataset |
|------|-------|---------|
| Mental health text classification (5-class) | DistilBERT-base-uncased | SWMH + r/offmychest |
| Empathetic response generation | meta-llama/Llama-3.2-3B-Instruct | Estwld/empathetic_dialogues_llm |

---

## 📊 Results

### Classification (DistilBERT on SWMH)

| Method | F1 | AUC | Precision | Recall | Trainable Params |
|--------|----|-----|-----------|--------|-----------------|
| **LoRA** | **0.6985** | **0.904** | **0.710** | **0.700** | 741,893 (1.09%) |
| ReFT | 0.6555 | 0.8786 | 0.669 | 0.655 | ~36,888 (~0.05%) |

**Per-class recall (ReFT):** anxiety=0.74, suicide_watch=0.66, bipolar=0.71, depression=0.55, no_stress=0.76

**Key finding:** ReFT uses **20x fewer parameters** than LoRA with a 4.3% F1 gap. Critical failure: bidirectional suicide_watch↔depression confusion (0.20/0.19) in ReFT — a high-stakes finding for mental health applications.

---

### Generation (Llama 3.2 3B on Empathetic Dialogues)

#### Training Metrics

| Method | Eval Loss | Train Loss | Trainable Params | Time |
|--------|-----------|------------|-----------------|------|
| **LoRA** | **1.032** | 1.043 | 24,313,856 (0.75%) | 16 min |
| ReFT | 1.132 | 1.634 | 688,240 (0.038%) | 24 min |

#### Evaluation Metrics (500 test samples, GPT-4o judge)

| Method | BERTScore F1 | EPIC Overall | Emotional Recognition | Emotional Validation | Supportive Intent |
|--------|-------------|-------------|----------------------|---------------------|------------------|
| **LoRA** | 0.7008 | **4.888** | **4.862** | **4.880** | **4.932** |
| ReFT | **0.7041** | 2.944 | 3.342 | 2.908 | 3.038 |

**Key findings:**
- LoRA achieves near-perfect EPIC empathy scores (~4.9/5) vs ReFT ~2.9/5 — a **66% relative gap**
- ReFT marginally outperforms LoRA on BERTScore F1 (0.7041 vs 0.7008) despite severe empathy deficits — demonstrating that **BERTScore alone is insufficient** for evaluating mental health dialogue quality
- ReFT exhibits repetition loops on emotionally intense prompts, indicating rank-4 subspace interventions are insufficient for coherent long-form empathetic generation
- ReFT uses **35x fewer parameters** than LoRA

---

### Ablation Studies (ReFT Generation)

| Config | BERTScore F1 | EPIC Overall | Finding |
|--------|-------------|-------------|---------|
| ReFT-500 `[0,-1]` (baseline) | **0.7041** | **2.944** | Best ReFT config |
| ReFT-800 `[0,-1]` | 0.6659 | 1.934 | Overfitting — more steps hurt |
| ReFT-500 `[0,1,2,-3,-2,-1]` | 0.6582 | 2.038 | Expanded positions hurt — rank bottleneck |

**Ablation conclusion:** Rank-4 is the fundamental bottleneck — neither more training steps nor more intervention positions improve ReFT's empathetic generation quality.

---

## 🗂️ Project Structure

```
Thesis_PeFT/
│
├── config.py                           # Global seeds, paths, hyperparameters
├── requirements.txt                    # Pinned dependencies (torch==2.4.1+cu124)
│
├── data/
│   ├── prepare_data_classification.py  # SWMH + r/offmychest pipeline
│   └── prepare_data_generation.py      # Estwld/empathetic_dialogues_llm pipeline
│                                       # (HuggingFace, correct multi-turn role assignments)
│
├── classification/
│   ├── train_lora.py                   # DistilBERT + LoRA (EXP-01)
│   └── train_reft.py                   # DistilBERT + manual LoReFT hooks (EXP-02)
│
├── generation/
│   ├── train_lora_llama.py             # Llama 3.2 3B + QLoRA (EXP-03)
│   ├── train_reft_llama.py             # Llama 3.2 3B + manual LoReFT hooks (EXP-04)
│   └── evaluate.py          # BERTScore + GPT-4o EPIC judge
│
├── notebooks/
│   └── exploratory/                    # EDA and result analysis
│
└── results/
    └── figures/                        # Plots, confusion matrices, charts
```

---

## 🚀 Reproducing This Work

### 1. Clone the repo

```bash
git clone https://github.com/chetnapriyadarshini/Thesis_PeFT.git
cd Thesis_PeFT
```

### 2. Install dependencies

```bash
# Install PyTorch with CUDA 12.4 first (required)
pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Then install remaining dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

> **Note:** Pinned versions are critical. The combination of `transformers==4.45.1` + `trl==0.11.4` + `bitsandbytes==0.43.3` is required for compatibility. See `requirements.txt` for full details.

### 3. Set environment variables

```bash
export HF_TOKEN=your_huggingface_token      # Required for Llama 3.2 gated access
export WANDB_API_KEY=your_wandb_key          # Required for experiment tracking
export OPENAI_API_KEY=your_openai_key        # Required for GPT-4o evaluation judge
```

### 4. Prepare datasets

```bash
# Classification (SWMH + r/offmychest)
python data/prepare_data_classification.py

# Generation (Estwld/empathetic_dialogues_llm — downloads from HuggingFace)
python data/prepare_data_generation.py
```

### 5. Run classification experiments

```bash
python classification/train_lora.py   # EXP-01: LoRA
python classification/train_reft.py   # EXP-02: ReFT
```

### 6. Run generation experiments

```bash
python generation/train_lora_llama.py  # EXP-03: LoRA
python generation/train_reft_llama.py  # EXP-04: ReFT
```

### 7. Evaluate generation models

```bash
python generation/evaluate_generation.py
```

> **Hardware:** Classification on Kaggle T4 (16GB). Generation on RunPod RTX 4090 (24GB). Minimum 16GB VRAM recommended for generation experiments.

---

## ⚙️ Implementation Notes

### ReFT — Manual Implementation
ReFT is implemented using **manual PyTorch forward hooks** rather than the official `pyreft`/`pyvene` libraries, which had 7+ compatibility errors with both DistilBERT and Llama 3.2.

The LoReFT formula applied at each intervention point:
```
h_new = h + Rᵀ(Wh + b − Rh)
```
Where R is an orthonormal low-rank projection matrix and W is a learned transform. Hooks are registered on `output_layer_norm` (DistilBERT) and `post_attention_layernorm` (Llama).

### Dataset — Generation
The original Kaggle `emotion-emotion_69k.csv` had systematic **role confusion** in multi-turn conversations — agent turns were being parsed as user turns. Replaced with [`Estwld/empathetic_dialogues_llm`](https://huggingface.co/datasets/Estwld/empathetic_dialogues_llm), which has correct user/assistant role assignments. Only conversations where the **last turn is the assistant** are retained as training pairs.

---

## 🛠️ Tech Stack

| Component | Library / Tool |
|-----------|---------------|
| Models | DistilBERT-base-uncased, meta-llama/Llama-3.2-3B-Instruct |
| LoRA | HuggingFace PEFT |
| ReFT | Manual PyTorch hooks (no pyreft dependency) |
| Quantization | BitsAndBytes 4-bit NF4 (QLoRA) |
| Training | PyTorch, HuggingFace Transformers, TRL, Accelerate |
| Evaluation | BERTScore, GPT-4o (EPIC judge), scikit-learn |
| Tracking | Weights & Biases |
| PII Redaction | Microsoft Presidio |
| Datasets | SWMH, r/offmychest, Estwld/empathetic_dialogues_llm |

---

## ⚠️ Ethical Considerations

This project uses mental health-related social media text. All data is:
- Sourced from publicly available, research-approved datasets
- Processed through PII redaction (Microsoft Presidio) before use
- Used strictly for academic research purposes
- Not used for clinical diagnosis or treatment

---

## 👤 Author

**Chetna Priyadarshini**
MSc Artificial Intelligence — Liverpool John Moores University
[uplcpriy@ljmu.ac.uk](mailto:uplcpriy@ljmu.ac.uk) | [LinkedIn](https://www.linkedin.com/in/chetnapriyadarshini)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
