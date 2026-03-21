# Comparing LoRA and ReFT for Mental Health NLP

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange)

Master's thesis research comparing **LoRA** (Low-Rank Adaptation) and **ReFT** 
(Representation Fine-Tuning) — two parameter-efficient fine-tuning (PEFT) methods — 
on mental health NLP tasks. Evaluated on text classification and empathetic 
response generation.

---

## 🔍 Research Question

> Can ReFT match or outperform LoRA on sensitive NLP tasks while using fewer 
> trainable parameters?

---

## 🧠 Tasks

| Task | Model | Dataset |
|------|-------|---------|
| Mental health text classification | DistilBERT | SWMH + IMDB |
| Empathetic response generation | Llama 3.2 3B | Empathetic Dialogues |

---

## ⚙️ Methods Compared

| Method | Type | Trainable Params |
|--------|------|-----------------|
| Full Fine-Tuning (baseline) | Full | 100% |
| LoRA | Weight-space | ~0.1–1% |
| ReFT | Representation-space | ~0.1–1% |

---

## 📊 Results

> 🔄 Experiments in progress — results will be updated here

### Classification (DistilBERT on SWMH)

| Method | F1 | AUC | Precision | Recall | Trainable Params |
|--------|----|-----|-----------|--------|-----------------|
| Baseline (no fine-tuning) | - | - | - | - | 0 |
| LoRA | - | - | - | - | - |
| ReFT | - | - | - | - | - |

### Generation (Llama 3.2 3B on Empathetic Dialogues)

| Method | BERTScore | EPIC | LLM-as-Judge (1–5) | Trainable Params |
|--------|-----------|------|--------------------|-----------------|
| Baseline (no fine-tuning) | - | - | - | 0 |
| LoRA | - | - | - | - |
| ReFT | - | - | - | - |

---

## 🗂️ Project Structure
```
mental-health-nlp-peft/
│
├── config.py                        # Global seeds, paths, hyperparameters
├── requirements.txt                 # All dependencies
│
├── data/
│   └── prepare_data.py              # Download, clean, PII redact, save splits
│
├── classification/
│   ├── train_lora.py                # DistilBERT + LoRA training
│   ├── train_reft.py                # DistilBERT + ReFT training
│   └── evaluate.py                  # F1, AUC, confusion matrix
│
├── generation/
│   ├── train_lora_llama.py          # Llama 3.2 + LoRA training
│   ├── train_reft_llama.py          # Llama 3.2 + ReFT training
│   └── evaluate.py                  # BERTScore, EPIC, LLM-as-Judge
│
├── notebooks/
│   └── exploratory/                 # EDA and result analysis notebooks
│
└── results/
    └── figures/                     # Plots, confusion matrices, charts
```

---

## 🚀 Reproducing This Work

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/mental-health-nlp-peft.git
cd mental-health-nlp-peft
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare data
```bash
python data/prepare_data.py
```

### 4. Run classification experiments
```bash
# LoRA
python classification/train_lora.py

# ReFT
python classification/train_reft.py
```

### 5. Run generation experiments
```bash
# LoRA
python generation/train_lora_llama.py

# ReFT
python generation/train_reft_llama.py
```

> **Note:** Training was run on Kaggle (T4 GPU, 16GB VRAM) and RunPod 
> (RTX 4090, 24GB VRAM). A GPU with at least 16GB VRAM is recommended.

---

## 🛠️ Tech Stack

- **Models:** DistilBERT, Llama 3.2 3B Instruct
- **PEFT:** [HuggingFace PEFT](https://github.com/huggingface/peft) (LoRA), 
  [PyReFT](https://github.com/stanfordnlp/pyreft) (ReFT)
- **Training:** PyTorch, HuggingFace Transformers, Unsloth, Accelerate
- **Evaluation:** BERTScore, EPIC, scikit-learn, W&B
- **Privacy:** Microsoft Presidio (PII redaction)
- **Datasets:** SWMH (HuggingFace), IMDB, Empathetic Dialogues

---

## ⚠️ Ethical Considerations

This project uses mental health-related social media text. All data is:
- Sourced from publicly available, research-approved datasets
- Processed through PII redaction (Microsoft Presidio) before use
- Used strictly for academic research purposes

---

## 👤 Author

**[Your Name]**  
MSc [AI/ML] — [Liverpool Johm Moores University]  
[uplcpriy@ljmu.ac.uk] | [www.linkedin.com/in/chetnapriyadarshini]

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

