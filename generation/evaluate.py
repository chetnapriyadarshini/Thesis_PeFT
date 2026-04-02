# =============================================================================
# generation/evaluate_generation.py
# Evaluation script for LoRA and ReFT generation models
#
# Metrics:
#   1. BERTScore  — semantic similarity between generated and reference responses
#   2. LLM-as-Judge (GPT-4o) on EPIC dimensions:
#        - Emotional Recognition  (1-5): Does the model identify the emotion correctly?
#        - Emotional Validation   (1-5): Does the model acknowledge and validate feelings?
#        - Supportive Intent      (1-5): Is the response genuinely supportive?
#        - Tone Appropriateness   (1-5): Is the tone warm, non-clinical, non-judgmental?
#        - Overall Empathy        (1-5): Holistic empathy score
#
# Models evaluated:
#   - EXP-03: Llama 3.2 3B + LoRA adapter
#   - EXP-04: Llama 3.2 3B + LoReFT interventions (manual hooks)
#
# Usage:
#   export OPENAI_API_KEY=your_key_here
#   python generation/evaluate_generation.py
#
# Install:
#   pip install bert-score openai
# =============================================================================

import os, sys, json, random, warnings, time
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"]   = "0"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from bert_score import score as bert_score
from openai import OpenAI

import config

# =============================================================================
# 0.  Configuration
# =============================================================================
random.seed(config.SEED)
torch.manual_seed(config.SEED)

MODEL_NAME     = "meta-llama/Llama-3.2-3B-Instruct"
LORA_PATH      = os.path.join(config.BASE_DIR, "results", "lora_generation", "lora_adapter")
REFT_PATH      = os.path.join(config.BASE_DIR, "results", "reft_generation", "reft_model")
DATA_DIR       = os.path.join(config.BASE_DIR, "data", "generation")
RESULTS_DIR    = os.path.join(config.BASE_DIR, "results", "evaluation")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SAMPLES      = 500
MAX_NEW_TOKENS = 150
REFT_LAYERS    = list(range(28))
REFT_RANK      = 4
REFT_POSITIONS = [0, -1]

SYSTEM_PROMPT = (
    "You are an empathetic mental health support assistant. "
    "A person has shared their situation and emotional state with you. "
    "Respond with warmth, understanding, and supportive intent. "
    "Acknowledge their feelings and provide gentle, non-clinical support."
)

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# =============================================================================
# 1.  Load test dataset
# =============================================================================
print("\n── Loading test dataset ────────────────────────────────────────────────")
dataset = load_from_disk(DATA_DIR)
test_data = dataset["test"]
print(f"Full test set: {len(test_data)} samples")

indices = random.sample(range(len(test_data)), N_SAMPLES)
test_samples = test_data.select(indices)
print(f"Sampled {N_SAMPLES} for evaluation")
print(f"\nSample prompt:\n{test_samples[0]['prompt']}")
print(f"\nSample reference:\n{test_samples[0]['response']}")

# =============================================================================
# 2.  Helpers
# =============================================================================
def load_base_model():
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
    model.config.use_cache = True
    model.eval()
    return model

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # left pad for generation
    return tokenizer

def generate_response(model, tokenizer, prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

# =============================================================================
# 3.  LoReFT module (same as training)
# =============================================================================
class LoReFTIntervention(nn.Module):
    def __init__(self, embed_dim, rank, positions):
        super().__init__()
        self.positions = positions
        self.R = nn.Linear(embed_dim, rank, bias=False)
        self.W = nn.Linear(embed_dim, rank, bias=True)

    def forward(self, h):
        h = h.clone()
        for pos in self.positions:
            h_pos = h[:, pos, :]
            difference   = self.W(h_pos) - self.R(h_pos)
            intervention = difference @ self.R.weight
            h[:, pos, :] = h_pos + intervention
        return h

# =============================================================================
# 4.  Generate — LoRA
# =============================================================================
print("\n── Generating LoRA responses ───────────────────────────────────────────")
tokenizer   = load_tokenizer()
base_model  = load_base_model()
lora_model  = PeftModel.from_pretrained(base_model, LORA_PATH)
lora_model.eval()
print(f"LoRA model loaded")

lora_responses = []
for i, sample in enumerate(test_samples):
    if i % 50 == 0:
        print(f"  {i}/{N_SAMPLES}...")
    lora_responses.append(generate_response(lora_model, tokenizer, sample["prompt"]))

print(f"LoRA generation complete.")
del lora_model, base_model
torch.cuda.empty_cache()

# =============================================================================
# 5.  Generate — ReFT
# =============================================================================
print("\n── Generating ReFT responses ───────────────────────────────────────────")
base_model = load_base_model()

embed_dim = base_model.config.hidden_size
base_model.reft_interventions = nn.ModuleList([
    LoReFTIntervention(embed_dim, REFT_RANK, REFT_POSITIONS)
    for _ in REFT_LAYERS
])

state_dicts = torch.load(
    os.path.join(REFT_PATH, "reft_interventions.pt"), map_location="cuda"
)
for i, iv in enumerate(base_model.reft_interventions):
    iv.load_state_dict(state_dicts[f"intervention_{i}"])
    iv.to("cuda")
    iv.eval()
    
base_model.reft_interventions.to(torch.float16) 
print(f"ReFT interventions loaded")

hooks = []
for idx, layer_idx in enumerate(REFT_LAYERS):
    layer_norm   = base_model.model.layers[layer_idx].post_attention_layernorm
    intervention = base_model.reft_interventions[idx]

    def make_hook(iv):
        def hook(module, input, output):
            if isinstance(output, tuple):
                return (iv(output[0]),) + output[1:]
            return iv(output)
        return hook

    hooks.append(layer_norm.register_forward_hook(make_hook(intervention)))

reft_responses = []
for i, sample in enumerate(test_samples):
    if i % 50 == 0:
        print(f"  {i}/{N_SAMPLES}...")
    reft_responses.append(generate_response(base_model, tokenizer, sample["prompt"]))

for h in hooks:
    h.remove()

print(f"ReFT generation complete.")
del base_model
torch.cuda.empty_cache()

# =============================================================================
# 6.  BERTScore
# =============================================================================
print("\n── Computing BERTScore ─────────────────────────────────────────────────")
references = [sample["response"] for sample in test_samples]

print("  Scoring LoRA...")
lora_P, lora_R, lora_F1 = bert_score(
    lora_responses, references,
    lang="en", model_type="distilbert-base-uncased",
    verbose=False, device="cuda"
)

print("  Scoring ReFT...")
reft_P, reft_R, reft_F1 = bert_score(
    reft_responses, references,
    lang="en", model_type="distilbert-base-uncased",
    verbose=False, device="cuda"
)

bertscore_results = {
    "lora": {
        "precision": round(lora_P.mean().item(), 4),
        "recall":    round(lora_R.mean().item(), 4),
        "f1":        round(lora_F1.mean().item(), 4),
    },
    "reft": {
        "precision": round(reft_P.mean().item(), 4),
        "recall":    round(reft_R.mean().item(), 4),
        "f1":        round(reft_F1.mean().item(), 4),
    }
}
print(f"\n  LoRA — P:{bertscore_results['lora']['precision']}  R:{bertscore_results['lora']['recall']}  F1:{bertscore_results['lora']['f1']}")
print(f"  ReFT — P:{bertscore_results['reft']['precision']}  R:{bertscore_results['reft']['recall']}  F1:{bertscore_results['reft']['f1']}")

# =============================================================================
# 7.  GPT-4o Judge — EPIC dimensions
# =============================================================================
JUDGE_SYSTEM = """You are an expert evaluator of empathetic dialogue in mental health support contexts.

Score the MODEL RESPONSE on 5 EPIC dimensions (1-5 each):

1. Emotional Recognition  — Does the response correctly identify the person's emotion?
2. Emotional Validation   — Does the response validate feelings without judgment?
3. Supportive Intent      — Does the response show genuine care and desire to help?
4. Tone Appropriateness   — Is the tone warm, gentle, and non-clinical?
5. Overall Empathy        — Holistic rating of empathetic quality.

Scale:
1 = Very poor (dismissive, cold, or harmful)
2 = Poor (minimal empathy, generic)
3 = Moderate (some empathy, room for improvement)
4 = Good (clearly empathetic and supportive)
5 = Excellent (highly empathetic, warm, nuanced)

Respond ONLY with valid JSON — no explanation, no markdown:
{"emotional_recognition": <1-5>, "emotional_validation": <1-5>, "supportive_intent": <1-5>, "tone_appropriateness": <1-5>, "overall_empathy": <1-5>}"""

def judge_response(prompt, reference, generated, retries=3):
    user_msg = f"SITUATION:\n{prompt}\n\nREFERENCE RESPONSE:\n{reference}\n\nMODEL RESPONSE:\n{generated}"
    for attempt in range(retries):
        try:
            completion = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0,
                max_tokens=80,
            )
            return json.loads(completion.choices[0].message.content.strip())
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None

print("\n── LLM-as-Judge (GPT-4o) — EPIC ───────────────────────────────────────")
lora_epic, reft_epic = [], []

for i in range(N_SAMPLES):
    if i % 50 == 0:
        print(f"  Judging {i}/{N_SAMPLES}...")

    prompt    = test_samples[i]["prompt"]
    reference = test_samples[i]["response"]

    ls = judge_response(prompt, reference, lora_responses[i])
    if ls: lora_epic.append(ls)

    rs = judge_response(prompt, reference, reft_responses[i])
    if rs: reft_epic.append(rs)

    time.sleep(0.3)  # rate limit buffer

DIMS = ["emotional_recognition", "emotional_validation",
        "supportive_intent", "tone_appropriateness", "overall_empathy"]

def avg_epic(scores):
    return {d: round(np.mean([s[d] for s in scores if d in s]), 3) for d in DIMS}

lora_epic_avg = avg_epic(lora_epic)
reft_epic_avg = avg_epic(reft_epic)

# =============================================================================
# 8.  Save results
# =============================================================================
print("\n── Saving results ──────────────────────────────────────────────────────")

full_results = {
    "config": {
        "model":     MODEL_NAME,
        "n_samples": N_SAMPLES,
        "judge":     "gpt-4o",
        "seed":      config.SEED,
    },
    "bertscore": bertscore_results,
    "epic": {
        "lora": lora_epic_avg,
        "reft": reft_epic_avg,
    },
    "per_sample": [
        {
            "idx":              i,
            "prompt":           test_samples[i]["prompt"],
            "reference":        test_samples[i]["response"],
            "lora_response":    lora_responses[i],
            "reft_response":    reft_responses[i],
            "lora_bertscore_f1": round(lora_F1[i].item(), 4),
            "reft_bertscore_f1": round(reft_F1[i].item(), 4),
            "lora_epic":        lora_epic[i] if i < len(lora_epic) else None,
            "reft_epic":        reft_epic[i] if i < len(reft_epic) else None,
        }
        for i in range(N_SAMPLES)
    ]
}

with open(os.path.join(RESULTS_DIR, "generation_eval_results_800steps.json"), "w") as f:
    json.dump(full_results, f, indent=2)

# Print summary
print("\n" + "="*62)
print("EVALUATION SUMMARY")
print("="*62)
print(f"\nBERTScore:")
print(f"  LoRA — P:{bertscore_results['lora']['precision']}  R:{bertscore_results['lora']['recall']}  F1:{bertscore_results['lora']['f1']}")
print(f"  ReFT — P:{bertscore_results['reft']['precision']}  R:{bertscore_results['reft']['recall']}  F1:{bertscore_results['reft']['f1']}")
print(f"\nEPIC (GPT-4o, 1-5 scale):")
for dim in DIMS:
    print(f"  {dim:28s}  LoRA={lora_epic_avg.get(dim,'—')}  ReFT={reft_epic_avg.get(dim,'—')}")
print("="*62)
print("\n✅ Evaluation complete.")
print(f"Full results → {RESULTS_DIR}/generation_eval_results_800steps.json")