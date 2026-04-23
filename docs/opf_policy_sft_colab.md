# OPF head-only SFT on Colab T4

Two-stage approach: **Viterbi calibration** first, then **head-only SFT**.

## Why two stages?

Calibration sweep results on the 300-example benchmark:

| Config | F1 | Recall | Precision | TP | FP | FN |
|---|---|---|---|---|---|---|
| Baseline | .8459 | .8165 | .8776 | 129 | 18 | 29 |
| v5 calibration | .8698 | .8671 | .8726 | 137 | 20 | 21 |

Calibration recovered 8 false negatives (literal tokens where the backbone had weak signal) for only 2 new false positives. Free recall.

The remaining 21 FN break down as:
- **6 quasi-identifiers** — no literal PII tokens, needs semantic reasoning
- **6 indirect requests** — intent detection, not token classification
- **2 spelled-out IDs** — partial backbone signal exists
- **2 SEA indirect/quasi** — semantic-level in Malay/Filipino
- **5 other** (encoded, split, multilingual, literal)

Head-only SFT can only help the cases where token-level signal exists: literal identifiers, multilingual scripts, obfuscated/spelled-out patterns. The SFT dataset is curated to include **only** these cases.

Quasi-identifiers and indirect requests are **excluded** because their whole-text span annotations teach the head to fire on common English tokens ("the", "she", "a"), increasing false positives.

## Files

- `calib/recall_v5.json` — Viterbi calibration (stage 1)
- `data/opf_policy_sft_train.jsonl` — head-only-safe training data
- `data/opf_policy_sft_val.jsonl` — validation data
- `data/opf_policy_label_space.json` — binary label space
- `data/opf_policy_sft_manifest.json` — dataset manifest
- `notebooks/opf_policy_sft_colab.ipynb` — Colab notebook
- `scripts/train_opf_head_only.py` — training script
- `scripts/build_opf_policy_sft_assets.py` — asset generator

## Step-by-step Colab instructions

### 1. Open a Colab T4 runtime

Go to Google Colab → Runtime → Change runtime type → **T4 GPU**.

### 2. Clone the repo

```python
!git clone --branch feature/openai-privacy-filter https://github.com/ivanreeve/vllm-qwen3guard-test.git
%cd vllm-qwen3guard-test
```

### 3. Install dependencies

```python
!pip install -U pip setuptools wheel
!pip install -r requirements.txt
```

### 4. Verify GPU

```python
import torch
assert torch.cuda.is_available(), "No GPU — change runtime type"
gpu = torch.cuda.get_device_properties(0)
print(f"{gpu.name}, {gpu.total_memory / 1024**3:.1f} GB")
```

### 5. Run calibration-only baseline (optional but recommended)

This gives you a comparison point to measure SFT improvement:

```bash
OPF_MOE_TRITON=0 python detect_pii.py \
  --opf-viterbi-calibration-path calib/recall_v5.json \
  --output results/privacy-filter-calib-v5.json \
  --verbose
```

### 6. Regenerate SFT assets (if you edited annotations)

```bash
python scripts/build_opf_policy_sft_assets.py
```

### 7. Train

```bash
OPF_MOE_TRITON=0 python scripts/train_opf_head_only.py \
  data/opf_policy_sft_train.jsonl \
  --validation-dataset data/opf_policy_sft_val.jsonl \
  --label-space-json data/opf_policy_label_space.json \
  --device cuda \
  --n-ctx 256 \
  --epochs 25 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --learning-rate 2e-4 \
  --weight-decay 0.0 \
  --max-grad-norm 1.0 \
  --output-dir checkpoints/opf_policy_sft_v1 \
  --overwrite-output
```

Notes:
- `n_ctx=256` — examples are short, saves memory.
- `learning-rate=2e-4` — higher than full-model because only the output head is trainable.
- `OPF_MOE_TRITON=0` — avoids Triton failures in constrained environments.
- Training takes ~5-10 minutes on T4.

### 8. Evaluate the SFT checkpoint

```bash
python detect_pii.py \
  --model checkpoints/opf_policy_sft_v1 \
  --output results/privacy-filter-sft.json \
  --verbose
```

### 9. Compare results

```python
import json
from pathlib import Path

calib = json.loads(Path("results/privacy-filter-calib-v5.json").read_text())
sft = json.loads(Path("results/privacy-filter-sft.json").read_text())

for key in ["accuracy", "precision", "recall", "f1"]:
    delta = sft["metrics"][key] - calib["metrics"][key]
    print(f"{key}: calib={calib['metrics'][key]:.4f}  sft={sft['metrics'][key]:.4f}  delta={delta:+.4f}")
```

## What to expect

**Realistic gains from head-only SFT (on top of v5 calibration):**
- TC-007 (SG name + NRIC) — likely recovered
- SEA-006 (Myanmar script) — likely recovered
- TC-061, TC-075, TC-084 (spelled-out) — possible, backbone has partial signal

**Will NOT improve (needs a different approach):**
- Quasi-identifiers (6 cases) — no token-level signal
- Indirect requests (6 cases) — intent detection, not NER
- Encoded (ROT13) — backbone can't decode
- SEA quasi/indirect — semantic reasoning in local languages

For those ~18 remaining cases, consider:
- A second-pass LLM classifier (Qwen3Guard, Claude) for semantic cases
- Full-model LoRA on an A100 (may teach some patterns into the backbone)
- A hybrid pipeline: OPF for literal spans + LLM for policy cases

## Training dataset composition

| Category | Train | Purpose |
|---|---|---|
| Literal identifiers | 11 | Anchors + missed literals |
| Multilingual identifiers | 6 | SEA script coverage |
| Spelled-out identifiers | 4 | Phonetic/word-form patterns |
| Obfuscated identifiers | 2 | Leetspeak-style evasion |
| Hard negatives | 30 | All baseline + calibration FP |
| **Total** | **44** | |

Positive/negative ratio: 19 positive, 25 negative (57% negative — biased toward preventing FP regression).

## Regenerating assets

If you edit the annotation spec in `build_opf_policy_sft_assets.py`:

```bash
python scripts/build_opf_policy_sft_assets.py
```

This regenerates all JSONL, label space, manifest, and notebook files.
