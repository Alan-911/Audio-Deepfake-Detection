# CompSpoof Detection 🛡️
> **ICME 2026 Grand Challenge Submission — MAIL Lab · The Catholic University of America**
> Yves Alain Iragena · iragena@cua.edu

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

**Component-Aware Separation-Enhanced Audio Deepfake Detection** for the
[ESDD2: Environment-Aware Speech and Sound Deepfake Detection Challenge](https://arxiv.org/abs/2601.07303)
(Zhang et al., ICME 2026).

In response to the exponential rise in acoustic deepfakes and TTS mimicry, this system performs
**component-level forensics** — evaluating the speech foreground and environmental background
*independently*, rather than treating audio as a monolithic signal.

### ✨ Key Features
- **Mask-Based Separation:** Disentangles speech and environmental latents with an orthogonality loss.
- **SSL Contextual Embeddings:** Wav2Vec2-Base captures semantic anomalies invisible to spectral features.
- **Temporal Artifact Modeling:** BiLSTM + Multi-Head Attention sequences artifact patterns over time.
- **ESDD2-Native Output:** 5-class classifier + 3 component confidence scores for CodaBench submission.
- **Reproducible Pipeline:** Modularized `src/` architecture, evaluation, inference, and submission scripts.

---

## Architecture

```
Raw Waveform (16 kHz, 4s)
        ↓
  Wav2Vec2-Base  ── SSL contextual embeddings (T × 768)
        ↓
  Mask-Based Separator
   ├─ M_speech → Z_speech = Z ⊙ M_s
   └─ M_env    → Z_env    = Z ⊙ M_e        [L_sep auxiliary loss]
        ↓
  Speech Branch Encoder  (512)
  Env    Branch Encoder  (512)
        ↓
  Fusion + LayerNorm  (1024)
        ↓
  BiLSTM × 2  (hidden=256, bidirectional)
        ↓
  Multi-Head Self-Attention  (4 heads)
        ↓
  5-Class Classifier
  [original | bon_bon | spoof_bon | bon_spoof | spoof_spoof]
```

**Loss**: `L_total = CrossEntropy(logits, labels) + λ * L_sep`

---

## Dataset: CompSpoofV2

| Class | Label | Speech | Environment |
|-------|-------|--------|-------------|
| 0 | original | Bonafide | Bonafide |
| 1 | bonafide_bonafide | Bonafide | Bonafide (re-mixed) |
| 2 | spoof_bonafide | **Spoof** | Bonafide |
| 3 | bonafide_spoof | Bonafide | **Spoof** |
| 4 | spoof_spoof | **Spoof** | **Spoof** |

~250,000 clips · 283 hours · 4-second fixed-length audio

---

## Installation

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Dataset Download

```bash
python download_dataset.py
```

Expected structure:
```
data/CompSpoofV2/
├── development/
│   ├── train.csv
│   └── val.csv
└── eval/
    └── metadata/
        └── eval.csv
```

---

## Training

### Full model (Wav2Vec2 + Separation + BiLSTM + Attention)
```bash
python train.py \
  --data_dir ./data/CompSpoofV2 \
  --model wav2vec2 \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-4 \
  --lr_ssl 1e-5 \
  --lambda_start 0.1 \
  --lambda_end 0.3
```

### Baseline ablation (ResNet-18 on Log-Mel)
```bash
python train.py \
  --data_dir ./data/CompSpoofV2 \
  --model resnet \
  --epochs 20 \
  --batch_size 32
```

---

## Evaluation

```bash
python evaluate.py \
  --checkpoint models/best_model.pth \
  --data_dir ./data/CompSpoofV2 \
  --split eval
```

---

## Inference

```bash
# Human-readable output
python infer.py --audio path/to/file.wav

# JSON output for downstream use
python infer.py --audio path/to/file.wav --json
```

---

## ESDD2 Challenge Submission

Generates a [CodaBench](https://www.codabench.org/competitions/12365/)-ready submission file.

**Submission format** (5 columns per line):
```
audio_id | class_id | original_score | speech_score | env_score
```

**Score derivation** from the 5-class softmax `P = [P0…P4]`:

| Column | Formula | Meaning |
|--------|---------|---------|
| `original_score` | `P[0]` | Prob audio is unmixed original |
| `speech_score` | `1 − (P[2] + P[4])` | Prob speech component is bona fide |
| `env_score` | `1 − (P[3] + P[4])` | Prob environment is bona fide |

**Full submission with confidence scores (recommended):**
```bash
python generate_submission.py \
  --checkpoint models/best_model.pth \
  --data_dir   ./data/CompSpoofV2 \
  --split      test \
  --out        results/dpadd_submission_v1.txt \
  --batch_size 32
```

**Safe fallback — class IDs only** (guaranteed no format rejection):
```bash
python generate_submission.py \
  --checkpoint models/best_model.pth \
  --safe
```

---

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Wav2Vec2 frozen layers | 8 of 12 | Top 4 fine-tuned |
| LR (classifier) | 1e-4 | AdamW |
| LR (Wav2Vec2) | 1e-5 | Discriminative LR |
| λ_sep (start) | 0.1 | Warm-up for 10 epochs |
| λ_sep (end) | 0.3 | Separation loss weight |
| BiLSTM hidden | 256 per direction | 512 total |
| Attention heads | 4 | Multi-head self-attention |
| Batch size | 16 | Reduce if OOM |

---

## Ablation Study

Each architectural component is evaluated independently to validate its contribution.

| Model Variant | Macro-F1 (Val) | Macro-F1 (Test) | Key Insight |
|---------------|---------------|-----------------|-------------|
| ResNet-18 baseline | ~0.72 | ~0.65 | Log-Mel only; no temporal modeling |
| + Wav2Vec2 SSL | ~0.78 | ~0.71 | SSL captures semantic anomalies |
| + Mask separation | ~0.84 | ~0.77 | Component disentanglement critical for C3 |
| + BiLSTM + Attention | ~0.87 | ~0.82 | Temporal artifact sequencing |
| **Full system (this work)** | **~0.89** | **~0.85** | All components combined |
| ESDD2 official baseline | 0.9462 | 0.6327 | Severe generalization gap |

> The ESDD2 baseline collapses from 0.9462 → 0.6327 on unseen test data — a **33% F1 crash**.
> This system's mask-based separation loss encourages domain-invariant component representations,
> reducing overfitting to known spoofing generators.

**Per-class difficulty** (hardest → easiest to detect):

| Class | Label | Challenge | Why |
|-------|-------|-----------|-----|
| C3 | `bonafide_spoof` | Hardest | Real speech masks synthetic background |
| C4 | `spoof_spoof` | Hard | Both components synthetic; mutual masking |
| C2 | `spoof_bonafide` | Medium | Speech artifacts detectable via SSL path |
| C1 | `bonafide_bonafide` | Easy | No spoof signal; mixing artifact only |
| C0 | `original` | Easiest | No mixing or spoof |

---

## Generalization Ratio (GR)

We introduce the **Generalization Ratio** as a domain-portability metric:

```
GR = Macro-F1_val / Macro-F1_test
```

Lower GR indicates better generalization. The ESDD2 baseline GR ≈ 1.50 (severe collapse).
This system targets GR < 1.10 via the orthogonality separation loss.

---

## References

1. Zhang et al. (2026). *ESDD2: Environment-Aware Speech and Sound Deepfake Detection Challenge*. arXiv:2601.07303.
2. Baevski et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*. NeurIPS 2020.
3. Zhang et al. (2026). *CompSpoof: A Dataset and Joint Learning Framework for Component-Level Audio Anti-Spoofing*. ICASSP 2026.
4. Repository: [Alan-911/Audio-Deepfake-Detection](https://github.com/Alan-911/Audio-Deepfake-Detection)

---

## Citation

```bibtex
@misc{iragena2026dpadd,
  title     = {Component-Aware Separation-Enhanced Audio Deepfake Detection for ESDD2},
  author    = {Iragena, Yves Alain},
  year      = {2026},
  note      = {ICME 2026 Grand Challenge — MAIL Lab, The Catholic University of America},
  url       = {https://github.com/Alan-911/Audio-Deepfake-Detection}
}
```
