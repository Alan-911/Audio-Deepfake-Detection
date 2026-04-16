# CompSpoof Detection 🛡️
> **Academic ML module classifying synthetic, deepfaked, and spoofed speech using PyTorch.**

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

In response to the exponential rise in acoustic deepfakes and TTS mimicry, this academic model processes sub-frequency anomalies within voice recordings to flag potentially spoofed speech securely.

### ✨ Key Features
- **Acoustic Profiling Layer:** Utilizes custom transformations to isolate micro-artifacts within generated speech.
- **High-Accuracy ML Model:** PyTorch deep neural classification optimized for generalized spoof attacks.
- **Dataset Agnostic:** Can function effectively over varying base languages and recording complexities.
- **Research Module Pipeline:** Cleanly modularized integration for future academic branches.

---

### 📺 Screenshots
> *[Placeholder: Insert an image/GIF or graph of the tensor accuracy classifications]*

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

## Expected Improvements vs Baseline

| Enhancement | Expected Macro-F1 gain |
|-------------|----------------------|
| Wav2Vec2 SSL features | +5–10% |
| Mask-based separation | +8–15% (esp. C2, C3) |
| BiLSTM temporal modeling | +3–7% |
| Multi-head attention | +2–5% |
| **Combined (full system)** | **+15–25%** |

---

## References

1. Zhang et al. (2026). *ESDD2: Environment-Aware Speech and Sound Deepfake Detection Challenge*. arXiv:2601.07303.
2. Baevski et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*. NeurIPS 2020.
3. Repository: [Alan-911/Audio-Deepfake-Detection](https://github.com/Alan-911/Audio-Deepfake-Detection)
