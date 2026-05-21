# DPADD-ES: Component-Aware Audio Deepfake Detection

> **ICME 2026 Grand Challenge (ESDD2) Submission** — Multimodal AI Laboratory, Catholic University of America

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![Macro-F1](https://img.shields.io/badge/Macro--F1-0.8923-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## The Problem

Existing audio deepfake detectors treat a recording as a single entity — but **synthetic artifacts live in the speech signal, not the background noise**. Mixing them causes catastrophic generalization failure: the ESDD2 baseline drops from **validation F1 = 0.9462 → test F1 = 0.6327** when encountering unseen spoof generators.

## Solution: Mask-Based Component Disentanglement

DPADD-ES separates speech and environmental representations into orthogonal latent spaces using a **Frobenius orthogonality loss**, forcing the model to learn artifact patterns that generalize across environments.

```
Audio Input
    │
    ▼
Wav2Vec2-base (SSL encoder)
    │
    ▼
Mask Separation Module ──── Frobenius Orthogonality Loss
    ├── Speech Stream
    └── Environment Stream
         │
         ▼
    BiLSTM (temporal encoding)
         │
         ▼
    Multi-Head Self-Attention
         │
         ▼
    5-Class Classifier
    (bonafide / spoof A–D)
```

## Results

| Model | Val Macro-F1 | Test Macro-F1 | GR Score |
|-------|-------------|---------------|----------|
| ESDD2 Baseline | 0.9462 | 0.6327 | ~1.50 |
| **DPADD-ES (ours)** | **0.9201** | **0.8923** | **<1.10** |

Per-class F1 range: 0.7819 – 0.9612. Mask separation module alone contributed +0.0573 F1 over ablated baseline.

## Dataset

**CompSpoofV2** — ~250,000 clips, 5 classes, ~283 hours of audio across 4 spoof generator types + bonafide.

```bash
python download_dataset.py   # auto-downloads and organizes splits
```

## Quick Start

```bash
git clone https://github.com/Alan-911/Audio-Deepfake-Detection
cd Audio-Deepfake-Detection
python -m venv venv && source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Train
python train.py --config configs/dpadd_es.yaml

# Evaluate
python evaluate.py --checkpoint models/best.pt --split test

# Inference on a single file
python infer.py --audio path/to/audio.wav
```

## Repo Structure

```
├── src/              # Model architecture, data loaders, loss functions
├── models/           # Saved checkpoints
├── results/          # Evaluation outputs and confusion matrices
├── configs/          # Training configuration files
├── train.py          # Training entrypoint
├── evaluate.py       # Evaluation script
├── infer.py          # Single-file inference
└── download_dataset.py
```

## Citation

If you use this work, please cite:

```
Iragena, Y. A. (2026). DPADD-ES: Component-Aware Deepfake Audio Detection
via Mask-Based Speech/Environment Disentanglement.
ICME 2026 Grand Challenge (ESDD2). Catholic University of America.
```

---

**Yves Alain Iragena** | MAIL Lab, Catholic University of America | iragena@cua.edu
