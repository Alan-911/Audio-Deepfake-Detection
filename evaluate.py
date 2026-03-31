"""
Evaluation Script — Component-Aware Audio Deepfake Detection
=============================================================
Evaluates a trained model checkpoint on the CompSpoofV2 eval split.

Metrics:
    - Macro-F1           (primary ESDD2 challenge metric)
    - EER (Original vs Rest)  (auxiliary ESDD2 metric)
    - Per-class F1, Precision, Recall
    - Confusion matrix heatmap
    - Multi-class ROC AUC curves

Usage:
    python evaluate.py --checkpoint models/best_model.pth
    python evaluate.py --checkpoint models/best_model.pth --model resnet  # baseline
"""

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, f1_score,
                             roc_curve, accuracy_score)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tabulate import tabulate

from src.dataset import CompSpoofDataset
from src.model import DeepfakeDetector, DeepfakeDetectorResNet
from src.plots import plot_roc_curve, plot_confusion_matrix


CLASS_NAMES = ["original", "bonafide_bonafide", "spoof_bonafide",
               "bonafide_spoof", "spoof_spoof"]


def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluating on {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    feature_mode = 'melspectrogram' if args.model == 'resnet' else 'waveform'
    csv_name     = os.path.join(args.split, "metadata", f"{args.split}.csv")
    dataset      = CompSpoofDataset(args.data_dir, csv_name,
                                    feature_mode=feature_mode, augment=False)
    if len(dataset) == 0:
        print(f"[!] No samples found for split '{args.split}'. Check data_dir and CSV path.")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.model == 'resnet':
        model = DeepfakeDetectorResNet(num_classes=5, pretrained=False).to(device)
    else:
        model = DeepfakeDetector(num_classes=5).to(device)

    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"[*] Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"[!] Checkpoint not found: {args.checkpoint}")
        return

    model.eval()

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            logits, _ = model(features)
            probs     = F.softmax(logits, dim=1)
            preds     = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    # ── Metrics ───────────────────────────────────────────────────────────────
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)

    # EER: Class 0 (genuine) vs all spoof classes
    y_true_orig = (all_labels != 0).astype(int)
    y_score_orig = all_probs[:, 1:].sum(axis=1)
    try:
        eer_orig = compute_eer(y_true_orig, y_score_orig)
    except Exception:
        eer_orig = float('nan')

    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T

    # ── Display ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ESDD2 AUDIO DEEPFAKE DETECTION — EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Macro-F1 (Primary):   {macro_f1:.4f}")
    print(f"  Accuracy:             {accuracy:.4f}")
    print(f"  EER (Orig vs Rest):   {eer_orig:.4f}  [auxiliary]")
    print("-" * 60)

    print("\n  Per-Class Results:")
    per_class = report_df.iloc[:5, :3].rename(
        index={c: f"C{i}: {c}" for i, c in enumerate(CLASS_NAMES)})
    print(tabulate(per_class, headers='keys', tablefmt='psql', floatfmt=".4f"))

    # ── Save plots ────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    plot_roc_curve(all_labels, all_probs, num_classes=5,
                   save_path=os.path.join(args.out_dir, "roc_curve.png"))
    plot_confusion_matrix(all_labels, all_preds, num_classes=5,
                          save_path=os.path.join(args.out_dir, "confusion_matrix.png"))

    # Save full report
    report_csv = os.path.join(args.out_dir, "classification_report.csv")
    report_df.to_csv(report_csv)
    print(f"\n[*] Results saved to: {args.out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth")
    parser.add_argument("--data_dir",   type=str, default="./data/CompSpoofV2")
    parser.add_argument("--split",      type=str, default="eval")
    parser.add_argument("--model",      type=str, default="wav2vec2",
                        choices=["wav2vec2", "resnet"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out_dir",    type=str, default="results")
    args = parser.parse_args()
    evaluate(args)
