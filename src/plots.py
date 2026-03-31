"""Plotting utilities for evaluation results."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

CLASS_NAMES = ["original", "bon_bon", "spoof_bon", "bon_spoof", "spoof_spoof"]


def plot_roc_curve(y_true, y_probs, num_classes=5, save_path="roc_curve.png"):
    y_bin  = label_binarize(y_true, classes=range(num_classes))
    colors = sns.color_palette("husl", num_classes)
    plt.figure(figsize=(10, 8))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f"C{i}: {CLASS_NAMES[i]} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — 5-Class ADD (One-vs-Rest)")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[*] ROC curve saved: {save_path}")


def plot_confusion_matrix(y_true, y_preds, num_classes=5, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_preds, labels=range(num_classes))
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f"C{i}" for i in range(num_classes)],
                yticklabels=[f"C{i}: {CLASS_NAMES[i]}" for i in range(num_classes)])
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix — 5-Class ADD")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[*] Confusion matrix saved: {save_path}")


def plot_training_history(log_path: str, save_path: str = "training_history.png"):
    """Plot training and validation loss + F1 from a training_log.json file."""
    import json
    with open(log_path) as f:
        history = json.load(f)

    epochs     = [r['epoch']     for r in history]
    train_f1   = [r['train_f1']  for r in history]
    val_f1     = [r['val_f1']    for r in history]
    train_loss = [r['train_loss'] for r in history]
    val_loss   = [r['val_loss']   for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_f1, label='Train Macro-F1', color='steelblue')
    ax1.plot(epochs, val_f1,   label='Val Macro-F1',   color='coral')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Macro-F1')
    ax1.set_title('Macro-F1 Score over Training')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_loss, label='Train Loss', color='steelblue')
    ax2.plot(epochs, val_loss,   label='Val Loss',   color='coral')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.set_title('Loss over Training')
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[*] Training history saved: {save_path}")
