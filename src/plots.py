import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

def plot_roc_curve(y_true, y_probs, num_classes=5, save_path="roc_curve.png"):
    """
    Plots a multi-class ROC Curve using One-vs-Rest strategy.
    
    y_true: [n_samples] list of true labels
    y_probs: [n_samples, num_classes] array of probabilities
    """
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    plt.figure(figsize=(10, 8))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    colors = sns.color_palette("husl", num_classes)
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label=f'Class {i} (AUC = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - 5-Class')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(save_path)
    plt.close()
    print(f"[*] ROC Curve saved to {save_path}")

def plot_confusion_matrix(y_true, y_preds, num_classes=5, save_path="confusion_matrix.png"):
    """
    Plots a detailed heatmap confusion matrix.
    """
    cm = confusion_matrix(y_true, y_preds, labels=range(num_classes))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f"C{i}" for i in range(num_classes)],
                yticklabels=[f"C{i}" for i in range(num_classes)])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix Heatmap')
    plt.savefig(save_path)
    plt.close()
    print(f"[*] Confusion Matrix saved to {save_path}")

def generate_analysis_table(metrics_dict):
    """
    metrics_dict should contain precision, recall, f1 for each class.
    """
    df = pd.DataFrame(metrics_dict).transpose()
    return df
