import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve
from tabulate import tabulate
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from src.dataset import CompSpoofDataset
from src.model import DeepfakeDetectorResNet
from src.plots import plot_roc_curve, plot_confusion_matrix

def compute_eer(y_true, y_score):
    """
    Computes the Equal Error Rate.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")

    # Load dataset using CSV
    # Structure: eval / metadata / eval.csv
    csv_name = os.path.join(args.split, "metadata", f"{args.split}.csv")
    dataset = CompSpoofDataset(args.data_dir, csv_name, max_length_s=4.0)
    
    if len(dataset) == 0:
        print(f"Error: No audio files found for {csv_name}.")
        return
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Init and load model
    model = DeepfakeDetectorResNet(num_classes=5, pretrained=False).to(device)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded model weights from {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found.")
        
    model.eval()
    
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            outputs = model(features)
            
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate Macro metrics
    mf1 = f1_score(all_labels, all_preds, average='macro')
    
    # Classification Report
    report_dict = classification_report(all_labels, all_preds, 
                                        target_names=[f"Class {i}" for i in range(5)], 
                                        output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    
    # --- ESDD2 AUXILIARY EER METRICS ---
    # EERoriginal: Class 0 (Genuine) vs All Others
    y_true_orig = (all_labels != 0).astype(int)
    y_score_orig = np.sum(all_probs[:, 1:], axis=1) # Sum of all spoof probabilities
    eer_orig = compute_eer(y_true_orig, y_score_orig)
    
    print("\n" + "="*50)
    print("ESDD2 - AUDIO DEEPFAKE DETECTION ANALYSIS")
    print("="*50)
    print(f"OVERALL MACRO-F1: {mf1:.4f}  <-- Main Challenge Metric")
    print(f"EER - Original:   {eer_orig:.4f}  <-- Auxiliary diagnostic")
    print("-"*50)
    
    print("\nDETAILED ANALYSIS TABLE:")
    print(tabulate(report_df.iloc[:5, :3], headers='keys', tablefmt='psql', floatfmt=".4f"))
    
    # Plotting
    os.makedirs(args.out_dir, exist_ok=True)
    plot_roc_curve(all_labels, all_probs, num_classes=5, save_path=os.path.join(args.out_dir, "roc_curve.png"))
    plot_confusion_matrix(all_labels, all_preds, num_classes=5, save_path=os.path.join(args.out_dir, "confusion_matrix.png"))
    
    print(f"\n[*] Results saved to {args.out_dir}/")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/latest.pth")
    parser.add_argument("--data_dir", type=str, default="./data/CompSpoofV2")
    parser.add_argument("--split", type=str, default="eval")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()
    evaluate(args)
