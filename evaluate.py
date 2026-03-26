import argparse
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from src.dataset import CompSpoofDataset
from src.model import DeepfakeDetectorResNet

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = CompSpoofDataset(os.path.join(args.data_dir, "eval"))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    model = DeepfakeDetectorResNet(num_classes=5).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    print("\n--- 5-Class Evaluation Results ---")
    print(classification_report(all_labels, all_preds, target_names=[f"Class {i}" for i in range(5)]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data/CompSpoofV2")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    evaluate(args)
