import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CompSpoofDataset
from src.model import DeepfakeDetectorResNet

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_dataset = CompSpoofDataset(args.data_dir, "development/train.csv")
    val_dataset = CompSpoofDataset(args.data_dir, "development/val.csv")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = DeepfakeDetectorResNet(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * features.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        print(f"Epoch {epoch}: Loss {total_loss/total:.4f}, Acc {100*correct/total:.2f}%")
        
        # Validation
        if len(val_dataset) > 0:
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            print(f"Val Acc: {100*val_correct/val_total:.2f}%")
            
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pth"))
        torch.save(model.state_dict(), os.path.join(args.save_dir, "latest.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/CompSpoofV2")
    parser.add_argument("--save_dir", type=str, default="./models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)
