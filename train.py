"""
Training Script — Component-Aware Audio Deepfake Detection
===========================================================
Trains the Wav2Vec2 + Mask Separation + BiLSTM + Attention model
on the CompSpoofV2 dataset for the ESDD2 5-class challenge.

Loss:
    L_total = CrossEntropy(logits, labels) + lambda_sep * L_sep

Optimiser:
    AdamW with discriminative learning rates:
      - Wav2Vec2 layers   : lr_ssl  (default 1e-5)
      - All other layers  : lr_base (default 1e-4)

Scheduler:
    CosineAnnealingLR (T_max = num_epochs)

Usage:
    python train.py --data_dir ./data/CompSpoofV2 --epochs 30
    python train.py --data_dir ./data/CompSpoofV2 --model resnet  # baseline ablation
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.dataset import CompSpoofDataset
from src.model import DeepfakeDetector, DeepfakeDetectorResNet
from src.separation import SeparationModule


# ─── Helper: build model ──────────────────────────────────────────────────────

def build_model(model_type: str, device: torch.device):
    if model_type == 'resnet':
        print("[*] Using ResNet-18 baseline (ablation A0)")
        return DeepfakeDetectorResNet(num_classes=5, pretrained=True).to(device)
    print("[*] Using full Wav2Vec2 + Separation + BiLSTM + Attention model")
    return DeepfakeDetector(num_classes=5).to(device)


# ─── Helper: one training epoch ───────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, lambda_sep, clip_norm=1.0):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for waveform, labels in tqdm(loader, desc="  Train", leave=False):
        waveform = waveform.to(device)
        labels   = labels.to(device)

        optimizer.zero_grad()
        logits, l_sep = model(waveform)

        l_cls   = criterion(logits, labels)
        loss    = l_cls + lambda_sep * l_sep
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

        total_loss += loss.item() * waveform.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n        = len(loader.dataset)
    avg_loss = total_loss / n
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, macro_f1


# ─── Helper: validation epoch ─────────────────────────────────────────────────

def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for waveform, labels in tqdm(loader, desc="  Val  ", leave=False):
            waveform = waveform.to(device)
            labels   = labels.to(device)
            logits, l_sep = model(waveform)
            loss = criterion(logits, labels)

            total_loss += loss.item() * waveform.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    n        = len(loader.dataset)
    avg_loss = total_loss / n
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, macro_f1


# ─── Main training loop ───────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    feature_mode = 'melspectrogram' if args.model == 'resnet' else 'waveform'
    print(f"[*] Feature mode: {feature_mode}")

    train_ds = CompSpoofDataset(args.data_dir, "development/train.csv",
                                feature_mode=feature_mode, augment=True)
    val_ds   = CompSpoofDataset(args.data_dir, "development/val.csv",
                                feature_mode=feature_mode, augment=False)

    # Weighted sampler for class imbalance
    if args.weighted_sampler and len(train_ds) > 0:
        sampler     = WeightedRandomSampler(train_ds.get_sample_weights(),
                                            num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=args.num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.model, device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    # Class weights for CrossEntropy (inverse-frequency)
    if args.class_weights and len(train_ds) > 0:
        cw = train_ds.class_weights
        weight_tensor = torch.tensor([cw[i] for i in range(5)], dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        print(f"[*] Class weights: {[f'{w:.2f}' for w in weight_tensor.cpu().tolist()]}")
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Optimiser ─────────────────────────────────────────────────────────────
    if args.model == 'resnet':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    else:
        param_groups = model.get_param_groups(lr_base=args.lr, lr_ssl=args.lr_ssl)
        optimizer    = optim.AdamW(param_groups, weight_decay=args.weight_decay)
        print(f"[*] Discriminative LR: base={args.lr}, SSL={args.lr_ssl}")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Output dirs ───────────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir,  exist_ok=True)
    log_file = os.path.join(args.log_dir, "training_log.json")
    history  = []
    best_f1  = 0.0

    print(f"\n[*] Starting training for {args.epochs} epochs")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        # Lambda warm-up for separation loss
        lambda_sep = SeparationModule.separation_loss_weight(
            epoch - 1, warmup_epochs=10,
            lambda_start=args.lambda_start, lambda_end=args.lambda_end
        ) if args.model != 'resnet' else 0.0

        train_loss, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, lambda_sep)

        val_loss, val_f1 = val_epoch(model, val_loader, criterion, device)

        scheduler.step()

        # Logging
        lr_current = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:>3}/{args.epochs} | "
              f"TrainLoss={train_loss:.4f} TrainF1={train_f1:.4f} | "
              f"ValLoss={val_loss:.4f} ValF1={val_f1:.4f} | "
              f"λ_sep={lambda_sep:.3f} | LR={lr_current:.2e}")

        record = dict(epoch=epoch, train_loss=train_loss, train_f1=train_f1,
                      val_loss=val_loss, val_f1=val_f1, lambda_sep=lambda_sep)
        history.append(record)
        with open(log_file, 'w') as f:
            json.dump(history, f, indent=2)

        # Checkpoints
        ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        torch.save(model.state_dict(), os.path.join(args.save_dir, "latest.pth"))

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            print(f"  ✓ New best Val Macro-F1: {best_f1:.4f} — saved best_model.pth")

    print("\n" + "=" * 60)
    print(f"[*] Training complete. Best Val Macro-F1: {best_f1:.4f}")
    print(f"[*] Checkpoints saved to: {args.save_dir}")
    print(f"[*] Training log:         {log_file}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Component-Aware ADD model on CompSpoofV2")

    # Data
    parser.add_argument("--data_dir",   type=str,   default="./data/CompSpoofV2")
    parser.add_argument("--save_dir",   type=str,   default="./models")
    parser.add_argument("--log_dir",    type=str,   default="./logs")

    # Model
    parser.add_argument("--model",      type=str,   default="wav2vec2",
                        choices=["wav2vec2", "resnet"],
                        help="wav2vec2 = full proposed model | resnet = baseline ablation")

    # Training
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--batch_size",     type=int,   default=16)
    parser.add_argument("--lr",             type=float, default=1e-4,
                        help="Base learning rate (non-SSL layers)")
    parser.add_argument("--lr_ssl",         type=float, default=1e-5,
                        help="Learning rate for Wav2Vec2 layers")
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--num_workers",    type=int,   default=4)

    # Separation loss
    parser.add_argument("--lambda_start",   type=float, default=0.1,
                        help="Starting separation loss weight")
    parser.add_argument("--lambda_end",     type=float, default=0.3,
                        help="Final separation loss weight (after 10-epoch warmup)")

    # Flags
    parser.add_argument("--weighted_sampler", action="store_true", default=True,
                        help="Use WeightedRandomSampler to handle class imbalance")
    parser.add_argument("--class_weights",    action="store_true", default=True,
                        help="Apply inverse-frequency class weights to CrossEntropy")

    args = parser.parse_args()
    train(args)
