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

Resume:
    A full checkpoint (model + optimizer + scheduler + epoch + best_f1 + history)
    is saved to models/latest_ckpt.pth after every epoch.
    Pass --resume to restart from the last saved state automatically.
    Model-only checkpoints (best_model.pth, checkpoint_epoch_N.pth) remain
    compatible with evaluate.py and infer.py.

Usage:
    python train.py --data_dir ./data/CompSpoofV2 --epochs 30
    python train.py --resume                        # auto-resume from latest_ckpt.pth
    python train.py --resume models/latest_ckpt.pth # explicit path
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
from src.plots import plot_training_history


# ─── Helper: build model ──────────────────────────────────────────────────────

def build_model(model_type: str, device: torch.device):
    if model_type == 'resnet':
        print("[*] Using ResNet-18 baseline (ablation A0)")
        return DeepfakeDetectorResNet(num_classes=5, pretrained=True).to(device)
    print("[*] Using full Wav2Vec2 + Separation + BiLSTM + Attention model")
    return DeepfakeDetector(num_classes=5).to(device)


# ─── Helper: save / load full checkpoint ──────────────────────────────────────

def save_full_checkpoint(path, model, optimizer, scheduler,
                          epoch, best_f1, history):
    """
    Save a full training checkpoint for resume support.
    Separate from model-only checkpoints used by evaluate.py / infer.py.
    """
    torch.save({
        'epoch':                epoch,
        'best_f1':              best_f1,
        'history':              history,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)


def load_full_checkpoint(path, model, optimizer, scheduler, device):
    """
    Restore model, optimizer, scheduler, epoch counter, best_f1, and history.
    Returns (start_epoch, best_f1, history).
    """
    print(f"[*] Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt['epoch'] + 1          # resume from the next epoch
    best_f1     = ckpt['best_f1']
    history     = ckpt.get('history', [])
    print(f"[*] Resumed at epoch {start_epoch} | Best Val F1 so far: {best_f1:.4f}")
    return start_epoch, best_f1, history


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
    log_file       = os.path.join(args.log_dir, "training_log.json")
    full_ckpt_path = os.path.join(args.save_dir, "latest_ckpt.pth")
    history        = []
    best_f1        = 0.0
    start_epoch    = 1

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume:
        resume_path = (args.resume
                       if isinstance(args.resume, str) and os.path.exists(args.resume)
                       else full_ckpt_path)
        if os.path.exists(resume_path):
            start_epoch, best_f1, history = load_full_checkpoint(
                resume_path, model, optimizer, scheduler, device)
            # Sync log file with restored history so it stays consistent
            with open(log_file, 'w') as f:
                json.dump(history, f, indent=2)
        else:
            print(f"[!] No checkpoint found at {resume_path}. Starting from scratch.")

    remaining = args.epochs - start_epoch + 1
    print(f"\n[*] Starting training — epochs {start_epoch}→{args.epochs} ({remaining} remaining)")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
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

        # Full checkpoint for resume support
        save_full_checkpoint(full_ckpt_path, model, optimizer, scheduler,
                             epoch, best_f1, history)

    print("\n" + "=" * 60)
    print(f"[*] Training complete. Best Val Macro-F1: {best_f1:.4f}")
    print(f"[*] Checkpoints saved to: {args.save_dir}")
    print(f"[*] Training log:         {log_file}")

    # Plot training curves
    history_plot = os.path.join(args.log_dir, "training_history.png")
    try:
        plot_training_history(log_file, save_path=history_plot)
    except Exception as e:
        print(f"[!] Could not save training history plot: {e}")


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
    parser.add_argument("--weighted_sampler", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use WeightedRandomSampler to handle class imbalance (default: on). Disable with --no-weighted_sampler")
    parser.add_argument("--class_weights",    action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Apply inverse-frequency class weights to CrossEntropy (default: on). Disable with --no-class_weights")

    # Resume
    parser.add_argument("--resume", nargs="?", const=True, default=False,
                        metavar="CHECKPOINT",
                        help="Resume training from a full checkpoint. "
                             "Pass a path to use a specific file, or use --resume alone "
                             "to auto-load models/latest_ckpt.pth")

    args = parser.parse_args()
    train(args)
