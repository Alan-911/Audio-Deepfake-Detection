"""
ESDD2 Challenge Submission Generator
=====================================
Generates a CodaBench-ready submission file from a trained DeepfakeDetector
checkpoint against the CompSpoofV2 test set.

Output format (per line):
    audio_id | class_id | original_score | speech_score | env_score

Score derivation from 5-class softmax P = [P0, P1, P2, P3, P4]:
    original_score = P[0]                   (prob audio is unmixed original)
    speech_score   = 1 - (P[2] + P[4])     (prob speech component is bona fide)
    env_score      = 1 - (P[3] + P[4])     (prob environment is bona fide)

Usage:
    # Full submission with confidence scores (recommended):
    python generate_submission.py --checkpoint models/best_model.pth

    # Safe fallback — class IDs only, no scores (use if model not trained on CompSpoofV2):
    python generate_submission.py --checkpoint models/best_model.pth --safe

    # Custom paths:
    python generate_submission.py \\
        --checkpoint models/best_model.pth \\
        --data_dir   ./data/CompSpoofV2 \\
        --split      test \\
        --out        results/dpadd_submission_v1.txt \\
        --batch_size 32
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import pandas as pd

from src.model import DeepfakeDetector

# ── Constants (must match dataset.py) ─────────────────────────────────────────
TARGET_SR    = 16_000
CLIP_SAMPLES = TARGET_SR * 4    # 64,000 samples = 4 seconds

# ── Preprocessing (mirrors CompSpoofDataset._load_waveform) ───────────────────

def load_audio(file_path: str) -> torch.Tensor:
    """Load, resample, pad/crop, and RMS-normalise a single audio file."""
    try:
        waveform, sr = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != TARGET_SR:
            waveform = T.Resample(sr, TARGET_SR)(waveform)
        waveform = waveform.squeeze(0)      # (T,)
    except Exception as e:
        print(f"  [WARN] Failed to load {file_path}: {e} — using zeros")
        return torch.zeros(CLIP_SAMPLES)

    # Pad or crop to exactly 4 seconds
    if waveform.shape[0] < CLIP_SAMPLES:
        waveform = F.pad(waveform, (0, CLIP_SAMPLES - waveform.shape[0]))
    else:
        waveform = waveform[:CLIP_SAMPLES]

    # RMS normalisation
    rms = waveform.pow(2).mean().sqrt().clamp(min=1e-9)
    return waveform / rms   # (64000,)


# ── Submission generation ──────────────────────────────────────────────────────

def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device     : {device}")
    print(f"[*] Checkpoint : {args.checkpoint}")
    print(f"[*] Data dir   : {args.data_dir}")
    print(f"[*] Split      : {args.split}")
    print(f"[*] Output     : {args.out}")
    print(f"[*] Safe mode  : {args.safe}")
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"[!] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    model = DeepfakeDetector(num_classes=5).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"[*] Model loaded successfully.")

    # ── Load test CSV ─────────────────────────────────────────────────────────
    # Expected paths (mirrors evaluate.py convention):
    #   {data_dir}/{split}/metadata/{split}.csv
    # Fallback: {data_dir}/{split}.csv
    csv_candidates = [
        os.path.join(args.data_dir, args.split, "metadata", f"{args.split}.csv"),
        os.path.join(args.data_dir, f"{args.split}.csv"),
        os.path.join(args.data_dir, "metadata", f"{args.split}.csv"),
    ]
    csv_path = None
    for candidate in csv_candidates:
        if os.path.exists(candidate):
            csv_path = candidate
            break

    if csv_path is None:
        print(f"[!] No CSV found. Tried:")
        for c in csv_candidates:
            print(f"    {c}")
        print("    Falling back to scanning audio files directly in data_dir.")
        audio_extensions = ('.wav', '.flac', '.mp3', '.ogg')
        audio_dir = os.path.join(args.data_dir, args.split) \
                    if os.path.isdir(os.path.join(args.data_dir, args.split)) \
                    else args.data_dir
        files = [f for f in os.listdir(audio_dir) if f.endswith(audio_extensions)]
        df = pd.DataFrame({'audio_path': [os.path.join(audio_dir, f) for f in files],
                           'filename':   files})
        print(f"[*] Found {len(df)} audio files by scan.")
    else:
        df = pd.read_csv(csv_path)
        print(f"[*] Loaded {len(df)} samples from {csv_path}")
        # Resolve relative audio paths
        if 'audio_path' not in df.columns:
            print(f"[!] CSV must have an 'audio_path' column. Columns found: {list(df.columns)}")
            sys.exit(1)
        df['audio_path'] = df['audio_path'].apply(
            lambda p: p if os.path.isabs(p) else os.path.join(args.data_dir, p)
        )
        # audio_id = filename (with extension), matching CodaBench examples
        df['filename'] = df['audio_path'].apply(os.path.basename)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # ── Batch inference ───────────────────────────────────────────────────────
    # Try tqdm for a progress bar; fall back to plain counter
    try:
        from tqdm import tqdm
        iterator = tqdm(df.iterrows(), total=len(df), desc="Inferencing", unit="clip")
    except ImportError:
        print("[*] tqdm not installed — progress printed every 500 samples.")
        iterator = df.iterrows()

    lines_written = 0
    errors        = 0

    with open(args.out, 'w') as fout:
        batch_waves  = []
        batch_ids    = []
        batch_size   = args.batch_size

        def flush_batch(waves, ids):
            nonlocal lines_written
            batch_tensor = torch.stack(waves).to(device)   # (B, 64000)
            with torch.no_grad():
                logits, _ = model(batch_tensor)
                probs = F.softmax(logits, dim=1).cpu()     # (B, 5)

            for audio_id, p in zip(ids, probs):
                pred_class = p.argmax().item()

                if args.safe:
                    line = f"{audio_id} | {pred_class} | - | - |-\n"
                else:
                    orig_score   = p[0].item()
                    speech_score = 1.0 - (p[2] + p[4]).item()
                    env_score    = 1.0 - (p[3] + p[4]).item()
                    line = (f"{audio_id} | {pred_class} | "
                            f"{orig_score:.4f} | {speech_score:.4f} | {env_score:.4f}\n")

                fout.write(line)
                lines_written += 1

        for idx, row in iterator:
            if not args.safe and not os.path.exists(row['audio_path']):
                print(f"  [WARN] Missing file: {row['audio_path']}")
                errors += 1
                # Write a safe fallback line so the submission stays aligned
                fout.write(f"{row['filename']} | 0 | - | - | -\n")
                lines_written += 1
                continue

            try:
                wave = load_audio(row['audio_path'])
            except Exception as e:
                print(f"  [WARN] Load error {row['filename']}: {e}")
                errors += 1
                fout.write(f"{row['filename']} | 0 | - | - | -\n")
                lines_written += 1
                continue

            batch_waves.append(wave)
            batch_ids.append(row['filename'])

            if len(batch_waves) == batch_size:
                flush_batch(batch_waves, batch_ids)
                batch_waves, batch_ids = [], []

            if not 'tqdm' in sys.modules and lines_written % 500 == 0 and lines_written > 0:
                print(f"  ... {lines_written}/{len(df)} processed")

        # Flush remaining
        if batch_waves:
            flush_batch(batch_waves, batch_ids)

    print()
    print("=" * 55)
    print(f"  Submission file written : {args.out}")
    print(f"  Total lines             : {lines_written}")
    print(f"  Errors / missing files  : {errors}")
    print(f"  Mode                    : {'SAFE (class IDs only)' if args.safe else 'FULL (3 confidence scores)'}")
    print("=" * 55)
    print()
    print("  First 3 lines of output:")
    with open(args.out) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(f"    {line.rstrip()}")
    print()
    print("[*] Upload this file to CodaBench: https://www.codabench.org/competitions/12365/")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ESDD2 CodaBench submission file from a trained checkpoint.")

    parser.add_argument("--checkpoint",  type=str, default="models/best_model.pth",
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--data_dir",    type=str, default="./data/CompSpoofV2",
                        help="Root directory of CompSpoofV2 dataset")
    parser.add_argument("--split",       type=str, default="test",
                        choices=["test", "eval", "validation"],
                        help="Dataset split to run inference on (default: test)")
    parser.add_argument("--out",         type=str, default="results/dpadd_submission_v1.txt",
                        help="Output submission file path")
    parser.add_argument("--batch_size",  type=int, default=16,
                        help="Inference batch size (default: 16, use 32+ on A100)")
    parser.add_argument("--safe",        action="store_true",
                        help="Safe mode: output class IDs only, use '-' for confidence scores")

    args = parser.parse_args()
    generate(args)
