"""
Inference Pipeline — Component-Aware Audio Deepfake Detection
=============================================================
Deployment-ready inference script. Accepts any audio file and returns
the 5-class prediction with per-class confidence scores.

Usage:
    python infer.py --audio path/to/audio.wav
    python infer.py --audio path/to/audio.wav --checkpoint models/best_model.pth
    python infer.py --audio path/to/audio.wav --json          # JSON output
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

from src.model import DeepfakeDetector

CLASS_NAMES = {
    0: "original          — both speech and environment are genuine",
    1: "bonafide_bonafide — genuine components, re-mixed",
    2: "spoof_bonafide    — SYNTHETIC SPEECH + real environment",
    3: "bonafide_spoof    — real speech + SYNTHETIC ENVIRONMENT",
    4: "spoof_spoof       — BOTH components are synthetic",
}

TARGET_SR    = 16_000
CLIP_SAMPLES = TARGET_SR * 4


def load_audio(path: str) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = T.Resample(sr, TARGET_SR)(waveform)
    waveform = waveform.squeeze(0)
    if waveform.shape[0] < CLIP_SAMPLES:
        waveform = F.pad(waveform, (0, CLIP_SAMPLES - waveform.shape[0]))
    else:
        waveform = waveform[:CLIP_SAMPLES]
    rms = waveform.pow(2).mean().sqrt().clamp(min=1e-9)
    return (waveform / rms).unsqueeze(0)   # (1, 64000)


def predict(audio_path: str, checkpoint: str,
            device: torch.device, as_json: bool = False):

    waveform = load_audio(audio_path).to(device)

    model = DeepfakeDetector(num_classes=5).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    with torch.no_grad():
        logits, _ = model(waveform)
        probs     = F.softmax(logits, dim=1).squeeze(0)

    pred_class = probs.argmax().item()
    result = {
        "audio_file":   os.path.basename(audio_path),
        "prediction":   pred_class,
        "label":        CLASS_NAMES[pred_class].split("—")[0].strip(),
        "description":  CLASS_NAMES[pred_class],
        "confidence":   round(probs[pred_class].item(), 4),
        "probabilities": {
            f"class_{i}_{name.split('—')[0].strip()}": round(probs[i].item(), 4)
            for i, name in CLASS_NAMES.items()
        }
    }

    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 55)
        print("  AUDIO DEEPFAKE DETECTION — INFERENCE RESULT")
        print("=" * 55)
        print(f"  File       : {result['audio_file']}")
        print(f"  Prediction : Class {pred_class} — {result['label']}")
        print(f"  Confidence : {result['confidence']:.2%}")
        print(f"\n  Description: {CLASS_NAMES[pred_class]}")
        print("\n  All class probabilities:")
        for cls_name, prob in result['probabilities'].items():
            bar = "█" * int(prob * 30)
            print(f"    {cls_name:<40} {prob:.4f}  {bar}")
        print("=" * 55)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on a single audio file")
    parser.add_argument("--audio",      type=str, required=True,
                        help="Path to input audio file (.wav, .flac, .mp3)")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--device",     type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Inference device")
    parser.add_argument("--json",       action="store_true",
                        help="Output result as JSON")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    predict(args.audio, args.checkpoint, device, as_json=args.json)
