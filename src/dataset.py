"""
CompSpoofV2 Dataset
===================
PyTorch Dataset for the ESDD2 5-class audio deepfake detection challenge.
Supports two feature modes:
    - 'waveform'   : raw PCM tensor (B, T) for Wav2Vec2-based models  ← DEFAULT
    - 'melspectrogram' : log-Mel (B, 1, 128, T) for ResNet baseline

Labels (CompSpoofV2):
    0 = original          (both components genuine, unprocessed)
    1 = bonafide_bonafide (both genuine, re-mixed)
    2 = spoof_bonafide    (synthetic speech + real environment)
    3 = bonafide_spoof    (real speech + synthetic environment)
    4 = spoof_spoof       (both components synthetic)

Reference: Zhang et al. (2026) - ESDD2 / CompSpoofV2.
"""

import os
import random
import pandas as pd
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset


# ─── Label mapping ────────────────────────────────────────────────────────────

LABEL_MAP = {
    'original':          0,
    'bonafide_bonafide': 1,
    'spoof_bonafide':    2,
    'bonafide_spoof':    3,
    'spoof_spoof':       4,
}

TARGET_SR   = 16_000
CLIP_SAMPLES = TARGET_SR * 4    # 4-second clips = 64,000 samples


# ─── Augmentation helpers ─────────────────────────────────────────────────────

def add_gaussian_noise(waveform: torch.Tensor,
                       snr_db_range=(15, 30)) -> torch.Tensor:
    """Add Gaussian noise at a random SNR within snr_db_range."""
    snr_db  = random.uniform(*snr_db_range)
    sig_pow = waveform.pow(2).mean().clamp(min=1e-9)
    noise   = torch.randn_like(waveform)
    nos_pow = noise.pow(2).mean().clamp(min=1e-9)
    scale   = (sig_pow / nos_pow) * (10 ** (-snr_db / 10))
    return waveform + scale.sqrt() * noise


def time_stretch(waveform: torch.Tensor, sr: int,
                 rate_range=(0.9, 1.1)) -> torch.Tensor:
    """Random time stretch via small-kernel resampling (worker-safe)."""
    rate      = random.uniform(*rate_range)
    orig_len  = waveform.shape[-1]
    # Use small coprime-friendly integers to avoid huge resampling kernels
    orig_freq = 100
    new_freq  = max(1, round(orig_freq / rate))
    resampler = T.Resample(orig_freq, new_freq)
    stretched = resampler(waveform.unsqueeze(0)).squeeze(0)
    # Restore original length
    if stretched.shape[0] < orig_len:
        stretched = torch.nn.functional.pad(
            stretched, (0, orig_len - stretched.shape[0]))
    else:
        stretched = stretched[:orig_len]
    return stretched


def pitch_shift(waveform: torch.Tensor, sr: int,
                semitones_range=(-2, 2)) -> torch.Tensor:
    """Approximate pitch shift via resampling."""
    n_semitones = random.uniform(*semitones_range)
    factor      = 2 ** (n_semitones / 12)
    orig_sr     = int(sr * factor)
    resampler   = T.Resample(orig_sr, sr)
    return resampler(waveform.unsqueeze(0)).squeeze(0)


def spec_augment(mel: torch.Tensor,
                 time_mask_param: int = 40,
                 freq_mask_param: int = 20) -> torch.Tensor:
    """Apply SpecAugment time + frequency masking to a Mel spectrogram."""
    freq_masker = T.FrequencyMasking(freq_mask_param=freq_mask_param)
    time_masker = T.TimeMasking(time_mask_param=time_mask_param)
    mel = freq_masker(mel)
    mel = time_masker(mel)
    return mel


# ─── Main Dataset ─────────────────────────────────────────────────────────────

class CompSpoofDataset(Dataset):
    """
    Args:
        root_dir    : root of the CompSpoofV2 directory
        csv_name    : relative path to the split CSV (e.g. 'development/train.csv')
        feature_mode: 'waveform' or 'melspectrogram'
        augment     : enable data augmentation (training only)
        max_length_s: clip length in seconds (default 4.0)
    """

    def __init__(self,
                 root_dir: str,
                 csv_name: str,
                 feature_mode: str = 'waveform',
                 augment: bool = False,
                 max_length_s: float = 4.0):
        self.root_dir    = root_dir
        self.feature_mode = feature_mode
        self.augment     = augment
        self.sample_rate = TARGET_SR
        self.max_samples = int(TARGET_SR * max_length_s)

        csv_path = os.path.join(root_dir, csv_name)
        if not os.path.exists(csv_path):
            print(f"Warning: CSV not found at {csv_path}. Using empty dataset.")
            self.df = pd.DataFrame(columns=['audio_path', 'label'])
        else:
            self.df = pd.read_csv(csv_path)
            print(f"Loaded {len(self.df)} samples from {csv_path}")

        # Compute inverse-frequency class weights for weighted sampling
        if len(self.df) > 0 and 'label' in self.df.columns:
            counts = self.df['label'].map(LABEL_MAP).fillna(4).value_counts()
            self.class_weights = {c: len(self.df) / (len(LABEL_MAP) * counts.get(c, 1))
                                  for c in range(len(LABEL_MAP))}
        else:
            self.class_weights = {c: 1.0 for c in range(len(LABEL_MAP))}

    def __len__(self):
        return len(self.df)

    def _load_waveform(self, file_path: str) -> torch.Tensor:
        """Load, resample, and pad/crop audio to self.max_samples."""
        try:
            waveform, sr = torchaudio.load(file_path)
            if waveform.shape[0] > 1:          # mix to mono
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform  = resampler(waveform)
            waveform = waveform.squeeze(0)     # (T,)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            waveform = torch.zeros(self.max_samples)

        # Pad or crop
        if waveform.shape[0] < self.max_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.max_samples - waveform.shape[0]))
        else:
            waveform = waveform[:self.max_samples]

        # RMS normalisation
        rms = waveform.pow(2).mean().sqrt().clamp(min=1e-9)
        waveform = waveform / rms

        return waveform   # (max_samples,)

    def _to_melspectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to log-Mel spectrogram (1, 128, T)."""
        mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=400,
            hop_length=160, n_mels=128, f_max=8000
        )
        mel = mel_transform(waveform.unsqueeze(0))      # (1, 128, T)
        log_mel = T.AmplitudeToDB()(mel)
        return log_mel

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        label_str = row['label']
        label     = LABEL_MAP.get(label_str, 4)
        file_path = os.path.join(self.root_dir, row['audio_path'])

        waveform = self._load_waveform(file_path)

        # ── Augmentation (training only) ──────────────────────────────────────
        if self.augment:
            if random.random() < 0.4:
                waveform = add_gaussian_noise(waveform)
            if random.random() < 0.3:
                try:
                    waveform = time_stretch(waveform, self.sample_rate)
                except Exception:
                    pass
            if random.random() < 0.2:
                try:
                    waveform = pitch_shift(waveform, self.sample_rate)
                except Exception:
                    pass
            # Re-pad/crop after augmentation
            if waveform.shape[0] < self.max_samples:
                waveform = torch.nn.functional.pad(
                    waveform, (0, self.max_samples - waveform.shape[0]))
            else:
                waveform = waveform[:self.max_samples]

        if self.feature_mode == 'melspectrogram':
            feature = self._to_melspectrogram(waveform)
            if self.augment:
                feature = spec_augment(feature)
            return feature, torch.tensor(label, dtype=torch.long)

        # Default: raw waveform for Wav2Vec2
        return waveform, torch.tensor(label, dtype=torch.long)

    def get_sample_weights(self) -> torch.Tensor:
        """
        Returns per-sample weights for WeightedRandomSampler.
        Use to address class imbalance during training.
        """
        labels  = self.df['label'].map(LABEL_MAP).fillna(4).astype(int).tolist()
        weights = [self.class_weights[l] for l in labels]
        return torch.tensor(weights, dtype=torch.float)
