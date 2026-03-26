import os
import pandas as pd
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

class CompSpoofDataset(Dataset):
    """
    PyTorch Dataset for 5-Class Audio Deepfake Detection (ESDD2 / CompSpoofV2).
    Uses the official CSV metadata for loading.
    
    Labels:
    - original: 0
    - bonafide_bonafide: 1
    - spoof_bonafide: 2
    - bonafide_spoof: 3
    - spoof_spoof: 4
    """
    LABEL_MAP = {
        'original': 0,
        'bonafide_bonafide': 1,
        'spoof_bonafide': 2,
        'bonafide_spoof': 3,
        'spoof_spoof': 4
    }

    def __init__(self, root_dir, csv_name, sample_rate=16000, max_length_s=4.0, transform=None):
        """
        root_dir: The directory containing 'mixed_audio', 'original_audio', etc.
        csv_name: Path to the metadata CSV (e.g. 'development/train.csv')
        """
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_length_s)
        self.transform = transform
        
        csv_path = os.path.join(root_dir, csv_name)
        if not os.path.exists(csv_path):
            print(f"Warning: CSV not found at {csv_path}")
            self.df = pd.DataFrame(columns=['audio_path', 'label'])
        else:
            self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_rel_path = row['audio_path']
        label_str = row['label']
        
        label = self.LABEL_MAP.get(label_str, 4) # Default to spoof_spoof if unknown
        file_path = os.path.join(self.root_dir, audio_rel_path)
        
        try:
            waveform, sr = librosa.load(file_path, sr=self.sample_rate)
        except Exception:
            waveform = np.zeros(self.max_length)
            sr = self.sample_rate
            
        if len(waveform) > self.max_length:
            waveform = waveform[:self.max_length]
        else:
            waveform = np.pad(waveform, (0, self.max_length - len(waveform)), 'constant')
            
        # Extract Mel spectrogram
        mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=128, fmax=8000)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        
        feature = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
        
        if self.transform:
            feature = self.transform(feature)
            
        return feature, torch.tensor(label, dtype=torch.long)
