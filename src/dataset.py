import os
import glob
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

class CompSpoofDataset(Dataset):
    """
    PyTorch Dataset for 5-Class Audio Deepfake Detection (ESDD2 / CompSpoofV2).
    """
    def __init__(self, root_dir, sample_rate=16000, max_length_s=4.0, transform=None):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_length_s)
        self.transform = transform
        self.file_paths = []
        self.labels = []
        
        if not os.path.exists(root_dir):
            return

        # Expectations: subfolders are 'class_0', 'class_1', ..., 'class_4'
        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        for label, class_name in enumerate(classes):
            class_dir = os.path.join(root_dir, class_name)
            for ext in ('*.wav', '*.flac'):
                files = glob.glob(os.path.join(class_dir, ext))
                self.file_paths.extend(files)
                self.labels.extend([label] * len(files))

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            waveform, sr = librosa.load(file_path, sr=self.sample_rate)
        except:
            waveform = np.zeros(self.max_length)
            sr = self.sample_rate
            
        if len(waveform) > self.max_length:
            waveform = waveform[:self.max_length]
        else:
            waveform = np.pad(waveform, (0, self.max_length - len(waveform)), 'constant')
            
        # Extract Mel spectrogram
        mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=128, fmax=8000)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        
        # (Channels, Mels, Time)
        feature = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
        
        if self.transform:
            feature = self.transform(feature)
            
        return feature, torch.tensor(label, dtype=torch.long)
