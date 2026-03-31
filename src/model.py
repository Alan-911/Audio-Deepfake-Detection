"""
Deepfake Detection Model — Component-Aware Architecture
========================================================
Implements the full proposed pipeline from the ESDD2 implementation plan:

    Raw waveform
        → Wav2Vec2-Base  (SSL contextual embeddings, T × 768)
        → SeparationModule  (speech mask + env mask, L_sep)
        → Speech branch encoder  (Linear → 512)
        → Env    branch encoder  (Linear → 512)
        → Fusion + LayerNorm  (concat → 1024)
        → BiLSTM × 2  (hidden=256, bidirectional)
        → Multi-Head Self-Attention  (4 heads)
        → 5-Class Classifier  (Linear 1024 → 5)

Loss: L_total = CrossEntropy(logits, labels) + lambda * L_sep

Reference: Zhang et al. (2026) ESDD2 / CompSpoofV2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config

from .separation import SeparationModule


# ─── Wav2Vec2 Feature Extractor ───────────────────────────────────────────────

class Wav2Vec2Encoder(nn.Module):
    """
    Wraps facebook/wav2vec2-base.
    - Freezes the bottom (num_freeze_layers) transformer blocks.
    - Fine-tunes the top layers at a lower learning rate.
    """

    PRETRAINED_ID = "facebook/wav2vec2-base"

    def __init__(self, num_freeze_layers: int = 8):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(self.PRETRAINED_ID)
        self._freeze_layers(num_freeze_layers)

    def _freeze_layers(self, n: int):
        # Always freeze the feature extractor CNN
        for p in self.model.feature_extractor.parameters():
            p.requires_grad = False
        for p in self.model.feature_projection.parameters():
            p.requires_grad = False
        # Freeze the bottom n transformer blocks
        for i, layer in enumerate(self.model.encoder.layers):
            if i < n:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform : (B, T_samples)  raw PCM, 16 kHz, ~64k samples for 4s clips
        Returns:
            embeddings : (B, T_frames, 768)
        """
        out = self.model(input_values=waveform)
        return out.last_hidden_state   # (B, T, 768)


# ─── Branch Encoder (one per component) ──────────────────────────────────────

class BranchEncoder(nn.Module):
    """
    Projects separated component latent (T, 768) → (T, 512).
    """
    def __init__(self, input_dim: int = 768, output_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)   # (B, T, 512)


# ─── BiLSTM Temporal Encoder ──────────────────────────────────────────────────

class BiLSTMEncoder(nn.Module):
    """2-layer bidirectional LSTM for sequential artifact modeling."""

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, 1024)
        Returns:
            out : (B, T, 512)  — hidden=256 × 2 directions
        """
        out, _ = self.lstm(x)
        return self.dropout(out)   # (B, T, 512)


# ─── Multi-Head Self-Attention ────────────────────────────────────────────────

class ArtifactAttention(nn.Module):
    """
    Multi-head self-attention over the LSTM output sequence.
    Learns to focus on artifact-rich temporal regions.
    Returns a single summary vector via weighted mean pooling.
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm  = nn.LayerNorm(embed_dim)
        self.score = nn.Linear(embed_dim, 1)   # temporal importance weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, 512)
        Returns:
            pooled : (B, 512) — attention-weighted summary
        """
        residual = x
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(attn_out + residual)          # residual + norm

        # Weighted mean pooling along time axis
        weights = torch.softmax(self.score(x), dim=1)   # (B, T, 1)
        pooled  = (weights * x).sum(dim=1)              # (B, 512)
        return pooled


# ─── Full Model ───────────────────────────────────────────────────────────────

class DeepfakeDetector(nn.Module):
    """
    Component-Aware Audio Deepfake Detector.

    Architecture:
        Wav2Vec2  →  SeparationModule  →  SpeechBranch + EnvBranch
        →  Fusion  →  BiLSTM  →  Attention  →  Classifier (5 classes)

    Loss:
        L_total = CrossEntropy + lambda * L_sep

    Usage:
        model = DeepfakeDetector()
        logits, l_sep = model(waveform)   # waveform: (B, 64000)
    """

    def __init__(self,
                 num_classes: int = 5,
                 num_freeze_layers: int = 8,
                 separation_hidden: int = 256,
                 branch_dim: int = 512,
                 lstm_hidden: int = 256,
                 lstm_layers: int = 2,
                 attn_heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()

        # 1. SSL encoder
        self.wav2vec2  = Wav2Vec2Encoder(num_freeze_layers)

        # 2. Separation module
        self.separator = SeparationModule(feature_dim=768,
                                          hidden_dim=separation_hidden)

        # 3. Dual-branch encoders
        self.speech_encoder = BranchEncoder(768, branch_dim)
        self.env_encoder    = BranchEncoder(768, branch_dim)

        # 4. Fusion
        fused_dim = branch_dim * 2                          # 1024
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.fusion_drop = nn.Dropout(0.2)

        # 5. Temporal modeling
        self.bilstm    = BiLSTMEncoder(fused_dim, lstm_hidden, lstm_layers, dropout)
        lstm_out_dim   = lstm_hidden * 2                    # 512 (bidirectional)

        # 6. Attention pooling
        self.attention = ArtifactAttention(lstm_out_dim, attn_heads, dropout=0.1)

        # 7. Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(lstm_out_dim, num_classes),
        )

    def forward(self, waveform: torch.Tensor):
        """
        Args:
            waveform : (B, T_samples)  raw PCM at 16 kHz

        Returns:
            logits  : (B, num_classes)
            l_sep   : scalar separation loss
        """
        # 1. SSL features
        z = self.wav2vec2(waveform)                         # (B, T, 768)

        # 2. Mask-based separation
        z_speech, z_env, l_sep = self.separator(z)         # (B, T, 768) each

        # 3. Branch encoders
        h_speech = self.speech_encoder(z_speech)           # (B, T, 512)
        h_env    = self.env_encoder(z_env)                  # (B, T, 512)

        # 4. Fusion
        fused = torch.cat([h_speech, h_env], dim=-1)       # (B, T, 1024)
        fused = self.fusion_drop(self.fusion_norm(fused))

        # 5. BiLSTM
        temporal = self.bilstm(fused)                      # (B, T, 512)

        # 6. Attention pooling
        pooled = self.attention(temporal)                  # (B, 512)

        # 7. Classification
        logits = self.classifier(pooled)                   # (B, 5)

        return logits, l_sep

    def get_param_groups(self, lr_base: float = 1e-4, lr_ssl: float = 1e-5):
        """
        Returns discriminative learning-rate parameter groups:
        - Wav2Vec2 fine-tune layers : lr_ssl  (10× smaller)
        - All other parameters      : lr_base
        """
        ssl_params   = list(self.wav2vec2.parameters())
        ssl_ids      = {id(p) for p in ssl_params}
        other_params = [p for p in self.parameters() if id(p) not in ssl_ids]
        return [
            {"params": other_params,           "lr": lr_base},
            {"params": ssl_params,             "lr": lr_ssl},
        ]


# ─── Legacy ResNet-18 Baseline (kept for ablation) ────────────────────────────

import torchvision.models as tv_models

class DeepfakeDetectorResNet(nn.Module):
    """
    Original ResNet-18 baseline.
    Kept in-tree for ablation study A0.
    Input: Log-Mel spectrogram (B, 1, 128, T).
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = tv_models.resnet18(weights=weights)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x: torch.Tensor):
        return self.resnet(x), torch.tensor(0.0, device=x.device)
