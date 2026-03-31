"""
Mask-Based Joint Source Separation Module
==========================================
Implements component-aware separation for the ESDD2 5-class deepfake detection task.
Based on: Zhang et al. (2026) - ESDD2 / CompSpoofV2 Framework.

Two learnable soft masks (M_speech, M_env) are estimated from a shared latent
representation Z and applied element-wise to disentangle speech and environmental
components before classification.

Separation loss (Frobenius orthogonality):
    L_sep = || Z_speech * Z_env ||_F
encourages the two masked representations to be orthogonal (non-overlapping).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskNetwork(nn.Module):
    """
    Lightweight MLP that estimates a soft mask in [0, 1] over the latent space.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()          # mask values ∈ [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: latent tensor of shape (B, T, D) or (B, D)
        Returns:
            mask of same shape as z, values in [0, 1]
        """
        return self.net(z)


class SeparationModule(nn.Module):
    """
    Joint mask-based source separation module.

    Given a shared latent Z (from Wav2Vec2 or similar encoder):
        Z_speech = Z ⊙ M_speech
        Z_env    = Z ⊙ M_env

    The Frobenius separation loss penalises overlap between the two masked
    representations during joint training with the classifier.

    Args:
        feature_dim : dimensionality of the shared latent space (e.g. 768 for Wav2Vec2-base)
        hidden_dim  : hidden size of each mask network (default 256)
    """

    def __init__(self, feature_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.speech_mask_net = MaskNetwork(feature_dim, hidden_dim)
        self.env_mask_net    = MaskNetwork(feature_dim, hidden_dim)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z : shared latent tensor, shape (B, T, D)

        Returns:
            z_speech : speech-dominant representation, shape (B, T, D)
            z_env    : environment-dominant representation, shape (B, T, D)
            l_sep    : scalar separation loss (Frobenius norm of element-wise product)
        """
        m_speech = self.speech_mask_net(z)   # (B, T, D) ∈ [0, 1]
        m_env    = self.env_mask_net(z)       # (B, T, D) ∈ [0, 1]

        z_speech = z * m_speech              # element-wise masking
        z_env    = z * m_env

        # Frobenius orthogonality loss: encourages z_speech and z_env to be non-overlapping
        l_sep = torch.norm(z_speech * z_env, p='fro') / (z.shape[0] * z.shape[1])

        return z_speech, z_env, l_sep

    @staticmethod
    def separation_loss_weight(epoch: int, warmup_epochs: int = 10,
                               lambda_start: float = 0.1,
                               lambda_end: float = 0.3) -> float:
        """
        Linear warm-up schedule for the separation loss weight lambda.
        Prevents the separation objective from overwhelming classification early on.

        Args:
            epoch         : current training epoch (0-indexed)
            warmup_epochs : number of epochs for warm-up
            lambda_start  : starting lambda value
            lambda_end    : target lambda value after warm-up

        Returns:
            current lambda weight (float)
        """
        if epoch >= warmup_epochs:
            return lambda_end
        progress = epoch / warmup_epochs
        return lambda_start + progress * (lambda_end - lambda_start)
