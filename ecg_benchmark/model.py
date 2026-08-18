"""
ECGTransformer Model (Single-Head, 5-Superclass Multi-Label)
ecg_benchmark/model.py
"""

import torch
import torch.nn as nn


class ECGTransformer(nn.Module):
    """
    Transformer-based 12-lead ECG classifier — 5-superclass version.

    Architecture:
        1. Conv1D Feature Extractor (12 -> 64 -> 256 channels)
        2. Patch Embedding (25 patches of 50 samples)
        3. Learnable Positional Encoding
        4. Transformer Encoder (6 layers, 8 heads, d_model=256, d_ff=1024, norm_first=True)
        5. Global Average Pooling
        6. Single 5-class classifier head (LayerNorm + Linear(256->128) + GELU + Linear(128->5))
    """

    def __init__(
        self,
        n_leads=12,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.1,
        patch_size=50,
        n_classes=5,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_classes  = n_classes

        # Conv1D Backbone
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, d_model, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )

        # Patch Embedding
        self.n_patches   = 1250 // patch_size   # 25 patches
        self.patch_embed = nn.Linear(patch_size * d_model, d_model)

        # Positional Encoding & Dropout
        self.pos_encoding = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.dropout      = nn.Dropout(dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 5-Class Classifier Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): shape (batch, 5000, 12)
        Returns:
            logits (torch.Tensor): shape (batch, 5) raw logits (no sigmoid)
        """
        B = x.shape[0]

        x = x.transpose(1, 2)            # (B, 12, 5000)
        x = self.conv1(x)                 # (B, 64, 2500)
        x = self.conv2(x)                 # (B, 256, 1250)
        x = x.transpose(1, 2)            # (B, 1250, 256)

        x = x.reshape(B, self.n_patches, self.patch_size * x.shape[-1])
        x = self.patch_embed(x)           # (B, 25, 256)

        x = x + self.pos_encoding
        x = self.dropout(x)

        x = self.transformer(x)           # (B, 25, 256)
        x = x.mean(dim=1)                 # (B, 256) global average pooling

        logits = self.classifier(x)       # (B, 5) raw logits
        return logits
