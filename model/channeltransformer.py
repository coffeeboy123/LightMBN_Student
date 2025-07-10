import torch, torch.nn as nn
from torch.nn import functional as F

class ChannelTransformer(nn.Module):
    def __init__(self, channels, dim_reduction=64, heads=4, depth=1):
        super().__init__()
        self.dim_reduce = nn.Linear(channels, dim_reduction)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_reduction, nhead=heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.dim_expand = nn.Linear(dim_reduction, channels)
        self.sigmoid = nn.Sigmoid()          # gating 형태로 곱할 때 사용

    def forward(self, x):
        b, c, h, w = x.shape                             # (B,C,H,W)
        token = x.view(b, c, -1).mean(dim=2)             # GAP → (B,C)
        token = self.dim_reduce(token)                   # (B,C)→(B,D)
        token = self.transformer(token)                  # MHSA 적용
        attn  = self.sigmoid(self.dim_expand(token))     # (B,C)
        attn  = attn.view(b, c, 1, 1)
        return x * attn + x                              # residual + weighting
    

import torch
import torch.nn as nn

class ChannelTransformer_Multi(nn.Module):
    def __init__(self, channels, dim_reduction=64, heads=4, depth=1):
        super().__init__()
        self.dim_reduce = nn.Linear(channels, dim_reduction)  # channel → D
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_reduction, nhead=heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.dim_expand = nn.Linear(dim_reduction, channels)  # D → channel
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape  # (B, C, H, W)

        # (B, C, H, W) → (B, H*W, C)
        x_reshaped = x.permute(0, 2, 3, 1).reshape(b, h * w, c)

        # (B, H*W, C) → (B, H*W, D)
        tokens = self.dim_reduce(x_reshaped)

        # Transformer Encoder
        tokens = self.transformer(tokens)  # (B, H*W, D)

        # Mean Pooling → (B, D)
        pooled = tokens.mean(dim=1)

        # (B, D) → (B, C)
        attn = self.sigmoid(self.dim_expand(pooled))

        # Attention 적용: (B, C, 1, 1)
        attn = attn.view(b, c, 1, 1)

        # Residual + attention
        return x * attn + x

