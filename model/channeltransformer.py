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
