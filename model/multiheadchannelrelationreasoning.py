import torch
from torch import nn
import torch.nn.functional as F

class MultiHeadChannelRelationReasoning(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.query = nn.Linear(channels, channels)
        self.key   = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

        self.out_proj = nn.Linear(channels, channels)
        self.gamma = nn.Parameter(torch.zeros(1))  # residual scaling

    def forward(self, x):
        b, c, h, w = x.shape
        x_avg = x.view(b, c, -1).mean(dim=2)  # (B, C)

        # Linear projection
        q = self.query(x_avg).view(b, self.num_heads, self.head_dim)  # (B, H, D)
        k = self.key(x_avg).view(b, self.num_heads, self.head_dim)    # (B, H, D)
        v = self.value(x_avg).view(b, self.num_heads, self.head_dim)  # (B, H, D)

        # Compute attention per head
        attn = torch.einsum('bhd,bhe->bhe', q, k) / (self.head_dim ** 0.5)  # (B, H, H)
        attn = F.softmax(attn, dim=-1)

        # Weighted sum of value vectors
        out = torch.einsum('bhe,bhd->bhd', attn, v)  # (B, H, D)
        out = out.reshape(b, self.channels)          # (B, C)

        out = self.out_proj(out).view(b, c, 1, 1)    # (B, C, 1, 1)
        return x + self.gamma * out.expand_as(x)
