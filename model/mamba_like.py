import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaLikeChannelBlock(nn.Module):
    def __init__(self, channels, state_dim=16, hidden_ratio=2, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.state_dim = state_dim
        self.hidden_dim = channels * hidden_ratio

        # Gating mechanism (like z in Mamba)
        self.gate_conv = nn.Conv2d(channels, channels, 1)
        self.sigmoid = nn.Sigmoid()

        # State interaction projection (x_proj in Mamba)
        self.state_proj = nn.Conv2d(channels, self.hidden_dim, 1)

        # Simulate decay / memory with depthwise conv
        self.memory_conv = nn.Conv2d(
            self.hidden_dim, self.hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            groups=self.hidden_dim  # depthwise
        )

        # Output projection
        self.out_proj = nn.Conv2d(self.hidden_dim, channels, 1)

        # Optional: normalization
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        # x: (B, C, H, W)
        gate = self.sigmoid(self.gate_conv(x))             # (B, C, H, W)
        proj = self.state_proj(x)                          # (B, C*hidden_ratio, H, W)
        memory = self.memory_conv(proj)                    # decay-like depthwise conv
        memory = memory * gate.repeat(1, memory.shape[1] // self.channels, 1, 1)
        out = self.out_proj(memory)                        # (B, C, H, W)
        return self.norm(out + x)                          # residual + norm
