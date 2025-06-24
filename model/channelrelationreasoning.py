from torch import nn
import torch
from torch.nn import functional as F

class ChannelRelationReasoning(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.gamma = nn.Parameter(torch.zeros(1))  # residual scaling

    def forward(self, x):
        b, c, h, w = x.shape
        x_avg = x.view(b, c, -1).mean(dim=2)  # (B, C)
        q = self.query(x_avg)                 # (B, C)
        k = self.key(x_avg)                  # (B, C)
        v = self.value(x_avg)                # (B, C)

        attn = torch.bmm(q.unsqueeze(2), k.unsqueeze(1)) / (c ** 0.5)  # (B, C, C)
        attn = F.softmax(attn, dim=-1)  # 채널 간 관계 (B, C, C)

        out = torch.bmm(attn, v.unsqueeze(2)).squeeze(2)  # (B, C)
        out = out.view(b, c, 1, 1)

        return x + self.gamma * out.expand_as(x)  # residual
