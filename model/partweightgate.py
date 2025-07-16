import torch
from torch import nn
from torch.nn import functional as F

class PartWeightGate(nn.Module):
    def __init__(self, in_dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_dim // 4, 1, 1)  # Scalar importance per part
        )

    def forward(self, x_head, x_upper, x_lower):
        w_head = self.fc(x_head)  # (B, 1, 1, 1)
        w_upper = self.fc(x_upper)
        w_lower = self.fc(x_lower)

        weights = torch.cat([w_head, w_upper, w_lower], dim=1)  # (B, 3, 1, 1)
        weights = F.softmax(weights, dim=1)  # Normalize across parts

        return weights  # shape: (B, 3, 1, 1)