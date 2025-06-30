# mamba_ssm 설치 필요 (pip install mamba-ssm)
from torch import nn

class MambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        from mamba_ssm.modules.mamba_simple import Mamba
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(1, d_model)
        self.mamba = Mamba(d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, C, 1)
        x_in = x
        x = self.in_proj(x)               # (B, C, d_model)
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        x = self.out_proj(x)             # (B, C, 1)
        return x + x_in                  # residual connection
