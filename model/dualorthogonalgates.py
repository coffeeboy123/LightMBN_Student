import torch
from torch import nn

class DualOrthogonalGates(nn.Module):
    """
    x: (B,C,H,W)
    출력: x0, x1, reg  (# reg: 정규화 항목 딕셔너리)
    - GAP→MLP로 per-sample 게이트 두 개 생성 (w0,w1 ∈ [0,1]^C)
    - 정규화: overlap(겹침↓), balance(두 게이트 균형), sparsity(선택성)
    """
    def __init__(self, in_ch=512, hidden=256):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_ch*2, 1, bias=True)  # 두 게이트 한 번에
        )

    def forward(self, x):
        d = self.gap(x)                 # (B,C,1,1)
        g = self.fc(d)                  # (B,2C,1,1)
        w0, w1 = torch.chunk(g, 2, dim=1)
        w0 = torch.sigmoid(w0)
        w1 = torch.sigmoid(w1)

        x0 = x * w0
        x1 = x * w1

        # 정규화 항목
        overlap  = (w0 * w1).mean()                 # 겹침↓
        balance  = (w0.mean() - w1.mean()).abs()    # 균형↑
        sparsity = 0.5*(w0.mean() + w1.mean())      # 선택성(너무 1로 몰리지 않게 조정)

        reg = {"overlap": overlap, "balance": balance, "sparsity": sparsity}
        return x0, x1, reg
