# ---------- 1) 입력 x → 브랜치 3중 스칼라 게이트 ----------
import torch
from torch import nn
import torch.nn.functional as F

class InputBranchGate(nn.Module):
    """
    입력 x만 보고 [global, partial, channel] 3개 스칼라 게이트를 산출.
    - 아주 얕은 CNN + GAP → 3-way logits → softmax
    - 출력 shape: (B, 3, 1, 1)
    """
    def __init__(self, in_ch=3, hidden=128, temperature=1.0):
        super().__init__()
        self.T = temperature
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(hidden, 3, 1, bias=True)  # 3 branches

        # 간단 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        z = self.enc(x)                    # (B, hidden, H/4, W/4)
        z = F.adaptive_avg_pool2d(z, 1)    # (B, hidden, 1, 1)
        logits = self.head(z) / self.T     # (B, 3, 1, 1)
        w = torch.softmax(logits, dim=1)   # (B, 3, 1, 1)
        return w
