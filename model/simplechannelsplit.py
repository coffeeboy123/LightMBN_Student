import torch
from torch import nn

class SimpleChannelSplit(nn.Module):
    """
    cha: (B, C, H, W) → c0, c1: (B, C, 1, 1)
    - 1x1 두 번으로 2개의 마스크 로짓 생성
    - softmax로 픽셀별 2-way 마스킹
    - 각 그룹을 (1,1)로 평균 풀링해 반환
    """
    def __init__(self, C, temperature=1.0):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(C, C // 4, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 4, 2, kernel_size=1, bias=True)  # 2개 마스크 로짓
        )
        self.t = float(temperature)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):  # (B,C,H,W)
        logits = self.head(x)                       # (B,2,H,W)
        m = F.softmax(logits / self.t, dim=1)       # (B,2,H,W)
        x0 = x * m[:, 0:1]                          # (B,C,H,W)
        x1 = x * m[:, 1:2]                          # (B,C,H,W)
        return self.avg(x0), self.avg(x1)           # (B,C,1,1), (B,C,1,1)
