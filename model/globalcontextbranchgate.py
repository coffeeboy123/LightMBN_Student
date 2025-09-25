import torch
from torch import nn
import torch.nn.functional as F

class GlobalContextBranchGate(nn.Module):
    """
    입력 x만 사용하지만, '더 전역적'인 문맥을 보도록 한 게이트.
    병렬 3경로:
      - Large 5x5 depthwise (stride=2)
      - Dilated 5x5 depthwise (dilation=2, stride=2)
      - Directional (1xK -> Kx1) separable path (세로/가로 장거리 정보)
    세 경로를 concat → 1x1 fuse → GAP → 3-way logits → softmax
    출력: (B, 3, 1, 1)
    """
    def __init__(self, in_ch=3, stem_ch=64, hidden=192, k_dir=9, learnable_temp=True, init_T=1.0):
        super().__init__()
        assert hidden % 3 == 0, "hidden은 3의 배수가 편합니다(각 분기 균등 채널)"
        self.logT = nn.Parameter(torch.log(torch.tensor([init_T], dtype=torch.float32))) if learnable_temp else None
        h_each = hidden // 3

        # Stem: 해상도 1/2로 줄이며 저차원 특징 추출
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, stem_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch), nn.ReLU(inplace=True),
        )

        # Branch A: Large Kernel (5x5) depthwise + pointwise
        self.large5 = nn.Sequential(
            nn.Conv2d(stem_ch, stem_ch, 5, stride=2, padding=2, groups=stem_ch, bias=False),
            nn.BatchNorm2d(stem_ch), nn.ReLU(inplace=True),
            nn.Conv2d(stem_ch, h_each, 1, bias=False),
            nn.BatchNorm2d(h_each), nn.ReLU(inplace=True),
        )

        # Branch B: Dilated (5x5, dilation=2) depthwise + pointwise  → 더 넓은 수용영역
        self.dil5 = nn.Sequential(
            nn.Conv2d(stem_ch, stem_ch, 5, stride=2, padding=4, dilation=2, groups=stem_ch, bias=False),
            nn.BatchNorm2d(stem_ch), nn.ReLU(inplace=True),
            nn.Conv2d(stem_ch, h_each, 1, bias=False),
            nn.BatchNorm2d(h_each), nn.ReLU(inplace=True),
        )

        # Branch C: Directional (1xK -> Kx1) separable path
        pad = k_dir // 2
        self.dir_path = nn.Sequential(
            # 가로 긴 문맥 (1xK)
            nn.Conv2d(stem_ch, stem_ch, (1, k_dir), stride=(1, 2), padding=(0, pad), groups=stem_ch, bias=False),
            nn.BatchNorm2d(stem_ch), nn.ReLU(inplace=True),
            # 세로 긴 문맥 (Kx1)
            nn.Conv2d(stem_ch, stem_ch, (k_dir, 1), stride=(2, 1), padding=(pad, 0), groups=stem_ch, bias=False),
            nn.BatchNorm2d(stem_ch), nn.ReLU(inplace=True),
            # pointwise 축소
            nn.Conv2d(stem_ch, h_each, 1, bias=False),
            nn.BatchNorm2d(h_each), nn.ReLU(inplace=True),
        )

        # Fuse & head
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Conv2d(hidden, 3, 1, bias=True)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        z = self.stem(x)          # (B, stem_ch, H/2, W/2)
        a = self.large5(z)        # (B, hidden/3, H/4, W/4)
        b = self.dil5(z)          # (B, hidden/3, H/4, W/4)
        c = self.dir_path(z)      # (B, hidden/3, H/4, W/4)
        u = torch.cat([a, b, c], dim=1)  # (B, hidden, H/4, W/4)

        u = self.fuse(u)          # (B, hidden, 1, 1)
        logits = self.head(u)     # (B, 3, 1, 1)
        if self.logT is not None:
            logits = logits / torch.exp(self.logT)
        w = torch.softmax(logits, dim=1)  # (B, 3, 1, 1)
        return w
