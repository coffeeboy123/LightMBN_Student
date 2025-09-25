import torch
from torch import nn
import torch.nn.functional as F

class GlobalContextBranchGateHeavy(nn.Module):
    """
    입력 x만 사용하되 '전역 문맥'을 강하게 잡도록 일반 conv 위주로 설계한 헤비 버전.
    병렬 3경로 (모두 일반 conv, groups=1):
      A) Large 5x5 stack (stride=2)  : 넓은 로컬+중거리 문맥
      B) Dilated 5x5 stack (rate=2)  : 파라미터 늘리며 수용영역 확장
      C) Directional (1xK -> Kx1)    : 가로/세로 장거리 문맥(일반 conv로)
    세 경로 concat → 1x1 fuse(확장) → GAP → 작은 MLP head → softmax → (B,3,1,1)
    """
    def __init__(
        self,
        in_ch=3,
        stem_ch=96,       # 64 -> 96 (표현력 ↑)
        path_ch=128,      # 각 분기 출력 채널 수
        k_dir=11,         # 방향성 커널 길이(7/9/11/13 추천)
        mlp_hidden=512,   # MLP hidden
        learnable_temp=True,
        init_T=1.0,
    ):
        super().__init__()
        self.logT = nn.Parameter(torch.log(torch.tensor([init_T], dtype=torch.float32))) if learnable_temp else None

        # Stem: 일반 3x3 conv로 해상도 1/2 다운 + 채널 확장
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, stem_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch), nn.ReLU(inplace=True),
            nn.Conv2d(stem_ch, stem_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch), nn.ReLU(inplace=True),
        )

        # A: Large 5x5 stack (stride=2)  → H/4, W/4
        self.large5 = nn.Sequential(
            nn.Conv2d(stem_ch, path_ch, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(path_ch), nn.ReLU(inplace=True),
            nn.Conv2d(path_ch, path_ch, 5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(path_ch), nn.ReLU(inplace=True),
        )

        # B: Dilated 5x5 stack (rate=2, stride=2)
        self.dil5 = nn.Sequential(
            nn.Conv2d(stem_ch, path_ch, 5, stride=2, padding=4, dilation=2, bias=False),
            nn.BatchNorm2d(path_ch), nn.ReLU(inplace=True),
            nn.Conv2d(path_ch, path_ch, 5, stride=1, padding=4, dilation=2, bias=False),
            nn.BatchNorm2d(path_ch), nn.ReLU(inplace=True),
        )

        # C: Directional (1xK -> Kx1) with 일반 conv
        pad = k_dir // 2
        self.dir_path = nn.Sequential(
            nn.Conv2d(stem_ch, path_ch, (1, k_dir), stride=(1, 2), padding=(0, pad), bias=False),
            nn.BatchNorm2d(path_ch), nn.ReLU(inplace=True),
            nn.Conv2d(path_ch, path_ch, (k_dir, 1), stride=(2, 1), padding=(pad, 0), bias=False),
            nn.BatchNorm2d(path_ch), nn.ReLU(inplace=True),
        )

        # Fuse: concat(A,B,C) → 1x1 확장(채널 혼합) → GAP
        fused_ch = path_ch * 3
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_ch, fused_ch, 1, bias=False),
            nn.BatchNorm2d(fused_ch), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        # Head: 작은 MLP (Conv1x1로 구현) → 3-way 로짓
        self.head = nn.Sequential(
            nn.Conv2d(fused_ch, mlp_hidden, 1, bias=False),
            nn.BatchNorm2d(mlp_hidden), nn.ReLU(inplace=True),
            nn.Conv2d(mlp_hidden, 3, 1, bias=True),
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        z = self.stem(x)          # (B, stem_ch, H/2, W/2)
        a = self.large5(z)        # (B, path_ch, H/4, W/4)
        b = self.dil5(z)          # (B, path_ch, H/4, W/4)
        c = self.dir_path(z)      # (B, path_ch, H/4, W/4)
        u = torch.cat([a, b, c], dim=1)  # (B, 3*path_ch, H/4, W/4)

        u = self.fuse(u)          # (B, 3*path_ch, 1, 1)
        logits = self.head(u)     # (B, 3, 1, 1)
        if self.logT is not None:
            logits = logits / torch.exp(self.logT)
        w = torch.softmax(logits, dim=1)  # (B, 3, 1, 1)
        return w
