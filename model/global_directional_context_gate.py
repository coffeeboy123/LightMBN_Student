# global_directional_context_gate.py
import torch
from torch import nn
import torch.nn.functional as F

class SE(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // r, 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(ch // r, ch, 1, bias=True), nn.Sigmoid()
        )
    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = self.fc(w)
        return x * w

class GlobalDirectionalContextGateHeavy(nn.Module):
    """
    큰 커널 + 방향성 컨볼루션 기반의 전역 컨텍스트 게이트 (티처용, heavy).
    입력 x -> (B,3,1,1) [global, partial, channel] 가중치.
    """
    def __init__(
        self,
        in_ch=3,
        stem_ch=128,      # 96~160 추천
        path_ch=160,      # 각 경로 출력 채널
        dir_k=15,         # 방향성 커널 길이 (11/13/15/17 스윕)
        mlp_hidden=512,   # 512~768 추천
        learnable_temp=True,
        init_T=1.0,
    ):
        super().__init__()
        self.logT = nn.Parameter(torch.log(torch.tensor([init_T], dtype=torch.float32))) if learnable_temp else None

        # Stem: 1/2 다운 + 채널 확장
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, stem_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch), nn.ReLU(inplace=True),
            nn.Conv2d(stem_ch, stem_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch), nn.ReLU(inplace=True),
        )

        # Path-A: Large 7x7 -> 11x11
        self.path_large = nn.Sequential(
            nn.Conv2d(stem_ch, path_ch, 7, stride=2, padding=3, bias=False),  # H/4, W/4
            nn.BatchNorm2d(path_ch), nn.ReLU(inplace=True),
            nn.Conv2d(path_ch, path_ch, 11, stride=1, padding=5, bias=False),
            nn.BatchNorm2d(path_ch), nn.ReLU(inplace=True),
        )

        # Path-B: Dilated Large (7x7 dil=2 -> 7x7 dil=3)
        self.path_dilated = nn.Sequential(
            nn.Conv2d(stem_ch, path_ch, 7, stride=2, padding=2*2+3, dilation=2, bias=False),
            nn.BatchNorm2d(path_ch), nn.ReLU(inplace=True),
            nn.Conv2d(path_ch, path_ch, 7, stride=1, padding=3*2+3, dilation=3, bias=False),
            nn.BatchNorm2d(path_ch), nn.ReLU(inplace=True),
        )

        # Path-C: Directional (1xK -> Kx1) 일반 conv, 총 stride 2
        pad = dir_k // 2
        self.path_dir = nn.Sequential(
            nn.Conv2d(stem_ch, path_ch, (1, dir_k), stride=(1, 2), padding=(0, pad), bias=False),
            nn.BatchNorm2d(path_ch), nn.ReLU(inplace=True),
            nn.Conv2d(path_ch, path_ch, (dir_k, 1), stride=(2, 1), padding=(pad, 0), bias=False),
            nn.BatchNorm2d(path_ch), nn.ReLU(inplace=True),
        )

        # SPP-lite: 1x1, 2x2, 3x3 매크로 컨텍스트
        self.spp_pool_sizes = (1, 2, 3)
        self.spp_convs = nn.ModuleList([
            nn.Conv2d(stem_ch, path_ch, 1, bias=False) for _ in self.spp_pool_sizes
        ])
        for m in self.spp_convs:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')

        fused_ch = path_ch * (3 + len(self.spp_pool_sizes))  # A/B/C + SPP(3)
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_ch, fused_ch, 1, bias=False),
            nn.BatchNorm2d(fused_ch), nn.ReLU(inplace=True),
            SE(fused_ch)
        )

        # Head: 1x1 MLP → 3
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fused_ch, mlp_hidden, 1, bias=False),
            nn.BatchNorm2d(mlp_hidden), nn.ReLU(inplace=True),
            nn.Conv2d(mlp_hidden, 3, 1, bias=True)
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x: (B,3,H,W)  ex) 384x128
        z = self.stem(x)                       # (B,stem_ch,H/2,W/2)

        a = self.path_large(z)                 # (B,path_ch,H/4,W/4)
        b = self.path_dilated(z)               # (B,path_ch,H/4,W/4)
        c = self.path_dir(z)                   # (B,path_ch,H/4,W/4)

        # SPP-lite (z 기반, 최종 해상도에 맞춰 업샘플)
        spp_feats = []
        for k, conv1x1 in zip(self.spp_pool_sizes, self.spp_convs):
            p = F.adaptive_avg_pool2d(z, k)    # (B,stem_ch,k,k)
            p = conv1x1(p)                     # (B,path_ch,k,k)
            p = F.interpolate(p, size=a.shape[-2:], mode='bilinear', align_corners=False)
            spp_feats.append(p)

        u = torch.cat([a, b, c] + spp_feats, dim=1)  # (B,fused_ch,H/4,W/4)
        u = self.fuse(u)                              # (B,fused_ch,H/4,W/4)
        logits = self.head(u)                         # (B,3,1,1)
        if self.logT is not None:
            logits = logits / torch.exp(self.logT)
        w = torch.softmax(logits, dim=1)             # (B,3,1,1)
        return w
