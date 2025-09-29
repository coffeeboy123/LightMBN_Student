import torch
from torch import nn
import torch.nn.functional as F

class GlobalDirectionalContextGateLite(nn.Module):
    """
    가벼운 전역 컨텍스트 게이트:
      - Large/Dilated/Directional 모두 depthwise → pointwise로 경량화
      - 출력: (B,3,1,1) [global, partial, channel] 게이트
    """
    def __init__(
        self,
        in_ch=3,
        stem_ch=64,       # 48/64/80 스윕
        path_ch=64,       # 48/64/80 스윕
        dir_k=13,         # 11/13/15 스윕 (가벼워도 충분히 큼)
        mlp_hidden=128,   # 128/192 스윕
        norm='bn',        # 'bn' or 'ln2d'
        learnable_temp=True,
        init_T=1.0,
        dropout_p=0.0,
    ):
        super().__init__()
        self.logT = nn.Parameter(torch.log(torch.tensor([init_T], dtype=torch.float32))) if learnable_temp else None

        Norm = nn.BatchNorm2d if norm == 'bn' else LayerNorm2d

        # Stem: H/2, W/2
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, stem_ch, 3, stride=2, padding=1, bias=False),
            Norm(stem_ch), nn.GELU(),
        )

        # 공통: DW → PW 블록
        def dw_pw_block(dw_ks, dw_stride, dw_pad, dilation=1):
            return nn.Sequential(
                nn.Conv2d(stem_ch, stem_ch, dw_ks, stride=dw_stride,
                          padding=dw_pad, dilation=dilation,
                          groups=stem_ch, bias=False),          # Depthwise
                Norm(stem_ch), nn.GELU(),
                nn.Conv2d(stem_ch, path_ch, 1, bias=False),     # Pointwise
                Norm(path_ch), nn.GELU(),
            )

        # Path-A: Large depthwise 11x11, stride=2 → (H/4,W/4)
        self.path_large = dw_pw_block(dw_ks=11, dw_stride=2, dw_pad=5, dilation=1)

        # Path-B: Dilated depthwise 7x7 (dil=2), stride=2 → (H/4,W/4)
        # padding=6 로 다른 경로와 정확히 정렬
        self.path_dilated = dw_pw_block(dw_ks=7, dw_stride=2, dw_pad=6, dilation=2)

        # Path-C: Directional depthwise (1xK → Kx1), 총 stride 2 → (H/4,W/4)
        pad = dir_k // 2
        self.path_dir = nn.Sequential(
            nn.Conv2d(stem_ch, stem_ch, (1, dir_k), stride=(1, 2), padding=(0, pad),
                      groups=stem_ch, bias=False),
            Norm(stem_ch), nn.GELU(),
            nn.Conv2d(stem_ch, stem_ch, (dir_k, 1), stride=(2, 1), padding=(pad, 0),
                      groups=stem_ch, bias=False),
            Norm(stem_ch), nn.GELU(),
            nn.Conv2d(stem_ch, path_ch, 1, bias=False),
            Norm(path_ch), nn.GELU(),
        )

        # Fuse: concat → 1x1 축소
        fused_in = path_ch * 3
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_in, path_ch, 1, bias=False),
            Norm(path_ch), nn.GELU(),
        )

        # Head: GAP → 1x1-MLP → 3
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(path_ch, mlp_hidden, 1, bias=True),
            nn.GELU(),
            nn.Dropout(p=dropout_p, inplace=False) if dropout_p > 0 else nn.Identity(),
            nn.Conv2d(mlp_hidden, 3, 1, bias=True),
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d,)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x: (B,3,H,W)
        z = self.stem(x)              # (B,stem_ch,H/2,W/2)
        a = self.path_large(z)        # (B,path_ch,H/4,W/4)
        b = self.path_dilated(z)      # (B,path_ch,H/4,W/4)
        c = self.path_dir(z)          # (B,path_ch,H/4,W/4)

        u = torch.cat([a, b, c], dim=1)  # (B,3*path_ch,H/4,W/4)
        u = self.fuse(u)                 # (B,path_ch,H/4,W/4)

        logits = self.head(u)            # (B,3,1,1)
        if self.logT is not None:
            logits = logits / torch.exp(self.logT)
        w = torch.softmax(logits, dim=1) # (B,3,1,1)
        return w


# 선택: LayerNorm2d (BatchNorm 대신 쓰고 싶을 때)
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias   = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps
    def forward(self, x):
        var = x.var(dim=(1,2,3), keepdim=True, unbiased=False)
        mean = x.mean(dim=(1,2,3), keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias
