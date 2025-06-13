import torch
import torch.nn as nn

class LocalGlobalFusionUpsampler3ch(nn.Module):
    """
    LGFU-3ch: 3채널 IR/LR → (3,384,128) 업샘플러
    """
    def __init__(self, in_ch=3, mid_ch=32):
        super().__init__()
        # 1) Local feature: 3×3 conv ×2
        self.local = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(True),
        )
        # 2) Global context: GAP → 1×1 conv → expand
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(True),
        )
        # 3) Fuse local + global
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_ch*2, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(True),
        )
        # 4) Upsample: PixelShuffle×2 → Bilinear(×3 H)
        self.to_ps = nn.Conv2d(mid_ch, 3 * 4, 3, padding=1)
        self.ps   = nn.PixelShuffle(2)  # 64→128
        self.up   = nn.Upsample((384,128),
                                mode='bilinear',
                                align_corners=False)

    def forward(self, x):
        # local
        lf = self.local(x)                       # [B, mid, 64,64]
        # global
        gf = self.global_pool(x)                 # [B, 3, 1, 1]
        gf = self.global_conv(gf)                # [B, mid,1,1]
        gf = gf.expand(-1, -1, 64, 64)           # [B, mid,64,64]
        # fuse
        f  = torch.cat([lf, gf], dim=1)          # [B, mid*2,64,64]
        f  = self.fuse(f)                        # [B, mid,64,64]
        # upsample
        u  = self.to_ps(f)                       # [B, 3*4,64,64]
        u  = self.ps(u)                          # [B, 3,128,128]
        u  = self.up(u)                          # [B, 3,384,128]
        return u

# 테스트
if __name__ == "__main__":
    m = LocalGlobalFusionUpsampler3ch()
    inp = torch.randn(1, 3, 64, 64)
    out = m(inp)
    print(inp.shape, "→", out.shape)  # [1,3,64,64] → [1,3,384,128]
