import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
    def forward(self, x):
        return x + self.block(x)

# Contrastive Laplacian Residual Upsampler (CLRU)
class CLRU(nn.Module):
    """
    Contrastive Laplacian Residual Upsampler
      - Input:  (B, 3, 64, 64)
      - Output: (B, 3, 384, 128)
    """
    def __init__(self, in_ch=3, mid_ch=32, num_res=2):
        super().__init__()
        # Edge branch: learnable Laplacian-like conv
        self.edge_branch = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        # Contrast branch: 1x1 conv to emphasize thermal contrast
        self.contrast_branch = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        # Fusion: 1x1 conv after addition
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(mid_ch) for _ in range(num_res)]
        )
        # Upsample: PixelShuffle + bilinear
        self.to_ps = nn.Conv2d(mid_ch, in_ch * 4, kernel_size=3, padding=1, bias=False)
        self.ps    = nn.PixelShuffle(upscale_factor=2)  # 64->128
        self.up    = nn.Upsample((384, 128), mode='bilinear', align_corners=False)

    def forward(self, x):
        # x: [B, 3, 64, 64]
        e = self.edge_branch(x)       # [B, mid_ch, 64,64]
        c = self.contrast_branch(x)   # [B, mid_ch, 64,64]
        f = e + c                     # fuse by addition
        f = self.fuse(f)              # [B, mid_ch, 64,64]
        f = self.res_blocks(f)        # residual refinement
        u = self.to_ps(f)             # [B, 3*4, 64,64]
        u = self.ps(u)                # [B, 3,128,128]
        u = self.up(u)                # [B, 3,384,128]
        return u

# Test
if __name__ == "__main__":
    inp = torch.randn(2, 3, 64, 64)
    model = CLRU()
    out = model(inp)
    print("Input shape :", inp.shape)
    print("Output shape:", out.shape)  # -> [2, 3, 384, 128]
