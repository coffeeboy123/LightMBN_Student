import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Modality Encoder
class ModalityEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=16):
        super(ModalityEncoder, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)

# 2. Structural Feature Extractor
class StructuralFeatureExtractor(nn.Module):
    def __init__(self, in_channels=16, out_channels=48):
        super(StructuralFeatureExtractor, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)

# 3. Upsample Block: PixelShuffle(×2) → Bilinear(H×3)
class PixelShuffleUpsample384x128(nn.Module):
    def __init__(self, in_channels=48):
        super(PixelShuffleUpsample384x128, self).__init__()
        self.conv = nn.Conv2d(in_channels, 3 * 4, kernel_size=3, padding=1)  # upscale ×2 = 4 channels per output
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)  # → 128×128
        self.bilinear_up = nn.Upsample(scale_factor=(3, 1), mode='bilinear', align_corners=False)  # → 384×128

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.bilinear_up(x)
        return x

# 전체 MASSR 구조
class MASSR_PixelShuffle(nn.Module):
    def __init__(self, in_channels=3):
        super(MASSR_PixelShuffle, self).__init__()
        self.me = ModalityEncoder(in_channels)
        self.sfe = StructuralFeatureExtractor()
        self.upsample = PixelShuffleUpsample384x128()

    def forward(self, x):
        x = self.me(x)
        x = self.sfe(x)
        x = self.upsample(x)
        return x
