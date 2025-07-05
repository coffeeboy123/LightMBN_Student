import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = x.mean(dim=[2, 3])  # Global average pooling → [B, C]
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EdgeEnhancer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.sobel_x = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.sobel_y = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        sobel_kernel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)
        self.sobel_x.weight.data = sobel_kernel_x.repeat(channels, 1, 1, 1)
        self.sobel_y.weight.data = sobel_kernel_y.repeat(channels, 1, 1, 1)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        gx = self.sobel_x(x)
        gy = self.sobel_y(x)
        grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        return x + grad  # residual edge emphasis

class IRLRPreprocessorV2(nn.Module):
    def __init__(self, input_channels=3, base_channels=32):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.edge_enhancer = EdgeEnhancer(base_channels)
        self.spectral_attn = SpectralAttention(base_channels)

        self.middle = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(6, 2), mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.edge_enhancer(x)
        x = self.spectral_attn(x)
        x = self.middle(x)
        x = self.upsample(x)
        return x
