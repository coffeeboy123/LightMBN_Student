import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoiseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 3, padding=1)
        )
    def forward(self, x):
        return self.net(x)

class IRChannelPreprocess(nn.Module):
    def __init__(self, out_h=384, out_w=128):
        super().__init__()
        self.denoise = DenoiseModule()
        self.out_h = out_h
        self.out_w = out_w

    def lowpass_filter(self, x, kernel_size=15):
        return F.avg_pool2d(x, kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        # x: (B, 3, 64, 64)
        c1 = x[:, 0:1, :, :]                     # 채널1: 원본
        c2 = self.denoise(x[:, 1:2, :, :])       # 채널2: 노이즈 제거
        c3 = self.lowpass_filter(x[:, 2:3, :, :])# 채널3: 저주파 추출

        out = torch.cat([c1, c2, c3], dim=1)     # (B, 3, 64, 64)
        # 384x128로 resize
        out = F.interpolate(out, size=(self.out_h, self.out_w), mode='bilinear', align_corners=False)
        return out
