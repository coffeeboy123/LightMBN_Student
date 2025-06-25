import torch
import torch.nn as nn

class GroupSEBlock(nn.Module):
    def __init__(self, channels, groups=4, reduction=16):
        super().__init__()
        assert channels % groups == 0
        self.groups = groups
        self.channels_per_group = channels // groups
        self.se_blocks = nn.ModuleList([
            SEBlock(self.channels_per_group, reduction) for _ in range(groups)
        ])
        
    def forward(self, x):
        # x: (B, C, H, W)
        splits = torch.split(x, self.channels_per_group, dim=1) # 각 group별 (B, Cg, H, W)
        outs = [block(feat) for block, feat in zip(self.se_blocks, splits)]
        return torch.cat(outs, dim=1)

# SEBlock 정의 (위에서 제공한 코드 재사용)
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# 사용 예시 (채널 branch에)
self.channel_branch = nn.Sequential(
    ...기존 OSNet 블록...,
    GroupSEBlock(512, groups=4)  # 4그룹이면 각 그룹 128채널
)
