from torch import nn

class PartAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        att_map = self.att(x)            # (B, C, H, W)
        out = x * att_map                # apply attention
        return self.norm(out + x)        # residual + normalization
