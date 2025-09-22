import torch
from torch import nn

class LearnableChannelSplit(nn.Module):
    """
    입력 feature를 픽셀 단위로 2그룹으로 나누는 soft mask 라우팅.
    x: (B, C, H, W) -> x0, x1: (B, C, H, W), m: (B, 2, H, W)
    """
    def __init__(self, C, temperature=1.0):
        super().__init__()
        self.mask_head = nn.Sequential(
            nn.Conv2d(C, C // 4, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 4, 2, kernel_size=1, bias=True)   # 2-way mask logits
        )
        self.temperature = temperature

    def forward(self, x):
        # (B, 2, H, W)
        logits = self.mask_head(x)
        m = F.softmax(logits / self.temperature, dim=1)
        # 그룹별 라우팅
        x0 = x * m[:, 0:1, :, :]
        x1 = x * m[:, 1:2, :, :]
        return x0, x1, m


def mask_split_regularizer(m):
    """
    m: (B, 2, H, W) soft mask
    - sharp: 각 픽셀에서 0.5/0.5에 머무르지 않도록 (샤프하게)
    - balance: 전체적으로 한 그룹에만 쏠리지 않도록
    - overlap: 두 그룹의 공존(=중복) 억제
    """
    # 샤프니스(낮을수록 좋음): p*(1-p)의 평균
    sharp = (m * (1.0 - m)).mean()

    # 밸런스: 두 그룹 평균이 0.5에 가깝도록
    mean_g = m.mean(dim=(0, 2, 3))  # (2,)
    balance = ((mean_g - 0.5) ** 2).mean()

    # 오버랩: 픽셀별 두 그룹이 동시에 큰 값 갖지 않도록
    overlap = (m[:, 0:1] * m[:, 1:2]).mean()

    return sharp, balance, overlap
