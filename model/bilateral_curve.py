import torch
import torch.nn as nn
import torch.nn.functional as F

class BilateralCurveSplit(nn.Module):
    """
    입력:  x (B,C,H,W)
    출력:  c0_vec, c1_vec, m (B,C,1,1), (B,C,1,1), (B,2,H,W)
    추가 loss 없이, δ 보정으로 면적≈50:50 유지
    """
    def __init__(self, in_ch=512, tau_init=0.5, newton_steps=2, eps=1e-6):
        super().__init__()
        self.tau = tau_init
        self.newton_steps = newton_steps
        self.eps = eps
        # 폭 평균 후 H축 컨텍스트로 경계 curve 예측
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//4, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_ch//4, 1, kernel_size=3, padding=1, bias=True)  # -> (B,1,H,1)
        )
        # 초기 편향 0 근처
        nn.init.zeros_(self.head[-1].bias)

    @torch.no_grad()
    def set_tau(self, tau: float):
        self.tau = float(tau)

    def _masked_gap(self, x, m):
        num = (x * m).sum(dim=(2,3), keepdim=True)
        den = (m.sum(dim=(2,3), keepdim=True) + self.eps)
        return num / den

    def forward(self, x):
        B, C, H, W = x.shape
        # 경계 b(y) 예측
        yfeat = x.mean(dim=3, keepdim=True)          # (B,C,H,1)
        b = self.head(yfeat)                          # (B,1,H,1), 실수 좌표
        # 좌표 그리드 (픽셀 단위)
        xs = torch.arange(W, device=x.device, dtype=x.dtype).view(1,1,1,W)  # (1,1,1,W)
        # δ(글로벌 x-shift)로 면적 0.5 맞추기: 뉴턴 1-2회
        delta = x.new_zeros(B,1,1,1)                 # per-sample scalar shift
        tau = max(self.tau, 1e-6)

        # 초기 마스크
        def m0_from_delta(d):
            z = (xs - (b + d)) / tau                  # (B,1,H,W)
            return torch.sigmoid(z)

        m0 = m0_from_delta(delta)
        # 뉴턴 업데이트
        for _ in range(self.newton_steps):
            S = m0.mean(dim=(1,2,3), keepdim=True)    # (B,1,1,1)
            g = m0 * (1.0 - m0)                       # σ'(z)
            dS = (g / tau).mean(dim=(1,2,3), keepdim=True) + 1e-6
            delta = delta - (S - 0.5) / dS
            m0 = m0_from_delta(delta)

        m1 = 1.0 - m0
        # masked GAP -> 좌/우 벡터
        c0_vec = self._masked_gap(x, m0)              # (B,C,1,1)
        c1_vec = self._masked_gap(x, m1)
        m = torch.cat([m0, m1], dim=1)                # (B,2,H,W)
        return c0_vec, c1_vec, m
