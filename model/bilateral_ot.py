import torch
import torch.nn as nn
import torch.nn.functional as F

def _cosine_cost(x, a, eps=1e-6):
    # x: (B, HW, C), a: (2, C)
    x = F.normalize(x, dim=-1)
    a = F.normalize(a, dim=-1)
    # cost = 1 - cos
    return 1 - torch.einsum('bhc,kc->bhk', x, a)  # (B, HW, 2)

class BilateralOTSplit(nn.Module):
    """
    입력 x: (B, C, H, W)
    출력 c0_vec, c1_vec: (B, C, 1, 1)
    - 두 개의 learnable prototype a∈R^{2×C}로 HW개의 토큰을 균형(b=[0.5,0.5]) 할당
    - Sinkhorn-Knopp로 transport plan P 계산 (균형 분할 + 부드러운 경계)
    - 추가 loss 불필요
    """
    def __init__(self, in_ch=512, eps=0.07, iters=7):
        super().__init__()
        self.anchors = nn.Parameter(torch.randn(2, in_ch))  # 좌/우 프로토타입
        nn.init.normal_(self.anchors, std=0.02)
        self.eps = eps       # entropic regularization (온도 느낌)
        self.iters = iters   # sinkhorn 반복 횟수
        self._tiny = 1e-8

    @torch.no_grad()
    def set_eps(self, eps: float):
        self.eps = float(eps)

    def sinkhorn(self, C):
        B, HW, K = C.shape
        assert K == 2
        Kmat = torch.exp(-C / max(self.eps, self._tiny)) + self._tiny  # (B, HW, 2)
        a = torch.full((B, HW), 1.0 / HW, device=C.device, dtype=Kmat.dtype)
        b = torch.full((B, 2), 0.5, device=C.device, dtype=Kmat.dtype)

        u = torch.ones_like(a)
        v = torch.ones_like(b)

        for _ in range(self.iters):
            # (B,HW,2) * (B,1,2) -> (B,HW,2) sum_k -> (B,HW)
            Kv = (Kmat * v.unsqueeze(1)).sum(dim=2) + self._tiny
            u = a / Kv

            # ❌ 기존: (Kmat.transpose(1,2) * u.unsqueeze(2)).sum(dim=1)
            # ✅ 수정: (B,2,HW) * (B,1,HW) -> sum_HW -> (B,2)
            KTu = (Kmat.transpose(1, 2) * u.unsqueeze(1)).sum(dim=2) + self._tiny
            v = b / KTu

        P = Kmat * u.unsqueeze(2) * v.unsqueeze(1)  # (B, HW, 2)
        return P

    def forward(self, x):
        B, C, H, W = x.shape
        X = x.flatten(2).transpose(1, 2)         # (B, HW, C)
        Cmat = _cosine_cost(X, self.anchors)     # (B, HW, 2)
        P = self.sinkhorn(Cmat)                  # (B, HW, 2), balanced

        # 픽셀-후행 확률 p(j|i) = P_{i,j} / a_i, a_i=1/HW → p = P * HW
        m = (P * H * W).transpose(1, 2).reshape(B, 2, H, W)  # (B,2,H,W)
        m0 = m[:, 0:1]
        m1 = m[:, 1:2]

        # masked GAP
        num0 = (x * m0).sum(dim=(2, 3), keepdim=True); den0 = m0.sum(dim=(2, 3), keepdim=True) + 1e-6
        num1 = (x * m1).sum(dim=(2, 3), keepdim=True); den1 = m1.sum(dim=(2, 3), keepdim=True) + 1e-6
        c0_vec = num0 / den0   # (B, C, 1, 1)
        c1_vec = num1 / den1
        return c0_vec, c1_vec, m  # m은 원하면 시각화 가능
