# modules/per_sample_soft_split.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PerSampleSoftSplit(nn.Module):
    """
    입력 x(B,C,H,W)에 대해, 샘플마다 채널을 좌/우로 '유연하게' 분할.
    - 채널 요약 v = GAP(x) -> (B,C)
    - score = Linear(v) -> (B,C)
    - 분포 기반 임계값 t: 각 샘플의 중앙경향(평균/근사중앙값)
    - p_left = sigmoid((score - t)/tau),  p_right = 1 - p_left
    - 좌/우 출력: x * p_left, x * p_right (채널 유지)
    """
    def __init__(self, C=512, tau=0.5, use_median=False, reg_w_entropy=0.0, reg_w_balance=0.0):
        super().__init__()
        self.tau = tau
        self.use_median = use_median
        self.reg_w_entropy = reg_w_entropy
        self.reg_w_balance = reg_w_balance
        self.fc = nn.Linear(C, C, bias=False)  # 아주 가벼운 선형 변환

    def forward(self, x):
        B, C, H, W = x.shape
        v = F.adaptive_avg_pool2d(x, (1,1)).flatten(1)      # (B,C)
        s = self.fc(v)                                      # (B,C)
        # per-sample 정규화로 분포 안정화
        s = (s - s.mean(dim=1, keepdim=True)) / (s.std(dim=1, keepdim=True) + 1e-5)

        if self.use_median:
            # 채널 스코어의 근사 중앙값: 정확 median은 비용↑, 여기선 kthvalue로 근사
            k = C // 2
            t = s.kthvalue(k, dim=1, keepdim=True).values   # (B,1)
        else:
            t = s.mean(dim=1, keepdim=True)                 # (B,1)  # 평균을 임계값으로

        p_left = torch.sigmoid((s - t) / self.tau)          # (B,C)
        p_right = 1.0 - p_left

        # (B,C,1,1)로 확장해 채널별 소프트 마스크
        m_left = p_left.unsqueeze(-1).unsqueeze(-1)
        m_right = p_right.unsqueeze(-1).unsqueeze(-1)

        x_left  = x * m_left        # (B,C,H,W)
        x_right = x * m_right       # (B,C,H,W)

        # 선택적 정규화 손실(가볍게): 엔트로피↓(결정성↑), 좌/우 균형↑
        loss_reg = x.new_tensor(0.)
        if self.reg_w_entropy > 0:
            p = torch.stack([p_left, p_right], dim=-1)      # (B,C,2)
            ent = -(p.clamp_min(1e-8) * p.clamp_min(1e-8).log()).sum(dim=-1).mean()
            loss_reg = loss_reg + self.reg_w_entropy * ent
        if self.reg_w_balance > 0:
            q = p_left.mean(dim=1)  # (B,)
            bal = ((q - 0.5) ** 2).mean()
            loss_reg = loss_reg + self.reg_w_balance * bal

        return x_left, x_right, loss_reg
