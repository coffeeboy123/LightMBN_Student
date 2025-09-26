# spectral_spatial_branch_gate.py
import torch
from torch import nn
import torch.nn.functional as F

class SpectralSpatialBranchGate(nn.Module):
    """
    IR/LR에 맞춘 입력-분포 게이트:
      - FFT 스펙트럼의 저/중/고 대역 에너지 (radial bands)
      - 스펙트럼 엔트로피
      - 공간 밝기 mean/std
    => 3-way softmax (B,3,1,1)
    """
    def __init__(self, resize_hw=(64, 64), r1=0.10, r2=0.30, mlp_hidden=256,
                 learnable_temp=True, init_T=1.0):
        super().__init__()
        self.Ht, self.Wt = resize_hw
        self.r1 = r1
        self.r2 = r2
        self.logT = nn.Parameter(torch.log(torch.tensor([init_T], dtype=torch.float32))) if learnable_temp else None

        in_dim = 3 + 1 + 2   # low/mid/high + spec_entropy + mean/std
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden), nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, 3)
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def _to_gray(x):
        if x.size(1) == 3:
            r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
            return 0.2989*r + 0.5870*g + 0.1140*b
        return x

    def _radial_masks(self, B, H, W, device):
        # 반지름(0~1) 정규화된 원형 마스크 3개 생성
        yy, xx = torch.meshgrid(torch.linspace(-1,1,H,device=device),
                                torch.linspace(-1,1,W,device=device), indexing='ij')
        rr = torch.sqrt(xx**2 + yy**2)  # (H,W), 0~sqrt(2)
        rr = rr / rr.max()              # 0~1
        low = (rr <= self.r1).float()
        mid = ((rr > self.r1) & (rr <= self.r2)).float()
        hig = (rr > self.r2).float()
        # 배치 브로드캐스트용
        return low.unsqueeze(0).expand(B, -1, -1), \
               mid.unsqueeze(0).expand(B, -1, -1), \
               hig.unsqueeze(0).expand(B, -1, -1)

    def forward(self, x):
        B, C, H, W = x.shape
        # 1) 그레이 & 리사이즈
        gray = self._to_gray(x)                     # (B,1,H,W)
        g = F.interpolate(gray, size=(self.Ht, self.Wt), mode='bilinear', align_corners=False)  # (B,1,Ht,Wt)

        # 2) FFT magnitude
        spec = torch.fft.rfft2(g.squeeze(1), norm='ortho')     # (B,Ht, Wt//2+1), complex
        mag  = torch.abs(spec)                                  # (B,Ht, Wt//2+1)
        # rfft2는 절반 평면이므로 좌우 대칭 가정으로 마스크도 같은 크기에 맞춤
        Hf, Wf = mag.shape[1], mag.shape[2]
        low, mid, hig = self._radial_masks(B, Hf, Wf, mag.device)  # (B,Hf,Wf)

        # 3) 대역 에너지 (정규화)
        eps = 1e-8
        total = (mag**2).sum(dim=(1,2)) + eps                       # (B,)
        low_e = ((mag**2) * low).sum(dim=(1,2)) / total             # (B,)
        mid_e = ((mag**2) * mid).sum(dim=(1,2)) / total             # (B,)
        hig_e = ((mag**2) * hig).sum(dim=(1,2)) / total             # (B,)

        # 4) 스펙트럼 엔트로피 (에너지 분포의 불확실성)
        p = (mag**2) / total.view(B,1,1)
        spec_entropy = -(p.clamp_min(eps) * (p.clamp_min(eps)).log()).sum(dim=(1,2)) / (Hf*Wf)  # (B,)

        # 5) 공간 도메인 밝기 통계
        mu  = g.mean(dim=(2,3)).squeeze(1)                          # (B,)
        std = g.std (dim=(2,3)).squeeze(1)                          # (B,)

        feat = torch.stack([low_e, mid_e, hig_e, spec_entropy, mu, std], dim=1)  # (B,6)

        logits = self.mlp(feat)                                      # (B,3)
        if self.logT is not None:
            logits = logits / torch.exp(self.logT)
        w = torch.softmax(logits, dim=1).view(B,3,1,1)
        return w
