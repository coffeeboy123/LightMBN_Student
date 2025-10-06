import torch
from torch import nn
import torch.nn.functional as F

class BranchQualityGate(nn.Module):
    """
    입력 x + 브랜치 피처(glo, par, cha)를 모두 보고
    [global, partial, channel] 3개 스칼라 게이트(B,3,1,1)를 산출.
    """
    def __init__(
        self,
        in_ch=3,          # 입력 채널
        feat_ch=512,      # 브랜치 피처 채널(OSNet 분기 출력 기준)
        token_ch=128,     # 각 토큰 임베딩 채널
        hidden=256,       # fusion MLP 은닉 채널
        learnable_temp=True,
        init_T=1.0,
        detach_branch=False,  # True면 브랜치 피처에서 게이트로 gradient 차단
    ):
        super().__init__()
        self.detach_branch = detach_branch
        self.logT = nn.Parameter(torch.log(torch.tensor([init_T], dtype=torch.float32))) if learnable_temp else None

        # 입력 x 요약(전역 토큰)
        self.x_encoder = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, token_ch, 1, bias=False),
            nn.BatchNorm2d(token_ch), nn.ReLU(inplace=True),
        )

        # 브랜치 토큰(각각 동일 모양의 작은 1x1)
        def branch_token():
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(feat_ch, token_ch, 1, bias=False),
                nn.BatchNorm2d(token_ch), nn.ReLU(inplace=True),
            )
        self.g_token = branch_token()
        self.p_token = branch_token()
        self.c_token = branch_token()

        # 토큰 결합 후 MLP(1x1 conv 두 층)
        fused_in = token_ch * 4  # x,g,p,c
        self.fuse_head = nn.Sequential(
            nn.Conv2d(fused_in, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 1, bias=True)  # 3 branches
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, g, p, c):
        if self.detach_branch:
            g = g.detach(); p = p.detach(); c = c.detach()

        tx = self.x_encoder(x)   # (B,token_ch,1,1)
        tg = self.g_token(g)     # (B,token_ch,1,1)
        tp = self.p_token(p)     # (B,token_ch,1,1)
        tc = self.c_token(c)     # (B,token_ch,1,1)

        u = torch.cat([tx, tg, tp, tc], dim=1)  # (B,4*token_ch,1,1)
        logits = self.fuse_head(u)              # (B,3,1,1)

        if self.logT is not None:
            logits = logits / torch.exp(self.logT)
        w = torch.softmax(logits, dim=1)        # (B,3,1,1)
        return w
