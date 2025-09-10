# loss/complementarity.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def _std_norm(z: torch.Tensor, eps: float = 1e-4):
    z = z - z.mean(dim=0, keepdim=True)
    z = z / (z.std(dim=0, keepdim=True) + eps)
    return z

def _var_reg(z: torch.Tensor, gamma: float = 1.0):
    std = z.std(dim=0)
    return F.relu(gamma - std).pow(2).mean()

class ComplementarityLoss(nn.Module):
    def __init__(self,
                 diag_weight: float = 0.5,
                 offdiag_weight: float = 1.0,
                 var_lambda: float = 1.0,
                 var_gamma: float = 1.0):
        super().__init__()
        self.diag_weight = diag_weight
        self.offdiag_weight = offdiag_weight
        self.var_lambda = var_lambda
        self.var_gamma = var_gamma

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        # (B, D) 보장
        z_a = z_a.view(z_a.size(0), -1)
        z_b = z_b.view(z_b.size(0), -1)

        B = z_a.size(0)
        za = _std_norm(z_a)
        zb = _std_norm(z_b)

        C = (za.T @ zb) / max(B - 1, 1)  # (D, D)
        diag = torch.diagonal(C)         # (D,)
        offdiag = C - torch.diag(diag)   # diag만 0으로 만들어둠
        D = C.size(0)

        # ▶ 평균 기반 정규화: 대각은 D로, 비대각은 (D*D - D)로 나눔
        diag_mse = diag.pow(2).mean()
        offdiag_mse = offdiag.pow(2).sum() / (D * D - D)

        loss_comp = self.diag_weight * diag_mse + self.offdiag_weight * offdiag_mse
        loss_var = _var_reg(z_a, self.var_gamma) + _var_reg(z_b, self.var_gamma)
        return loss_comp + self.var_lambda * loss_var
