"""
PH-MetaFormer: 可分离能量函数
H(q, p) = V_net(q) + 0.5 * p^T * diag(1/m) * p

从 Week2 迁移，适配任意 d_latent。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class SeparableEnergy(nn.Module):
    """
    H(q,p) = V(q) + K(p)
    - V(q): 势能 MLP — 可学任意势能地形
    - K(p): 动能 = 0.5 * Σ(p_i² / m_i) — 对角质量矩阵, 解析梯度
    """
    def __init__(self, d_latent: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.d_latent = d_latent
        # 谱归一化: 限制 Lipschitz 常数, 防能量地形过陡
        self.V_net = nn.Sequential(
            spectral_norm(nn.Linear(d_latent, hidden_dim)), nn.SiLU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)), nn.SiLU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)), nn.SiLU(),
            spectral_norm(nn.Linear(hidden_dim // 2, 1)),
        )
        # 对角质量矩阵参数: inv_mass > 0, 上限 clamp
        self.inv_mass_raw = nn.Parameter(torch.ones(d_latent))

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.V(q) + self.K(p)

    def V(self, q: torch.Tensor) -> torch.Tensor:
        return self.V_net(q).squeeze(-1).clamp(min=0.0, max=50.0)

    def K(self, p: torch.Tensor) -> torch.Tensor:
        inv_mass = F.softplus(self.inv_mass_raw).clamp(max=10.0)
        return 0.5 * torch.sum(p * inv_mass * p, dim=-1).clamp(max=100.0)

    def grad_V(self, q: torch.Tensor) -> torch.Tensor:
        """计算 dV/dq，保持计算图完整（不 detach、不造叶子节点）"""
        with torch.autocast(device_type='cuda', enabled=False):
            q_fp32 = q.float()
            is_training = q_fp32.requires_grad

            if not is_training:
                # eval 模式: q 没有梯度链, 安全地创建临时叶子节点
                with torch.enable_grad():
                    q_temp = q_fp32.detach().requires_grad_(True)
                    V = self.V_net(q_temp).sum()
                    grad = torch.autograd.grad(V, q_temp)[0]
                return grad.to(q.dtype)

            # 训练模式: q_fp32 是计算图的一部分，严禁 detach/requires_grad_！
            with torch.enable_grad():
                V = self.V_net(q_fp32).sum()
                grad = torch.autograd.grad(V, q_fp32, create_graph=True)[0]
            return grad.to(q.dtype)

    def grad_K(self, p: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type='cuda', enabled=False):
            p_fp32 = p.float()
            inv_mass = F.softplus(self.inv_mass_raw).clamp(max=10.0).float()
            return (inv_mass * p_fp32).to(p.dtype)

    @property
    def mass(self) -> torch.Tensor:
        return 1.0 / F.softplus(self.inv_mass_raw).clamp(max=10.0)
