"""
PH-MetaFormer: Port-Hamiltonian 动力学 + Leapfrog 辛积分器

从 Week2 迁移。核心公式:
  ż = (J - D)·∇H(z)
  J: 辛矩阵 [[0,I],[-I,0]]
  D: 对角耗散矩阵 (可学习)

支持可选的 J_net (反对称保守残差) 和 R_net (半正定耗散残差)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PHDynamics(nn.Module):
    """Port-Hamiltonian 动力学，含可选的 J_net / R_net 残差"""

    def __init__(
        self,
        d_latent: int = 24,
        use_learned_residuals: bool = False,
    ):
        super().__init__()
        self.d = d_latent
        self.use_learned_residuals = use_learned_residuals

        # ── 固定辛矩阵 J = [[0, I], [-I, 0]] ──
        Z = torch.zeros(d_latent, d_latent)
        I = torch.eye(d_latent)
        self.register_buffer('J_fixed', torch.cat([
            torch.cat([Z, I], dim=1),
            torch.cat([-I, Z], dim=1)
        ], dim=0))

        # ── 可学习对角耗散 D = diag(α_q, α_p), α ≥ 0 ──
        self.alpha_q_raw = nn.Parameter(torch.zeros(d_latent))
        self.alpha_p_raw = nn.Parameter(torch.zeros(d_latent))

        # ── 可选: 结构化学习残差 ──
        if use_learned_residuals:
            # J_net: 反对称保守流 = L_J - L_J^T
            self.L_J_raw = nn.Parameter(torch.randn(2 * d_latent, 2 * d_latent) * 0.01)
            # R_net: 半正定耗散 = L_R @ L_R^T
            self.L_R_raw = nn.Parameter(torch.randn(2 * d_latent, 2 * d_latent) * 0.01)
        else:
            self.L_J_raw = None
            self.L_R_raw = None

    def get_J_net(self) -> torch.Tensor:
        if self.L_J_raw is None:
            return torch.zeros(2 * self.d, 2 * self.d, device=self.J_fixed.device)
        L = torch.tril(self.L_J_raw, diagonal=-1)
        return L - L.T

    def get_R_net(self) -> torch.Tensor:
        if self.L_R_raw is None:
            return torch.zeros(2 * self.d, 2 * self.d, device=self.J_fixed.device)
        L = torch.tril(self.L_R_raw)
        # 强制对角线为正 (保证半正定)
        L_diag = F.softplus(torch.diag(L))
        L = L - torch.diag(torch.diag(L)) + torch.diag(L_diag)
        return L @ L.T

    def get_dissipation(self) -> torch.Tensor:
        """对角耗散 D, sigmoid 缩放上限 0.05"""
        alpha_q = torch.sigmoid(self.alpha_q_raw) * 0.05
        alpha_p = torch.sigmoid(self.alpha_p_raw) * 0.05
        return torch.diag(torch.cat([alpha_q, alpha_p]))

    def forward(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        energy_net,
        dt: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        执行一步 PH 动力学: ż = (J_fixed + J_net - D - R_net)·∇H

        Args:
            q, p: 广义坐标和动量, 各 (B, d_latent)
            energy_net: SeparableEnergy 实例
            dt: 积分步长

        Returns:
            dq, dp: 各 (B, d_latent) 的增量
        """
        MAX_DERIV = 50.0  # 放宽物理限速: 允许台风等剧烈变化
        grad_q = torch.clamp(energy_net.grad_V(q), -MAX_DERIV, MAX_DERIV)
        grad_p = torch.clamp(energy_net.grad_K(p), -MAX_DERIV, MAX_DERIV)
        grad_H = torch.cat([grad_q, grad_p], dim=-1).unsqueeze(-1)  # (B, 2d, 1)

        # [修正] 只计算非保守残差: leapfrog_step 已经处理了 J_fixed!
        Residual = self.get_J_net() - self.get_dissipation() - self.get_R_net()

        dz = Residual @ grad_H  # (B, 2d, 1)
        dz = dz.squeeze(-1)  # (B, 2d)

        dz = torch.clamp(dz, -MAX_DERIV, MAX_DERIV) * dt
        dq, dp = dz[:, :, :self.d], dz[:, :, self.d:]
        return dq, dp


# ── 独立函数：Leapfrog 积分器和 rollout ──

def leapfrog_step(
    q: torch.Tensor,
    p: torch.Tensor,
    energy_net,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Störmer-Verlet 辛积分, O(dt²)"""
    MAX_DERIV = 50.0  # 放宽物理限速
    grad_V_q = energy_net.grad_V(q)
    grad_V_q = torch.clamp(grad_V_q, -MAX_DERIV, MAX_DERIV)
    p_half = p - 0.5 * dt * grad_V_q

    grad_K_ph = energy_net.grad_K(p_half)
    grad_K_ph = torch.clamp(grad_K_ph, -MAX_DERIV, MAX_DERIV)
    q_new = q + dt * grad_K_ph

    grad_V_qn = energy_net.grad_V(q_new)
    grad_V_qn = torch.clamp(grad_V_qn, -MAX_DERIV, MAX_DERIV)
    p_new = p_half - 0.5 * dt * grad_V_qn
    return q_new, p_new


def ph_rollout(
    q0: torch.Tensor,
    p0: torch.Tensor,
    energy_net,
    dynamics: PHDynamics,
    n_steps: int,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PH 多步 rollout: Operator Splitting 版本。

    修复: 保守 Leapfrog 和耗散修正分离，不互相污染。
      Step 1: 纯保守 Leapfrog → (q*, p*)
      Step 2: 耗散/J_net/R_net 修正 → dq, dp (单独积分)

    这保证了:
      - 保守步的辛性质不被动量/耗散破坏
      - 耗散项在"固定能量"下单独作用
      - 避免标准 Leapfrog + 耗散直接混合导致的失稳

    Returns:
        q_seq: (B, n_steps+1, d_latent)
        p_seq: (B, n_steps+1, d_latent)
    """
    q_seq, p_seq = [q0], [p0]
    q, p = q0, p0

    for _ in range(n_steps):
        # ── Step 1: 纯保守 Leapfrog (保持辛结构) ──
        q_star, p_star = leapfrog_step(q, p, energy_net, dt)

        # ── Step 2: 耗散 + 残差修正 (在 (q*, p*) 上单独积分) ──
        dq, dp = dynamics(q_star, p_star, energy_net, dt)

        q = q_star + dq
        p = p_star + dp

        q_seq.append(q)
        p_seq.append(p)

    return torch.stack(q_seq, dim=1), torch.stack(p_seq, dim=1)
