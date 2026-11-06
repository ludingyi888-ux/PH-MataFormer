"""
PH-MetaFormer: 完整模型

三模块串行:
  1. SpatioTemporalMetaFormerEncoder   — 时空编码, 输出 [q₀, p₀] + skips
  2. PHDynamics + Leapfrog rollout     — 隐空间 PH 时间演化
  3. MetaFormerDecoder                  — 对称解码, FiLM 条件注入

Loss: L_pixel (MSE + SSIM) + L_energy_drift + L_symp + L_vic
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metaformer_encoder import SpatioTemporalMetaFormerEncoder
from .decoder import MetaFormerDecoder
from .energy_net import SeparableEnergy
from .ph_dynamics import PHDynamics, ph_rollout


class PHMetaFormer(nn.Module):
    """
    PH-MetaFormer: 时空 MetaFormer 编码 + PH 动力学演化 + MetaFormer 解码

    用途: FY-4A 卫星云图 4 帧 → 4 帧预测
    """
    def __init__(
        self,
        in_channels: int = 4,
        T_in: int = 4,
        T_out: int = 4,
        H: int = 256,
        W: int = 640,
        dims: tuple = (80, 160, 320, 480),
        depths: tuple = (3, 3, 9, 3),
        d_latent: int = 24,
        dt: float = 0.5,
        use_learned_residuals: bool = False,
        n_heads: int = 8,
        dropout: float = 0.1,
        ablation_mode: str = "",  # "" | "no_ph" | "no_pde" | "no_timedecay"
    ):
        super().__init__()
        self.T_in = T_in
        self.T_out = T_out
        self.H = H
        self.W = W
        self.d_latent = d_latent
        self.dt = dt
        self.ablation_mode = ablation_mode

        # ── 1. Encoder ──
        self.encoder = SpatioTemporalMetaFormerEncoder(
            in_channels=in_channels, T_in=T_in,
            H=H, W=W, dims=dims, depths=depths,
            n_heads=n_heads, dropout=dropout,
            d_latent=d_latent,
        )

        # ── 2. PH 组件 ──
        self.energy_net = SeparableEnergy(d_latent=d_latent)
        self.dynamics = PHDynamics(
            d_latent=d_latent,
            use_learned_residuals=use_learned_residuals,
        )

        # ── 3. Decoder ──
        self.decoder = MetaFormerDecoder(
            enc_dims=dims,
            depths=depths,
            d_latent=d_latent,
            out_channels=in_channels,
            out_frames=1,
            n_heads=n_heads,
            dropout=dropout,
            ablation_mode=ablation_mode,
        )

        # 编码器最底层的空间分辨率
        self.H4, self.W4 = H // 32, W // 32  # 4 次 2× 下采样

    def forward(
        self,
        x: torch.Tensor,
        compute_energy: bool = False,
    ) -> dict:
        """
        Args:
            x: (B, T_in, C, H, W) 输入云图序列
            compute_energy: 是否返回能量值 (训练时)

        Returns:
            dict with:
              preds:   (B, T_out, C, H, W) 预测帧
              q_seq:   (B, T_out+1, d_latent) PH 轨迹
              p_seq:   (B, T_out+1, d_latent)
              H_seq:   (B, T_out+1) 能量值 (if compute_energy)
        """
        B = x.shape[0]

        # ── 1. 编码 → [q₀, p₀] + skips ──
        q0, p0, skips = self.encoder(x)

        # ── 2. PH 时间演化 (消融: no_ph 跳过) ──
        if self.ablation_mode == "no_ph":
            # 消融: 恒等复制, 无物理演化
            N4 = q0.shape[1]
            q_seq = q0.unsqueeze(1).expand(B, self.T_out+1, N4, self.d_latent).contiguous()
            p_seq = p0.unsqueeze(1).expand(B, self.T_out+1, N4, self.d_latent).contiguous()
        else:
            q_seq, p_seq = ph_rollout(
                q0, p0, self.energy_net, self.dynamics,
                n_steps=self.T_out, dt=self.dt,
            )

        # ── 3. 逐帧解码 ──
        preds = self.decoder(skips, q_seq, p_seq, self.T_out, (self.H4, self.W4))

        out = {'preds': preds, 'q_seq': q_seq, 'p_seq': p_seq}

        if compute_energy:
            H_seq = self.energy_net(q_seq.reshape(-1, self.d_latent),
                                    p_seq.reshape(-1, self.d_latent))
            # 空间平均, 恢复 (B, T_out+1)
            H_seq = H_seq.reshape(B, self.T_out + 1, -1).mean(dim=-1)
            out['H_seq'] = H_seq

        return out

    # ── Loss 函数 ──

    def compute_loss(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        q_seq: torch.Tensor,
        p_seq: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> dict:
        """Loss: MSE + 课程退火的 GDL/L1/PH 约束"""
        losses = {}

        # ── 1. 基础图像 Loss ──
        losses['pixel'] = F.mse_loss(preds, target)

        # 计算 GDL 和 L1 (权重在退火策略中控制)
        dx_p = preds[..., 1:] - preds[..., :-1]
        dy_p = preds[..., 1:, :] - preds[..., :-1, :]
        dx_t = target[..., 1:] - target[..., :-1]
        dy_t = target[..., 1:, :] - target[..., :-1, :]
        diff_x = (dx_p - dx_t).float(); diff_y = (dy_p - dy_t).float()
        eps = 1e-6
        losses['gdl'] = torch.mean(torch.sqrt(diff_x**2 + eps**2) - eps) + \
                         torch.mean(torch.sqrt(diff_y**2 + eps**2) - eps)
        losses['l1'] = F.l1_loss(preds, target)

        # ── 2. 物理潜空间 Loss ──
        q_flat = q_seq.reshape(-1, self.d_latent)
        var_q = torch.var(q_flat, dim=0, unbiased=False)
        std_q = torch.sqrt(var_q + 1e-6).mean()
        p_flat = p_seq.reshape(-1, self.d_latent)
        var_p = torch.var(p_flat, dim=0, unbiased=False)
        std_p = torch.sqrt(var_p + 1e-6).mean()
        # vic: 连续斥力+吸引力 — 无硬阈值安全区，永远有梯度
        repulsive_q = 0.1 / (std_q + 1e-4)
        repulsive_p = 0.1 / (std_p + 1e-4)
        attractive_q = 0.01 * (std_q ** 2)
        attractive_p = 0.01 * (std_p ** 2)
        losses['vic'] = repulsive_q + repulsive_p + attractive_q + attractive_p

        # canonical: 余弦夹角 + 归一化强度 + 最小范数 — 三重防平凡解
        dq_dt = (q_seq[:, 1:] - q_seq[:, :-1]) / self.dt
        p_mid = p_seq[:, :-1]
        expected_dq_dt = self.energy_net.grad_K(p_mid)
        # 1. 方向一致 (余弦相似度)
        cos_sim = F.cosine_similarity(
            dq_dt.reshape(-1, self.d_latent),
            expected_dq_dt.reshape(-1, self.d_latent), dim=-1
        )
        loss_angle = 1.0 - cos_sim.mean()
        # 2. 强度比例一致
        norm_dq = torch.norm(dq_dt, dim=-1) + 1e-6
        norm_exp = torch.norm(expected_dq_dt, dim=-1) + 1e-6
        loss_norm = F.smooth_l1_loss(norm_dq / norm_exp, torch.ones_like(norm_dq))
        # 3. 禁止双双归零 + 马赫数限速
        loss_min = F.relu(0.05 - norm_dq).mean() + F.relu(0.05 - norm_exp).mean()
        loss_maxspeed = F.relu(norm_dq - 15.0).mean() + F.relu(norm_exp - 15.0).mean()
        losses['canonical'] = loss_angle + 0.1 * loss_norm + 0.5 * loss_min + 0.01 * loss_maxspeed

        H_all = self.energy_net(
            q_seq.reshape(-1, self.d_latent),
            p_seq.reshape(-1, self.d_latent)
        )
        H_all = H_all.reshape(q_seq.shape[0], q_seq.shape[1], -1).mean(dim=-1)
        losses['energy_drift'] = F.mse_loss(H_all[:, 1:], H_all[:, :-1])

        # symp: J_net = L - L^T 代数是反对称的，永远是0，删除
        losses['symp'] = torch.tensor(0.0, device=q_seq.device)

        # ── 3.5. 潜空间一致性: Encoder(Decoder(q_T)) ≈ q_T ──
        # 防止 Decoder 学到的映射偏离 Encoder 的语义
        last_pred = preds[:, -1:]  # (B, 1, C, H, W) 最后一帧预测
        fake_seq = last_pred.repeat(1, self.T_in, 1, 1, 1)  # 复制为4帧序列
        q_re, p_re, _ = self.encoder(fake_seq)
        losses['consistency'] = F.mse_loss(q_re, q_seq[:, -1]) + F.mse_loss(p_re, p_seq[:, -1])

        # ── 4. PH 能量引导的抗模糊 Loss ──
        # 核心: 高 PH 能量 → 大气活跃 → 预测帧应有更多高频细节
        # 只惩罚"预测比真值模糊"的方向, 避免过度锐化
        H_frame = H_all[:, 1:]  # (B, T_out)
        H_min = H_frame.min(dim=1, keepdim=True)[0]
        H_max = H_frame.max(dim=1, keepdim=True)[0]
        H_norm = (H_frame - H_min) / (H_max - H_min + 1e-6)  # (B, T_out) in [0,1]
        H_norm = H_norm.detach()  # 只做权重, 不回传梯度到 energy_net

        # 每帧梯度幅值 (空间梯度均值 = 锐度代理)
        grad_pred_x = (preds[..., 1:] - preds[..., :-1]).abs().mean(dim=(-1, -2, -3))
        grad_pred_y = (preds[..., :, 1:] - preds[..., :, :-1]).abs().mean(dim=(-1, -2, -3))
        grad_pred = grad_pred_x + grad_pred_y  # (B, T_out)
        grad_targ_x = (target[..., 1:] - target[..., :-1]).abs().mean(dim=(-1, -2, -3))
        grad_targ_y = (target[..., :, 1:] - target[..., :, :-1]).abs().mean(dim=(-1, -2, -3))
        grad_targ = grad_targ_x + grad_targ_y

        # 模糊惩罚: 仅当 pred 锐度 < 真值锐度 时激活
        grad_ratio = grad_pred / (grad_targ + 1e-6)  # 1.0=匹配, <1=偏模糊
        blur_mask = F.relu(1.0 - grad_ratio)  # 只惩罚模糊, 不惩罚过度锐化
        # 能量加权 + 时域加权: 后期帧 2x
        t_w = torch.linspace(1.0, 2.0, self.T_out, device=preds.device).view(1, -1)
        losses['sharpness'] = (t_w * H_norm * blur_mask).mean()

        # ── 5. 课程退火: GDL/L1 从0缓慢爬升 ──
        progress = min(step / max(total_steps * 0.5, 1), 1.0)
        warmup_progress = min(step / max(total_steps * 0.2, 1), 1.0)

        weight_l1 = warmup_progress * 0.3
        weight_gdl = warmup_progress * 0.5
        weight_vic = 1e-3 + progress * 0.1
        weight_canon = 0.0 if self.ablation_mode == "no_pde" else (1e-3 + progress * 0.1)
        weight_drift = 0.0 if self.ablation_mode == "no_pde" else (warmup_progress * progress * 0.1)
        weight_cons = warmup_progress * progress * 0.05  # 潜空间闭环, 后期激活
        weight_sharp = warmup_progress * progress * 0.1  # 能量引导锐度, 后期激活

        total = (
            losses['pixel'] +
            weight_l1 * losses['l1'] +
            weight_gdl * losses['gdl'] +
            weight_vic * losses['vic'] +
            weight_canon * losses['canonical'] +
            weight_drift * losses['energy_drift'] +
            weight_cons * losses['consistency'] +
            weight_sharp * losses['sharpness']
        )

        losses['total'] = total
        losses['ph_weight'] = max(weight_vic, weight_canon, weight_drift)

        return losses
