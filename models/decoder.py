"""
PH-MetaFormer: MetaFormer 风格解码器

对称于编码器的 4 阶段设计:
  Stage 4' → Stage 3' → Stage 2' → Stage 1'
  每阶段: 上采样 + SpatioTemporalBlock + skip connection

PH 状态 [q_t, p_t] 通过 FiLM 注入到每层解码器来条件化帧生成。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from .metaformer_encoder import SpatioTemporalBlock, StarReLU, ResScale


class Upsample(nn.Module):
    """空间 2× 上采样 (interpolate+conv, 无棋盘格)"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, padding=1, bias=False)

    def forward(self, X: torch.Tensor, H: int, W: int) -> tuple:
        B, T, N, D = X.shape
        X_img = X.reshape(B * T, H, W, D).permute(0, 3, 1, 2)
        X_img = F.interpolate(X_img, scale_factor=2.0, mode='bilinear', align_corners=False)
        X_img = self.conv(X_img)
        _, C_out, H_out, W_out = X_img.shape
        X_out = X_img.reshape(B * T, C_out, -1).transpose(1, 2)
        return X_out.reshape(B, T, -1, C_out), H_out, W_out


class FiLMModulate(nn.Module):
    """
    用 PH 状态 [q, p] 对特征做 FiLM 调制。
    gamma, beta = MLP([q, p])
    输出 = X * gamma + beta
    """
    def __init__(self, ph_dim: int, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * ph_dim, feat_dim * 2, bias=False),
            StarReLU(),
            nn.Linear(feat_dim * 2, feat_dim * 2, bias=False),
        )

    def forward(self, X: torch.Tensor, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        B, T, N, D = X.shape
        # 空间平均作为全局调制信号
        q_g = q.mean(dim=1) if q.dim() == 3 else q
        p_g = p.mean(dim=1) if p.dim() == 3 else p
        phase = torch.cat([q_g, p_g], dim=-1)
        gamma_beta = self.net(phase)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = torch.tanh(gamma) * 2.0
        beta = torch.tanh(beta)
        return X * gamma.unsqueeze(1).unsqueeze(1) + beta.unsqueeze(1).unsqueeze(1)


class SkipGate(nn.Module):
    """
    Skip 特征物理门控。

    问题: skip 特征携带强历史静态纹理，可能导致解码器惰性学习，
    忽略 FiLM 的物理控制信号。

    修复: 在用 skip 之前，用物理状态 [q,p] 生成一个门控信号 g ∈ [0,1]，
    对 skip 做调制: skip' = skip ⊙ g。这样解码器不能无条件依赖 skip，
    必须通过物理状态"请求"历史的纹理信息。
    """
    def __init__(self, ph_dim: int, skip_dim: int):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(2 * ph_dim, skip_dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        q_g = q.mean(dim=1) if q.dim() == 3 else q
        p_g = p.mean(dim=1) if p.dim() == 3 else p
        phase = torch.cat([q_g, p_g], dim=-1)
        gate = self.gate_net(phase)  # (B, D_skip)
        return skip * gate.unsqueeze(1).unsqueeze(1)


class DecoderStage(nn.Module):
    """
    解码器的一个 stage:
      [Upsample] → FiLM 注入 → L 个 SpatioTemporalBlock → [Gated Skip 拼接]
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        skip_dim: int,
        depth: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
        d_latent: int = 24,
        use_temporal: bool = True,
        has_upsample: bool = True,
    ):
        super().__init__()
        self.has_upsample = has_upsample
        self.has_skip = (skip_dim > 0)

        if has_upsample:
            self.upsample = Upsample(in_dim, out_dim)
        else:
            self.project = nn.Linear(in_dim, out_dim, bias=False)

        self.film = FiLMModulate(d_latent, out_dim)

        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(out_dim, n_heads=n_heads, dropout=dropout,
                               use_temporal=use_temporal)
            for _ in range(depth)
        ])

        if self.has_skip:
            self.skip_gate = SkipGate(d_latent, skip_dim)
            self.skip_fuse = nn.Sequential(
                nn.LayerNorm(out_dim + skip_dim, elementwise_affine=False),
                nn.Linear(out_dim + skip_dim, out_dim, bias=False),
            )

    def forward(self, X, skip, q, p, H_in, W_in):
        B, T, N, D = X.shape

        if self.has_upsample:
            X, H_out, W_out = self.upsample(X, H_in, W_in)
        else:
            X = X.reshape(B * T * N, D)
            X = self.project(X).reshape(B, T, N, -1)
            H_out, W_out = H_in, W_in

        X = self.film(X, q, p)

        for blk in self.blocks:
            if self.training and X.requires_grad:
                def _fw(b, h, w):
                    def _f(x_in):
                        return b(x_in, h, w)
                    return _f
                X = cp.checkpoint(_fw(blk, H_out, W_out), X, use_reentrant=False)
            else:
                X = blk(X, H_out, W_out)

        if self.has_skip and skip is not None:
            # 物理门控: 解码器不能无条件使用 skip
            skip = self.skip_gate(skip, q, p)
            X = torch.cat([X, skip], dim=-1)
            B_s, T_s, N_s, D_merged = X.shape
            X = X.reshape(B_s * T_s * N_s, D_merged)
            X = self.skip_fuse(X)
            X = X.reshape(B_s, T_s, N_s, -1)

        return X, H_out, W_out


class MetaFormerDecoder(nn.Module):
    """
    MetaFormer 风格解码器。

    对称编码器的 4 个 stage, 从最粗到最细逐层上采样。
    每层接收 PH 状态 [q_t, p_t] 做 FiLM 条件注入。
    最终输出预测的未来帧。
    """
    def __init__(
        self,
        enc_dims: tuple = (80, 160, 320, 480),  # 编码器 D1→D2→D3→D4 (正序)
        depths: tuple = (3, 9, 3, 3),
        d_latent: int = 24,
        out_channels: int = 4,
        out_frames: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        ablation_mode: str = "",
    ):
        super().__init__()
        self.out_channels = out_channels
        self.out_frames = out_frames
        self.ablation_mode = ablation_mode

        # 解码器维度: D4→D3→D2→D1 (倒序)
        dec_dims = tuple(reversed(enc_dims))  # (D4, D3, D2, D1)

        # 各 stage 配置
        # Stage 0: 从 PH 投影 → D4, 无 skip, 无上采样
        # Stage i>0: D_i-1 → D_i, skip=D_i, 有上采样
        n_stages = len(dec_dims)
        stage_in_dims  = [dec_dims[0]] + list(dec_dims[:-1])
        stage_out_dims = list(dec_dims)
        stage_skip_dims = [0] + list(dec_dims[1:])
        stage_depths   = list(reversed(depths))

        self.stages = nn.ModuleList()
        for i in range(n_stages):
            has_upsample = (i > 0)
            use_temporal = (i < 2)
            self.stages.append(
                DecoderStage(stage_in_dims[i], stage_out_dims[i],
                            stage_skip_dims[i], stage_depths[i],
                            n_heads=n_heads, dropout=dropout,
                            d_latent=d_latent, use_temporal=use_temporal,
                            has_upsample=has_upsample)
            )

        # 最终输出投影: D1 → out_channels × out_frames, + 最终上采样
        self.final_dim = dec_dims[-1]
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.final_dim, elementwise_affine=False),
            nn.Linear(self.final_dim, out_channels * out_frames, bias=False),
        )
        # 从 H/4×W/4 上采样回 H×W (interpolate→conv, 无棋盘格)
        self.final_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)

        # 初始投影: PH 状态 → D4 维度, 广播到 H4×W4 个 token
        self.initial_dim = dec_dims[0]  # D4
        self.initial_proj = nn.Sequential(
            nn.Linear(2 * d_latent, self.initial_dim, bias=False),
            StarReLU(),
        )

    def forward(self, skips, q_seq, p_seq, T_out, initial_HW):
        """
        时空递归解码: 每步继承上一步的隐状态, 防长时衰减
        """
        B = q_seq.shape[0]
        H_cur_init, W_cur_init = initial_HW
        N4 = H_cur_init * W_cur_init

        preds = []
        # 每层 stage 的隐状态, 在时间步之间传递
        hidden_states = {i: None for i in range(len(self.stages))}

        for t in range(T_out):
            q_t = q_seq[:, t + 1]
            p_t = p_seq[:, t + 1]

            phase = torch.cat([q_t, p_t], dim=-1)
            head_feat = self.initial_proj(phase)
            X = head_feat.unsqueeze(1)
            H_cur, W_cur = H_cur_init, W_cur_init

            for i, stage in enumerate(self.stages):
                skip_i = None
                if i > 0:
                    raw_skip = skips[-(i + 1)][:, -1:]
                    if self.ablation_mode == "no_timedecay":
                        # 消融: 无时间衰减, 无递归, 全量 skip = static U-Net
                        skip_i = raw_skip
                    else:
                        decay = max(0.3, 1.0 - t * 0.15)
                        if hidden_states[i] is not None:
                            beta = min(0.7, t * 0.2)
                            skip_i = (1 - beta) * (raw_skip * decay) + beta * hidden_states[i]
                        else:
                            skip_i = raw_skip * decay

                X, H_cur, W_cur = stage(X, skip_i, q_t, p_t, H_cur, W_cur)
                hidden_states[i] = X.detach()

            B_s, T_s, N_s, D_s = X.shape
            X_flat = X.reshape(B_s * T_s * N_s, D_s)
            pixels = self.output_proj(X_flat)
            pixels = pixels.reshape(B_s, 1, self.out_channels, H_cur, W_cur)
            pixels = pixels.squeeze(1)
            pixels = F.interpolate(pixels, scale_factor=4.0, mode='bilinear', align_corners=False)
            pixels = self.final_conv(pixels)
            preds.append(pixels)

        return torch.stack(preds, dim=1)
