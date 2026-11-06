"""
PH-MetaFormer: 时空 MetaFormer 编码器

严格遵循 MetaFormer 论文 (Yu et al., TPAMI 2024) 的架构原则:
  - 4 阶段层级设计 (Conv k7s4 stem + Conv k3s2 下采样)
  - Block = SpatialMixer → TemporalMixer → ChannelMLP
  - StarReLU 激活, ResScale 缩放, 无偏置

SpatialMixer: 倒置可分离卷积 (ConvFormer 风格, k=7, exp=2)
TemporalMixer: 多头自注意力 (跨帧, 每 patch 独立)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


# ═══════════════════════════════════════════════════
# 激活函数 & 缩放
# ═══════════════════════════════════════════════════

class StarReLU(nn.Module):
    """StarReLU(x) = s * (ReLU(x))^2 + b, 比 GELU 少 71% FLOPs"""
    def __init__(self, dim: int = None):
        super().__init__()
        self.s = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.s * F.relu(x).pow(2) + self.b


class ResScale(nn.Module):
    """缩放残差分支: X' = λ_r ⊙ X + F(Norm(X)), λ 初始化为 1"""
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, residual, branch_out):
        return self.scale * residual + branch_out


# ═══════════════════════════════════════════════════
# Token Mixer: 空间 (ConvFormer 风格)
# ═══════════════════════════════════════════════════

class InvertedSeparableConv(nn.Module):
    """
    倒置可分离卷积 token mixer。
    同 ConvFormer: Conv_pw1 → DepthwiseConv → Conv_pw2
    配置: kernel=7, expansion=2
    """
    def __init__(self, dim: int, kernel_size: int = 7, expansion: int = 2):
        super().__init__()
        hidden = dim * expansion
        self.pw1 = nn.Conv2d(dim, hidden, 1, bias=False)
        self.dw  = nn.Conv2d(hidden, hidden, kernel_size,
                             padding=kernel_size // 2, groups=hidden, bias=False)
        self.pw2 = nn.Conv2d(hidden, dim, 1, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (B, N, D) → (B, D, H, W)
        B, N, D = x.shape
        x = x.transpose(1, 2).reshape(B, D, H, W)
        x = self.pw2(self.dw(F.relu(self.pw1(x))))
        return x.reshape(B, D, N).transpose(1, 2)  # → (B, N, D)


# ═══════════════════════════════════════════════════
# Token Mixer: 时间 (跨帧 MultiheadAttention)
# ═══════════════════════════════════════════════════

class TemporalAttention(nn.Module):
    """
    时间维度 token mixer。
    对每个空间 patch 独立: 在 T 帧之间做自注意力。
    """
    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, T, N, D)
        B, T, N, D = X.shape
        X_t = X.permute(0, 2, 1, 3).reshape(B * N, T, D)
        attn_out, _ = self.attn(X_t, X_t, X_t)
        return attn_out.reshape(B, N, T, D).permute(0, 2, 1, 3)  # → (B, T, N, D)


# ═══════════════════════════════════════════════════
# 时空 MetaFormer Block
# ═══════════════════════════════════════════════════

class SpatioTemporalBlock(nn.Module):
    """
    一个时空 MetaFormer block:

      X'  = X   + ResScale * SpatialMixer(Norm(X))
      X'' = X'  + ResScale * TemporalMixer(Norm(X'))
      X'''= X'' + ResScale * ChannelMLP(Norm(X''))

    其中 ChannelMLP = StarReLU(Norm(X'')·W1)·W2
    """
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_temporal: bool = True,
    ):
        super().__init__()
        self.use_temporal = use_temporal
        mlp_hidden = int(dim * mlp_ratio)

        # ── Norms ──
        self.norm_s = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_t = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_c = nn.LayerNorm(dim, elementwise_affine=False)

        # ── Token Mixers ──
        self.spatial_mixer = InvertedSeparableConv(dim, kernel_size=7)
        if use_temporal:
            self.temporal_mixer = TemporalAttention(dim, n_heads, dropout)

        # ── Channel MLP ──
        self.W1 = nn.Linear(dim, mlp_hidden, bias=False)
        self.W2 = nn.Linear(mlp_hidden, dim, bias=False)
        self.act = StarReLU()

        # ── ResScale (共享尺度, 或各分支独立) ──
        self.scale_s = ResScale(dim)
        self.scale_t = ResScale(dim)
        self.scale_c = ResScale(dim)

    def forward(self, X: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            X: (B, T, N, D) 时空 token
            H, W: 当前 stage 的空间分辨率

        Returns:
            X: (B, T, N, D)
        """
        B, T, N, D = X.shape

        # ── Step 1: 空间混合 (每帧内) ──
        X_flat = X.reshape(B * T, N, D)
        X_flat = self.scale_s(X_flat,
            self.spatial_mixer(self.norm_s(X_flat), H, W))
        X = X_flat.reshape(B, T, N, D)

        # ── Step 2: 时间混合 (每 patch 跨帧) ──
        if self.use_temporal:
            X = self.scale_t(X, self.temporal_mixer(self.norm_t(X)))

        # ── Step 3: 通道混合 ──
        X_flat = X.reshape(B * T * N, D)
        X_flat = self.scale_c(X_flat,
            self.W2(self.act(self.W1(self.norm_c(X_flat)))))
        return X_flat.reshape(B, T, N, D)


# ═══════════════════════════════════════════════════
# 下采样模块
# ═══════════════════════════════════════════════════

class Downsample(nn.Module):
    """空间下采样 (不改变时间维度)"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, 2, 1, bias=False)

    def forward(self, X: torch.Tensor, H: int, W: int) -> tuple:
        B, T, N, D = X.shape
        X_img = X.reshape(B * T, H, W, D).permute(0, 3, 1, 2)
        X_img = self.conv(X_img)
        _, C_out, H_out, W_out = X_img.shape
        X_out = X_img.reshape(B * T, C_out, -1).transpose(1, 2)
        return X_out.reshape(B, T, -1, C_out), H_out, W_out


# ═══════════════════════════════════════════════════
# 完整时空 MetaFormer 编码器
# ═══════════════════════════════════════════════════

class SpatioTemporalMetaFormerEncoder(nn.Module):
    """
    时空 MetaFormer 编码器。

    4 阶段层级架构, 各阶段先 2× 空间下采样再 L_i 个时空 block。
    输入 T 帧多通道云图, 输出多尺度特征 + 相空间初始条件 [q₀, p₀]。

    配置参考 MetaFormer 论文 Table II ConvFormer-S18, 等比缩放。
    """
    def __init__(
        self,
        in_channels: int = 4,
        T_in: int = 4,
        H: int = 256,
        W: int = 640,
        dims: tuple = (80, 160, 320, 480),    # D1-D4
        depths: tuple = (3, 3, 9, 3),          # L1-L4
        n_heads: int = 8,
        dropout: float = 0.1,
        d_latent: int = 24,                    # 相空间维度
    ):
        super().__init__()
        self.T_in = T_in
        self.d_latent = d_latent

        # ── Stem: Conv k7s4 每帧独立处理 ──
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], 7, 4, 3, bias=False),
            nn.LayerNorm(dims[0], elementwise_affine=False),
        )
        H1, W1 = H // 4, W // 4

        # ── 构建各 Stage ──
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.HW = [(H1, W1)]

        in_dim = dims[0]
        current_H, current_W = H1, W1

        for i, (dim, depth) in enumerate(zip(dims, depths)):
            # Stage blocks
            stage_blocks = nn.ModuleList([
                SpatioTemporalBlock(dim, n_heads=n_heads, dropout=dropout,
                                   use_temporal=(i < 3))  # Stage 4 可选不用 temporal
                for _ in range(depth)
            ])
            self.stages.append(stage_blocks)

            # 下采样 (除最后一层)
            if i < len(dims) - 1:
                self.downsamples.append(Downsample(dim, dims[i + 1]))
                current_H //= 2
                current_W //= 2
                self.HW.append((current_H, current_W))

        # ── 相空间投影头 (Tanh 限制范围, 防能量爆炸) ──
        self.phase_head = nn.Sequential(
            nn.LayerNorm(dims[-1], elementwise_affine=False),
            nn.Linear(dims[-1], 256, bias=False),
            StarReLU(),
            nn.Linear(256, 2 * d_latent, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, C, H, W) 输入云图序列

        Returns:
            q0, p0: 各 (B, d_latent) — 初始相空间状态
            skips:  list of (B, T, N_i, D_i) — 各 stage 输出特征 (给 decoder)
            H, W  :  各 stage 的空间分辨率
        """
        B, T, C, H, W = x.shape

        # ── Stem: 每帧独立卷积, 保持真实时间维度 ──
        x = x.reshape(B * T, C, H, W)
        x = self.stem[0](x)  # Conv2d: (B*T, D1, H1, W1)
        H_cur, W_cur = self.HW[0]
        x = x.permute(0, 2, 3, 1)  # (B*T, H1, W1, D1)
        x = self.stem[1](x)         # LayerNorm(D1)
        X = x.reshape(B, T, -1, x.shape[-1])  # (B, T, N1, D1) — 真实T维!

        skips = []

        for i, stage_blocks in enumerate(self.stages):
            for blk in stage_blocks:
                if self.training and X.requires_grad:
                    def _fw(b, h, w):
                        def _f(x_in):
                            return b(x_in, h, w)
                        return _f
                    X = cp.checkpoint(_fw(blk, H_cur, W_cur), X, use_reentrant=False)
                else:
                    X = blk(X, H_cur, W_cur)

            skips.append(X)

            # 下采样
            if i < len(self.downsamples):
                X, H_cur, W_cur = self.downsamples[i](X, H_cur, W_cur)

        # ── 空间场: q₀, p₀ 保留网格维度 (B, N4, d_latent) ──
        z_spatial = X[:, -1]              # (B, N4, D4) — 不取均值!
        qp = self.phase_head(z_spatial)   # (B, N4, 2 * d_latent)
        q0, p0 = qp.chunk(2, dim=-1)      # 各 (B, N4, d_latent)

        return q0, p0, skips

    @property
    def encoder_dim(self):
        """编码器最终输出维度"""
        return self.phase_head[-1].out_features // 2
