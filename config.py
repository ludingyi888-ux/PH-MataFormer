"""
PH-MetaFormer 训练配置
"""
from dataclasses import dataclass, field


@dataclass
class PHMetaFormerConfig:
    # ── 数据 (合并 V3-V6: 3226帧 IR10.8, 128×320, min-max 归一化) ──
    data_path: str = r"D:\curelsummer\week3\data\processed\fy4a_ir108_all128.dat"
    split_path: str = r"D:\curelsummer\week3\data\processed\split_all128.npz"
    meta_path: str = r"D:\curelsummer\week3\data\processed\metadata_all128.npz"
    shape_path: str = r"D:\curelsummer\week3\data\processed\shape_all128.npy"

    # ── 输入尺寸 ──
    in_channels: int = 1       # IR10.8 单通道
    T_in: int = 4
    T_out: int = 4             # 4 帧预测
    stride: int = 2            # 采样步长 2h → 输入跨度 6h, 预测跨度 8h
    H: int = 128
    W: int = 320

    # ── 模型架构 ──
    dims: tuple = (64, 128, 256, 384)   # 相比原设计等比缩减
    depths: tuple = (2, 2, 6, 2)
    d_latent: int = 256
    dt: float = 0.5
    use_learned_residuals: bool = True   # J_net/R_net 可学习残差
    n_heads: int = 8
    dropout: float = 0.1

    # ── 训练 ──
    batch_size: int = 8        # T_out=4, stride=2 显存宽裕
    lr: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 1e-5
    epochs: int = 150          # stride=2, 2h 步长 8h 窗口
    grad_clip: float = 5.0

    # ── 保存 ──
    checkpoint_dir: str = "checkpoints_s2"  # stride=2 专用目录
    log_interval: int = 20
    save_interval: int = 10
    pretrained: str = "checkpoints/best.pt"  # 从 300-epoch T_out=4 stride=1 模型迁移

    # ── 消融实验 ──
    ablation_mode: str = ""  # "" | "no_ph" | "no_pde" | "no_timedecay"


# 全分辨率配置 (256×640) — 用 128×320 权重微调
@dataclass
class PHMetaFormerConfigFull(PHMetaFormerConfig):
    # 全分辨率数据
    data_path: str = r"D:\curelsummer\week3\data\processed\fy4a_ir108_all256.dat"
    split_path: str = r"D:\curelsummer\week3\data\processed\split_all256.npz"
    meta_path: str = r"D:\curelsummer\week3\data\processed\metadata_all256.npz"
    shape_path: str = r"D:\curelsummer\week3\data\processed\shape_all256.npy"
    H: int = 256
    W: int = 640
    batch_size: int = 1
    grad_accum_steps: int = 4
    lr: float = 1e-5          # 微调低学习率
    min_lr: float = 1e-6
    epochs: int = 50          # 微调 50 epoch
    pretrained: str = "checkpoints/best.pt"  # 128×320 预训练权重
