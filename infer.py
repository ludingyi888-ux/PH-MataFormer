"""
PH-MetaFormer 单通道 IR10.8 推理可视化
"""
import numpy as np
import torch, torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse

# ── 气象红外增强色阶 ──
ir_colors = [
    (0.0, 0.0, 0.0),       # 黑 (暖地表)
    (0.0, 0.0, 0.3),       # 深蓝
    (0.0, 0.3, 0.6),       # 蓝
    (0.0, 0.6, 0.8),       # 青
    (0.0, 0.8, 0.4),       # 绿
    (0.8, 0.8, 0.0),       # 黄
    (1.0, 0.5, 0.0),       # 橙
    (1.0, 0.0, 0.0),       # 红
    (1.0, 0.7, 0.7),       # 浅红
    (1.0, 1.0, 1.0),       # 白 (冷云顶)
]
ir_cmap = LinearSegmentedColormap.from_list('IR_enhanced', ir_colors, N=256)

parser = argparse.ArgumentParser()
parser.add_argument('--full', action='store_true')
parser.add_argument('--ckpt', type=str, default='checkpoints/best.pt')
args = parser.parse_args()

if args.full:
    from config import PHMetaFormerConfigFull; cfg = PHMetaFormerConfigFull()
else:
    from config import PHMetaFormerConfig; cfg = PHMetaFormerConfig()

from models.ph_metaformer import PHMetaFormer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Loading {args.ckpt} ...")
model = PHMetaFormer(
    in_channels=cfg.in_channels, T_in=cfg.T_in, T_out=cfg.T_out,
    H=cfg.H, W=cfg.W, dims=cfg.dims, depths=cfg.depths,
    d_latent=cfg.d_latent, dt=cfg.dt,
    n_heads=cfg.n_heads, dropout=cfg.dropout,
).to(device)
model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
model.eval()
print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

from train import FY4ADataset
val_ds = FY4ADataset(cfg, mode='val')
print(f"Val samples: {len(val_ds)}")

# ── 元数据: 获取 BT 反归一化参数 ──
meta_path = cfg.meta_path
import os
meta = np.load(meta_path)
bt_min, bt_max = float(meta['bt_min']), float(meta['bt_max'])
print(f"BT range: {bt_min:.0f} - {bt_max:.0f} K")

def denorm(x):
    """min-max 反归一化到 BT (K): [0,1] → [bt_min, bt_max]"""
    return x * (bt_max - bt_min) + bt_min

# ── 在多个验证样本上推理 ──
sample_indices = [0, len(val_ds)//3, len(val_ds)//2, 2*len(val_ds)//3, len(val_ds)-1]
all_mae = []

for si, idx in enumerate(sample_indices):
    x, y_true = val_ds[idx]
    x = F.interpolate(x, size=(cfg.H, cfg.W), mode='bilinear', align_corners=False)
    y_true = F.interpolate(y_true, size=(cfg.H, cfg.W), mode='bilinear', align_corners=False)
    x_in = x.unsqueeze(0).to(device)

    with torch.no_grad():
        y_pred = model(x_in)['preds'][0].cpu()

    x_np = x.numpy()
    y_pred_np = y_pred.numpy()
    y_true_np = y_true.numpy()

    # IR10.8 BT 范围
    vmin, vmax = 190, 310  # 标准亮温范围
    T_out = cfg.T_out
    stride = getattr(cfg, 'stride', 1)
    ncols = T_out
    fig_w = max(16, ncols * 3.5)
    fig_h = 16 if ncols <= 4 else 12

    fig, axes = plt.subplots(4, ncols, figsize=(fig_w, fig_h), facecolor='black')
    if ncols == 1:
        axes = axes.reshape(4, 1)

    # Row 0: 输入
    stride = getattr(cfg, 'stride', 1)
    for t in range(cfg.T_in):
        axes[0, t].imshow(denorm(x_np[t, 0]), cmap=ir_cmap, vmin=vmin, vmax=vmax)
        axes[0, t].set_title(f'Input T{(t-cfg.T_in+1)*stride:+d}h', fontsize=9, color='white')
        axes[0, t].axis('off')
    # 多余列留空
    for t in range(cfg.T_in, ncols):
        axes[0, t].axis('off')

    # Row 1: 预测
    for t in range(T_out):
        axes[1, t].imshow(denorm(y_pred_np[t, 0]), cmap=ir_cmap, vmin=vmin, vmax=vmax)
        axes[1, t].set_title(f'Pred T+{(t+1)*stride}h', fontsize=9, color='white')
        axes[1, t].axis('off')

    # Row 2: 真值
    for t in range(T_out):
        axes[2, t].imshow(denorm(y_true_np[t, 0]), cmap=ir_cmap, vmin=vmin, vmax=vmax)
        axes[2, t].set_title(f'Truth T+{(t+1)*stride}h', fontsize=9, color='white')
        axes[2, t].axis('off')

    # Row 3: 误差图 |pred - truth|
    for t in range(T_out):
        err = np.abs(denorm(y_pred_np[t, 0]) - denorm(y_true_np[t, 0]))
        im = axes[3, t].imshow(err, cmap='hot', vmin=0, vmax=15)
        axes[3, t].set_title(f'|Err| T+{(t+1)*stride}h', fontsize=9, color='white')
        axes[3, t].axis('off')

    # 标注
    axes[0, 0].set_ylabel('INPUT', fontsize=12, weight='bold', color='cyan')
    axes[1, 0].set_ylabel('PRED', fontsize=12, weight='bold', color='lime')
    axes[2, 0].set_ylabel('TRUTH', fontsize=12, weight='bold', color='yellow')
    axes[3, 0].set_ylabel('ERROR', fontsize=12, weight='bold', color='red')

    mae = np.abs(denorm(y_pred_np[:, 0]) - denorm(y_true_np[:, 0])).mean()
    all_mae.append(mae)
    plt.suptitle(f'PH-MetaFormer IR10.8 — Sample {idx} | MAE={mae:.2f}K',
                 fontsize=14, weight='bold', color='white')
    plt.tight_layout()
    fname = f'checkpoints/pred_sample_{si}.png'
    plt.savefig(fname, dpi=120, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Sample {idx}: MAE={mae:.2f}K -> {fname}")

# ── 汇总 ──
print(f"\n{'='*50}")
print(f"Overall MAE (5 samples): {np.mean(all_mae):.2f} ± {np.std(all_mae):.2f} K")
print(f"Best: {np.min(all_mae):.2f}K, Worst: {np.max(all_mae):.2f}K")

# ── 全验证集统计 ──
print("\nComputing full validation metrics...")
val_mse = 0.0
val_mae_total = 0.0
n_batches = 0
from torch.utils.data import DataLoader
val_loader = DataLoader(val_ds, cfg.batch_size, shuffle=False, drop_last=False, num_workers=0)

with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        B = x.shape[0]
        x = F.interpolate(x.flatten(0, 1), size=(cfg.H, cfg.W),
            mode='bilinear', align_corners=False).contiguous().view(B, cfg.T_in, -1, cfg.H, cfg.W)
        y = F.interpolate(y.flatten(0, 1), size=(cfg.H, cfg.W),
            mode='bilinear', align_corners=False).contiguous().view(B, cfg.T_out, -1, cfg.H, cfg.W)
        out = model(x)
        pred = out['preds']
        val_mse += F.mse_loss(pred, y).item() * B
        val_mae_total += F.l1_loss(pred, y).item() * B
        n_batches += 1

n_val = len(val_ds) if val_loader.drop_last else len(val_ds)
val_mse /= n_val
val_mae_total /= n_val

# 转换到物理单位
bt_range = bt_max - bt_min
val_rmse_k = np.sqrt(val_mse) * bt_range
val_mae_k = val_mae_total * bt_range

print(f"Val MSE (norm): {val_mse:.6f}")
print(f"Val RMSE: {val_rmse_k:.2f} K")
print(f"Val MAE:  {val_mae_k:.2f} K")
print(f"\nDone!")
