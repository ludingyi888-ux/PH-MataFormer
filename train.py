"""
PH-MetaFormer 训练脚本 V4 — 终极防崩溃版

修复:
  1. GDL Loss 的 abs(0) 奇点导致梯度 NaN (改用 pseudo-Huber)
  2. AMP 未使用 GradScaler 导致 AdamW 除零崩溃
"""
import os, sys, time, argparse
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from datetime import datetime

from config import PHMetaFormerConfig
from models.ph_metaformer import PHMetaFormer


class FY4ADataset(Dataset):
    """FY-4A 时序数据集 (连续索引, 已知数据已按时间排序, 支持 stride 采样)"""

    def __init__(self, cfg: PHMetaFormerConfig, mode: str = 'train'):
        shape_path = getattr(cfg, 'shape_path', r"D:\curelsummer\week3\data\processed\shape_v3.npy")
        data_shape = tuple(np.load(shape_path))
        self.data = np.memmap(cfg.data_path, dtype='float32', mode='r', shape=data_shape)
        n_samples, self.n_ch, self.h_data, self.w_data = data_shape
        self.T_in, self.T_out = cfg.T_in, cfg.T_out
        self.stride = getattr(cfg, 'stride', 1)
        self.target_H, self.target_W = cfg.H, cfg.W
        self.need_resize = (self.h_data != self.target_H) or (self.w_data != self.target_W)

        # 加载 split
        split_path = cfg.split_path
        if not os.path.exists(split_path):
            split_path = split_path.replace('_v3', '')
        split = np.load(split_path)
        self.indices = split['train_idx'] if mode == 'train' else split['val_idx']

        # 过滤太靠近末尾的索引
        n_consec = (self.T_in + self.T_out - 1) * self.stride + 1
        self.indices = self.indices[self.indices <= len(self.data) - n_consec]

        # 检测数据是否已归一化
        sample = self.data[self.indices[:min(100, len(self.indices))]]
        sr = np.nanmax(sample) - np.nanmin(sample)
        self.is_normalized = (0.5 < sr < 2.0)

        if self.is_normalized:
            self.mean = torch.zeros(1, self.n_ch, 1, 1)
            self.std = torch.ones(1, self.n_ch, 1, 1)
        else:
            self.mean = torch.from_numpy(np.nanmean(sample, axis=(0,2,3)).astype(np.float32)).view(1,-1,1,1)
            self.std = torch.from_numpy(np.maximum(np.nanstd(sample, axis=(0,2,3)),1e-6).astype(np.float32)).view(1,-1,1,1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = int(self.indices[idx])
        n_consec = (self.T_in + self.T_out - 1) * self.stride + 1
        if start + n_consec > len(self.data):
            start = len(self.data) - n_consec
        # 读取连续帧, 再按 stride 跳采样
        frames_np = np.array(self.data[start:start + n_consec:self.stride])
        frames = torch.from_numpy(frames_np)
        frames = torch.nan_to_num(frames, nan=0.0)
        if not self.is_normalized:
            frames = (frames - self.mean) / self.std
        return frames[:self.T_in], frames[self.T_in:]


def train(cfg: PHMetaFormerConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    train_ds = FY4ADataset(cfg, mode='train')
    val_ds = FY4ADataset(cfg, mode='val')
    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, drop_last=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, cfg.batch_size, shuffle=False, drop_last=True,
                            num_workers=0, pin_memory=True)
    print(f"Train: {len(train_ds)} windows, Val: {len(val_ds)} windows")

    model = PHMetaFormer(
        in_channels=cfg.in_channels, T_in=cfg.T_in, T_out=cfg.T_out,
        H=cfg.H, W=cfg.W, dims=cfg.dims, depths=cfg.depths,
        d_latent=cfg.d_latent, dt=cfg.dt,
        use_learned_residuals=cfg.use_learned_residuals,
        n_heads=cfg.n_heads, dropout=cfg.dropout,
        ablation_mode=getattr(cfg, 'ablation_mode', ''),
    ).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # 加载预训练权重 (低分辨率→全分辨率)
    pretrained = getattr(cfg, 'pretrained', '')
    if pretrained and os.path.exists(pretrained):
        print(f"Loading pretrained from {pretrained}")
        pt = torch.load(pretrained, map_location=device)
        if 'model' in pt:
            pt = pt['model']
        missing, unexpected = model.load_state_dict(pt, strict=False)
        if missing:
            print(f"  [Warn] Missing: {list(missing)[:3]}...")
        print("  Pretrained loaded.")

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)

    # [修复核心]: 必须实例化 GradScaler!
    scaler = GradScaler('cuda')

    total_steps = (len(train_loader) * cfg.epochs) // getattr(cfg, 'grad_accum_steps', 1)
    grad_accum = getattr(cfg, 'grad_accum_steps', 1)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # ── 断点续训: 加载最新 checkpoint ──
    start_epoch, best_val_loss, global_step = 1, float('inf'), 0
    latest_ckpt = None
    import glob
    ckpt_files = sorted(glob.glob(os.path.join(cfg.checkpoint_dir, 'ckpt_*.pt')))
    if ckpt_files:
        latest_ckpt = ckpt_files[-1]
        print(f"Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(ckpt['model'].keys())
        missing = model_keys - ckpt_keys
        if missing:
            print(f"  [Warn] New params (random init): {missing}")
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        if 'best_val_loss' in ckpt:
            best_val_loss = ckpt['best_val_loss']
        if 'global_step' in ckpt:
            global_step = ckpt['global_step']
        # 重建 scheduler 以适配可能变化的 T_max
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
        for _ in range(ckpt['epoch']):
            scheduler.step()
        print(f"  -> Start from epoch {start_epoch}, best_val={best_val_loss:.6f}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        epoch_losses = {}
        t_start = time.time()
        nan_count = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            global_step += 1

            # GPU resize (contiguous fix)
            if train_ds.need_resize:
                B = x.shape[0]
                x = F.interpolate(x.flatten(0,1), size=(cfg.H,cfg.W),
                    mode='bilinear', align_corners=False).contiguous().view(B, cfg.T_in, -1, cfg.H, cfg.W)
                y = F.interpolate(y.flatten(0,1), size=(cfg.H,cfg.W),
                    mode='bilinear', align_corners=False).contiguous().view(B, cfg.T_out, -1, cfg.H, cfg.W)

            with autocast('cuda'):
                out = model(x, compute_energy=True)
                losses = model.compute_loss(
                    out['preds'], y, out['q_seq'], out['p_seq'],
                    step=global_step, total_steps=total_steps,
                )
                loss = losses['total'] / grad_accum

            # NaN 保护: 跳过坏批次
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                print(f"  [!] NaN/Inf at E{epoch} batch {batch_idx}, skipping", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            for k, v in losses.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                epoch_losses[k] = epoch_losses.get(k, 0.0) + val

            if batch_idx % cfg.log_interval == 0:
                ph_w = losses['ph_weight']
                if isinstance(ph_w, torch.Tensor):
                    ph_w = ph_w.item()
                print(f"E{epoch:3d} [{batch_idx:4d}] "
                      f"pixel={losses['pixel'].item():.4f} "
                      f"gdl={losses['gdl'].item():.4f} "                      f"l1={losses['l1'].item():.4f} "                      f"total={losses['total'].item():.4f} "
                      f"ph_w={ph_w:.3f}", flush=True)

        scheduler.step()

        n_batches = len(train_loader)
        nan_ratio = nan_count / n_batches
        avg = {k: v / n_batches for k, v in epoch_losses.items()}
        elapsed = time.time() - t_start
        print(f"\n=== Epoch {epoch:3d} ({elapsed:.1f}s) ===")
        for k, v in avg.items():
            print(f"  {k}: {v:.6f}")
        print(f"  lr: {scheduler.get_last_lr()[0]:.2e}")
        print(f"  NaN: {nan_count}/{n_batches} ({nan_ratio:.1%})\n")

        # 自动回滚: NaN 率超过 30% → 恢复上一个 checkpoint 并停训
        if nan_ratio > 0.3:
            print(f"\n  *** NaN ratio {nan_ratio:.1%} > 30%! Auto-rollback ***")
            last_ckpt = sorted([f for f in os.listdir(cfg.checkpoint_dir) if f.startswith('ckpt_')])
            if last_ckpt:
                ckpt_path = os.path.join(cfg.checkpoint_dir, last_ckpt[-1])
                ckpt = torch.load(ckpt_path, map_location=device)
                # 恢复 model state
                if 'model' in ckpt:
                    model.load_state_dict(ckpt['model'], strict=False)
                else:
                    model.load_state_dict(ckpt, strict=False)
                print(f"  -> Restored from {last_ckpt[-1]}")
                torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, 'best_rollback.pt'))
                print(f"  -> Saved as best_rollback.pt")
            print(f"  *** Training stopped at epoch {epoch} ***")
            break

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            val_mse = 0.0
            with torch.no_grad(), autocast('cuda'):
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    if train_ds.need_resize:
                        Bv = x.shape[0]
                        x = F.interpolate(x.flatten(0,1), size=(cfg.H,cfg.W),
                            mode='bilinear', align_corners=False).contiguous().view(Bv, cfg.T_in, -1, cfg.H, cfg.W)
                        y = F.interpolate(y.flatten(0,1), size=(cfg.H,cfg.W),
                            mode='bilinear', align_corners=False).contiguous().view(Bv, cfg.T_out, -1, cfg.H, cfg.W)
                    out = model(x)
                    val_mse += F.mse_loss(out['preds'], y).item()
            val_mse /= len(val_loader)
            print(f"  Val MSE: {val_mse:.6f}")

            if val_mse < best_val_loss:
                best_val_loss = val_mse
                torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, 'best.pt'))
                print(f"  -> Saved best model (MSE={val_mse:.6f})")

        if epoch % cfg.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'global_step': global_step,
            }, os.path.join(cfg.checkpoint_dir, f'ckpt_{epoch:03d}.pt'))

    print(f"\nDone. Best val MSE: {best_val_loss:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    if args.full:
        from config import PHMetaFormerConfigFull; cfg = PHMetaFormerConfigFull()
    else:
        cfg = PHMetaFormerConfig()
    train(cfg)
