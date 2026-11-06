"""
ConvLSTM 基准训练脚本

与 PH-MetaFormer 使用相同数据和配置，公平对比
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

from config import PHMetaFormerConfig
from models.convlstm import ConvLSTMBaseline

# 复用 FY4ADataset
from train import FY4ADataset


def train(cfg: PHMetaFormerConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_ds = FY4ADataset(cfg, mode='train')
    val_ds = FY4ADataset(cfg, mode='val')
    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, drop_last=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, cfg.batch_size, shuffle=False, drop_last=True,
                            num_workers=0, pin_memory=True)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    model = ConvLSTMBaseline(
        in_channels=cfg.in_channels, T_in=cfg.T_in, T_out=cfg.T_out,
        H=cfg.H, W=cfg.W,
        hidden_dims=(128, 128, 128, 128),
    ).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
    scaler = GradScaler('cuda')

    save_dir = 'checkpoints_convlstm'
    os.makedirs(save_dir, exist_ok=True)

    start_epoch, best_val_loss = 1, float('inf')

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        t_start = time.time()
        total_loss = 0.0
        n_batches = 0
        nan_count = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # GPU resize if needed
            if train_ds.need_resize:
                B = x.shape[0]
                x = F.interpolate(x.flatten(0,1), size=(cfg.H, cfg.W),
                    mode='bilinear', align_corners=False).contiguous().view(B, cfg.T_in, -1, cfg.H, cfg.W)
                y = F.interpolate(y.flatten(0,1), size=(cfg.H, cfg.W),
                    mode='bilinear', align_corners=False).contiguous().view(B, cfg.T_out, -1, cfg.H, cfg.W)

            with autocast('cuda'):
                out = model(x)
                loss = F.mse_loss(out['preds'], y)

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                if nan_count <= 3:
                    print(f"  [NaN] E{epoch} batch {batch_idx}, skip")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            n_batches += 1

            if batch_idx % 40 == 0:
                print(f"E{epoch:3d} [{batch_idx:4d}] loss={loss.item():.6f}", flush=True)

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t_start

        nan_ratio = nan_count / max(len(train_loader), 1)
        if nan_ratio > 0.3:
            print(f"NaN ratio {nan_ratio:.1%}, stopping")
            break

        print(f"\nEpoch {epoch:3d} | loss={avg_loss:.6f} | lr={scheduler.get_last_lr()[0]:.2e} | NaN:{nan_count}/{len(train_loader)} | {elapsed:.0f}s")

        # Validation
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            val_mse = 0.0
            with torch.no_grad(), autocast('cuda'):
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    if train_ds.need_resize:
                        Bv = x.shape[0]
                        x = F.interpolate(x.flatten(0,1), size=(cfg.H, cfg.W),
                            mode='bilinear', align_corners=False).contiguous().view(Bv, cfg.T_in, -1, cfg.H, cfg.W)
                        y = F.interpolate(y.flatten(0,1), size=(cfg.H, cfg.W),
                            mode='bilinear', align_corners=False).contiguous().view(Bv, cfg.T_out, -1, cfg.H, cfg.W)
                    out = model(x)
                    val_mse += F.mse_loss(out['preds'], y).item()
            val_mse /= len(val_loader)
            print(f"  Val MSE: {val_mse:.6f}")

            if val_mse < best_val_loss:
                best_val_loss = val_mse
                torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))
                print(f"  -> Saved best (MSE={val_mse:.6f})")

        if epoch % cfg.save_interval == 0:
            ckpt = {
                'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                'best_val_loss': best_val_loss,
            }
            torch.save(ckpt, os.path.join(save_dir, f'ckpt_{epoch:03d}.pt'))

    print(f"\nDone. Best val MSE: {best_val_loss:.6f}")


if __name__ == '__main__':
    cfg = PHMetaFormerConfig()
    cfg.checkpoint_dir = 'checkpoints_convlstm'
    # Use same config but adjust checkpoint dir
    train(cfg)
