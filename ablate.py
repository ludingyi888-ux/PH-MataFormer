"""
PH-MetaFormer 消融实验

测试每个组件的贡献:
  1. 无 PH 动力学 (替换为恒等映射)
  2. 无递归解码器 (去除 hidden_states)
  3. 无 Skip 连接 (decay=0)
  4. 无 FiLM 调制
  5. 旧版时间衰减 (decay=max(0,1-t*0.3))
  6. 完整模型
"""
import numpy as np
import torch, torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import copy, time, argparse

from config import PHMetaFormerConfig
cfg = PHMetaFormerConfig()
from models.ph_metaformer import PHMetaFormer
from models.ph_dynamics import ph_rollout
from train import FY4ADataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 气象红外色阶 ──
ir_colors = [
    (0.0,0.0,0.0),(0.0,0.0,0.3),(0.0,0.3,0.6),(0.0,0.6,0.8),
    (0.0,0.8,0.4),(0.8,0.8,0.0),(1.0,0.5,0.0),(1.0,0.0,0.0),
    (1.0,0.7,0.7),(1.0,1.0,1.0),
]
ir_cmap = LinearSegmentedColormap.from_list('IR', ir_colors, N=256)

meta = np.load(cfg.meta_path)
bt_min, bt_max = float(meta['bt_min']), float(meta['bt_max'])

def denorm(x):
    return x * (bt_max - bt_min) + bt_min

# ── 消融变体 ──

def ablate_no_ph(model, x):
    """无 PH 动力学：q_seq/p_seq 直接恒等复制"""
    B = x.shape[0]
    T_out = model.T_out
    d = model.d_latent
    q0, p0, skips = model.encoder(x)
    N4 = q0.shape[1]
    # 恒等 rollout
    q_seq = q0.unsqueeze(1).expand(B, T_out+1, N4, d).contiguous()
    p_seq = p0.unsqueeze(1).expand(B, T_out+1, N4, d).contiguous()
    preds = model.decoder(skips, q_seq, p_seq, T_out, (model.H4, model.W4))
    return {'preds': preds}

def ablate_no_recurrent(model, x):
    """无递归解码器：disconnect hidden_states between timesteps"""
    q0, p0, skips = model.encoder(x)
    q_seq, p_seq = ph_rollout(q0, p0, model.energy_net, model.dynamics, model.T_out, model.dt)
    preds = _decoder_no_recurrent(model.decoder, skips, q_seq, p_seq, model.T_out, (model.H4, model.W4))
    return {'preds': preds}

def _decoder_no_recurrent(decoder, skips, q_seq, p_seq, T_out, initial_HW):
    """解码器: 无隐状态传递"""
    B = q_seq.shape[0]
    H_cur_init, W_cur_init = initial_HW
    preds = []
    for t in range(T_out):
        q_t = q_seq[:, t+1]; p_t = p_seq[:, t+1]
        phase = torch.cat([q_t, p_t], dim=-1)
        head_feat = decoder.initial_proj(phase)
        X = head_feat.unsqueeze(1)
        H_cur, W_cur = H_cur_init, W_cur_init
        for i, stage in enumerate(decoder.stages):
            skip_i = None
            if i > 0:
                raw_skip = skips[-(i+1)][:, -1:]
                decay = max(0.3, 1.0 - t*0.15)
                skip_i = raw_skip * decay
            X, H_cur, W_cur = stage(X, skip_i, q_t, p_t, H_cur, W_cur)
        B_s, T_s, N_s, D_s = X.shape
        X_flat = X.reshape(B_s*T_s*N_s, D_s)
        pixels = decoder.output_proj(X_flat)
        pixels = pixels.reshape(B_s, 1, decoder.out_channels, H_cur, W_cur)
        pixels = pixels.squeeze(1)
        pixels = F.interpolate(pixels, scale_factor=4.0, mode='bilinear', align_corners=False)
        pixels = decoder.final_conv(pixels)
        preds.append(pixels)
    return torch.stack(preds, dim=1)

def ablate_no_skip(model, x):
    """无 Skip 连接"""
    q0, p0, skips = model.encoder(x)
    q_seq, p_seq = ph_rollout(q0, p0, model.energy_net, model.dynamics, model.T_out, model.dt)
    preds = _decoder_no_skip(model.decoder, skips, q_seq, p_seq, model.T_out, (model.H4, model.W4))
    return {'preds': preds}

def _decoder_no_skip(decoder, skips, q_seq, p_seq, T_out, initial_HW):
    """解码器: 无 skip 连接"""
    B = q_seq.shape[0]
    H_cur_init, W_cur_init = initial_HW
    preds = []
    hidden_states = {i: None for i in range(len(decoder.stages))}
    for t in range(T_out):
        q_t = q_seq[:, t+1]; p_t = p_seq[:, t+1]
        phase = torch.cat([q_t, p_t], dim=-1)
        head_feat = decoder.initial_proj(phase)
        X = head_feat.unsqueeze(1)
        H_cur, W_cur = H_cur_init, W_cur_init
        for i, stage in enumerate(decoder.stages):
            # skip_i = None 始终
            skip_i = None
            X, H_cur, W_cur = stage(X, None, q_t, p_t, H_cur, W_cur)
            hidden_states[i] = X.detach()
        B_s, T_s, N_s, D_s = X.shape
        X_flat = X.reshape(B_s*T_s*N_s, D_s)
        pixels = decoder.output_proj(X_flat)
        pixels = pixels.reshape(B_s, 1, decoder.out_channels, H_cur, W_cur)
        pixels = pixels.squeeze(1)
        pixels = F.interpolate(pixels, scale_factor=4.0, mode='bilinear', align_corners=False)
        pixels = decoder.final_conv(pixels)
        preds.append(pixels)
    return torch.stack(preds, dim=1)

def ablate_no_film(model, x):
    """无 FiLM 调制"""
    q0, p0, skips = model.encoder(x)
    q_seq, p_seq = ph_rollout(q0, p0, model.energy_net, model.dynamics, model.T_out, model.dt)
    preds = _decoder_no_film(model.decoder, skips, q_seq, p_seq, model.T_out, (model.H4, model.W4))
    return {'preds': preds}

def _decoder_no_film(decoder, skips, q_seq, p_seq, T_out, initial_HW):
    """解码器: FiLM 调制 替换为恒等"""
    B = q_seq.shape[0]
    H_cur_init, W_cur_init = initial_HW
    preds = []
    hidden_states = {i: None for i in range(len(decoder.stages))}
    for t in range(T_out):
        q_t = q_seq[:, t+1]; p_t = p_seq[:, t+1]
        phase = torch.cat([q_t, p_t], dim=-1)
        head_feat = decoder.initial_proj(phase)
        X = head_feat.unsqueeze(1)
        H_cur, W_cur = H_cur_init, W_cur_init
        for i, stage in enumerate(decoder.stages):
            skip_i = None
            if i > 0:
                raw_skip = skips[-(i+1)][:, -1:]
                decay = max(0.3, 1.0 - t*0.15)
                if hidden_states[i] is not None:
                    beta = min(0.7, t*0.2)
                    skip_i = (1-beta)*(raw_skip*decay) + beta*hidden_states[i]
                else:
                    skip_i = raw_skip*decay
            # 跳过 FiLM: 直接用 stage 的 upsample+blocks, 不放 FiLM
            X = _stage_no_film(stage, X, skip_i, q_t, p_t, H_cur, W_cur)
            if hasattr(stage, 'has_upsample') and stage.has_upsample:
                H_cur, W_cur = H_cur*2, W_cur*2
            hidden_states[i] = X.detach()
        B_s, T_s, N_s, D_s = X.shape
        X_flat = X.reshape(B_s*T_s*N_s, D_s)
        pixels = decoder.output_proj(X_flat)
        pixels = pixels.reshape(B_s, 1, decoder.out_channels, H_cur, W_cur)
        pixels = pixels.squeeze(1)
        pixels = F.interpolate(pixels, scale_factor=4.0, mode='bilinear', align_corners=False)
        pixels = decoder.final_conv(pixels)
        preds.append(pixels)
    return torch.stack(preds, dim=1)

def _stage_no_film(stage, X, skip, q, p, H, W):
    """DecoderStage 无 FiLM"""
    if stage.has_upsample:
        X, H_out, W_out = stage.upsample(X, H, W)
    else:
        B, T, N, D = X.shape
        X = stage.project(X)
        H_out, W_out = H, W
    for blk in stage.blocks:
        X = blk(X, H_out, W_out)
    if skip is not None and stage.has_skip:
        if skip.shape[-1] != X.shape[-1]:
            skip = F.linear(skip, torch.eye(X.shape[-1], device=X.device)[:skip.shape[-1], :])
        X = X + skip
    return X

def ablate_old_decay(model, x):
    """旧版时间衰减 decay=max(0, 1-t*0.3)"""
    q0, p0, skips = model.encoder(x)
    q_seq, p_seq = ph_rollout(q0, p0, model.energy_net, model.dynamics, model.T_out, model.dt)
    preds = _decoder_old_decay(model.decoder, skips, q_seq, p_seq, model.T_out, (model.H4, model.W4))
    return {'preds': preds}

def _decoder_old_decay(decoder, skips, q_seq, p_seq, T_out, initial_HW):
    B = q_seq.shape[0]
    H_cur_init, W_cur_init = initial_HW
    preds = []
    for t in range(T_out):
        q_t = q_seq[:, t+1]; p_t = p_seq[:, t+1]
        phase = torch.cat([q_t, p_t], dim=-1)
        head_feat = decoder.initial_proj(phase)
        X = head_feat.unsqueeze(1)
        H_cur, W_cur = H_cur_init, W_cur_init
        for i, stage in enumerate(decoder.stages):
            skip_i = None
            if i > 0:
                raw_skip = skips[-(i+1)][:, -1:]
                decay = max(0.0, 1.0 - t*0.3)  # 旧版衰减
                skip_i = raw_skip * decay
            X, H_cur, W_cur = stage(X, skip_i, q_t, p_t, H_cur, W_cur)
        B_s, T_s, N_s, D_s = X.shape
        X_flat = X.reshape(B_s*T_s*N_s, D_s)
        pixels = decoder.output_proj(X_flat)
        pixels = pixels.reshape(B_s, 1, decoder.out_channels, H_cur, W_cur)
        pixels = pixels.squeeze(1)
        pixels = F.interpolate(pixels, scale_factor=4.0, mode='bilinear', align_corners=False)
        pixels = decoder.final_conv(pixels)
        preds.append(pixels)
    return torch.stack(preds, dim=1)


def evaluate(name, pred_fn, model, val_ds, n_samples=100):
    """评估消融变体"""
    model.eval()
    mae_total, mse_total = 0.0, 0.0
    # 分帧统计
    frame_mae = {i: 0.0 for i in range(4)}
    count = 0
    indices = list(range(0, len(val_ds), max(1, len(val_ds)//n_samples)))[:n_samples]
    t0 = time.time()
    with torch.no_grad(), torch.autocast('cuda'):
        for idx in indices:
            x, y_true = val_ds[idx]
            x, y_true = x.unsqueeze(0).to(device), y_true.unsqueeze(0).to(device)
            out = pred_fn(model, x)
            y_pred = out['preds']
            err = torch.abs(denorm(y_pred) - denorm(y_true)).cpu().numpy()
            mae_total += err.mean()
            mse_total += (err**2).mean()
            for f in range(4):
                frame_mae[f] += err[0, f].mean()
            count += 1
    elapsed = time.time() - t0
    results = {
        'MAE': mae_total / count,
        'RMSE': np.sqrt(mse_total / count),
        'T+5h': frame_mae[0] / count,
        'T+6h': frame_mae[1] / count,
        'T+7h': frame_mae[2] / count,
        'T+8h': frame_mae[3] / count,
        'time': elapsed,
    }
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='checkpoints/best.pt')
    parser.add_argument('--n', type=int, default=100, help='验证样本数')
    args = parser.parse_args()

    print(f"Loading {args.ckpt} ...")
    model = PHMetaFormer(
        in_channels=cfg.in_channels, T_in=cfg.T_in, T_out=cfg.T_out,
        H=cfg.H, W=cfg.W, dims=cfg.dims, depths=cfg.depths,
        d_latent=cfg.d_latent, dt=cfg.dt,
        n_heads=cfg.n_heads, dropout=cfg.dropout,
        use_learned_residuals=cfg.use_learned_residuals,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    model.eval()
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    val_ds = FY4ADataset(cfg, mode='val')
    print(f"Val samples: {len(val_ds)}")

    experiments = {
        'Full Model':           lambda m, x: m(x),
        'No PH Dynamics':       ablate_no_ph,
        'No Recurrent':         ablate_no_recurrent,
        'No Skip Connections':  ablate_no_skip,
        'No FiLM Modulate':     ablate_no_film,
        'Old Decay (t×0.3)':    ablate_old_decay,
    }

    print(f"\n{'='*70}")
    print(f"  Ablation Study ({args.n} samples)")
    print(f"{'='*70}")
    print(f"{'Config':<22s} {'MAE(K)':>8s} {'RMSE(K)':>8s} {'T+5h':>7s} {'T+6h':>7s} {'T+7h':>7s} {'T+8h':>7s}")
    print(f"{'-'*70}")

    results_all = {}
    for name, fn in experiments.items():
        r = evaluate(name, fn, model, val_ds, n_samples=args.n)
        results_all[name] = r
        print(f"{name:<22s} {r['MAE']:8.2f} {r['RMSE']:8.2f} "
              f"{r['T+5h']:7.2f} {r['T+6h']:7.2f} {r['T+7h']:7.2f} {r['T+8h']:7.2f}")

    # ── 相对影响 ──
    baseline = results_all['Full Model']['MAE']
    print(f"\n{'='*70}")
    print(f"  Impact Analysis (ΔMAE from Full Model: {baseline:.2f}K)")
    print(f"{'='*70}")
    for name, r in results_all.items():
        if name == 'Full Model':
            continue
        delta = r['MAE'] - baseline
        bar = '+' * max(1, int(abs(delta)*10)) if delta > 0 else ''
        print(f"  {name:<22s}: {delta:+.2f}K  {bar}")

    # ── 逐帧趋势 ──
    print(f"\n{'='*70}")
    print(f"  Per-Frame Degradation (T+5h → T+8h)")
    print(f"{'='*70}")
    for name, r in results_all.items():
        degradation = r['T+8h'] - r['T+5h']
        print(f"  {name:<22s}: Δ={degradation:+.2f}K "
              f"({r['T+5h']:.2f}→{r['T+6h']:.2f}→{r['T+7h']:.2f}→{r['T+8h']:.2f})")

    print(f"\nDone! ({sum(r['time'] for r in results_all.values()):.0f}s total)")
