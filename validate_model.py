"""
PH-MetaFormer: 模型验证脚本

检查:
  1. 模型能否正常实例化
  2. Forward pass shape 是否正确
  3. Loss 计算是否正常
  4. 参数量统计
"""
import torch
import sys
sys.path.insert(0, '.')

from models.ph_metaformer import PHMetaFormer


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # 用小尺寸测试
    cfg = dict(
        in_channels=4, T_in=4, T_out=4,
        H=128, W=320,
        dims=(64, 128, 256, 384),
        depths=(2, 2, 4, 2),
        d_latent=24,
        dt=0.5,
        use_learned_residuals=False,
        n_heads=8,
        dropout=0.1,
    )

    model = PHMetaFormer(**cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {n_params:.2f}M")
    print(f"  Encoder:  {sum(p.numel() for p in model.encoder.parameters())/1e6:.2f}M")
    print(f"  Energy:   {sum(p.numel() for p in model.energy_net.parameters())/1e3:.1f}K")
    print(f"  Dynamics: {sum(p.numel() for p in model.dynamics.parameters())/1e3:.1f}K")
    print(f"  Decoder:  {sum(p.numel() for p in model.decoder.parameters())/1e6:.2f}M")

    # ── Test 1: Forward pass ──
    print("\n── Test 1: Forward pass ──")
    B = 2
    x = torch.randn(B, cfg['T_in'], cfg['in_channels'], cfg['H'], cfg['W']).to(device)
    out = model(x, compute_energy=True)

    print(f"  Input:  {x.shape}")
    print(f"  Preds:  {out['preds'].shape}     (expect: B={B}, T_out={cfg['T_out']}, C={cfg['in_channels']})")
    print(f"  q_seq:  {out['q_seq'].shape}     (expect: B={B}, T_out+1={cfg['T_out']+1}, d={cfg['d_latent']})")
    print(f"  p_seq:  {out['p_seq'].shape}")
    print(f"  H_seq:  {out['H_seq'].shape}")

    assert out['preds'].shape == (B, cfg['T_out'], cfg['in_channels'], cfg['H'], cfg['W']), \
        f"Pred shape mismatch: {out['preds'].shape}"
    assert out['q_seq'].shape == (B, cfg['T_out'] + 1, cfg['d_latent']), \
        f"q_seq shape mismatch: {out['q_seq'].shape}"
    print("  ✓ All shapes correct (full resolution)")

    # ── Test 2: Loss computation ──
    print("\n── Test 2: Loss ──")
    target = torch.randn(B, cfg['T_out'], cfg['in_channels'], cfg['H'], cfg['W']).to(device)
    losses = model.compute_loss(
        out['preds'], target, out['q_seq'], out['p_seq'],
        step=100, total_steps=1000,
    )
    for k, v in losses.items():
        print(f"  {k}: {v.item():.6f}" if isinstance(v, torch.Tensor) else f"  {k}: {v:.4f}")
    assert torch.isfinite(losses['total']), "Loss is NaN/Inf!"
    print("  ✓ Loss finite")

    # ── Test 3: Backward pass ──
    print("\n── Test 3: Backward ──")
    losses['total'].backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Total grad norm: {grad_norm:.2f}")
    assert grad_norm > 0, "No gradients!"
    print("  ✓ Gradients flow")

    # ── Test 4: Energy conservation (no dissipation) ──
    print("\n── Test 4: Energy drift ──")
    H_seq = out['H_seq']
    drift = (H_seq[:, -1] - H_seq[:, 0]).abs().mean().item()
    print(f"  Energy drift over {cfg['T_out']} steps: {drift:.6f}")
    # 允许小幅漂移 (耗散项存在)
    assert drift < H_seq.abs().mean().item() * 2, f"Energy exploded! drift={drift:.4f}"
    print("  ✓ Energy stable")

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)


if __name__ == '__main__':
    main()
