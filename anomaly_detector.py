"""
PH-MetaFormer 隐空间轨迹异常预警系统 V2

原理：
  正常天气系统的隐空间轨迹 [q(t), p(t)] 在相空间中围绕一个"吸引子"区域分布。
  异常天气（快速增强的台风、突发对流等）的轨迹会偏离正常吸引子。
  
核心思路（对齐课题要求）：
  1. 收集正常样本的潜轨迹 → 构建"正常吸引子"
  2. 计算新样本轨迹与吸引子的距离 → 预警信号
  3. 像素 MAE 仅作为辅助验证，主导信号来自潜空间
  
预警层级：
  - 潜轨迹距离（主导）: 轨迹偏离正常吸引子的马氏距离
  - 物理一致性（辅助）: canonical 方程偏离、能量漂移
  - 像素验证（参考）: 预测 MAE 用于确认
"""
import torch
import numpy as np
import sys, os, json, argparse
from scipy.spatial.distance import mahalanobis

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PHMetaFormerConfig
from models.ph_metaformer import PHMetaFormer
from train import FY4ADataset


class TrajectoryAnomalyDetector:
    def __init__(self, ckpt_path='checkpoints/best.pt', n_components=32):
        self.cfg = PHMetaFormerConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_components = n_components

        self.model = PHMetaFormer(
            in_channels=self.cfg.in_channels, T_in=self.cfg.T_in, T_out=self.cfg.T_out,
            H=self.cfg.H, W=self.cfg.W, dims=self.cfg.dims, depths=self.cfg.depths,
            d_latent=self.cfg.d_latent, dt=self.cfg.dt,
            use_learned_residuals=self.cfg.use_learned_residuals,
            n_heads=self.cfg.n_heads, dropout=self.cfg.dropout,
        ).to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device), strict=False)
        self.model.eval()

        self.meta = np.load(self.cfg.meta_path)
        self.bt_range = float(self.meta['bt_max']) - float(self.meta['bt_min'])

        self.fitted = False
        self.stats = {}

    # ── 轨迹提取 ──
    def extract_trajectory(self, x):
        """从输入提取完整的 (q, p, H) 轨迹"""
        x_in = x.unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.autocast('cuda'):
            out = self.model(x_in, compute_energy=True)

        q_seq = out['q_seq'][0].detach()  # (T_out+1, N4, d_latent)
        p_seq = out['p_seq'][0].detach()
        H_seq = out['H_seq'][0].detach()
        preds = out['preds'][0].detach()

        # 全局潜轨迹：空间平均 q, p
        T = q_seq.shape[0]
        q_global = q_seq.mean(dim=1).cpu().numpy()  # (T, d_latent)
        p_global = p_seq.mean(dim=1).cpu().numpy()

        # 物理诊断量
        dq = (q_seq[1:] - q_seq[:-1]) / self.cfg.dt  # (T_out, N4, d)
        p_mid = p_seq[:-1]
        with torch.autocast('cuda'):
            gK = self.model.energy_net.grad_K(p_mid.reshape(-1, 256)).detach().reshape(
                self.cfg.T_out, -1, self.cfg.d_latent)

        cos_sim = torch.nn.functional.cosine_similarity(
            dq.reshape(-1, 256), gK.reshape(-1, 256), dim=-1)
        speed = dq.norm(dim=-1).mean(dim=1)  # (T_out,)

        return {
            'q_global': q_global,           # (T, 256)  广义坐标轨迹
            'p_global': p_global,           # (T, 256)  动量轨迹
            'H_seq': H_seq.cpu().numpy(),   # (T,)      能量轨迹
            'cos_canon': cos_sim.mean().item(),    # 正则一致性
            'cos_min': cos_sim.min().item(),
            'speed_mean': speed.mean().item(),     # 演化速度
            'speed_max': speed.max().item(),
            'preds': preds.cpu().numpy(),   # 用于像素验证
        }

    # ── 轨迹扁平化 → 固定维特征向量 ──
    def _trajectory_to_features(self, traj):
        """将轨迹转为固定维特征向量"""
        feats = []
        # q 轨迹的均值 + 首尾差分 (捕捉趋势)
        feats.append(traj['q_global'].mean(axis=0))     # 256
        feats.append(traj['q_global'][-1] - traj['q_global'][0])  # 256
        # p 轨迹
        feats.append(traj['p_global'].mean(axis=0))     # 256
        feats.append(traj['p_global'][-1] - traj['p_global'][0])  # 256
        # 标量指标
        feats.append(np.array([
            traj['H_seq'].mean(), traj['H_seq'].std(),
            traj['cos_canon'], traj['cos_min'],
            traj['speed_mean'], traj['speed_max'],
        ]))
        return np.concatenate([f.flatten() for f in feats])  # ~1030 维

    # ── 拟合正常吸引子 ──
    def fit(self, n_samples=250):
        """在验证集上建立隐轨迹的正常统计分布"""
        print(f"Fitting trajectory-based anomaly detector on {n_samples} samples...")
        val_ds = FY4ADataset(self.cfg, mode='val')
        all_feat = []

        indices = np.linspace(0, len(val_ds)-1, min(n_samples, len(val_ds)), dtype=int)
        for idx in indices:
            x, _ = val_ds[idx]
            traj = self.extract_trajectory(x)
            all_feat.append(self._trajectory_to_features(traj))

        X = np.array(all_feat)
        self.stats['mean'] = X.mean(axis=0)
        self.stats['std'] = X.std(axis=0) + 1e-8

        # PCA 降维 → 正常轨迹的"吸引子"
        X_centered = (X - self.stats['mean']) / self.stats['std']
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.stats['V'] = Vt[:self.n_components]      # (k, D) 投影矩阵
        self.stats['S'] = S[:self.n_components]         # 奇异值

        # 计算所有训练样本到吸引子的马氏距离
        X_proj = X_centered @ self.stats['V'].T        # (N, k)
        X_proj_scaled = X_proj / (S[:self.n_components] / np.sqrt(len(X)) + 1e-8)
        distances = np.linalg.norm(X_proj_scaled, axis=1)

        self.stats['dist_mean'] = float(distances.mean())
        self.stats['dist_std'] = float(distances.std())
        self.stats['dist_p95'] = float(np.percentile(distances, 95))
        self.stats['dist_p99'] = float(np.percentile(distances, 99))
        self.stats['n_samples'] = len(X)
        self.stats['feat_dim'] = X.shape[1]

        self.fitted = True
        print(f"  Feature dim: {X.shape[1]} → PCA {self.n_components}")
        print(f"  Attractor radius (p95): {self.stats['dist_p95']:.3f}")
        print(f"  Attractor radius (p99): {self.stats['dist_p99']:.3f}")
        return self.stats

    # ── 轨迹异常检测 ──
    def predict(self, x, y_true=None):
        """对单样本的潜轨迹进行异常评估"""
        if not self.fitted:
            raise RuntimeError("Must call fit() first")

        traj = self.extract_trajectory(x)
        feat = self._trajectory_to_features(traj)
        feat_centered = (feat - self.stats['mean']) / self.stats['std']

        # 投影到吸引子空间
        proj = feat_centered @ self.stats['V'].T
        proj_scaled = proj / (self.stats['S'] / np.sqrt(self.stats['n_samples']) + 1e-8)
        traj_distance = float(np.linalg.norm(proj_scaled))

        # 多维度异常分数
        z_traj = (traj_distance - self.stats['dist_mean']) / (self.stats['dist_std'] + 1e-8)

        # 物理一致性检查
        phys_score = 0.0
        if traj['cos_canon'] < 0.99:
            phys_score += (0.99 - traj['cos_canon']) * 100
        if traj['H_seq'].std() > 0.5:
            phys_score += traj['H_seq'].std() * 5

        # 综合判定（主导：轨迹距离）
        is_anomaly = z_traj > 2.0 or phys_score > 1.0

        # 像素验证
        pixel_mae = None
        if y_true is not None:
            y_in = y_true.unsqueeze(0).to(self.device)
            pixel_mae = (traj['preds'] - y_in.cpu().numpy()).__abs__().mean() * self.bt_range

        # 认定级别
        if z_traj > 3.0:
            level = 'Anomaly'
        elif z_traj > 2.0:
            level = 'Caution'
        else:
            level = 'Normal'

        return {
            'level': level,
            'traj_distance': traj_distance,
            'z_traj': float(z_traj),
            'phys_score': float(phys_score),
            'cos_canon': traj['cos_canon'],
            'H_std': traj['H_seq'].std(),
            'speed_mean': traj['speed_mean'],
            'pixel_mae': float(pixel_mae) if pixel_mae else None,
        }

    def save(self, path='checkpoints/anomaly_detector.npz'):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        np.savez(path, **{k: v for k, v in self.stats.items() if isinstance(v, np.ndarray)})
        # Also save scalars
        scalar_stats = {k: v for k, v in self.stats.items() if not isinstance(v, np.ndarray)}
        with open(path.replace('.npz', '_meta.json'), 'w') as f:
            json.dump(scalar_stats, f, indent=2)
        print(f"Detector saved to {path}")

    def load(self, path='checkpoints/anomaly_detector.npz'):
        self.stats = dict(np.load(path, allow_pickle=True))
        for k in list(self.stats.keys()):
            self.stats[k] = self.stats[k].item() if self.stats[k].ndim == 0 else self.stats[k]
        meta_path = path.replace('.npz', '_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.stats.update(json.load(f))
        self.fitted = True
        print(f"Detector loaded from {path}")


def run_demo():
    print("=" * 60)
    print("PH-MetaFormer Latent Trajectory Anomaly Detector V2")
    print("=" * 60)

    detector = TrajectoryAnomalyDetector(ckpt_path='checkpoints/best.pt')
    detector.fit(n_samples=250)

    val_ds = FY4ADataset(detector.cfg, mode='val')
    results = []
    n_scan = min(500, len(val_ds))

    print(f"\nScanning {n_scan} samples for latent trajectory anomalies...")
    for idx in range(n_scan):
        x, y = val_ds[idx]
        r = detector.predict(x, y)
        results.append((idx, r))

    normals = [r for _, r in results if r['level'] == 'Normal']
    cautions = [r for _, r in results if r['level'] == 'Caution']
    anomalies = [r for _, r in results if r['level'] == 'Anomaly']

    traj_dists = [r['traj_distance'] for _, r in results]
    print(f"\n{'='*60}")
    print(f"Trajectory distance stats: mean={np.mean(traj_dists):.3f}, std={np.std(traj_dists):.3f}")
    print(f"Results: Normal={len(normals)}, Caution={len(cautions)}, Anomaly={len(anomalies)}")
    print(f"{'='*60}")

    if anomalies:
        print("\n=== Anomaly Samples (latent trajectory deviation) ===")
        sorted_anom = sorted([(i, r) for i, r in results if r['level'] == 'Anomaly'],
                             key=lambda x: x[1]['z_traj'], reverse=True)
        for idx, r in sorted_anom[:5]:
            print(f"\n  Sample {idx}: z_traj={r['z_traj']:.2f}, dist={r['traj_distance']:.3f}")
            print(f"  Physics: cos={r['cos_canon']:.4f}, H_std={r['H_std']:.4f}, speed={r['speed_mean']:.2f}")
            if r['pixel_mae']:
                print(f"  Pixel MAE: {r['pixel_mae']:.2f}K (reference only)")

    if cautions:
        print(f"\n=== Caution Samples ===")
        sorted_cau = sorted([(i, r) for i, r in results if r['level'] == 'Caution'],
                            key=lambda x: x[1]['z_traj'], reverse=True)[:3]
        for idx, r in sorted_cau:
            print(f"  Sample {idx}: z_traj={r['z_traj']:.2f}, MAE={r.get('pixel_mae','N/A')}")

    detector.save()
    return detector, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run full demo')
    parser.add_argument('--fit', action='store_true')
    parser.add_argument('--scan', action='store_true')
    parser.add_argument('--ckpt', type=str, default='checkpoints/best.pt')
    args = parser.parse_args()

    if not (args.fit or args.scan):
        args.demo = True

    if args.demo:
        run_demo()
    else:
        detector = TrajectoryAnomalyDetector(ckpt_path=args.ckpt)
        if args.fit:
            detector.fit(n_samples=250)
            detector.save()
        if args.scan:
            if not detector.fitted:
                detector.load()
            val_ds = FY4ADataset(detector.cfg, mode='val')
            for idx in range(min(100, len(val_ds))):
                x, y = val_ds[idx]
                r = detector.predict(x, y)
                if r['level'] != 'Normal':
                    print(f"  Sample {idx}: {r['level']} (z_traj={r['z_traj']:.2f}, MAE={r.get('pixel_mae','N/A')})")

