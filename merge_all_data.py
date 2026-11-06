"""
合并 V3-V6 数据集 + 降采样到 128×320 (cv2.INTER_AREA)

生成:
  fy4a_ir108_all128.dat    — 统一时序 memmap
  metadata_all128.npz       — 时间戳等元数据
  split_all128.npz          — train/val 索引
  shape_all128.npy          — 形状信息
"""
import os, sys, time
import numpy as np
import cv2

DATA_DIR = r"D:\curelsummer\week3\data\processed"
VERSIONS = ['v3', 'v4', 'v5', 'v6']
TARGET_H, TARGET_W = 128, 320
OUT_NAME = "fy4a_ir108_all128"
PREFIX = "all128"  # 用于 metadata/split/shape 的短前缀


def load_version(ver):
    """加载一个版本的所有数据，无 split 文件则自动 80/20 划分"""
    dat_path = os.path.join(DATA_DIR, f"fy4a_ir108_{ver}.dat")
    meta = np.load(os.path.join(DATA_DIR, f"metadata_{ver}.npz"))
    shape_info = np.load(os.path.join(DATA_DIR, f"shape_{ver}.npy"))
    data = np.memmap(dat_path, dtype='float32', mode='r', shape=tuple(shape_info))
    N = data.shape[0]

    split_path = os.path.join(DATA_DIR, f"split_{ver}.npz")
    if os.path.exists(split_path):
        split = np.load(split_path)
        train_idx = split['train_idx'].astype(np.int64)
        val_idx = split['val_idx'].astype(np.int64)
    else:
        rng = np.random.default_rng(42)
        perm = rng.permutation(N)
        n_train = int(N * 0.8)
        train_idx = np.sort(perm[:n_train]).astype(np.int64)
        val_idx = np.sort(perm[n_train:]).astype(np.int64)
        print(f"(auto-split: {len(train_idx)}/{len(val_idx)})", end=' ', flush=True)
    return data, meta, train_idx, val_idx


def main():
    all_data = []
    all_ts = []
    train_offset = 0
    val_offset = 0
    all_train_idx = []
    all_val_idx = []

    for ver in VERSIONS:
        print(f"Loading {ver}...", end=' ', flush=True)
        data, meta, t_idx, v_idx = load_version(ver)
        N = data.shape[0]
        print(f"{N} frames, shape={data.shape}")

        # 逐帧降采样
        downsampled = np.zeros((N, 1, TARGET_H, TARGET_W), dtype='float32')
        for i in range(N):
            downsampled[i, 0] = cv2.resize(
                data[i, 0], (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA
            )
        all_data.append(downsampled)
        all_ts.extend(meta['timestamps'].tolist())

        all_train_idx.extend((t_idx + train_offset).tolist())
        all_val_idx.extend((v_idx + val_offset).tolist())
        train_offset += N
        val_offset += N
        del data, downsampled

    # 合并
    print(f"\nMerging {len(all_ts)} frames total...")
    full_data = np.concatenate(all_data, axis=0).astype('float32')
    del all_data

    # 保存 .dat
    out_dat = os.path.join(DATA_DIR, f"{OUT_NAME}.dat")
    shape = full_data.shape
    with open(out_dat, 'wb') as f:
        f.seek(int(np.prod(shape)) * 4 - 1)
        f.write(b'\x00')
    mm = np.memmap(out_dat, dtype='float32', mode='r+', shape=shape)
    chunk = 200
    for i in range(0, len(full_data), chunk):
        end = min(i + chunk, len(full_data))
        mm[i:end] = full_data[i:end]
        print(f"  Writing {end}/{len(full_data)}", flush=True)
    mm.flush()
    print(f"  Saved: {out_dat} ({os.path.getsize(out_dat)/1024/1024:.0f} MB)")

    # 保存 metadata
    np.savez(
        os.path.join(DATA_DIR, f"metadata_{OUT_NAME}.npz"),
        timestamps=np.array(all_ts),
        channels=['IR10.8'],
        channel_names=['IR10.8'],
        wavelengths=[10.8],
        shape=shape,
        bt_min=180.0,
        bt_max=320.0,
    )
    print(f"  Metadata: {len(all_ts)} timestamps, bt=[180,320]K")

    # 保存 split
    np.savez(
        os.path.join(DATA_DIR, f"split_{OUT_NAME}.npz"),
        train_idx=np.array(all_train_idx, dtype=np.int64),
        val_idx=np.array(all_val_idx, dtype=np.int64),
    )
    print(f"  Split: {len(all_train_idx)} train / {len(all_val_idx)} val")

    # 保存 shape
    np.save(os.path.join(DATA_DIR, f"shape_{OUT_NAME}.npy"), np.array(shape))
    print(f"  Shape: {shape}")

    # 统计
    print(f"\n=== Done ===")
    print(f"  Total: {len(all_ts)} frames")
    print(f"  Train: {len(all_train_idx)}, Val: {len(all_val_idx)}")
    print(f"  Size: {os.path.getsize(out_dat)/1024/1024:.0f} MB")


if __name__ == '__main__':
    main()
