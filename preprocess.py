"""
FY-4A IR10.8 预处理（参数化，合并原 preprocess_v3~v6）

方法: DN → CALChannel LUT → 亮温 → REGC 裁剪 → min-max 归一化 → INTER_AREA 降采样

用法:
  python preprocess.py --version v3 --batches batch1,batch2,batch3,batch4
  python preprocess.py --version v4 --batches batch3,batch4,batch_extra
  python preprocess.py --version v5 --batches batch5,batch6,batch7
  python preprocess.py --version v6 --batches batch8,batch9,batch10
"""
import os, sys, glob, time, argparse
import numpy as np
import h5py
import cv2

# ─── 配置 ───
DATA_DIR = r"D:\curelsummer\week3\data"
OUT_DIR = os.path.join(DATA_DIR, "processed")

TARGET_H, TARGET_W = 256, 640
REGC_ROW = (838, 1930)
REGC_COL = (0, 2748)
BT_MIN, BT_MAX = 180.0, 320.0

# 各版本的默认批次目录（--batches 可覆盖）
VERSION_BATCHES = {
    'v3': ['batch1', 'batch2', 'batch3', 'batch4'],
    'v4': ['batch3', 'batch4', 'batch_extra'],
    'v5': ['batch5', 'batch6', 'batch7'],
    'v6': ['batch8', 'batch9', 'batch10'],
}


def process_file(fp):
    with h5py.File(fp, 'r') as f:
        dn = f['NOMChannel11'][:].astype(np.int32)
        cal = f['CALChannel11'][:]        # (4096,) LUT
        bt = cal[np.clip(dn, 0, 4095)].astype(np.float32)
        bt[dn >= 65000] = np.nan
        if bt.shape[0] == 2748:  # 全圆盘 2748×2748, REGC 1092×2748
            bt = bt[REGC_ROW[0]:REGC_ROW[1], REGC_COL[0]:REGC_COL[1]]
        bt = np.clip(bt, BT_MIN, BT_MAX)
        bt = (bt - BT_MIN) / (BT_MAX - BT_MIN)
    return cv2.resize(bt, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='v3',
                        choices=list(VERSION_BATCHES.keys()))
    parser.add_argument('--batches', type=str, default=None,
                        help='逗号分隔的批次目录，默认按 --version 取内置映射')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    args = parser.parse_args()

    ver = args.version
    data_dir = args.data_dir
    out_dir = os.path.join(data_dir, "processed")
    out_name = f"fy4a_ir108_{ver}.dat"

    batches = (args.batches.split(',') if args.batches
               else VERSION_BATCHES[ver])

    all_files = []
    for b in batches:
        fs = sorted(glob.glob(os.path.join(data_dir, b, '*.HDF')))
        all_files.extend(fs)
        print(f'{b}: {len(fs)} files')
    N = len(all_files)
    print(f'Total: {N}')

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    shape = (N, 1, TARGET_H, TARGET_W)
    with open(out_path, 'wb') as f:
        f.seek(int(np.prod(shape)) * 4 - 1)
        f.write(b'\x00')
    data = np.memmap(out_path, dtype='float32', mode='r+', shape=shape)

    timestamps, bt_all, errs = [], [], 0
    t0 = time.time()
    for i, fp in enumerate(all_files):
        try:
            bt = process_file(fp)
            data[i, 0] = bt
            v = bt[~np.isnan(bt)]
            bt_all.extend([v.min(), v.max(), v.mean()])
            bn = os.path.basename(fp)
            ts = bn.split('_')[9] if len(bn.split('_')) > 9 else f'{i}'
            timestamps.append(ts)
        except Exception as e:
            errs += 1
            if errs <= 5:
                print(f'ERR[{i}] {os.path.basename(fp)}: {e}')
        if (i + 1) % 200 == 0 or i == N - 1:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N - i - 1) / 60 if i > 0 else 0
            print(f'[{i+1}/{N}] {(i+1)/N*100:.0f}% ETA:{eta:.1f}min errs={errs}',
                  flush=True)

    data.flush()
    bt_all = np.array(bt_all)
    print(f'\nIR10.8: {bt_all[::3].min():.3f}-{bt_all[::3].max():.3f} (norm), '
          f'mean={bt_all[2::3].mean():.3f}, errs={errs}')

    np.savez(os.path.join(out_dir, f'metadata_{ver}.npz'),
             timestamps=np.array(timestamps, dtype='U14'),
             channels=np.array([11]),
             channel_names=np.array(['IR10.8']),
             wavelengths=np.array([10.8]),
             shape=np.array(shape), bt_min=BT_MIN, bt_max=BT_MAX)
    np.save(os.path.join(out_dir, f'shape_{ver}.npy'), np.array(shape))
    print(f'Done: {os.path.getsize(out_path)/1e6:.0f}MB')


if __name__ == '__main__':
    main()
