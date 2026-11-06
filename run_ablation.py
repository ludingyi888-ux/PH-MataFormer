"""
PH-MetaFormer 三组训练级消融实验（串行执行）

实验 1: no_ph     — 无 PH 动力学，恒等复制 q0/p0
实验 2: no_pde    — 无 PDE 损失 (canonical=0, drift=0)
实验 3: no_timedecay — 无时间衰减+递归，全量静态 skip

用法: python run_ablation.py          # 全部三个
      python run_ablation.py --exp no_ph  # 单个
"""
import os, sys, subprocess

BASE_DIR = r"D:\curelsummer\week4"

experiments = [
    ("abl_no_ph", "no_ph", "无 PH 动力学：恒等复制 q0/p0"),
    ("abl_no_pde", "no_pde", "无 PDE 损失：canonical=drift=0"),
    ("abl_no_timedecay", "no_timedecay", "无时间衰减：全量静态 skip"),
]

def run_one(name, mode, desc):
    print(f"\n{'='*60}")
    print(f"  {name}: {desc}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, "-u", "run_ablation_one.py", "--mode", mode],
        cwd=BASE_DIR
    )
    print(f"  [{name}] exit: {result.returncode}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="all")
    args = parser.parse_args()

    for name, mode, desc in experiments:
        if args.exp == "all" or args.exp == mode:
            run_one(name, mode, desc)

    print("\n  All done!")
