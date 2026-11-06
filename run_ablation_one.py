"""
单组消融实验训练入口
用法: python run_ablation_one.py --mode no_ph
"""
import os, sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True,
                    choices=['no_ph', 'no_pde', 'no_timedecay'])
args = parser.parse_args()

from config import PHMetaFormerConfig
cfg = PHMetaFormerConfig()
cfg.ablation_mode = args.mode
cfg.checkpoint_dir = f"ablation_checkpoints/abl_{args.mode}"
cfg.epochs = 100

# 清空旧 checkpoint
import glob
for f in glob.glob(os.path.join(cfg.checkpoint_dir, "*.pt")):
    os.remove(f)
os.makedirs(cfg.checkpoint_dir, exist_ok=True)

print(f"=== Ablation: {args.mode} ===")
print(f"Checkpoint dir: {cfg.checkpoint_dir}")

from train import train
train(cfg)
