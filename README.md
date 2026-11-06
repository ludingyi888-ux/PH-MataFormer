# PH-MetaFormer

基于 **Port-Hamiltonian 框架**与**隐空间映射**的 FY-4A 卫星云图长时序预测模型。

- 任务：给定过去 4 帧 IR10.8 云图，预测未来 4 帧（4–8 小时窗口）
- 核心：隐空间 Port-Hamiltonian 动力学 + Leapfrog 辛积分，保证长期预测不发散
- 最佳成绩：Val MAE = 4.83 K（300 epochs，零 NaN）

## 文档

| 文档 | 内容 |
|------|------|
| [`01_研究背景.md`](01_研究背景.md) | 课题背景、核心挑战、参考文献 |
| [`02_算法框架.md`](02_算法框架.md) | 模型架构与 PH 动力学详解 |
| [`03_使用说明.md`](03_使用说明.md) | 环境、数据、训练、推理操作手册 |

## 快速开始

环境：Python 3.11+ / PyTorch (CUDA) / numpy / opencv-python / scipy / scikit-learn / matplotlib / h5py

```powershell
python run_train.py               # 训练
python infer.py                   # 推理可视化
python anomaly_detector.py --demo # 异常轨迹预警
python train_convlstm.py          # ConvLSTM 基线
python run_ablation.py            # 消融实验
```

完整流程（数据下载 → 预处理 → 训练 → 推理）见 [`03_使用说明.md`](03_使用说明.md)。

## 项目结构

```
├── models/                  # 模型核心模块
│   ├── ph_metaformer.py     # 主模型 + compute_loss
│   ├── metaformer_encoder.py# SpatioTemporal 编码器
│   ├── decoder.py           # 解码器（FiLM/SkipGate）
│   ├── energy_net.py        # SeparableEnergy 能量函数
│   ├── ph_dynamics.py       # PH 动力学（Leapfrog/算子分裂）
│   └── convlstm.py          # ConvLSTM 基线
├── config.py                # 训练配置
├── train.py                 # 训练主逻辑（含 FY4ADataset）
├── infer.py                 # 推理可视化
├── anomaly_detector.py      # 异常轨迹预警
├── preprocess.py            # 数据预处理（参数化）
├── merge_all_data.py        # 数据合并 + 降采样
└── ...                      # 消融、基线、下载等脚本
```
