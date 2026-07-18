# CIFAR-10 图像分类
pytorch入门训练

使用 PyTorch 搭建 CNN 模型，在 CIFAR-10 数据集上进行图像分类训练，学习学习率衰减（StepLR）、早停（early stopping）、检查点（checkpoint）

## 项目结构

```
PythonProject/
├── cifar_train.py        # 训练脚本
├── checkpoints/          # 模型权重
│   ├── best_cifar10.pth  # 准确率最高时的模型
│   └── checkpoint.pth    # 每轮保存完整训练状态，用于断点恢复
├── data/                 # CIFAR-10 数据集
├── images/               # TensorBoard截图
├── logs/                 # TensorBoard日志
├── 实验日志.md            # 实验记录与分析
├── 环境配置文档.md         # 开发环境配置说明
├── README.md
└── .gitignore
```

## 环境要求

- Python 3.11
- PyTorch 2.x（CUDA支持）
- torchvision
- tensorboard

详细配置见[环境配置文档.md](环境配置文档.md)

## 快速开始

```bash
# 1. 激活环境
conda activate pytorch

# 2. 运行训练
python cifar_train.py

# 3. 查看 TensorBoard 可视化
tensorboard --logdir=./logs --port=6007
```

训练完成后，在浏览器打开 http://localhost:6007 查看训练曲线。

## 模型结构

```
Conv2d(3→16, 3x3) → ReLU → MaxPool2d(2x2)
Conv2d(16→32, 3x3) → ReLU → MaxPool2d(2x2)
Flatten → Linear(2048→128) → ReLU → Linear(128→10)
```

## 实验记录

| 实验 | 训练轮数 | LR 策略 | 早停 | Checkpoint | Accuracy |
|------|---------|---------|------|-----------|----------|
| 实验一 | 5 | 固定 0.001 | 无 | 仅存最佳模型 | 53.55% |
| 实验二 | 10 | StepLR(step=2, γ=0.5) | 无 | 仅存最佳模型 | 52.46% |
| 实验三 | 20 | StepLR(step=4, γ=0.5) | patience=3 | best + latest | 59.79% |

详细分析见 [实验日志.md](实验日志.md)

## 技术要点

- **StepLR 学习率衰减**：每step_size个epoch将lr乘以gamma，前期大lr快速收敛，后期小lr精细调优
- **Early Stopping 早停**：连续patience个epoch验证准确率无提升则停止训练，避免浪费时间
- **Checkpoint 检查点**：
  - `best_cifar10.pth`：准确率最高时的模型，用于推理部署
  - `checkpoint.pth`：每epoch保存完整训练状态（model+optimizer+scheduler），用于断点恢复

## 作者

王铭硕
山东大学控制科学与工程学院
