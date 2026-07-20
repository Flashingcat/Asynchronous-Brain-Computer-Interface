# EEG Data Augmentation for Motor Imagery BCI

基于 BrainprintNet 的数据增强方法，为 BNCI2014001 异步运动想象 BCI 实现 17 种在线增强。

## 方法列表

### 单样本增强（13 种）

| 方法 | 说明 |
|------|------|
| `noise_trial` | 均匀噪声，幅度按 trial std 缩放 |
| `mult_trial` | 幅度随机缩放（×0.95~1.05） |
| `neg` | 幅值取反平移（负波变正） |
| `freq_shift` | Hilbert 频域移频 |
| `noise_ch` | 随机选通道加高斯噪声 |
| `flip_ch` | 随机选通道幅值反转 |
| `scale_ch` | 随机选通道幅度缩放 |
| `time_reverse` | 时间轴反转 |
| `mirror` | 左右半球通道镜像交换 |
| `noise_gaussian` | 全通道高斯噪声（std=0.1） |
| `noise_sp` | 全通道椒盐噪声（p=0.01） |
| `noise_poisson` | 全通道泊松噪声（λ=1.0） |
| `noise_pink` | 全通道粉红噪声（Voss-McCartney 算法） |

### 双样本增强（4 种）

| 方法 | 说明 |
|------|------|
| `mixup` | 同类样本混合 |
| `ch_mixure` | 同类样本通道级混合 |
| `trial_mixup` | 同类样本 trial 级混合 |
| `dwta` | 动态时间规整平均 |

## 使用方式

### 训练

```bash
# 单一增强
python BCI_Competition/code/augmentation/train_augmented.py \
    --model eegnet --subjects 1 --augment noise_trial --binary-epochs 30 --mi-epochs 30

# 多种增强组合
python BCI_Competition/code/augmentation/train_augmented.py \
    --model eegnet --subjects 1 --augment noise_trial freq_shift --binary-epochs 30 --mi-epochs 30

# 全部方法
python BCI_Competition/code/augmentation/train_augmented.py \
    --model eegnet --subjects 1 --augment all --binary-epochs 30 --mi-epochs 30
```

### 评测

```bash
python BCI_Competition/code/eval/evaluate_test_session.py \
    --algorithm argmax \
    --checkpoints results/checkpoints/augmented/s01_eegnet_noise_trial_final.pt \
    --output results/eegnet_noise_trial_metrics.json
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `augmentations.py` | 所有增强类实现 |
| `dataset.py` | 在线增强数据集包装器 |
| `train_augmented.py` | 独立训练脚本（兼容原 train 接口） |
| `__init__.py` | 模块导出 |

## 设计原则

- **在线增强**：每个 sample 以 p=0.5 独立触发增强，不做离线拼接
- **不修改原训练代码**：增强逻辑完全独立，原 `train_hierarchical_oof.py` 不受影响
- **可组合**：支持任意方法组合，pipeline 中每种方法按顺序独立触发

## 参考

BrainprintNet: https://github.com/hustmx721/BrainprintNet/blob/main/src/data/augmentation.py
