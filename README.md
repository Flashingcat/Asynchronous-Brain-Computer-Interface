# Asynchronous Brain-Computer Interface

基于 BNCI2014001 数据集的异步脑机接口（BCI）识别系统，采用两阶段流水线架构。

## 项目结构

```
Asynchronous-Brain-Computer-Interface/
  BCI_Competition/
    code/
      datasets/        # 数据集下载
      preprocessing/   # 数据预处理（滑窗构建）
      train/           # 两阶段训练
      eval/            # 评估指标
      models/          # 模型定义
        model_factory.py
        models/
          eegnet.py / eegnet_attn.py / shallowconvnet.py
          deepcnn.py / conformer.py / deformer.py
    data/
      public/          # 原始数据
      processed/       # 预处理后的窗口数据
    results/
      checkpoints/     # 模型权重
      tables/          # 评估结果
    docs/              # 文档
```

## 两阶段架构

```text
Stage 1: idle vs task 二分类
  输入：EEG 窗口 → 输出：idle / task
Stage 2: 4-class MI 运动想象分类
  输入：task 窗口 → 输出：left_hand / right_hand / feet / tongue
最终：合成 5 类输出（idle + 4 类 MI）
```

## 快速开始

```bash
# 1. 下载数据
python BCI_Competition/code/datasets/download_bnci2014001.py --subjects 1

# 2. 预处理（滑动窗口）
python BCI_Competition/code/preprocessing/build_async_windows.py

# 3. 训练
python BCI_Competition/code/train/train_eegnet_async.py --model eegnet

# 4. 评估
python BCI_Competition/code/eval/run_evaluation.py --model eegnet
```

## 可用模型

| 模型 | 特点 | 参数量 | 正确率 | FP/min | Kappa | 综合分 |
|------|------|:------:|:------:|:------:|:-----:|:------:|
| conformer | CNN + Transformer | 313K | 95.1% | 49.70 | **0.527** | **0.592** |
| eegnet_attn | EEGNet + 通道注意力 | 10K | **97.3%** | 57.27 | 0.496 | **0.592** |
| eegnet | 轻量 CNN 基线 | **4K** | 91.5% | 50.20 | 0.477 | 0.570 |
| deepcnn | 深层卷积网络 | 163K | 85.3% | 32.22 | 0.464 | 0.545 |
| shallowconvnet | 浅层卷积网络 | 1,058K | 85.3% | **26.57** | 0.454 | 0.542 |
| deformer | 纯 Transformer | — | 83.0% | 96.67 | 0.324 | 0.497 |
| dbconformer | 双分支 Conformer | — | — | — | — | — |

## 评估指标

提供事件级和窗口级两套指标：

- **事件级**：正确率、误分类率、遗漏率、FP/min、延迟
- **窗口级**：Cohen's Kappa、Macro F1、混淆矩阵、各类 Precision/Recall/F1
- **检测性能**：Binary AUC、ITR（信息传输率）
- **稳定性**：预测切换率、阈值扫描（TPR/FPR 曲线）
- **综合评分**：加权合并的多维度评分

详细说明见 [evaluation_metrics.md](BCI_Competition/docs/evaluation_metrics.md)。

## 环境要求

- Python 3.10+
- PyTorch >= 2.0
- 依赖安装：`pip install -r BCI_Competition/requirements.txt`

## 参考文献

- [BNCI Competition 2014](http://bnci-horizon-2020.eu/)
- Lawhern et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces.
- Schirrmeister et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization.
