# 异步BCI评估指标

## 概述

本文档描述 BNCI2014001 异步 BCI 实验中使用的评估框架。系统采用**两阶段流水线**：

1. **Stage 1（二分类）**：区分 idle（空闲）和 task（任务）
2. **Stage 2（4分类 MI）**：左手 / 右手 / 脚 / 舌头

数据以**2秒滑动窗口**、**0.5秒步长**处理，每个窗口得到一个预测值（0 = idle，1-4 = MI 类别）。

评估脚本：`BCI_Competition/code/eval/run_evaluation.py`

---

## 指标

### 1. 事件级指标

**事件**是指连续且标签相同的非零窗口序列（一次完整的 MI 试验）。事件是衡量用户真实意图的基本单位。

| 指标 | 说明 | 范围 |
|------|------|------|
| **正确率** | 事件中至少有一个窗口预测出正确的类别 | 0-100% |
| **误分类率** | 事件中有任务预测但从未猜对类别 | 0-100% |
| **遗漏率** | 事件中完全没有任务预测（系统完全没检测到） | 0-100% |
| **误触发率(FP/min)** | idle 期间出现任务预测的次数，按分钟标准化 | 0+（越低越好） |
| **延迟（中位数/均值）** | 从事件开始到第一次正确预测的时间（秒） | 0+（越低越好） |
| **Bootstrap 95% CI** | 自助法重采样2000次，得到正确率的95%置信区间 | 反映指标稳定性 |

**为什么要事件级评估？** 异步 BCI 中，用户会持续执行运动想象任务，系统只需要在试验过程中至少正确检测到一次即可，不要求每个窗口都分对。事件级指标更贴近实际使用场景。

**Bootstrap 置信区间**通过对事件进行有放回重采样（2000次），每次重新计算正确率，取2.5和97.5百分位数得到区间。区间越窄说明指标越稳定。

### 2. 窗口级指标

每个滑动窗口视为独立的分类结果。这些是标准的多分类指标。

| 指标 | 说明 | 范围 |
|------|------|------|
| **Cohen's Kappa** | 校正了随机一致性的预测-标签一致性。BCI 竞赛官方指标 | -1~1（越高越好） |
| **Macro F1** | 各类 F1 的无加权平均，各类别的重要性相同 | 0-1（越高越好） |
| **各类精确率 Precision** | `TP / (TP + FP)` — 预测为该类的结果中，有多少是对的 | 0-1 |
| **各类召回率 Recall** | `TP / (TP + FN)` — 真实的该类样本中，有多少被找到 | 0-1 |
| **各类 F1** | 精确率和召回率的调和平均 | 0-1 |

### 3. 混淆矩阵

行代表真实类别，列代表预测类别。帮助识别哪些类别容易相互混淆。

例如：如果 `feet` 的预测大量出现在 `tongue` 真实行里，说明模型很难区分脚和舌头。

### 4. 决策稳定性

| 指标 | 说明 |
|------|------|
| **预测切换次数** | 相邻窗口之间预测值变化的次数 |
| **切换率** | 切换次数 / (总窗口数-1) — 预测变化的频繁程度 |
| **Idle->Task 跳变** | 模型从空闲状态进入任务状态的次数 |

切换率高说明输出抖动剧烈，实际使用中会让用户感到疲劳和困惑。

### 5. 信息传输率 (ITR)

信息传输率是 BCI 领域的标准指标，衡量系统每秒能传输多少比特信息。

```
ITR (bits/symbol) = log2(M) + P*log2(P) + (1-P)*log2((1-P)/(M-1))
ITR (bits/min) = ITR(bits/symbol) * 60 / 平均事件时长(s)
```

其中 P 为事件级正确率，M 为类别数。脚本同时计算 M=4（仅MI类）和 M=5（含idle）两个版本。

| 指标 | 说明 | 范围 |
|------|------|------|
| **ITR (4-class MI)** | 仅考虑 4 类运动想象的信息传输率 | 0+（越高越好） |
| **ITR (5-class +idle)** | 含 idle 在内的 5 类信息传输率 | 0+（越高越好） |

### 6. Stage 1 二分类 AUC

使用 sklearn 的 `roc_auc_score` 计算 Stage1 二分类器（idle vs task）的 ROC 曲线下面积。AUC 衡量模型区分 idle 和 task 的能力，不依赖于具体阈值。

| 指标 | 说明 | 范围 |
|------|------|------|
| **Binary AUC** | Stage1 idle vs task 的 ROC 曲线下面积 | 0.5-1（越高越好） |

- AUC = 0.5：随机猜测
- AUC = 0.8-0.9：区分能力良好
- AUC > 0.9：区分能力优秀

### 7. 阈值扫描分析

通过扫描置信度阈值（0.00 ~ 0.95），观察 TPR（真正率）和 FPR（假正率）的变化，帮助选择最佳工作点。

| 指标 | 说明 |
|------|------|
| **Youden 最优阈值** | 最大化 TPR - FPR 的阈值 |
| **Youden 点 TPR/FPR** | 最优阈值对应的 TPR 和 FPR |
| **低 FPR 工作点** | FPR ≤ 0.1 时保持最高 TPR 的阈值 |

**Youden's J statistic**：`J = TPR - FPR`，值越大说明该阈值下检测性能越好。

实际应用中，可以根据对误触发的容忍度选择阈值：
- 追求低误触发（如医疗康复）→ 选高阈值，FPR 低但 TPR 也会下降
- 追求高检出率（如通信控制）→ 选低阈值，TPR 高但 FPR 也高

### 8. 综合评分

将多个指标加权合并为一个分数，便于模型快速比较。

```
综合分 = 0.25 × 正确率
       + 0.20 × (1 − FP/min ÷ 10, 截断至0)
       + 0.20 × max(0, Kappa)
       + 0.15 × MacroF1
       + 0.10 × (1 − 切换率)
       + 0.10 × (1 − 延迟/5, 截断至0)
```

权重分配上侧重正确率和误触发控制，兼顾分类质量（Kappa、F1）、稳定性和速度。

---

## 使用方法

```bash
# 评估单个模型
python BCI_Competition/code/eval/run_evaluation.py --model eegnet

# 带置信度阈值（排除低置信度预测）
python BCI_Competition/code/eval/run_evaluation.py --model conformer --threshold 0.3

# 可用模型
python BCI_Competition/code/eval/run_evaluation.py --model eegnet
python BCI_Competition/code/eval/run_evaluation.py --model eegnet_attn
python BCI_Competition/code/eval/run_evaluation.py --model shallowconvnet
python BCI_Competition/code/eval/run_evaluation.py --model deepcnn
python BCI_Competition/code/eval/run_evaluation.py --model conformer
python BCI_Competition/code/eval/run_evaluation.py --model deformer
```

输出：控制台汇总 + JSON 文件保存到 `BCI_Competition/results/tables/`。

---

## 结果解读

- **正确率高 + FP/min低**：理想状态——既能抓住事件又不会频繁误报
- **延迟 < 0.5s**：系统快速检测到意图。延迟高意味着用户需要更久保持运动想象
- **Kappa > 0.5**：超过随机水平的一致性较好（0.4-0.6为中等，> 0.6为优秀）
- **Idle 召回率高**：对降低误触发很重要——idle 占了数据近一半
- **Binary AUC > 0.8**：区分 idle 和 task 的能力良好
- **ITR > 20 bits/min**：对4类 MI 来说属于可用水平
- **Bootstrap 置信区间窄**：指标稳定可靠
- **左右手混淆**：运动想象 BCI 常见问题，由于大脑皮层映射区域重叠
- 通过阈值扫描选择工作点：如果 FP/min 过高，可以适当提高置信度阈值

---

## 完整模型对比

| 模型 | 正确率 | 误分类 | 遗漏 | FP/min | 延迟 | Kappa | Macro F1 | ITR(4cls) | AUC | 综合分 |
|------|:------:|:-----:|:----:|:------:|:----:|:----:|:--------:|:---------:|:---:|:------:|
| eegnet | 91.5% | 7.1% | 1.3% | 50.20 | 0.30s | 0.477 | 0.582 | 27.00 | 0.825 | 0.570 |
| conformer | 95.1% | 4.5% | 0.4% | 49.70 | 0.31s | 0.527 | 0.619 | 30.61 | 0.854 | 0.592 |

Conformer 在几乎所有维度上都优于 eegnet。两个模型的 FP/min 都偏高（约50），说明 idle 和 task 的区分还有改进空间。

---

## 参考文献

- BNCI Competition 2014: http://bnci-horizon-2020.eu/
- Cohen's Kappa: Cohen (1960). A coefficient of agreement for nominal scales.
- Macro F1: Sokolova & Lapalme (2009). A systematic analysis of performance measures for classification tasks.
- Youden's J statistic: Youden (1950). Index for rating diagnostic tests.
- Wolpaw et al. (2000). Brain-computer interface technology: a review of the first international meeting. (ITR standard)
