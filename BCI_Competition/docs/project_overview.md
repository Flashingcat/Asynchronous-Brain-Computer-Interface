# 项目概述：异步脑机接口（Asynchronous BCI）

## 1. 数据集：BNCI2014001

### 数据来源
- 公开的竞赛数据集，MOABB 库封装
- 9 名受试者（subject 1-9）
- 每个受试者有 **2 个 session**（训练 session + 测试 session）

### 采集设备
- **电极**：22 通道 EEG，按国际 10-20 系统放置
- **采样率**：256 Hz（预处理时重采样到 128 Hz）

### 实验范式（运动想象 Motor Imagery）
受试者根据屏幕提示，想象对应的身体运动，无需实际执行：

| 标签 | 类别 | 说明 |
|------|------|------|
| 0 | idle | 休息状态，无任务 |
| 1 | left_hand | 想象左手运动 |
| 2 | right_hand | 想象右手运动 |
| 3 | feet | 想象双脚运动 |
| 4 | tongue | 想象舌头运动 |

### 单次 trial 流程
```
 fixation (2s) → cue (1s 提示出现) → motor imagery (3.5s) → pause (1.5-2.5s)
 |<——————— 窗口对齐这段 ————————>|
 任务态窗口: cue onset 之后的 4s
```

## 2. 项目目标：异步解码

### 同步 vs 异步

| 同步 BCI | **异步 BCI（本项目）** |
|----------|----------------------|
| 系统知道用户"正在做任务" | **系统需要自己判断**用户是在休息还是在做任务 |
| 只做运动想象分类 | 先判 idle/task，再做分类 |

### 两阶段架构

```
                    ┌─────────────┐
                    │  EEG 输入    │
                    │  22ch × 256 │
                    └──────┬──────┘
                           ▼
                  ┌────────────────┐
     Stage 1      │  idle vs task  │
    二分类         │  (二分类)       │
                  └───────┬────────┘
                          │
                    ┌─────┴─────┐
                    │           │
                  idle        task
                              │
                              ▼
                    ┌────────────────┐
     Stage 2        │ left/right/    │
    四分类           │ feet/tongue    │
                    │ (四分类)        │
                    └────────────────┘

          最终输出: idle(0) / left_hand(1) /
                    right_hand(2) / feet(3) / tongue(4)
```

### 为什么是异步？
实际场景中用户不会一直做任务——大部分时间处于 idle 状态。系统必须能区分"用户在想但没动"和"用户在做运动想象"，否则会误触发。

**应用场景**：脑控轮椅、康复外骨骼、打字系统等连续控制的场景。

## 3. 数据处理

### 滑窗参数
| 参数 | 值 | 含义 |
|------|----|------|
| 窗口长度 | 2.0 s | 每次输入 256 个时间点 |
| 滑步步长 | 0.5 s | 相邻窗口重叠 1.5s |
| 频带 | 8-30 Hz | 保留 mu + beta 节律 |
| 任务窗口 | cue onset → +4s | 对齐运动想象时段 |
| idle 窗口 | 不与任务窗口重叠的 2s 段 | 从所有 run 中采集 |

### 数据划分
```
train session → train (除验证 run 外) + validation (指定 1 个 run)
test session  → test（全部）
```

### 归一化
按训练集各通道的 mean/std，对全部数据做标准化：
```python
normalized = (features - mean) / std
```

## 4. 模型架构

使用 EEGNet，专为 EEG 信号设计的轻量卷积网络：

```
Input: [1, 22, 256]  (1×通道×时间)
  │
  ├─ Conv2D (1→F1, 1×kernLength)      ← 时间滤波
  ├─ DepthwiseConv2D (F1→F1*D, chans×1) ← 空间滤波（跨通道）
  ├─ ELU + AvgPool + Dropout
  │
  ├─ SeparableConv2D (F1*D→F2)        ← 时间特征提取
  ├─ ELU + AvgPool + Dropout
  │
  └─ Linear → num_classes (2 或 4)
```

参数：`F1=8, D=2, F2=16`，参数量极小（~3K）。

项目另支持 5 种模型：`ShallowConvNet`、`DeepCNN`、`Conformer`、`Deformer`、`DBConformer`。

## 5. 训练流程

### Stage 1：idle vs task 二分类
- 输入：全部窗口（idle + 四类运动想象）
- 标签：0 → idle, 1+ → task（二值化）
- 损失：带类别权重的 CrossEntropy

### Stage 2：MI 四分类
- 输入：仅任务态窗口（label > 0）
- 标签：1/2/3/4 映射为 0/1/2/3
- 损失：带类别权重的 CrossEntropy（缓解类别不平衡）

### 推理
```
Stage 1 输出 idle → 最终结果: 0
Stage 1 输出 task → 进入 Stage 2 → 映射回 1/2/3/4
```

## 6. 评估指标

三组指标分别评估：

| 指标组 | 内容 |
|--------|------|
| final_5class | 最终 5 分类的 accuracy / balanced accuracy / 混淆矩阵 |
| stage1_binary | Stage 1 的 idle vs task 二分类 |
| stage2_mi | 仅在真实任务窗口上的 MI 四分类 |

重点关注 **final_5class balanced accuracy**，因为类别严重不平衡（idle 远多于 task）。
