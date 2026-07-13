# EEGNet 250 Hz OOF 基线训练矩阵

## 1. 本轮目的和边界

本轮先在 BNCI2014001 Subject 1 的训练 session（session 0）内建立首个可审计的 EEGNet 正式候选基线，随后用完全相同的被试内流程在 Subject 2–9 上分别独立重复，用于检查协议的可运行性和稳定性。这些矩阵回答“模型在同一被试中未参与拟合的完整 run 上表现如何”，不属于跨被试训练，不产生最终测试 session 结果，也不等同于严格在线仿真结果。

固定边界如下：

- 采样率：原生 250 Hz，不执行重采样；
- 模型：原仓库 `EEGNetClassifier`；
- 验证：六折留一完整 run，fold 编号等于验证 run 编号；
- Stage 1：`IDLE/Task` 二分类，使用共同正式 Stage 1 窗口；
- Stage 2：四类 MI 分类，使用上述窗口中的真实 Task 子集；
- 伪迹、滤波启动排除和标准化均沿用已冻结清单；
- 测试 session（session 1）在 OOF 训练、调参和 epoch 选择期间禁止访问。

正式训练不直接打开同时包含两个 session 的离线索引、标准化清单或信号存储，而只消费冻结的 session0-only bundle。bundle 构建是独立的数据边界准备步骤：它可核验既有联合上游，但输出只保留对应被试的 session 0 数据；以 Subject 1 为例，包含 2454 个 Stage 1 窗口、1365 个 Stage 2 窗口，以及因果/零相位各 21 个训练 session segment。训练器缺少 bundle 时直接失败，不会自动回访联合上游。

## 2. 要训练的 72 条轨迹

训练轨迹数为：

```text
6 folds × 2 stages × 2 training domains × 3 seeds = 72
```

| 轴 | 固定取值 |
|---|---|
| fold | `0,1,2,3,4,5` |
| stage | `1,2` |
| training domain | `causal, zero_phase` |
| seed | `42,43,44` |

一条轨迹只训练一个模型。验证输入域不增加训练次数：

- 因果训练模型只在因果验证窗口上输出 OOF logit；
- 零相位训练模型同时在零相位和因果验证窗口上输出 OOF logit；
- 两种验证输入使用完全相同的窗口身份、标签和顺序，只改变滤波输入及与之绑定的训练 fold 标准化参数。

由此，同一批零相位训练轨迹同时支持传统离线基线和“零相位训练、因果推理”消融。当前矩阵不训练 128 Hz 模型，也不扩展其他网络。

## 3. 固定训练参数

| 项目 | 值 |
|---|---|
| 最大 epoch | 50 |
| batch size | 64 |
| validation batch size | 256 |
| optimizer | AdamW |
| learning rate | `1e-3` |
| weight decay | `1e-4` |
| loss | 仅按当前 fold 训练标签计算权重的 balanced cross-entropy |
| early stopping | 不启用 |
| 数值精度 | FP32，不启用 AMP/TF32 |
| DataLoader workers | 0 |
| 确定性 | PyTorch deterministic algorithms；逐 epoch 固定洗牌种子 |

Stage 1 和 Stage 2 独立训练。所有输入保持 `(22, 500)`，即 22 个 EEG 通道和 2 秒原生 250 Hz 信号。

本轮保存全部 50 个 epoch 的验证原始 logit，而不是在每个 fold 内独立早停。具体主指标及 epoch 选择规则要在指标轮冻结后，使用六个 held-out run 的 OOF 结果统一确定；不得按测试 session 结果选择 epoch。若后续需要某个较早 epoch 的 final-fit 模型，将在六个训练 run 上按冻结 epoch 从头训练，而不是把某个 fold checkpoint 当最终模型。

## 4. 每条轨迹的审计产物

每个作业目录至少包含：

```text
job_config.json       数据/代码/超参数合同及其 SHA-256
latest.pt             模型、优化器、随机状态、history 和累计 OOF logit
oof_predictions.npz   验证窗口元数据、标签及逐 epoch 原始 logit
history.json          逐 epoch 训练损失和验证诊断指标
status.json           当前 epoch、状态和模型张量哈希
completed.json        50 epoch 完成标记
```

`oof_predictions.npz` 中的 logit 形状为 `[epoch, validation_window, class]`。Stage 1 的最后一维为 2，Stage 2 为 4。文件保留验证窗口 structured metadata，后续可按 run、类别和 MI 相对偏移重新分组，而无需重新推理。

检查点和派生文件均采用临时文件加原子替换，正常进程中断不会留下半个正式 checkpoint。恢复时必须同时匹配实验合同、bundle/源码哈希及 Python、PyTorch、CUDA、cuDNN、设备类型、GPU 名称和计算能力等执行指纹；不匹配时拒绝覆盖。每个 epoch 使用由 `seed + epoch` 决定的独立洗牌序列，同时恢复 Python、NumPy、PyTorch CPU/CUDA 随机状态，使中断前后的 dropout 轨迹可复现。`completed.json` 在 checkpoint 和全部公开产物完成后最后提交；重跑完整作业只读核验，派生产物缺失时才从已绑定 checkpoint 重建。

## 5. 启动门槛和命令

正式矩阵启动前必须先通过：

1. 单元测试和全量回归；
2. session0-only bundle 的窗口、segment、哈希、移动读取和打开文件隔离测试；
3. 一个真实 Subject 1 / fold 0 / Stage 1 / causal / seed 42 GPU 全流程；
4. 连续训练与“epoch 1 原子检查点后暂停、跨进程恢复”的模型、优化器、全部 RNG 和 OOF logit 逐值等价检查；
5. 一个真实 Stage 2 / zero-phase 训练并同时输出 zero-phase/causal 验证 logit 的 GPU 全流程；
6. 独立子智能体对数据隔离、恢复语义和产物结构的严格审核。

正式训练前先显式构建 bundle：

```powershell
python BCI_Competition/code/preprocessing/build_oof_training_bundle.py --subjects 1
```

恢复预检示例：

```powershell
python BCI_Competition/code/train/preflight_eegnet_oof.py `
  --output-root BCI_Competition/results/checkpoints/eegnet_oof_preflight_native250_v2
```

72 条正式候选矩阵：

```powershell
python BCI_Competition/code/train/train_eegnet_oof.py
```

再次执行同一命令会跳过完成作业，并从未完成作业的最近一个完整 epoch 继续。训练进度记录在矩阵根目录的 `matrix_status.json`；任何作业失败时矩阵立即停止，不静默跳过或改变协议。

## 6. 2026-07-13 实际运行记录

Subject 1 的正式候选矩阵在 RTX 5070 Laptop 上完成 72/72 条轨迹。由于首批速度允许，随后选择扩展被试而不是扩展模型：Subject 2–9 使用相同 EEGNet、250 Hz、六折、双训练域和三随机种子，各自在本被试的训练 session 内独立训练和验证。这样扩展是为了检查同一评估协议在多个被试上的可运行性和稳定性，不是跨被试训练，也不支持跨被试迁移或跨被试模型泛化结论。

边界和产物：

- Subject 1 是当前顶层协议的首个正式候选验证范围；Subject 2–9 是多被试独立重复得到的协议可运行性证据；
- 9 个被试各 72 条，共 648 条轨迹、32400 个训练 epoch；
- 9 个矩阵均为 `status=complete`，每个被试包含 72 个唯一规格；
- 全部作业使用同一源码身份和同一 RTX 5070 执行指纹；
- 648 份 checkpoint、history、OOF、status 和完成标记哈希全部匹配；
- 每份 OOF 均包含 50 个 epoch，Stage 1/2 logit 末维分别为 2/4；
- 临时文件为 0，统一审计未发现产物问题；
- 所有训练进程只消费各被试的 session0-only bundle；独立的 bundle 构建阶段可以核验同时包含两个 session 的既有上游；本轮没有生成测试 session 预测。

各训练 session 的共同训练窗口量：

| Subject | Stage 1 | Stage 2 |
|---:|---:|---:|
| 1 | 2454 | 1365 |
| 2 | 2425 | 1350 |
| 3 | 2423 | 1350 |
| 4 | 2343 | 1310 |
| 5 | 2353 | 1310 |
| 6 | 1898 | 1095 |
| 7 | 2430 | 1355 |
| 8 | 2352 | 1320 |
| 9 | 2096 | 1185 |

Subject 1 在 39/72 条完成后遇到一次 Windows 对 `matrix_status.json` 原子替换的临时 `WinError 5`。错误发生前，当前作业 epoch 38 的 checkpoint、history 和 OOF 已完整写入。矩阵按设计停止；随后使用完全相同的源码、bundle、超参数和执行指纹从 epoch 39 恢复，最终 72/72 完成。该事件证明了 epoch 边界恢复路径，但不等同于随机时刻断电测试。Subject 2–9 的最终 stderr 均为空。

### 6.1 固定 epoch 50 的描述性 OOF 结果

下表只用于确认训练健康和观察被试差异：每个 seed 先拼接六个 held-out run，再计算 balanced accuracy，表中为三个 seed 的均值。它没有选择最佳 epoch，不是冻结后的正式模型选择结果。

| Subject | S1 因果 | S1 零相位 | S1 零训→因果 | S2 因果 | S2 零相位 | S2 零训→因果 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.801 | 0.804 | 0.791 | 0.694 | 0.689 | 0.675 |
| 2 | 0.796 | 0.752 | 0.754 | 0.484 | 0.500 | 0.481 |
| 3 | 0.911 | 0.911 | 0.903 | 0.792 | 0.798 | 0.787 |
| 4 | 0.802 | 0.804 | 0.769 | 0.427 | 0.411 | 0.409 |
| 5 | 0.725 | 0.719 | 0.711 | 0.422 | 0.474 | 0.412 |
| 6 | 0.780 | 0.772 | 0.773 | 0.395 | 0.373 | 0.377 |
| 7 | 0.903 | 0.901 | 0.901 | 0.700 | 0.706 | 0.708 |
| 8 | 0.786 | 0.787 | 0.785 | 0.724 | 0.728 | 0.729 |
| 9 | 0.762 | 0.756 | 0.748 | 0.633 | 0.626 | 0.631 |
| 被试宏平均 | 0.807 | 0.801 | 0.793 | 0.586 | 0.590 | 0.579 |
| 被试间样本标准差（`ddof=1`） | 0.061 | 0.066 | 0.066 | 0.153 | 0.153 | 0.159 |

结果目录：Subject 1 位于 `results/checkpoints/eegnet_oof_native250_v1`；Subject 2–9 位于 `results/checkpoints/eegnet_oof_extension_sXX_native250_v1`。原始逐 epoch logit 是后续正式指标和 epoch 选择的唯一输入；在指标轮冻结前，不从本表推导“最佳滤波”或“最佳 epoch”结论。

### 6.2 源码快照与正式状态

本轮 648 个作业及 GPU 预检都记录了启动时的 Git 状态：基准 commit 为 `3974ed702ee7e1a78dcb38733582d286cb0b0244`，并且 `dirty=true`。该 commit 本身不包含本轮新增训练器，因此当前结果只能称为“正式候选”，不能仅凭旧 commit 宣称为最终正式结果。

作业合同同时保存了实际执行的四份关键源码 SHA-256，648 个作业记录完全一致：

| 运行源码 | SHA-256 |
|---|---|
| `code/train/train_eegnet_oof.py` | `4b7637a35e864185decd36e2f853ab31d32fed125f1d512a888fee2a3049b599` |
| `code/train/oof_training_bundle.py` | `f5a2deb40b64187dcbbce34b5e4c382ebb637177851776713eb96c4e99e80f56` |
| `code/models/model_factory.py` | `9e6b6af936f088cf0ed3cb25f52cf59d460159019b50beaf7b2b7b7c93173a60` |
| `code/models/models/eegnet.py` | `73c97e1bae388ad599025c61cde27f4d451d12b3b83fa898232d6fe90d3fdaed` |

用户审批并提交本轮源码后，必须在结果根目录补写 post-hoc source snapshot：记录新 commit、确认其工作树干净，并保存上述四份运行源码的字节级副本。snapshot 必须证明副本 SHA-256 与作业合同逐项一致；若 Git 的换行规范化导致 commit blob 与运行字节不同，还必须同时记录 blob SHA-256、LF 规范化 SHA-256 和差异类型，并证明除换行外内容一致，不得把它误写成原始字节完全相同。完成该映射后，Subject 1 训练矩阵产物的源码 provenance 才完整，可作为后续正式评估的输入；它本身仍不是正式评估结果。正式结果还必须等待主指标和 epoch 选择规则在后续指标轮冻结。Subject 2–9 仍只表示多被试独立重复，不改变实验范式的结论边界。
