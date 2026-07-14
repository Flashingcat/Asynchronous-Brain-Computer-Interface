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

- Subject 1–9 现均属于顶层评估协议的验证范围；各被试仍为独立的被试内训练与 OOF 验证，不构成跨被试训练；
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

上述映射现已由 `results/checkpoints/eegnet_oof_native250_v1/posthoc_source_snapshot/manifest.json` 完成：它覆盖 9 个矩阵的 648 个作业，保存四份实际运行源码的字节级副本，并明确披露其中两份 Git blob 仅存在 CRLF/LF 差异。训练产物因此具备可供后续评估消费的源码 provenance；但训练矩阵本身仍不是在线系统结果，也不自动决定最终 epoch 或模型选择规则。Subject 1–9 的结果始终表示被试内独立重复，不构成跨被试训练或迁移结论。

### 6.3 九被试固定 epoch 50 单窗在线 OOF 基线

在不重新训练的前提下，因果主基线复用 9 个被试的 324 个 causal checkpoint：每个被试 2 个 Stage、6 个 fold、3 个 seed。评估只消费各被试的 session0-only bundle，测试 session 未读取。完整连续库存共包含 37,020 个两秒窗口和 2,328 个干净 MI 事件；每个 checkpoint 均先逐行重算其冻结验证窗口，当前机器上 324/324 份 epoch 50 OOF logit 与训练产物逐元素完全相等。

跨被试汇总先在同一 seed 下对 9 个被试等权宏平均，再对 seed 42/43/44 报告描述性均值和总体标准差；不按被试事件数或窗口数做微平均：

| 决策方式 | 正确事件率 | 事件检出率 | 检出后分类准确率 | MISS率 | IDLE误指令/分钟 | 正确事件中位延迟/s |
|---|---:|---:|---:|---:|---:|---:|
| 无状态单窗诊断 | 0.3291 | 0.9897 | 0.3326 | 0.0103 | 52.9094 | 0.9890 |
| `READY/WAIT_IDLE` 单窗基线 | 0.2257 | 0.5804 | 0.3880 | 0.4196 | 7.0483 | 1.6110 |

该结果说明单窗 argmax 状态机能显著减少空闲期误输出，但会漏掉较多事件；它是后续多窗聚合、置信度阈值和复位规则的最低对照，不代表最终系统。代码已由独立子智能体严格审核并提交为 `ba35de6`，随后从干净工作树完整重跑；正式候选清单位于 `results/tables/s01_s09_epoch50_causal_single_window_oof_clean_ba35de6_v2/run_manifest.json`，状态为 `CLEAN_COMMIT_FORMAL_CANDIDATE`。

### 6.4 联合硬标签多窗口诊断矩阵

第一版多窗口基线完全复用 6.3 节的原始逐窗 logit，不重新训练，也不访问测试 session。每窗先形成 `IDLE=0` 或 MI 类别 `1..4` 的联合硬标签，再按第 7.2 节的 `N,K` 网格运行 `READY/WAIT_IDLE` 状态机。MI 输出和 IDLE 复位采用同一组 `N,K`，每次状态转换后清空缓存；所有 cell 均报告，不做最优参数选择。

复现实跑命令：

```powershell
python BCI_Competition/code/eval/run_hard_vote_matrix.py `
  --input-root BCI_Competition/results/tables/s01_s09_epoch50_causal_single_window_oof_clean_ba35de6_v2 `
  --output-root BCI_Competition/results/tables/s01_s09_epoch50_causal_hard_vote_matrix_clean_cc2e026_v1
```

下表仍是“同一 seed 内九被试等权宏平均，再对三个 seed 求均值”的描述性结果：

| 策略 | 正确事件率 | 事件检出率 | 检出后分类准确率 | MISS率 | IDLE误指令/分钟 | 正确事件中位延迟/s |
|---|---:|---:|---:|---:|---:|---:|
| `n1_k1` 单窗参考 | 0.2257 | 0.5804 | 0.3880 | 0.4196 | 7.0483 | 1.6110 |
| `n2_k2` | 0.3268 | 0.7310 | 0.4439 | 0.2690 | 2.3446 | 1.9798 |
| `n3_k2` | 0.3285 | 0.7465 | 0.4374 | 0.2535 | 2.3057 | 1.9345 |
| `n3_k3` | 0.3556 | 0.6430 | 0.5380 | 0.3570 | 1.6258 | 2.5156 |
| `n4_k3` | 0.3754 | 0.6908 | 0.5316 | 0.3092 | 1.3573 | 2.4816 |
| `n4_k4` | 0.2789 | 0.4178 | 0.6308 | 0.5822 | 1.8047 | 2.9713 |
| `n5_k3` | 0.3853 | 0.7229 | 0.5232 | 0.2771 | 1.2442 | 2.4953 |
| `n5_k4` | 0.3015 | 0.4653 | 0.6189 | 0.5347 | 1.8662 | 2.9565 |
| `n5_k5` | 0.1617 | 0.2222 | 0.6765 | 0.7778 | 1.8155 | 3.2143 |

硬投票相对单窗基线明显改变了三组相互制约的量：较低阈值通常保留更多事件，较高阈值提高已触发事件的类别纯度但增加 MISS 和延迟；由于同一 `K` 还控制 `WAIT_IDLE` 复位，IDLE 误指令率不要求随 `K` 单调下降。`n5_k3` 在本矩阵中具有最高的描述性正确事件率，但当前协议明确禁止直接把它宣布为最终工作点。

代码经独立审核后提交为 `cc2e026`，并从干净工作树重跑至 `results/tables/s01_s09_epoch50_causal_hard_vote_matrix_clean_cc2e026_v1`，状态为 `CLEAN_COMMIT_DIAGNOSTIC_MATRIX`。每个被试和 seed 均保存联合硬标签、八组完整状态/输出轨迹和逐事件指标；顶层同时保存包含单窗锚点的逐 seed 与跨 seed CSV。顶层及各被试清单还记录实际 Python 解释器、环境目录、Python/NumPy 版本、主机和平台，环境身份不依赖外部聊天记录。“诊断矩阵”身份不会因为干净提交而自动变成已选择的正式工作点。

### 6.5 可撤销候选态内核

在提交 `6c3849d` 的候选态内核轮次，新增 `READY/TASK_CANDIDATE/WAIT_IDLE` 三状态策略，把 Stage 1 开门与维持、Stage 2 四分类提交、输出后的重复触发锁定分别处理。该层只消费调用方已经因果计算好的 `task_on`、`task_hold`、`stage2_commit_class`、`idle_reset`；在该轮结束时尚未选择任何分数变换、EWMA 系数、阈值或最大候选窗数，也没有产生新的九被试 OOF 候选策略结果。

该内核轮只用可手算轨迹冻结转换语义和候选诊断指标，并要求后续策略分别定义 Stage 1 和 Stage 2 的证据生成规则，再在训练 session OOF 上完整报告候选打开、撤销、超时、提交、事件正确率、误指令和延迟；不得把单元测试中的候选窗数当作正式超参数。首批 logit 证据策略和描述性结果现见 6.6 节，但不改变 6.5 节只描述底层状态合同的范围。

为允许后续策略安全复用 6.3 节的冻结 logits，历史输入验证改为核对 `ba35de6` 运行自身的固定 master/child 源码哈希合同和全部产物哈希，而不是要求历史生成源码等于当前已扩展的评估器源码。新运行仍默认绑定当前源码。最终有效兼容回归目录为 `results/tables/hard_vote_candidate_mode_compat_precommit_v3`；其中 27 个轨迹 NPZ、27 个指标 JSON 和两份汇总 CSV 与 `cc2e026` 干净结果均字节级一致。该回归只证明旧策略未被改变，不构成新的性能实验。

### 6.6 候选态 logit 策略诊断矩阵

本轮不重新训练，而是从 6.3 节冻结的 session 0 OOF logits 构造第 7.4 节八种候选策略。最终有效预提交运行命令为：

```powershell
python BCI_Competition/code/eval/run_candidate_logit_matrix.py `
  --input-root BCI_Competition/results/tables/s01_s09_epoch50_causal_single_window_oof_clean_ba35de6_v2 `
  --output-root BCI_Competition/results/tables/s01_s09_epoch50_causal_candidate_logit_matrix_precommit_v2
```

运行状态为 `PRECOMMIT_DIAGNOSTIC_MATRIX`，Subject 1–9、seed 42/43/44 全部完成，只加载 session 0。九被试在每个 seed 内等权宏平均，再对三个配对 seed 求均值的主要结果如下；单窗两状态基线仅作为已有参考，不属于八个候选 cell：

| 策略 | 正确事件率 | 事件触发率 | 触发后分类准确率 | IDLE 误指令/分钟 | 正确事件平均延迟/s |
|---|---:|---:|---:|---:|---:|
| 单窗两状态参考 | 0.2257 | 0.5804 | 0.3880 | 7.048 | 1.724 |
| `raw_current` | 0.2739 | 0.5082 | 0.5238 | 1.821 | 2.067 |
| `stage1_ewma` | 0.2790 | 0.4561 | 0.5810 | 1.357 | 2.291 |
| `stage2_mean` | 0.2579 | 0.4171 | 0.5922 | 1.294 | 2.341 |
| `dual_ewma` | 0.2833 | 0.4073 | 0.6515 | 1.184 | 2.582 |
| `dual_ewma_drop_abort` | 0.2816 | 0.4037 | 0.6545 | 1.085 | 2.589 |
| `rolling_stable` | 0.2604 | 0.3370 | 0.7138 | 1.219 | 2.840 |
| `probability_curvature` | 0.2206 | 0.2830 | 0.7063 | 1.085 | 2.840 |
| `dual_ewma_high_precision` | 0.2299 | 0.2904 | 0.7597 | 1.161 | 2.811 |

结果验证了三类预期取舍，但不能据此挑选最终策略。相对直接单窗参考，候选态普遍大幅降低 IDLE 误指令并提高触发后的四分类纯度，代价是触发率下降和延迟增加；`dual_ewma_drop_abort` 相比 `dual_ewma` 进一步降低误指令，但正确事件率也略降；稳定性、曲率和高精度条件继续提高已触发事件的类别准确率，同时产生更多 MISS。表中数值来自用于探索参数的同一 OOF 矩阵，因此只能描述机制，不能把数值最高的 cell 称为无偏“最优工作点”。

运行器另存了不读取标签或事件的 111,060 个模型-窗口分数尺度摘要：Stage 1 Task 概率中位数为 0.6708，Stage 2 top-1 概率中位数为 0.5463；segment 内一阶差分有 110,325 项，二阶差分有 109,590 项。每个 seed 的 NPZ 逐策略保存分数、差分、曲率、四项证据、候选年龄、状态和输出，JSON 保存完整事件及候选诊断指标。`precommit_v1` 生成后又补强了计数 dtype、源码闭包和运行身份冻结，已由 v2 取代，不得作为最终证据目录。

### 6.7 EEGNet 隐藏特征提取预检

隐藏特征策略没有直接修改历史模型源码，而是由新入口验证 `LogitAdapter` 包装结构后调用底层 EEGNet 的 `return_features=True`。特征固定为 `block2` 输出展平后、线性分类头之前的 240 维向量；模型处于 evaluation 模式，因此 dropout 不参与随机丢弃。

提交 `8737c75` 后，Subject 1、seed 42 已从干净工作树用 RTX 5070 Laptop GPU 重跑至 `results/tables/s01_seed42_epoch50_feature_preflight_clean_8737c75_v1`。六个 fold 各自使用匹配 run 的 Stage 1/2 checkpoint，共 12 个作业和 4,356 个连续窗口。包装层 logit、底层特征接口 logit、训练时保存的 epoch 50 验证 OOF logit以及 6.3 节冻结连续 logit的最大绝对误差均为 0；两组特征形状均为 `(4356,240)`，全部为有限 `float32`。

该预检不含决策和指标选择。Stage 1/2 是不同网络，各 fold 也独立训练并使用各自训练 run 的标准化统计，因此隐藏特征不属于一个跨 Stage、跨 fold 的全局统一坐标系。后续直接距离只允许在同一 Stage、同一 run 和连续 segment 内计算。

### 6.8 S1/seed42 隐藏特征门控 pilot

首批机制 pilot 以 6.6 节的 `dual_ewma_drop_abort` 为共同 logit 骨架，只用 Stage 2 的单位化隐藏特征变化否决类别提交。阈值来自 S1/seed42 OOF 的无标签候选局部尺度观察；所有八个 cell 并列报告，因此结果只能用于判断机制是否值得扩展，不能用于宣布最终策略。

| 策略 | 正确事件率 | 事件触发率 | 触发后分类准确率 | IDLE 误指令/分钟 | 正确事件平均延迟/s |
|---|---:|---:|---:|---:|---:|
| `logit_only_reference` | 0.3480 | 0.5128 | 0.6786 | 1.395 | 2.493 |
| `velocity_loose` | 0.3370 | 0.4945 | 0.6815 | 1.448 | 2.595 |
| `velocity_strict` | 0.2821 | 0.3590 | 0.7857 | 1.555 | 2.746 |
| `velocity_consecutive` | 0.2967 | 0.4176 | 0.7105 | 1.985 | 2.829 |
| `prototype_loose` | 0.3443 | 0.4982 | 0.6912 | 1.502 | 2.655 |
| `prototype_strict` | 0.2894 | 0.3700 | 0.7822 | 1.448 | 2.784 |
| `acceleration_loose` | 0.3004 | 0.4249 | 0.7069 | 1.985 | 2.794 |
| `acceleration_strict` | 0.2454 | 0.3004 | 0.8171 | 1.716 | 2.972 |

在这一名被试和一个 seed 上，特征门控没有提高正确事件率。严格门控提高了触发后分类准确率，但同时增加 MISS、候选超时和延迟；更严格的提交条件也不保证 IDLE 误指令率单调下降，因为延迟提交会改变进入 `WAIT_IDLE`、复位和后续重开的整条轨迹。基于该 pilot，当前不立即批量提取 9 被试×3 seed 特征；是否扩展应先由本轮独立审核和用户对“更高类别纯度换取更高漏检”的系统目标判断决定。

### 6.9 MI 提交与复位独立析因矩阵

本轮不重新训练，复用 6.3 节九被试、三 seed 的 session 0 OOF logits，把 Stage 2 top-1 提交阈值 `0.55/0.625/0.70` 与 Stage 1 `WAIT_IDLE` 复位 profile `0.20/0.30/0.40 × 连续1/2窗` 做成 18 个 cell。其余条件固定为 `dual_ewma_drop_abort`。最终有效预提交结果目录为 `results/tables/s01_s09_epoch50_causal_commit_reset_matrix_precommit_v3`，状态是 `PRECOMMIT_DIAGNOSTIC_MATRIX`。v1 被严格配置类型检查取代；v2 又因漏记传递调用的输入子清单验证器源码哈希而被 v3 取代。三版数值一致，但后续只引用源码闭环完整的 v3。该矩阵未访问测试 session，也不得从同一矩阵选择 cell 后声称无偏性能。

固定 `r030_l2` 复位规则时，只提高 Stage 2 top-1 提交阈值：

| cell | 正确事件率 | 事件触发率 | 触发后分类准确率 | IDLE误指令/分钟 | 中位延迟/s |
|---|---:|---:|---:|---:|---:|
| `c055_r030_l2` | 0.2947 | 0.4229 | 0.6547 | 1.1693 | 2.593 |
| `c0625_r030_l2` | 0.2666 | 0.3574 | 0.6972 | 1.1376 | 2.633 |
| `c070_r030_l2` | 0.2382 | 0.3008 | 0.7551 | 1.0701 | 2.826 |

固定 `c055` 提交规则时，只改变复位阈值和连续确认：

| 复位 profile | 正确事件率 | FAR | 整事件锁定率 | 提前复位率 |
|---|---:|---:|---:|---:|
| `r020_l1` | 0.2816 | 1.0848 | 0.2566 | 0.0136 |
| `r020_l2` | 0.2641 | 1.0097 | 0.3029 | 0.0054 |
| `r030_l1` | 0.3162 | 1.2720 | 0.1750 | 0.0286 |
| `r030_l2` | 0.2947 | 1.1693 | 0.2236 | 0.0098 |
| `r040_l1` | 0.3362 | 1.4593 | 0.1223 | 0.0625 |
| `r040_l2` | 0.3247 | 1.3491 | 0.1537 | 0.0241 |

六个固定复位 profile 内，提交阈值从 `0.55` 增至 `0.625/0.70` 时 FAR 均单调下降，同时触发率和正确事件率下降、触发后分类准确率上升。固定提交阈值时，放宽复位阈值会减少整事件锁定并提高正确事件率，但增加 FAR 和提前复位；连续两窗确认会压低 FAR 和提前复位，同时重新增加锁定和 MISS。这证明提交和复位必须分别评价，不能继续用一个共享门槛解释两种作用。

所有 18 个 cell 的 FAR 均已独立拆成两项且逐项相加回正式 FAR；“MI 内打开候选、MI 后提交”占主要部分。例如回归锚点 `c055_r020_l1` 的总 FAR 为 `1.0848`，其中延迟溢出为 `0.7221/min`，其他 IDLE 误指令为 `0.3627/min`。该锚点在 9 被试×3 seed 的 27 组正式事件指标和决策库存哈希上逐项复现 6.6 节 `dual_ewma_drop_abort`。这些结果用于解释机制，尚未冻结最终工作点。
