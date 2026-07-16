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

以上当前入口会新建显式记录伪迹/segment 合同的 v2 bundle 与 v2 checkpoint 目录。第 6 节记录的是本轮此前已完成并由精确 manifest/source 哈希绑定的历史 v1 产物；v1 只用于复核既有结果，不与新建 v2 文件混写。

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

该历史结果按当时的 v1 合同从每个 trial 的 5 个离线 Task 窗口恢复事件。后续审计已新增独立 session0 clean event 真值侧车，并确认九被试的 2328 个事件、逐 run 数量、事件哈希、segment 和在线窗口与历史库存完全相同，所以本节成绩无需重算；旧恢复路径只保留为 `legacy_task_window_reconstruction` 兼容入口。当前既有 v1 bundle 的新运行要求显式真值侧车和 v3 库存合同；未来新建 v2 bundle 对应 v4。

历史 checkpoint 绑定的 `oof_training_bundle.py` SHA 为 `f5a2deb40b64187dcbbce34b5e4c382ebb637177851776713eb96c4e99e80f56`，早于当前 v1/v2 清单合同扩展；当前 reader 的 LF SHA 为 `6c71f5c2647fa5b032087e08bbee2ae60e59400fe15afe27da7b821fd3a17066`。新 runner 不是通用跳过源码门禁，而是同时锁定这两份源码及它们的完整 unified-diff SHA，且仅对精确 v1 bundle 启用 `audited_v1_reader_contract_extension_v1`。回归测试确认 model-facing 定义 AST、fold rows/哈希、store records 不变，并用两个 reader 逐元素比较九被试 37,020 个在线窗口；Subject 1/seed42 真实端到端诊断中 12/12 个 checkpoint 的冻结 OOF logit 重算最大绝对误差为 0，v3 严格指标与旧 v1 逐项一致。

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

### 6.10 正确候选提前提交的延迟来源诊断

本轮固定使用 `c055_r020_l1` 历史锚点，只消费九被试、三 seed 的 session 0 OOF logits。当前有效预提交目录为 `results/tables/s01_s09_epoch50_causal_candidate_latency_diagnostic_precommit_v5`，状态是 `PRECOMMIT_DIAGNOSTIC_ONLY`。v1 缺少 class-label-free 首 crossing 风险；v2 补入该风险和输入身份复核后又进一步修正“救回 MISS”分母；v3 没有绑定最终源码；v4 的元数据没有明确 class-label-free 诊断仍使用真值事件边界，均已作废。锚点正确事件率 `0.2816` 与 6.9 节逐项一致，证明诊断没有改变原决策轨迹。

按“每 seed 内九被试等权，再跨三个配对 seed”汇总，基线 `correct_latency_median_seconds` 的均值为 `2.576s`；它是被试级中位数的等权均值，不是把所有正确事件合并后的 pooled median。各被试中位数的 Stage 1 开门延迟均值为 `1.173s`，候选驻留中位数均值为 `1.157s`；扣除开门窗不计入且随后最少两窗形成的固定 `1.0s` 后，额外候选等待中位数均值只有 `0.157s`。这些聚合后的分位数不能直接逐项相加，但每条事件都已断言满足“总延迟=开门延迟+候选驻留”。因果滤波响应滞后保留在端到端分数时机中，没有减去固定群延迟；真实墙钟计算耗时未测量。

| 事件局部诊断 | `min1`：开门窗可立即提交 | `min2`：开门窗计作第一窗 |
|---|---:|---:|
| truth-aware 正确置信度覆盖率 | 0.2917 | 0.2789 |
| 基线正确事件中存在更早正确证据的比例 | 0.4043 | 0.3997 |
| 正 headroom 中位数 | 0.942s | 0.500s |
| 事件边界内 class-label-free 首 crossing 覆盖率 | 0.4336 | 0.4096 |
| 事件边界内 class-label-free 首 crossing 类别准确率 | 0.5387 | 0.5775 |
| 事件边界内 class-label-free 首 crossing 正确事件率 | 0.2410 | 0.2470 |
| 基线正确事件被错误首 crossing 截断的比例 | 0.1532 | 0.1011 |
| truth oracle 前已有错误 crossing 的比例 | 0.1167 | 0.0786 |

truth-aware 诊断说明模型分数中确实存在提前 `0.5–1.0s` 的正确证据，但它会跳过更早的错误高置信类别。class-label-free 首 crossing 在真值事件边界和 margin 内选择时不读取类别真值，其正确事件率仍低于 `0.2816` 基线，因此不能直接把开门窗纳入并立即提交；由于事件边界仍由真值事后限定，它也不能代表完整在线策略。正确类别的 MI 后溢出事件约占全部可评分事件的 `0.0272`；在其中基线原本为 MISS 的事件里，truth-aware `min1/min2` 分别有约 `0.3680/0.2667` 可在事件内找到正确置信度，但这仍是乐观可救回上限。已经正确或错误触发后又产生的溢出不进入“救回 MISS”分母。

本轮 label-free 统计只在真实基线候选区间和事件局部选择首 crossing，没有对全部 IDLE 候选重放“提前提交后进入 WAIT_IDLE”的后续状态，所以不是完整快速策略的 FAR 或系统成绩。它支持下一轮研究带更高首窗门槛的快速通道，而不支持直接把通用提交条件从两窗改成一窗。

### 6.11 Fast-0、Fast-1 与慢通道兜底矩阵

本轮把 6.10 节的局部 crossing 诊断改成完整状态轨迹：Fast-0 可在开门窗原子提交，Fast-1 只联合开门窗和紧接的下一窗，二者均未提交时继续历史慢通道。三套缓存彼此分离；无快速通道 cell 已逐窗断言完全复现 `c055_r020_l1` 的状态、输出和慢通道分数。复位仍为 Stage 1 Task 概率 `<=0.20` 的单窗确认，候选上限仍是开门后的八窗。

最终有效目录为 `results/tables/s01_s09_epoch50_causal_fast_path_matrix_contractfix_v4`，顶层及九个被试清单均为 `PASS`，共覆盖九被试、三个配对 seed、七个并列 cell，明确记录 `test_session_access=forbidden_and_not_loaded`、`artifact_policy=official_trial_exclusion` 和独立的 `segment_policy`。`precommit_v1` 是外层运行时限过短留下的未完成目录；v2 的数值虽经独立复算无误，但底层形式合同尚未拒绝“第二个及更晚候选窗伪装成 Fast-1”，且 CSV 和 NPZ 的审计字段不完整；v3 修复了这些问题，v4 再补齐顶层伪迹合同传播并绑定最终 PR 候选源码。

| cell | 正确事件率 | 事件触发率 | 触发后分类准确率 | FAR/分钟 | 正确事件中位延迟/s | 快速命令占比 |
|---|---:|---:|---:|---:|---:|---:|
| `anchor_no_fast` | 0.2816 | 0.4037 | 0.6545 | 1.0848 | 2.576 | 0.0000 |
| `f0_balanced` | 0.2759 | 0.4021 | 0.6495 | 1.0472 | 2.527 | 0.0906 |
| `f1_balanced` | 0.2808 | 0.4046 | 0.6527 | 1.0669 | 2.554 | 0.0522 |
| `f01_balanced` | 0.2757 | 0.4018 | 0.6494 | 1.0472 | 2.522 | 0.1053 |
| `f0_strict` | 0.2806 | 0.4043 | 0.6534 | 1.0469 | 2.553 | 0.0438 |
| `f1_strict` | 0.2823 | 0.4051 | 0.6540 | 1.0689 | 2.565 | 0.0263 |
| `f01_strict` | 0.2805 | 0.4043 | 0.6532 | 1.0489 | 2.549 | 0.0503 |

Fast-0 的 `balanced/strict` 快速命令触发后分类准确率分别为 `0.6576/0.8104`，但完整轨迹的正确事件率都没有超过锚点；这说明单窗极高 Stage 2 分数仍会抢先截断部分原本可由慢通道正确提交的事件。Fast-1 在本描述性矩阵中更保守：`strict` 快速命令只占 `2.63%`，其触发后分类准确率为 `0.8554`，完整正确事件率从 `0.2816` 变为 `0.2823`。这个绝对变化很小，且阈值与成绩来自同一 OOF 数据，不能称为已证实提升。路径条件准确率并非每个被试都有定义：`f0_strict` 的 Fast-0 在 seed 42/43/44 分别覆盖 `6/5/6` 个被试，`f1_strict` 的 Fast-1 覆盖 `5/5/6` 个被试，`f01_strict` 的 Fast-1 仅覆盖 `3/5/4` 个被试。

路径归因还表明，`f01_strict` 的 Fast-0/Fast-1 命令占比分别为 `4.38%/0.65%`，两条快速路径的 FAR 分别为 `0.0302/0.0020` 次每有效 IDLE 分钟；其余 `94.97%` 命令仍由慢通道产生。相对锚点，两者都正确且当前更早的事件比例为 `4.14%`，但所有共同正确事件的配对提前量中位数仍是 `0s`，所以不能只看总体延迟中位数宣称普遍提速。

当前最重要的机制结论是：快速通道确实能提前一小部分高置信事件，但 Fast-0 的抢先错误会抵消收益；Fast-1 严格门槛更安全，却只覆盖很少命令。下一轮若要冻结实际方案，应先决定是否取消或重做 Fast-0，再把候选方案与复位轴分开，并使用不参与本轮阈值探索的数据做确认。

严格审核修复后，v3 与 v2 的顶层 summary 完全相等，27 份逐 seed 指标 JSON 字节相同，旧 NPZ 中 4,671 个数组逐元素相同；v3 只新增逐窗类别/复位审计数组及 CSV 有效分母。全部 189 组实际轨迹中的 Fast-1 原本就只发生在候选开门后的紧接下一窗，因此形式约束补强没有改变任何既有成绩。

### 6.12 LD-GRU 的 Stage 1 三 token 配对消融

本实验回答一个很窄的问题：在候选区间已经由固定 Stage 1 规则打开、撤销和超时的前提下，LD-GRU 是否还需要看到三项 Stage 1 数值。三项分别是原始 Task-IDLE logit margin、Stage 1 EWMA Task 概率及其一阶差分。`full` 保留三项；`mask_stage1` 在训练集统计量完成标准化后把这三维固定为 0，即替换成训练被试均值。两组的候选区间、Stage 2 输入、候选年龄、模型结构、训练划分和随机种子均相同，所以它不是“有无 Stage 1 状态机”的比较。

每个外层被试只用于最终评价；其余八名被试执行内层 LOSO 选择 checkpoint 和最终训练 epoch。九被试、seed 42/43/44 严格配对，仍只消费 session 0 OOF 分数，不读取 session 1。下表先在每个 seed 内对九被试等权，再对三个 seed 求均值；差值为 `mask_stage1 - full`。

| LD-GRU 版本 | token | 正确事件率 | 事件触发率 | 触发后分类准确率 | MISS率 | FAR/分钟 | 正确事件平均延迟/s |
|---|---|---:|---:|---:|---:|---:|---:|
| `stop_only` | full | 0.3041 | 0.5539 | 0.5323 | 0.4461 | 0.8498 | 2.3471 |
| `stop_only` | mask | 0.2914 | 0.5536 | 0.5087 | 0.4464 | 0.8712 | 2.3491 |
| `stop_only` | 配对 mask-full | -0.0127 | -0.0004 | -0.0184 | +0.0004 | +0.0214 | +0.0053 |
| `stop_residual` | full | 0.3059 | 0.5546 | 0.5340 | 0.4454 | 0.8564 | 2.3727 |
| `stop_residual` | mask | 0.2916 | 0.5520 | 0.5109 | 0.4480 | 0.8809 | 2.3558 |
| `stop_residual` | 配对 mask-full | -0.0143 | -0.0026 | -0.0173 | +0.0026 | +0.0245 | -0.0141 |

两种 LD-GRU 头得到相同方向的结论：保留三项 Stage 1 token 使正确事件率高约 1.3–1.4 个百分点、配对触发后分类准确率高约 1.7–1.8 个百分点，并略微降低 FAR；对触发率、MISS 和延迟的影响很小。因此 Stage 1 数值是有用的辅助证据，但不是当前策略性能的核心来源。两个 LD-GRU 版本的正确事件率仍低于 `n5_k3` 的 0.3853，不过 FAR 更低。表中 `full/mask` 是各自正式宏平均；“配对 mask-full”严格按相同被试与 seed 先作差，因此在存在无有效分母的路径指标上不要求等于两列展示均值的简单相减。

正式 `full` 与 `mask_stage1` 结果分别位于远端 `full_token_combo_remote_55615c5/canonical_full_v1` 和 `mask_stage1_remote_bb527dc/canonical_full_v1`；配对比较清单为 `stage1_token_ablation_compare_20260716/comparison_manifest.json`，状态为 `PASS`。并行重建还与原串行 `full` 运行完成 54/54 个最终作业的指标和轨迹逐数组一致性检查。对应源码快照分别为 `55615c5` 和 `bb527dc`。

### 6.13 GRU 掌管全部决策证据

本实验与 6.12 节不同：不再由手写 Stage 1 规则打开、维持、撤销候选区间，也不存在候选超时。GRU 在每个连续有效窗口上运行，同时学习 Task/IDLE 证据、四类 MI 条件分布和回到 IDLE 的证据。系统只保留 `READY/WAIT_IDLE` 两状态安全外壳：`READY` 达到提交阈值时输出一次 MI，随后进入 `WAIT_IDLE`；达到 GRU 的 IDLE 复位阈值后才能回到 `READY`。这个外壳只阻止重复指令，不替代 GRU 做识别。

连续 token 共 10 维：Stage 1 原始 Task-IDLE margin 及其差分、四类中心化 Stage 2 logits 及各自差分；变化量和 GRU 隐状态均在干净 segment 边界重置。模型为 10 维输入、16 维隐层、1 层 GRU，共 1,429 个参数，输出一个 Task logit 和四个条件类别 logit。联合概率定义为 `P(IDLE)=1-P(Task)`、`P(MI_c)=P(Task)×P(c|Task)`；训练使用平衡的 Task/IDLE 二元损失与四类条件损失。同一 MI 事件内，每个满足 overlap、margin 和 offset 规则、并在“此前尚未提交”这一反事实条件下可合法首次提交的窗口都标为对应 MI 类，并非只标时间上最早的一个窗口。提交阈值和复位阈值均在 `0.3..0.9` 网格内由内层被试 LOSO 选择，外层被试不参与训练和阈值选择。

| 决策策略 | 正确事件率 | 事件触发率 | 触发后分类准确率 | MISS率 | FAR/分钟 | 平均延迟/s | 中位延迟/s | 过早指令数/被试-seed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 全控制 GRU | 0.3592 | 0.8375 | 0.4322 | 0.1625 | 0.6316 | 1.7746 | 1.7237 | 16.26 |
| `n5_k3` 硬投票参考 | 0.3853 | 0.7229 | 0.5232 | 0.2771 | 1.2442 | 2.4613 | 2.4953 | 3.04 |
| `dual_ewma_drop_abort` 参考 | 0.2816 | 0.4037 | 0.6545 | 0.5963 | 1.0848 | 2.5887 | 2.5764 | — |

全控制 GRU 相比慢速 `dual_ewma_drop_abort` 同时提高正确事件率、降低 MISS、FAR 和延迟；相比 `n5_k3`，它漏检更少、FAR 约减半且平均提前约 0.69 秒，但正确事件率低 2.6 个百分点，触发后分类准确率也明显更低。关键问题是它产生了更多“允许匹配区间之前”的过早指令：高召回策略会抢先锁定错误类别或过早进入 `WAIT_IDLE`。因此这一版证明了学习型全控制方向可行，但还不能作为最终策略；下一步应优先约束提前提交，而不是简单提高统一门槛把召回优势一起抹掉。

远端源码快照为 `260b0de`；精确全流程预检在 `full_control_remote_260b0de/preflight_v3_s01_seed42` 两次同命令重跑得到相同最终产物哈希，正式九被试×三 seed 结果在 `full_control_remote_260b0de/canonical_full_v1`，顶层清单为 `PASS`。以上仍是 session 0 OOF 的嵌套跨被试决策层评价，不是 session 1 独立测试成绩，也不包含真实设备墙钟计算耗时。

两组远端正式结果及失败即停审计证据已下载到本地 `remote_transfer_ld_gru/remote_results_20260716`。归档 `completed_full_control_and_mask_20260716.tar.gz` 的 SHA256 为 `A5A7FFBA88F3CEDD5166C289B81610365BBBA5FEE8BA67BB579CE60C2D0830DF`；归档 `completed_full_token_and_pair_compare_20260716.tar.gz` 的 SHA256 为 `806E2BF1CE9F73E466D07D4B67FAE2B4DF6822E838A6168F95D5D3BE98B2E26A`。
