# BCI Competition Async Decoding

本项目包含两套用途不同的流程：

- **正式研究流程**：原生 250 Hz、完整 run 留一验证、因果在线回放，用于本项目的可审计评估研究；
- **legacy 演示流程**：原仓库的 128 Hz 窗口分类脚本，仅保留作兼容示例，不得用于正式结果。

正式流程先判断 EEG 是否为 `idle/task`，再对 Task 窗口进行四类运动想象分类；两个 Stage 在在线回放中逐窗同时计算，决策状态机只控制是否发出指令。

当前主流程使用两阶段方法：

```text
Stage 1: idle vs task 二分类
Stage 2: left_hand / right_hand / feet / tongue 运动想象分类
Final: 合成 idle / left_hand / right_hand / feet / tongue 五类输出
```

## 目录结构

```text
BCI_Competition/
  code/
    datasets/
      download_bnci2014001.py
      download_zhou2014.py
    preprocessing/
      build_async_windows.py
      build_zhou2014_windows.py
    train/
      train_eegnet_async.py
    eval/
      evaluate_async.py
    models/
      model_factory.py
      models/
        eegnet.py
        shallowconvnet.py
        deepcnn.py
        conformer.py
        deformer.py
        DBConfrmer.py
  data/
    public/
      BNCI2014001/
      Zhou2014/
    processed/
  results/
    checkpoints/
    tables/
  requirements.txt
```

## 环境安装

原仓库 `requirements.txt` 保留旧版 CUDA 11.3 兼容环境，不能用于本项目的 RTX 5070 正式基线。250 Hz OOF 基线固定使用：

```powershell
conda env create -f BCI_Competition/environment-bciml-repro.yml
conda activate bciml-repro
$env:PYTHONNOUSERSITE=1
```

本轮实测身份为 Python 3.10.20、PyTorch 2.11.0+cu128、CUDA runtime 12.8、cuDNN 9.19、NumPy 1.26.4、MNE 1.11.0 和 MOABB 1.2.0。正式训练合同还会绑定实际 GPU 名称和计算能力；不同执行指纹不得续跑同一输出目录。

环境验收：

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
```

## 正式 250 Hz 评估流程

正式协议与实验台账是本分支的主入口：

- [异步运动想象评估协议](docs/evaluation_protocol.md)
- [EEGNet 250 Hz OOF 基线与描述性策略实验](docs/baseline_training_matrix.md)

固定边界如下：

- BNCI2014001 原始 MAT 的 MI 为 trial 开始后 `2–6 s`；MOABB cue annotation 已位于该 MI 起点，不能再次加 2 秒；
- 22 个 EEG 通道、250 Hz、2 秒窗口、0.5 秒步长；
- 官方伪迹 trial 整段删除，删除两侧不得拼接，一个 run 保留为多个独立 segment；
- 每名被试只在 session 0 内做六折留一完整 run 的 OOF 训练/验证；session 1 在工作点冻结前保持封存；
- 在线回放消费每个有效 segment 的全部连续窗口，包括 MI/IDLE 边界窗口。

以下命令以已经下载好的 MAT 目录为输入，先构建 Subject 1 的完整正式数据链：

```powershell
conda activate bciml-repro
$env:PYTHONNOUSERSITE=1
$env:BNCI2014001_ROOT='D:\path\to\001-2014'

python BCI_Competition/code/preprocessing/build_protocol_index.py --data-root $env:BNCI2014001_ROOT --subjects 1
python BCI_Competition/code/preprocessing/build_offline_view.py --subjects 1
python BCI_Competition/code/preprocessing/build_validation_folds.py --subjects 1
python BCI_Competition/code/preprocessing/build_signal_store.py --data-root $env:BNCI2014001_ROOT --subjects 1
python BCI_Competition/code/preprocessing/build_causal_filter_store.py --subjects 1
python BCI_Competition/code/preprocessing/build_zero_phase_filter_store.py --subjects 1
python BCI_Competition/code/preprocessing/build_fold_normalization.py --subjects 1
python BCI_Competition/code/preprocessing/build_oof_training_bundle.py --subjects 1
```

训练前先运行真实 GPU 全流程预检，再启动固定 50 epoch 的 OOF 矩阵：

```powershell
$bundle = 'BCI_Competition/data/processed/bnci2014001_s01_oof_train_session0_native250_v2/manifest.json'
$checkpoints = 'BCI_Competition/results/checkpoints/eegnet_oof_native250_v2'
$inventoryDir = 'BCI_Competition/results/tables/online_inventory_contracts_v2'
$inventory = "$inventoryDir/bnci2014001_s01_session0_causal_online_v2.json"

python BCI_Competition/code/train/preflight_eegnet_oof.py `
  --training-bundle $bundle `
  --output-root BCI_Competition/results/checkpoints/eegnet_oof_preflight_native250_v2
python BCI_Competition/code/train/train_eegnet_oof.py `
  --subject 1 --training-bundle $bundle --output-root $checkpoints
python BCI_Competition/code/eval/freeze_online_inventory_contracts.py `
  --subjects 1 --bundle-root BCI_Competition/data/processed `
  --output-dir $inventoryDir --write-missing
python BCI_Competition/code/eval/run_epoch50_online_oof.py `
  --subject 1 --bundle-manifest $bundle --checkpoint-root $checkpoints `
  --inventory-contract $inventory `
  --output-root BCI_Competition/results/tables/s01_epoch50_causal_single_window_oof_v2
```

新建流程统一使用显式伪迹合同的 v2 身份；仓库 `config/evaluation/*_v1.json` 只绑定既有 v1 bundle 与历史 checkpoint，不能拿来验证新生成的 bundle。

完整回归必须显式提供真实数据根目录：

```powershell
$env:PYTHONNOUSERSITE=1
$env:BNCI2014001_ROOT='D:\path\to\001-2014'
python -m unittest discover -s BCI_Competition/tests -p 'test_*.py' -v
```

正式 checkpoint 通过合同哈希绑定对应的 session0-only bundle；fold 专属 mean/std 位于该 bundle 中，checkpoint 与 bundle 必须成套保存和使用。`results/checkpoints/*` 与 `results/tables/*` 默认被 Git 忽略，PR 只包含源码、冻结配置、测试和结论文档；数值表及 manifest 是本地可复核产物，不随 PR 上传。

## 下载数据集

### BNCI2014001

下载全部 subject：

```bat
python BCI_Competition\code\datasets\download_bnci2014001.py
```

只下载部分 subject：

```bat
python BCI_Competition\code\datasets\download_bnci2014001.py --subjects 1 2 3
```

数据会缓存到：

```text
BCI_Competition\data\public\BNCI2014001
```

### Zhou2014

MOABB 中该数据集类名是 `Zhou2016`，本项目按实验命名保存到 `Zhou2014` 目录。

下载全部 subject：

```bat
python BCI_Competition\code\datasets\download_zhou2014.py
```

只下载部分 subject：

```bat
python BCI_Competition\code\datasets\download_zhou2014.py --subjects 1 2
```

数据会缓存到：

```text
BCI_Competition\data\public\Zhou2014
```

## Legacy 128 Hz 数据预处理（非正式）

从本节开始均为原仓库兼容示例。它会直接按原始 session 生成 train/test 窗口，不满足本项目的 run-level OOF、session 1 封存和严格在线回放要求。

当前预处理脚本用于 BNCI2014001 subject 1：

```bat
python BCI_Competition\code\preprocessing\build_async_windows.py
```

生成文件：

```text
BCI_Competition\data\processed\bnci2014001_subject01_async.npz
BCI_Competition\data\processed\bnci2014001_subject01_async.json
```

`.npz` 中包含：

```text
X      EEG 窗口数据，形状为 (n_windows, 22, 256)
y      标签，0 idle / 1 left_hand / 2 right_hand / 3 feet / 4 tongue
split  划分标记，0 train / 1 validation / 2 test
```

当前窗口规则：

```text
采样率: 128 Hz
窗口长度: 2.0 s
滑窗步长: 0.5 s
任务态: MOABB cue onset 到 cue onset + 4s；对应原始 MAT trial 的 2-6s
idle: 所有不与任务态重叠的 2s 窗口
跨边界窗口: 丢弃，不参与训练
```

训练/验证/测试划分：

```text
train session 中 1 个 run -> split = 1 validation
train session 其余 run   -> split = 0 train
test session 全部 run     -> split = 2 test
```

默认使用 train session 的最后一个 run 做验证集：

```bat
python BCI_Competition\code\preprocessing\build_async_windows.py --val-run-index -1
```

### Zhou2014/Zhou2016 预处理

Zhou 数据集预处理入口：

```bat
python BCI_Competition\code\preprocessing\build_zhou2014_windows.py
```

默认处理全部 subject，并且每个 subject 使用 1 个 run 作为验证集，其余 run 作为训练集：

```text
split = 0  train
split = 1  validation
```

默认验证 run 是每个 subject 的最后一个 run：

```bat
python BCI_Competition\code\preprocessing\build_zhou2014_windows.py --val-run-index -1
```

也可以指定第 1 个 run 作为验证集：

```bat
python BCI_Competition\code\preprocessing\build_zhou2014_windows.py --val-run-index 0
```

只处理部分 subject：

```bat
python BCI_Competition\code\preprocessing\build_zhou2014_windows.py --subjects 1 2 --val-run-index -1
```

生成文件：

```text
BCI_Competition\data\processed\zhou2014_async.npz
BCI_Competition\data\processed\zhou2014_async.json
```

`.npz` 中包含：

```text
X        EEG 窗口数据
y        标签，0 idle / 1 left_hand / 2 right_hand / 3 feet
split    0 train / 1 validation
subject  每个窗口对应的 subject id
```

## Legacy 128 Hz 模型训练（非正式）

训练入口：

```bat
python BCI_Competition\code\train\train_eegnet_async.py --model eegnet
```

可选模型来自：

```text
BCI_Competition\code\models\models
```

当前支持：

```text
eegnet
shallowconvnet
deepcnn
conformer
deformer
dbconformer
```

示例：

```bat
python BCI_Competition\code\train\train_eegnet_async.py --model eegnet --binary-epochs 30 --mi-epochs 30 --batch-size 32 --seed 42
```

两阶段训练过程：

```text
Stage 1:
  原始标签 y == 0  -> idle
  原始标签 y > 0   -> task
  训练一个二分类网络

Stage 2:
  只使用 y > 0 的任务态窗口
  将 1/2/3/4 映射成 0/1/2/3
  训练一个四分类 MI 网络

推理:
  Stage 1 判为 idle -> 最终输出 0
  Stage 1 判为 task -> 进入 Stage 2，再映射回 1/2/3/4
```

训练输出：

```text
BCI_Competition\results\checkpoints\hierarchical_<model>_bnci2014001_async_subject01.pt
BCI_Competition\results\tables\hierarchical_<model>_async_predictions.npz
BCI_Competition\results\tables\hierarchical_<model>_async_metrics.json
BCI_Competition\results\tables\hierarchical_<model>_run_manifest.json
```

其中 checkpoint 的一个 `.pt` 文件内同时保存两套网络参数：

```text
binary_state_dict  Stage 1 idle/task 网络
mi_state_dict      Stage 2 MI 四分类网络
```

## Legacy 128 Hz 窗口评估（非正式）

评估入口：

```bat
python BCI_Competition\code\eval\evaluate_async.py --model eegnet
```

评估脚本会读取：

```text
BCI_Competition\results\tables\hierarchical_eegnet_async_predictions.npz
```

并重新生成：

```text
BCI_Competition\results\tables\hierarchical_eegnet_async_metrics.json
```

指标分三组：

```text
final_5class:
  最终 idle / left_hand / right_hand / feet / tongue 五分类指标

stage1_binary:
  Stage 1 idle vs task 二分类指标

stage2_mi_on_true_task_windows:
  只在真实任务态窗口上评估 Stage 2 的四分类 MI 指标
```

## Legacy 128 Hz 完整示例（非正式）

```bat
conda activate BCI2026
set PYTHONNOUSERSITE=1

python BCI_Competition\code\datasets\download_bnci2014001.py --subjects 1
python BCI_Competition\code\preprocessing\build_async_windows.py

python BCI_Competition\code\train\train_eegnet_async.py --model eegnet --binary-epochs 30 --mi-epochs 30 --batch-size 32 --seed 42
python BCI_Competition\code\eval\evaluate_async.py --model eegnet
```

也可以在项目目录内运行：

```bat
cd BCI_Competition
python code\datasets\download_bnci2014001.py --subjects 1
python code\preprocessing\build_async_windows.py
python code\train\train_eegnet_async.py --model eegnet
python code\eval\evaluate_async.py --model eegnet
```
