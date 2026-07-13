# BCI Competition Async Decoding

本项目用于构建异步脑机接口识别流程：先判断当前 EEG 窗口是 `idle` 还是 `task`，如果是 `task`，再进行运动想象分类。

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

## 数据预处理

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
split  划分标记，0 train / 1 test
```

当前窗口规则：

```text
采样率: 128 Hz
窗口长度: 2.0 s
滑窗步长: 0.5 s
任务态: cue onset 到 cue onset + 4s，也就是 BNCI2014001 trial 的 3-7s
idle: 所有不与任务态重叠的 2s 窗口
跨边界窗口: 丢弃，不参与训练
```

训练/测试划分使用原始 session：

```text
train session -> split = 0
test session  -> split = 1
```

## 选择模型并训练

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

## Eval 评估

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

## 常用完整流程

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
