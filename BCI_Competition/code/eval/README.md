# 同口径验证与测试

验证和测试共用 `session_evaluator.py`。正式口径默认为 `continuous`；`pure` 仅用于兼容旧窗口评估。
旧 NPZ 和旧检查点没有完整实验身份，不能进入正式评估，必须重新生成和训练。

首次使用或数据契约更新后，重新生成连续窗口：

```powershell
python BCI_Competition/code/preprocessing/build_oof_windows.py --subjects all
```

训练命令会输出稳定的 `experiment_id`，检查点写入 `results/checkpoints/<experiment_id>/`。
生成逐运行留一验证检查点并显式选择该实验目录：

```powershell
python BCI_Competition/code/train/train_hierarchical_oof.py --all-subjects --run-mode validation
python BCI_Competition/code/eval/evaluate_validation_oof.py --experiment-dir BCI_Competition/results/checkpoints/<experiment_id> --algorithm candidate
```

锁定模型和策略后，训练最终模型并评估测试会话：

```powershell
python BCI_Competition/code/train/train_hierarchical_oof.py --all-subjects --run-mode final
python BCI_Competition/code/eval/evaluate_test_session.py --experiment-dir BCI_Competition/results/checkpoints/<experiment_id> --algorithm candidate
```

两个评估入口不再递归扫描全部检查点；混合实验身份、重复受试者或不匹配数据会直接报错。
