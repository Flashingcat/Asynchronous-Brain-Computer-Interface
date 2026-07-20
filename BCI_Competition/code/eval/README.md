# 同口径验证与测试

验证和测试共用 `session_evaluator.py`。正式口径默认为 `continuous`；`pure` 仅用于兼容旧窗口评估。

首次使用或数据契约更新后，重新生成连续窗口：

```powershell
python BCI_Competition/code/preprocessing/build_oof_windows.py --subjects all
```

生成逐运行留一验证检查点并评估：

```powershell
python BCI_Competition/code/train/train_hierarchical_oof.py --all-subjects --run-mode validation
python BCI_Competition/code/eval/evaluate_validation_oof.py --algorithm candidate
```

锁定模型和策略后，训练最终模型并评估测试会话：

```powershell
python BCI_Competition/code/train/train_hierarchical_oof.py --all-subjects --run-mode final
python BCI_Competition/code/eval/evaluate_test_session.py --algorithm candidate
```

两个评估入口共享全部策略参数，并完整报告窗口指标、事件正确/错类/漏检、误触发、额外指令和延迟，不生成综合分。
