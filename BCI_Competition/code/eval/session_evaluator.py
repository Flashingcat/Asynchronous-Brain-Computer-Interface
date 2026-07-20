"""验证集和测试集共用的连续 session 评估核心。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .algorithms.argmax import predict as argmax_predict
from .algorithms.candidate import CandidateConfig, commands as candidate_commands
from .algorithms.fast_path import commands as fast_path_commands
from .algorithms.feature_gate import commands as feature_gate_commands
from .algorithms.hard_vote import commands as hard_vote_commands
from .metric import classification_metrics, continuous_event_metrics, legacy_event_metrics, policy_diagnostics
from models.model_factory import build_model


def add_policy_args(parser: argparse.ArgumentParser) -> None:
    """为验证和测试命令注册完全相同的在线决策参数。"""
    parser.add_argument("--algorithm", choices=("argmax", "hard_vote", "candidate", "fast", "feature"), default="candidate")
    parser.add_argument("--window-mode", choices=("continuous", "pure"), default="continuous")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vote-windows", type=int, default=3)
    parser.add_argument("--vote-threshold", type=int, default=2)
    parser.add_argument("--task-on", type=float, default=0.60)
    parser.add_argument("--task-hold", type=float, default=0.50)
    parser.add_argument("--idle-reset", type=float, default=0.40)
    parser.add_argument("--min-windows", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=4)
    parser.add_argument("--top-probability", type=float, default=0.50)
    parser.add_argument("--probability-gap", type=float, default=0.10)
    parser.add_argument("--stable-windows", type=int, default=2)
    parser.add_argument("--stage2-aggregation", choices=("current", "candidate_mean", "candidate_ewma"), default="candidate_mean")
    parser.add_argument("--stage2-alpha", type=float, default=0.5)
    parser.add_argument("--fast-probability", type=float, default=0.75)
    parser.add_argument("--fast-gap", type=float, default=0.25)
    parser.add_argument("--feature-max-change", type=float, default=0.50)
    parser.add_argument("--feature-consecutive", type=int, default=2)


def policy_config(args: argparse.Namespace) -> CandidateConfig:
    return CandidateConfig(
        args.task_on, args.task_hold, args.idle_reset, args.min_windows, args.max_windows,
        args.top_probability, args.probability_gap, args.stable_windows,
        args.stage2_aggregation, args.stage2_alpha, False, False,
        args.fast_probability, args.fast_gap, False,
        args.feature_max_change, args.feature_consecutive,
    )


def policy_identity(args: argparse.Namespace) -> dict:
    keys = (
        "algorithm", "window_mode", "vote_windows", "vote_threshold", "task_on", "task_hold",
        "idle_reset", "min_windows", "max_windows", "top_probability", "probability_gap",
        "stable_windows", "stage2_aggregation", "stage2_alpha", "fast_probability", "fast_gap",
        "feature_max_change", "feature_consecutive",
    )
    return {key: getattr(args, key) for key in keys}


def load_session_data(path: Path, subject: int, session: int, window_mode: str, run: int | None = None) -> dict:
    """按受试者、session 和可选 run 读取有序窗口及对应事件。"""
    with np.load(path) as source:
        required = {
            "X", "y", "subject", "session", "split", "run", "segment", "decision_sample", "is_pure",
            "event_subject", "event_run", "event_label", "event_start", "event_stop", "sampling_rate",
        }
        missing = required.difference(source.files)
        if missing:
            raise RuntimeError(f"data is missing {sorted(missing)}")
        if session == 0 and "event_session" not in source.files:
            raise RuntimeError("training-session evaluation requires event_session; rebuild the dataset")

        split = 0 if session == 0 else 2
        mask = (source["subject"] == subject) & (source["session"] == session) & (source["split"] == split)
        if run is not None:
            mask &= source["run"] == run
        if window_mode == "pure":
            mask &= source["is_pure"]
        indices = np.flatnonzero(mask)
        if not len(indices):
            raise RuntimeError(f"subject {subject} session {session} has no selected windows")

        # 原数组按 session/run/segment/time 写入；这里显式校验，避免状态机吃到乱序窗口。
        order = np.lexsort((source["decision_sample"][indices], source["segment"][indices], source["run"][indices]))
        if not np.array_equal(order, np.arange(len(indices))):
            raise RuntimeError("selected windows are not ordered by run, segment, and decision_sample")
        runs = source["run"][indices].astype(np.int64)
        segments = source["segment"][indices].astype(np.int64)
        streams = np.zeros(len(indices), dtype=np.int64)
        if len(indices) > 1:
            streams[1:] = np.cumsum((runs[1:] != runs[:-1]) | (segments[1:] != segments[:-1]))

        event_mask = source["event_subject"] == subject
        if "event_session" in source.files:
            event_mask &= source["event_session"] == session
        if run is not None:
            event_mask &= source["event_run"] == run
        return {
            "index": indices,
            "X": source["X"][indices].astype(np.float32),
            "y": source["y"][indices].astype(np.int64),
            "run": runs,
            "streams": streams,
            "decision_sample": source["decision_sample"][indices].astype(np.int64),
            "sampling_rate": int(source["sampling_rate"].item()),
            "events": {
                "run": source["event_run"][event_mask].astype(np.int64),
                "label": source["event_label"][event_mask].astype(np.int64),
                "start": source["event_start"][event_mask].astype(np.int64),
                "stop": source["event_stop"][event_mask].astype(np.int64),
            },
        }


def infer(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch_size: int, need_features: bool) -> tuple[np.ndarray, np.ndarray | None]:
    logits, features = [], []
    model.eval()
    backbone = getattr(model, "model", None)
    if need_features and backbone is None:
        raise RuntimeError("feature policy requires a feature-returning backbone")
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start:start + batch_size]).to(device)
            result = backbone(batch, return_features=True) if need_features else model(batch)
            if need_features:
                result, hidden = result
                features.append(hidden.cpu().numpy())
            logits.append(result.cpu().numpy())
    return np.concatenate(logits), None if not features else np.concatenate(features)


def infer_checkpoint(checkpoint: dict, data: dict, args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    required = {"model", "binary_state_dict", "mi_state_dict", "mean", "std"}
    missing = required.difference(checkpoint)
    if missing:
        raise RuntimeError(f"checkpoint is missing {sorted(missing)}")
    x = ((data["X"] - checkpoint["mean"]) / checkpoint["std"]).astype(np.float32)
    stage1 = build_model(checkpoint["model"], 2, x.shape[1], x.shape[2]).to(device)
    stage2 = build_model(checkpoint["model"], 4, x.shape[1], x.shape[2]).to(device)
    stage1.load_state_dict(checkpoint["binary_state_dict"])
    stage2.load_state_dict(checkpoint["mi_state_dict"])
    stage1_logits, _ = infer(stage1, x, device, args.batch_size, False)
    stage2_logits, features = infer(stage2, x, device, args.batch_size, args.algorithm == "feature")
    return stage1_logits, stage2_logits, features


def run_policy(stage1: np.ndarray, stage2: np.ndarray, streams: np.ndarray, features: np.ndarray | None, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, tuple[str | None, ...]]:
    """从同一组 logits 生成密集分类和在线指令。"""
    dense = argmax_predict(stage1, stage2)
    reasons: tuple[str | None, ...] = tuple([None] * len(dense))
    if args.algorithm == "argmax":
        return dense, np.where(dense == 0, -1, dense), reasons
    if args.algorithm == "hard_vote":
        commands = hard_vote_commands(stage1, stage2, window_count=args.vote_windows, vote_threshold=args.vote_threshold, run_ids=streams)
        return dense, commands, reasons
    if args.algorithm == "feature":
        if features is None:
            raise RuntimeError("feature inference failed")
        output = feature_gate_commands(stage1, stage2, features, policy_config(args), run_ids=streams)
    elif args.algorithm == "fast":
        output = fast_path_commands(stage1, stage2, policy_config(args), run_ids=streams)
    else:
        output = candidate_commands(stage1, stage2, policy_config(args), run_ids=streams)
    return dense, output.commands, output.reasons


def score(data: dict, dense: np.ndarray, commands: np.ndarray, reasons: tuple[str | None, ...], window_mode: str) -> dict:
    """按统一窗口与事件定义生成一份完整报告。"""
    event_report = (
        legacy_event_metrics(data["y"], commands, data["streams"])
        if window_mode == "pure"
        else continuous_event_metrics(
            data["y"], commands, data["run"], data["decision_sample"], data["events"], data["sampling_rate"]
        )
    )
    report = {**classification_metrics(data["y"], dense), **event_report}
    if any(reason is not None for reason in reasons):
        report["diagnostics"] = policy_diagnostics(reasons)
    return report
