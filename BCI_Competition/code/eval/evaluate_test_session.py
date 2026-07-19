"""Final test-session evaluation for final two-stage checkpoints.

Uses only session=1/split=2, never OOF.  It supports one or many final .pt
checkpoints (for example, one per random seed) and evaluates argmax, hard-vote,
candidate, Fast-0/Fast-1, or feature-gated candidate policies.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE, CODE_ROOT = Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from algorithms.argmax import predict as argmax_predict
from algorithms.candidate import CandidateConfig, commands as candidate_commands
from algorithms.fast_path import commands as fast_path_commands
from algorithms.feature_gate import commands as feature_gate_commands
from algorithms.hard_vote import commands as hard_vote_commands
from metric import classification_metrics, event_metrics, policy_diagnostics, seed_summary
from models.model_factory import build_model


PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "bnci2014001_oof_windows.npz"
DEFAULT_PATTERN = PROJECT_ROOT / "results" / "checkpoints" / "**" / "*final.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoints", type=Path, nargs="+", help="one or more final .pt files")
    parser.add_argument("--checkpoint-glob", default=str(DEFAULT_PATTERN), help="used when --checkpoints is omitted")
    parser.add_argument("--algorithm", choices=("argmax", "hard_vote", "candidate", "fast", "feature"), default="candidate")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "test_session_metrics.json")
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
    parser.add_argument("--stage2-alpha", type=float, default=0.5, help="EWMA alpha; used only by candidate_ewma")
    parser.add_argument("--fast-probability", type=float, default=0.75)
    parser.add_argument("--fast-gap", type=float, default=0.25)
    parser.add_argument("--feature-max-change", type=float, default=0.50)
    parser.add_argument("--feature-consecutive", type=int, default=2)
    return parser.parse_args()


def find_checkpoints(args: argparse.Namespace) -> list[Path]:
    paths = args.checkpoints or [Path(item) for item in sorted(glob.glob(args.checkpoint_glob, recursive=True))]
    paths = [path.resolve() for path in paths]
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("no final checkpoint found; pass --checkpoints explicitly")
    return paths


def load_subject_test_data(path: Path, subject: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        required = {"X", "y", "subject", "session", "split", "run"}
        missing = required.difference(data.files)
        if missing:
            raise RuntimeError(f"data file is missing {sorted(missing)}")
        mask = (data["subject"] == subject) & (data["session"] == 1) & (data["split"] == 2)
        if not mask.any():
            raise RuntimeError(f"subject {subject} has no labelled test-session rows")
        return data["X"][mask].astype(np.float32), data["y"][mask].astype(np.int64), data["run"][mask].astype(np.int64)


def infer(model: torch.nn.Module, features: np.ndarray, device: torch.device, batch_size: int, *, need_features: bool) -> tuple[np.ndarray, np.ndarray | None]:
    logits, hidden = [], []
    model.eval()
    backbone = getattr(model, "model", None)
    if need_features and backbone is None:
        raise RuntimeError("feature policy requires a model exposing its backbone")
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start:start + batch_size]).to(device)
            values = backbone(batch, return_features=True) if need_features else model(batch)
            if need_features:
                values, feature_values = values
                hidden.append(feature_values.cpu().numpy())
            logits.append(values.cpu().numpy())
    return np.concatenate(logits), None if not hidden else np.concatenate(hidden)


def config_from_args(args: argparse.Namespace) -> CandidateConfig:
    return CandidateConfig(
        task_on_probability=args.task_on, task_hold_probability=args.task_hold,
        idle_reset_probability=args.idle_reset, min_candidate_windows=args.min_windows,
        max_candidate_windows=args.max_windows, top_probability=args.top_probability,
        probability_gap=args.probability_gap, stable_windows=args.stable_windows,
        stage2_aggregation=args.stage2_aggregation, stage2_alpha=args.stage2_alpha,
        fast0=False, fast1=False,
        fast_probability=args.fast_probability, fast_gap=args.fast_gap,
        feature_gate=False, feature_max_change=args.feature_max_change,
        feature_consecutive=args.feature_consecutive,
    )


def evaluate_checkpoint(path: Path, args: argparse.Namespace, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "subject", "binary_state_dict", "mi_state_dict", "mean", "std"}
    missing = required.difference(checkpoint)
    if missing:
        raise RuntimeError(f"{path.name} is not a final two-stage checkpoint; missing {sorted(missing)}")
    raw_x, y_true, runs = load_subject_test_data(args.data, int(checkpoint["subject"]))
    mean, std = np.asarray(checkpoint["mean"], dtype=np.float32), np.asarray(checkpoint["std"], dtype=np.float32)
    if mean.shape != (1, raw_x.shape[1], 1) or std.shape != mean.shape or np.any(std <= 0):
        raise RuntimeError(f"{path.name} has incompatible normalization statistics")
    x = ((raw_x - mean) / std).astype(np.float32)
    stage1 = build_model(checkpoint["model"], 2, x.shape[1], x.shape[2]).to(device)
    stage2 = build_model(checkpoint["model"], 4, x.shape[1], x.shape[2]).to(device)
    stage1.load_state_dict(checkpoint["binary_state_dict"], strict=True)
    stage2.load_state_dict(checkpoint["mi_state_dict"], strict=True)
    stage1_logits, _ = infer(stage1, x, device, args.batch_size, need_features=False)
    need_features = args.algorithm == "feature"
    stage2_logits, stage2_features = infer(stage2, x, device, args.batch_size, need_features=need_features)
    if args.algorithm == "argmax":
        prediction = argmax_predict(stage1_logits, stage2_logits)
        report = event_metrics(y_true, np.where(prediction == 0, -1, prediction), runs)
        report.update(classification_metrics(y_true, prediction))
    elif args.algorithm == "hard_vote":
        output = hard_vote_commands(stage1_logits, stage2_logits, window_count=args.vote_windows, vote_threshold=args.vote_threshold, run_ids=runs)
        report = event_metrics(y_true, output, runs)
        report.update(classification_metrics(y_true, np.where(output == -1, 0, output)))
    else:
        config = config_from_args(args)
        if args.algorithm == "fast":
            output = fast_path_commands(stage1_logits, stage2_logits, config, run_ids=runs)
        elif args.algorithm == "feature":
            if stage2_features is None:
                raise RuntimeError("feature inference returned no hidden features")
            output = feature_gate_commands(stage1_logits, stage2_logits, stage2_features, config, run_ids=runs)
        else:
            output = candidate_commands(stage1_logits, stage2_logits, config, run_ids=runs)
        report = event_metrics(y_true, output.commands, runs)
        report.update(classification_metrics(y_true, np.where(output.commands == -1, 0, output.commands)))
        report["diagnostics"] = policy_diagnostics(output.reasons)
    report.update({"checkpoint": str(path), "subject": int(checkpoint["subject"]), "model": checkpoint["model"]})
    return report


def run(args: argparse.Namespace) -> dict:
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    reports = [evaluate_checkpoint(path, args, device) for path in find_checkpoints(args)]
    result = {
        "split": "labelled_test_session_only", "algorithm": args.algorithm,
        "reports": reports, "summary": seed_summary(reports),
        "warning": "Final hold-out results: do not select models or thresholds after reading this file.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run(parse_args())
