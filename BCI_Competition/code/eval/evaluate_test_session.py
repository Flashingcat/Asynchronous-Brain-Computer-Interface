"""Evaluate final two-stage checkpoints on the labelled test session."""

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
from metric import (
    classification_metrics,
    continuous_event_metrics,
    legacy_event_metrics,
    policy_diagnostics,
    report_summary,
)
from models.model_factory import build_model


PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "bnci2014001_oof_windows.npz"
DEFAULT_PATTERN = PROJECT_ROOT / "results" / "checkpoints" / "**" / "*final.pt"
METRIC_DEFINITIONS = {
    "continuous": "original annotation events and decision-sample latency",
    "pure": "legacy pure-window event blocks and compressed 0.5-second latency",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoints", type=Path, nargs="+")
    parser.add_argument("--checkpoint-glob", default=str(DEFAULT_PATTERN))
    parser.add_argument("--algorithm", choices=("argmax", "hard_vote", "candidate", "fast", "feature"), default="candidate")
    parser.add_argument("--window-mode", choices=("continuous", "pure"), default="continuous")
    parser.add_argument("--output", type=Path)
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
    return parser.parse_args()


def checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    paths = args.checkpoints or [Path(item) for item in sorted(glob.glob(args.checkpoint_glob, recursive=True))]
    paths = [path.resolve() for path in paths]
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("no final checkpoint found")
    return paths


def load_test_data(path: Path, subject: int, window_mode: str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        required = {
            "X", "y", "subject", "session", "split", "run", "segment", "decision_sample", "is_pure",
            "event_subject", "event_run", "event_label", "event_start", "event_stop", "sampling_rate",
        }
        missing = required.difference(data.files)
        if missing:
            raise RuntimeError(f"data is missing {sorted(missing)}")

        mask = (data["subject"] == subject) & (data["session"] == 1) & (data["split"] == 2)
        if window_mode == "pure":
            mask &= data["is_pure"]
        if not mask.any():
            raise RuntimeError(f"subject {subject} has no test-session windows")

        event_mask = data["event_subject"] == subject
        runs, segments = data["run"][mask], data["segment"][mask]
        streams = np.zeros(len(runs), dtype=np.int64)
        if len(streams) > 1:
            streams[1:] = np.cumsum((runs[1:] != runs[:-1]) | (segments[1:] != segments[:-1]))

        return {
            "X": data["X"][mask].astype(np.float32),
            "y": data["y"][mask].astype(np.int64),
            "run": runs,
            "streams": streams,
            "decision_sample": data["decision_sample"][mask],
            "sampling_rate": int(data["sampling_rate"].item()),
            "events": {
                "run": data["event_run"][event_mask],
                "label": data["event_label"][event_mask],
                "start": data["event_start"][event_mask],
                "stop": data["event_stop"][event_mask],
            },
        }


def infer(model: torch.nn.Module, features: np.ndarray, device: torch.device, batch_size: int, need_features: bool) -> tuple[np.ndarray, np.ndarray | None]:
    logits, hidden = [], []
    model.eval(); backbone = getattr(model, "model", None)
    if need_features and backbone is None:
        raise RuntimeError("feature policy requires a feature-returning backbone")
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start:start + batch_size]).to(device)
            result = backbone(batch, return_features=True) if need_features else model(batch)
            if need_features:
                result, values = result; hidden.append(values.cpu().numpy())
            logits.append(result.cpu().numpy())
    return np.concatenate(logits), None if not hidden else np.concatenate(hidden)


def candidate_config(args: argparse.Namespace) -> CandidateConfig:
    return CandidateConfig(args.task_on, args.task_hold, args.idle_reset, args.min_windows, args.max_windows, args.top_probability, args.probability_gap, args.stable_windows, args.stage2_aggregation, args.stage2_alpha, False, False, args.fast_probability, args.fast_gap, False, args.feature_max_change, args.feature_consecutive)


def evaluate(path: Path, args: argparse.Namespace, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "subject", "binary_state_dict", "mi_state_dict", "mean", "std"}
    missing = required.difference(checkpoint)
    if missing:
        raise RuntimeError(f"{path.name} is missing {sorted(missing)}")
    data = load_test_data(args.data, int(checkpoint["subject"]), args.window_mode)
    x = ((data["X"] - checkpoint["mean"]) / checkpoint["std"]).astype(np.float32)
    stage1 = build_model(checkpoint["model"], 2, x.shape[1], x.shape[2]).to(device)
    stage2 = build_model(checkpoint["model"], 4, x.shape[1], x.shape[2]).to(device)
    stage1.load_state_dict(checkpoint["binary_state_dict"]); stage2.load_state_dict(checkpoint["mi_state_dict"])
    stage1_logits, _ = infer(stage1, x, device, args.batch_size, False)
    stage2_logits, features = infer(stage2, x, device, args.batch_size, args.algorithm == "feature")
    dense = argmax_predict(stage1_logits, stage2_logits)
    output = None
    if args.algorithm == "argmax":
        commands = np.where(dense == 0, -1, dense)
    elif args.algorithm == "hard_vote":
        commands = hard_vote_commands(stage1_logits, stage2_logits, window_count=args.vote_windows, vote_threshold=args.vote_threshold, run_ids=data["streams"])
    elif args.algorithm == "fast":
        output = fast_path_commands(stage1_logits, stage2_logits, candidate_config(args), run_ids=data["streams"]); commands = output.commands
    elif args.algorithm == "feature":
        if features is None: raise RuntimeError("feature inference failed")
        output = feature_gate_commands(stage1_logits, stage2_logits, features, candidate_config(args), run_ids=data["streams"]); commands = output.commands
    else:
        output = candidate_commands(stage1_logits, stage2_logits, candidate_config(args), run_ids=data["streams"]); commands = output.commands
    if args.window_mode == "pure":
        event_report = legacy_event_metrics(data["y"], commands, data["streams"])
    else:
        event_report = continuous_event_metrics(
            data["y"],
            commands,
            data["run"],
            data["decision_sample"],
            data["events"],
            data["sampling_rate"],
        )
    report = {
        **classification_metrics(data["y"], dense),
        **event_report,
        "checkpoint": str(path),
        "subject": int(checkpoint["subject"]),
        "model": checkpoint["model"],
        "seed": checkpoint.get("seed"),
    }
    if output is not None:
        report["diagnostics"] = policy_diagnostics(output.reasons)
    return report


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    reports = [evaluate(path, args, device) for path in checkpoint_paths(args)]
    result = {
        "split": "test_session",
        "window_mode": args.window_mode,
        "metric_definition": METRIC_DEFINITIONS[args.window_mode],
        "algorithm": args.algorithm,
        "reports": reports,
        "summary": report_summary(reports),
    }
    output = args.output or PROJECT_ROOT / "results" / f"test_session_{args.window_mode}_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run(parse_args())
