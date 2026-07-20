"""Evaluate leave-one-run-out checkpoints on the continuous training session."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE, CODE_ROOT = Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from eval.metric import report_summary
from eval.session_evaluator import (
    add_policy_args,
    infer_checkpoint,
    load_session_data,
    policy_identity,
    run_policy,
    score,
)


PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "bnci2014001_oof_windows.npz"
DEFAULT_PATTERN = PROJECT_ROOT / "results" / "checkpoints" / "simple_oof" / "*fold*_validation.pt"
METRIC_DEFINITIONS = {
    "continuous": "original annotation events and decision-sample latency",
    "pure": "legacy pure-window event blocks and compressed 0.5-second latency",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoints", type=Path, nargs="+")
    parser.add_argument("--checkpoint-glob", default=str(DEFAULT_PATTERN))
    parser.add_argument("--output", type=Path)
    add_policy_args(parser)
    return parser.parse_args()


def checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    paths = args.checkpoints or [Path(item) for item in sorted(glob.glob(args.checkpoint_glob, recursive=True))]
    paths = [path.resolve() for path in paths]
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("no validation fold checkpoint found")
    return paths


def combine(parts: list[dict]) -> dict:
    """按运行合并 OOF 输出；stream 编号保持全局唯一。"""
    streams, offset = [], 0
    for part in parts:
        current = part["data"]["streams"]
        streams.append(current + offset)
        offset += int(current.max()) + 1
    return {
        "y": np.concatenate([part["data"]["y"] for part in parts]),
        "run": np.concatenate([part["data"]["run"] for part in parts]),
        "streams": np.concatenate(streams),
        "decision_sample": np.concatenate([part["data"]["decision_sample"] for part in parts]),
        "sampling_rate": parts[0]["data"]["sampling_rate"],
        "events": {
            key: np.concatenate([part["data"]["events"][key] for part in parts])
            for key in ("run", "label", "start", "stop")
        },
    }


def evaluate_subject(paths: list[Path], args: argparse.Namespace, device: torch.device) -> dict:
    parts, checkpoints = [], []
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        required = {"subject", "held_out_run", "model", "seed"}
        missing = required.difference(checkpoint)
        if missing:
            raise RuntimeError(f"{path.name} is missing {sorted(missing)}")
        checkpoints.append((path, checkpoint))

    subjects = {int(checkpoint["subject"]) for _, checkpoint in checkpoints}
    models = {checkpoint["model"] for _, checkpoint in checkpoints}
    seeds = {checkpoint["seed"] for _, checkpoint in checkpoints}
    runs = [int(checkpoint["held_out_run"]) for _, checkpoint in checkpoints]
    if any(len(values) != 1 for values in (subjects, models, seeds)) or len(runs) != len(set(runs)):
        raise RuntimeError("one subject requires one model/seed and one checkpoint per run")
    subject = next(iter(subjects))
    expected_runs = sorted(set(load_session_data(args.data, subject, 0, args.window_mode)["run"].astype(int).tolist()))
    if sorted(runs) != expected_runs:
        raise RuntimeError(f"subject {subject} fold coverage mismatch: expected {expected_runs}, got {sorted(runs)}")

    # 每个折只负责自己的留出运行，策略状态也在运行内部独立演进。
    for path, checkpoint in sorted(checkpoints, key=lambda item: int(item[1]["held_out_run"])):
        run = int(checkpoint["held_out_run"])
        data = load_session_data(args.data, subject, 0, args.window_mode, run)
        stage1, stage2, features = infer_checkpoint(checkpoint, data, args, device)
        dense, commands, reasons = run_policy(stage1, stage2, data["streams"], features, args)
        parts.append({"data": data, "dense": dense, "commands": commands, "reasons": reasons, "path": path})

    data = combine(parts)
    dense = np.concatenate([part["dense"] for part in parts])
    commands = np.concatenate([part["commands"] for part in parts])
    reasons = tuple(reason for part in parts for reason in part["reasons"])
    return {
        **score(data, dense, commands, reasons, args.window_mode),
        "subject": subject,
        "model": next(iter(models)),
        "seed": next(iter(seeds)),
        "folds": expected_runs,
        "checkpoints": [str(part["path"]) for part in parts],
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in checkpoint_paths(args):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        required = {"subject"}
        missing = required.difference(checkpoint)
        if missing:
            raise RuntimeError(f"{path.name} is missing {sorted(missing)}")
        grouped[int(checkpoint["subject"])].append(path)
    reports = [evaluate_subject(grouped[subject], args, device) for subject in sorted(grouped)]
    result = {
        "split": "train_session_leave_one_run_out",
        "data": str(args.data.resolve()),
        "window_mode": args.window_mode,
        "metric_definition": METRIC_DEFINITIONS[args.window_mode],
        "algorithm": args.algorithm,
        "policy": policy_identity(args),
        "reports": reports,
        "summary": report_summary(reports),
    }
    output = args.output or PROJECT_ROOT / "results" / f"validation_oof_{args.window_mode}_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run(parse_args())
