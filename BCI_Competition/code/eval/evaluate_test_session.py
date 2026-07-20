"""Evaluate final two-stage checkpoints on the labelled test session."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

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
    parser.add_argument("--output", type=Path)
    add_policy_args(parser)
    return parser.parse_args()


def checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    paths = args.checkpoints or [Path(item) for item in sorted(glob.glob(args.checkpoint_glob, recursive=True))]
    paths = [path.resolve() for path in paths]
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("no final checkpoint found")
    return paths


def evaluate(path: Path, args: argparse.Namespace, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "subject" not in checkpoint:
        raise RuntimeError(f"{path.name} is missing subject")
    data = load_session_data(args.data, int(checkpoint["subject"]), 1, args.window_mode)
    stage1, stage2, features = infer_checkpoint(checkpoint, data, args, device)
    dense, commands, reasons = run_policy(stage1, stage2, data["streams"], features, args)
    return {
        **score(data, dense, commands, reasons, args.window_mode),
        "checkpoint": str(path),
        "subject": int(checkpoint["subject"]),
        "model": checkpoint["model"],
        "seed": checkpoint.get("seed"),
        "runs": sorted(set(data["run"].astype(int).tolist())),
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    reports = [evaluate(path, args, device) for path in checkpoint_paths(args)]
    result = {
        "split": "test_session",
        "data": str(args.data.resolve()),
        "window_mode": args.window_mode,
        "metric_definition": METRIC_DEFINITIONS[args.window_mode],
        "algorithm": args.algorithm,
        "policy": policy_identity(args),
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
