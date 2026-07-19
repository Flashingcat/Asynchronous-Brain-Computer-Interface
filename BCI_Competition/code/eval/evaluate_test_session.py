"""Final test-session evaluation for final two-stage checkpoints.

Uses only session=1/split=2, never OOF.  It supports one or many final .pt
checkpoints (for example, one per random seed) and evaluates argmax, hard-vote,
candidate, Fast-0/Fast-1, or feature-gated candidate policies.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import platform
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
from metric import classification_metrics, event_metrics, grouped_summary, policy_diagnostics
from models.model_factory import build_model, model_source_id


PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "bnci2014001_oof_windows.npz"
DEFAULT_PATTERN = PROJECT_ROOT / "results" / "checkpoints" / "**" / "*final.pt"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
REQUIRED_SCHEMA = "bnci2014001_causal_windows_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoints", type=Path, nargs="+", help="one or more final .pt files")
    parser.add_argument("--checkpoint-glob", default=str(DEFAULT_PATTERN), help="used when --checkpoints is omitted")
    parser.add_argument("--algorithm", choices=("argmax", "hard_vote", "candidate", "fast", "feature"), default="candidate")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
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
    paths = list(dict.fromkeys(path.resolve() for path in paths))
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("no final checkpoint found; pass --checkpoints explicitly")
    return paths


def load_subject_test_data(path: Path, subject: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        required = {
            "X", "y", "subject", "session", "split", "run", "segment", "window_stop", "event",
            "event_subject", "event_session", "event_run", "event_id", "event_label", "event_start",
            "schema_version", "dataset_id", "sampling_rate",
        }
        missing = required.difference(data.files)
        if missing:
            raise RuntimeError(f"data file is missing {sorted(missing)}")
        if str(data["schema_version"].item()) != REQUIRED_SCHEMA:
            raise RuntimeError(f"expected data schema {REQUIRED_SCHEMA}")
        mask = (data["subject"] == subject) & (data["session"] == 1) & (data["split"] == 2)
        if not mask.any():
            raise RuntimeError(f"subject {subject} has no labelled test-session rows")
        event_mask = (data["event_subject"] == subject) & (data["event_session"] == 1)
        return {
            "X": data["X"][mask].astype(np.float32),
            "y": data["y"][mask].astype(np.int64),
            "run": data["run"][mask].astype(np.int64),
            "segment": data["segment"][mask].astype(np.int64),
            "event": data["event"][mask].astype(np.int64),
            "window_stop": data["window_stop"][mask].astype(np.int64),
            "sampling_rate": int(data["sampling_rate"].item()),
            "dataset_id": str(data["dataset_id"].item()),
            "events": {
                "run": data["event_run"][event_mask].astype(np.int64),
                "event": data["event_id"][event_mask].astype(np.int64),
                "label": data["event_label"][event_mask].astype(np.int64),
                "start": data["event_start"][event_mask].astype(np.int64),
            },
        }


def continuous_ids(runs: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Assign one reset identity to each uninterrupted run segment."""
    runs, segments = np.asarray(runs), np.asarray(segments)
    if runs.ndim != 1 or segments.shape != runs.shape:
        raise ValueError("runs and segments must be aligned vectors")
    output = np.zeros(len(runs), dtype=np.int64)
    if len(runs) > 1:
        output[1:] = np.cumsum((runs[1:] != runs[:-1]) | (segments[1:] != segments[:-1]))
    return output


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


def policy_config(args: argparse.Namespace) -> dict:
    # 仅记录当前算法真正消费的阈值，避免无关参数改变评估身份。
    if args.algorithm == "argmax":
        return {}
    if args.algorithm == "hard_vote":
        return {"vote_windows": args.vote_windows, "vote_threshold": args.vote_threshold}
    config = {
        "task_on": args.task_on, "task_hold": args.task_hold, "idle_reset": args.idle_reset,
        "min_windows": args.min_windows, "max_windows": args.max_windows,
        "top_probability": args.top_probability, "probability_gap": args.probability_gap,
        "stable_windows": args.stable_windows, "stage2_aggregation": args.stage2_aggregation,
        "stage2_alpha": args.stage2_alpha,
    }
    if args.algorithm == "fast":
        config.update({"fast_probability": args.fast_probability, "fast_gap": args.fast_gap})
    if args.algorithm == "feature":
        config.update({"feature_max_change": args.feature_max_change, "feature_consecutive": args.feature_consecutive})
    return config


# 评估身份固定数据、checkpoint 内容、当前模型/评估源码和运行时；同名路径不能掩盖内容变化。
def evaluation_config(args: argparse.Namespace, checkpoints: list[tuple[Path, dict]]) -> dict:
    with np.load(args.data) as data:
        if "dataset_id" not in data:
            raise RuntimeError("data file has no dataset_id")
        dataset_id = str(data["dataset_id"].item())
    return {
        "data": str(args.data.resolve()),
        "dataset_id": dataset_id,
        "data_sha256": file_sha256(args.data),
        "evaluator_source_id": evaluator_source_fingerprint(),
        "checkpoints": [
            {
                "path": str(path), "sha256": file_sha256(path),
                "model_source_id": model_source_id(str(checkpoint.get("model", ""))),
            }
            for path, checkpoint in checkpoints
        ],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "algorithm": args.algorithm,
        "batch_size": args.batch_size,
        "device": args.device,
        "policy": policy_config(args),
    }


def config_fingerprint(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluator_source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(HERE.rglob("*.py")):
        digest.update(path.relative_to(HERE).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def score_commands(data: dict, commands: np.ndarray) -> dict:
    report = event_metrics(
        data["y"], commands, data["run"], data["event"], data["window_stop"], data["events"],
        sampling_rate=data["sampling_rate"],
    )
    report.update(classification_metrics(data["y"], np.where(commands == -1, 0, commands)))
    return report


def evaluate_checkpoint(
    path: Path,
    checkpoint: dict,
    args: argparse.Namespace,
    device: torch.device,
    data_sha256: str,
    checkpoint_sha256: str,
    current_model_source_id: str,
) -> dict:
    required = {
        "run_id", "model", "subject", "seed", "training_config",
        "binary_state_dict", "mi_state_dict", "mean", "std",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise RuntimeError(f"{path.name} is not a final two-stage checkpoint; missing {sorted(missing)}")
    data = load_subject_test_data(args.data, int(checkpoint["subject"]))
    training_config = dict(checkpoint["training_config"])
    if training_config.get("dataset_id") != data["dataset_id"]:
        raise RuntimeError(f"{path.name} was trained on a different dataset_id")
    if training_config.get("data_sha256") != data_sha256:
        raise RuntimeError(f"{path.name} was trained on different data content")
    if training_config.get("model") != checkpoint["model"] or int(training_config.get("seed", -1)) != int(checkpoint["seed"]):
        raise RuntimeError(f"{path.name} has inconsistent model/seed provenance")
    if training_config.get("model_source_id") != current_model_source_id:
        raise RuntimeError(f"{path.name} was trained with different model source")
    raw_x = data["X"]
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
    reset_ids = continuous_ids(data["run"], data["segment"])
    if args.algorithm == "argmax":
        prediction = argmax_predict(stage1_logits, stage2_logits)
        report = score_commands(data, np.where(prediction == 0, -1, prediction))
    elif args.algorithm == "hard_vote":
        output = hard_vote_commands(
            stage1_logits, stage2_logits, window_count=args.vote_windows,
            vote_threshold=args.vote_threshold, run_ids=reset_ids,
        )
        report = score_commands(data, output)
    else:
        config = config_from_args(args)
        if args.algorithm == "fast":
            output = fast_path_commands(stage1_logits, stage2_logits, config, run_ids=reset_ids)
        elif args.algorithm == "feature":
            if stage2_features is None:
                raise RuntimeError("feature inference returned no hidden features")
            output = feature_gate_commands(stage1_logits, stage2_logits, stage2_features, config, run_ids=reset_ids)
        else:
            output = candidate_commands(stage1_logits, stage2_logits, config, run_ids=reset_ids)
        report = score_commands(data, output.commands)
        report["diagnostics"] = policy_diagnostics(output.reasons)
    report.update({
        "checkpoint": str(path), "run_id": checkpoint["run_id"],
        "checkpoint_sha256": checkpoint_sha256,
        "subject": int(checkpoint["subject"]), "model": checkpoint["model"],
        "seed": int(checkpoint["seed"]), "training_config": training_config,
    })
    return report


def run(args: argparse.Namespace) -> dict:
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint_paths = find_checkpoints(args)
    checkpoints = [(path, torch.load(path, map_location="cpu", weights_only=False)) for path in checkpoint_paths]
    config = evaluation_config(args, checkpoints)
    evaluation_id = f"{args.algorithm}_{config_fingerprint(config)}"
    output = args.output or TABLE_DIR / f"test_{evaluation_id}.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace it: {output}")
    checkpoint_records = {Path(item["path"]): item for item in config["checkpoints"]}
    reports = [
        evaluate_checkpoint(
            path, checkpoint, args, device, config["data_sha256"],
            checkpoint_records[path]["sha256"], checkpoint_records[path]["model_source_id"],
        )
        for path, checkpoint in checkpoints
    ]
    result = {
        "evaluation_id": evaluation_id, "split": "labelled_test_session_only",
        "algorithm": args.algorithm, "evaluation_config": config,
        "reports": reports, "summary": grouped_summary(reports),
        "warning": "Final hold-out results: do not select models or thresholds after reading this file.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run(parse_args())
