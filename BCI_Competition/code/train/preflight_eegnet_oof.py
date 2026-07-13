"""执行真实 GPU OOF 分支覆盖与跨进程恢复等价性预检。"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from oof_training_bundle import BUNDLE_ID, file_hash
from train_eegnet_oof import atomic_json, canonical_hash, tensor_state_hash, utc_now


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINER = Path(__file__).with_name("train_eegnet_oof.py")
CAUSAL_JOB = "stage1_causal_fold0_seed42"
ZERO_PHASE_JOB = "stage2_zero_phase_fold0_seed42"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-bundle", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3)
    return parser.parse_args()


# ---------- 生产 CLI 覆盖：因果分支做恢复等价，零相位 Stage2 覆盖四分类和双验证域 ----------
def job_command(args: argparse.Namespace, output: Path, stage: int,
                domain: str) -> list[str]:
    return [
        sys.executable, str(TRAINER),
        "--training-bundle", str(args.training_bundle.resolve()),
        "--output-root", str(output.resolve()),
        "--subject", "1", "--folds", "0", "--stages", str(stage),
        "--train-domains", domain, "--seeds", "42",
        "--epochs", str(args.epochs), "--batch-size", "64",
        "--validation-batch-size", "256", "--learning-rate", "0.001",
        "--weight-decay", "0.0001", "--device", args.device,
    ]


def run_command(command: list[str], log_path: Path) -> None:
    result = subprocess.run(command, cwd=PROJECT_ROOT.parent, check=False,
                            capture_output=True, text=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        result.stdout + ("\n[stderr]\n" + result.stderr if result.stderr else ""),
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"预检子进程失败 returncode={result.returncode}; log={log_path}"
        )


# ---------- 逐值比较：模型、优化器、全部 RNG、公开 logit 与 checkpoint 必须相互一致 ----------
def nested_equal(left, right, path: str = "root") -> None:
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise RuntimeError(f"张量不等价: {path}")
    elif isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise RuntimeError(f"字典键不等价: {path}")
        for key in left:
            nested_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        if len(left) != len(right):
            raise RuntimeError(f"序列长度不等价: {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            nested_equal(a, b, f"{path}[{index}]")
    elif isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
            raise RuntimeError(f"数组不等价: {path}")
    elif left != right:
        raise RuntimeError(f"值不等价: {path}: {left!r} != {right!r}")


def load_and_crosscheck_job(job_dir: Path, expected_domains: tuple[str, ...]) -> dict:
    checkpoint_path = job_dir / "latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    history = json.loads((job_dir / "history.json").read_text(encoding="utf-8"))
    completed = json.loads((job_dir / "completed.json").read_text(encoding="utf-8"))
    config = json.loads((job_dir / "job_config.json").read_text(encoding="utf-8"))
    if (checkpoint.get("format_version") != 1 or
            checkpoint["contract_sha256"] != config["contract_sha256"] or
            completed["contract_sha256"] != config["contract_sha256"] or
            checkpoint["epoch"] != len(history) or checkpoint["history"] != history or
            completed["completed_epochs"] != len(history) or
            completed["checkpoint_sha256"] != file_hash(checkpoint_path)):
        raise RuntimeError(f"作业 checkpoint/config/history/completed 不自洽: {job_dir}")
    model_hash = tensor_state_hash(checkpoint["model_state_dict"])
    if completed["model_tensor_sha256"] != model_hash:
        raise RuntimeError("完成标记模型哈希错误")

    with np.load(job_dir / "oof_predictions.npz", allow_pickle=False) as data:
        if data["epochs"].tolist() != list(range(1, len(history) + 1)):
            raise RuntimeError("公开 OOF epoch 轴错误")
        if set(checkpoint["oof_logits"]) != set(expected_domains):
            raise RuntimeError("checkpoint 验证输入域错误")
        shapes = {}
        for domain in expected_domains:
            public_logits = data[f"{domain}_logits"]
            checkpoint_logits = checkpoint["oof_logits"][domain]
            if not np.array_equal(public_logits, checkpoint_logits):
                raise RuntimeError(f"公开 OOF 与 checkpoint 不一致: {domain}")
            shapes[domain] = list(public_logits.shape)
        row_count = len(data["validation_rows"])
        if len(data["validation_labels"]) != row_count:
            raise RuntimeError("OOF 行和标签数量不一致")
    return {
        "checkpoint": checkpoint,
        "history": history,
        "model_tensor_sha256": model_hash,
        "logit_shapes": shapes,
        "validation_window_count": row_count,
        "contract_sha256": config["contract_sha256"],
        "execution_fingerprint": config["contract"]["optimization"]["execution_fingerprint"],
    }


def compare_resumed(continuous: Path, resumed: Path) -> dict:
    left = load_and_crosscheck_job(continuous / CAUSAL_JOB, ("causal",))
    right = load_and_crosscheck_job(resumed / CAUSAL_JOB, ("causal",))
    for key in ("epoch", "history", "oof_logits", "model_state_dict",
                "optimizer_state_dict", "rng_state"):
        nested_equal(left["checkpoint"][key], right["checkpoint"][key], key)
    if left["contract_sha256"] != right["contract_sha256"]:
        raise RuntimeError("连续训练与恢复训练合同不同")
    return {
        "history_equal": True,
        "oof_arrays_bitwise_equal": True,
        "model_state_bitwise_equal": True,
        "optimizer_state_bitwise_equal": True,
        "rng_state_bitwise_equal": True,
        "model_tensor_sha256": left["model_tensor_sha256"],
        "causal_logits_shape": left["logit_shapes"]["causal"],
        "completed_epochs": len(left["history"]),
        "contract_sha256": left["contract_sha256"],
        "execution_fingerprint": left["execution_fingerprint"],
    }


def artifact_inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "preflight_manifest.json"
    }


def git_provenance() -> dict:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT.parent,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT.parent,
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    return {"commit": sha, "dirty": bool(dirty), "dirty_entries": dirty}


def main() -> None:
    args = parse_args()
    if args.epochs < 2:
        raise ValueError("恢复预检至少需要 2 个 epoch")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("正式 GPU preflight 必须显式使用可用 CUDA 设备，拒绝 CPU PASS")
    if args.training_bundle is None:
        args.training_bundle = (
            PROJECT_ROOT / "data" / "processed" /
            BUNDLE_ID.format(subject=1) / "manifest.json"
        )
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"预检输出目录必须为空或不存在: {root}")
    root.mkdir(parents=True, exist_ok=True)

    continuous, resumed = root / "continuous", root / "resumed"
    zero_phase = root / "stage2_zero_phase"
    command_continuous = job_command(args, continuous, 1, "causal")
    command_pause = job_command(args, resumed, 1, "causal") + ["--stop-after-epoch", "1"]
    command_resume = job_command(args, resumed, 1, "causal")
    command_zero_phase = job_command(args, zero_phase, 2, "zero_phase")
    commands = [command_continuous, command_pause, command_resume, command_zero_phase]

    manifest = {
        "preflight_id": "bnci2014001_s01_eegnet_oof_gpu_preflight_native250_v2",
        "status": "RUNNING",
        "host": {"hostname": platform.node(), "remote": False,
                 "cwd": str(PROJECT_ROOT.parent.resolve())},
        "protocol": {
            "subject": 1, "session": 0, "fold": 0, "seed": 42,
            "epochs": args.epochs, "sampling_rate": 250,
            "covered_jobs": [
                {"stage": 1, "train_domain": "causal",
                 "validation_domains": ["causal"], "resume_equivalence": True},
                {"stage": 2, "train_domain": "zero_phase",
                 "validation_domains": ["zero_phase", "causal"],
                 "resume_equivalence": False},
            ],
            "expected_unique_job_count": 2,
            "test_session_access": "forbidden",
        },
        "training_bundle": {
            "path": str(args.training_bundle.resolve()),
            "sha256": file_hash(args.training_bundle.resolve()),
        },
        "git": git_provenance(),
        "commands": commands,
        "interruption_mode": "clean_pause_after_atomic_epoch_1_checkpoint",
        "started_at_utc": utc_now(),
    }
    atomic_json(root / "preflight_manifest.json", manifest)
    try:
        run_command(command_continuous, root / "continuous.log")
        run_command(command_pause, root / "pause.log")
        run_command(command_resume, root / "resume.log")
        run_command(command_zero_phase, root / "stage2_zero_phase.log")
        resume_verification = compare_resumed(continuous, resumed)
        zero_verification = load_and_crosscheck_job(
            zero_phase / ZERO_PHASE_JOB, ("zero_phase", "causal")
        )
        manifest.update({
            "status": "PASS",
            "resume_verification": resume_verification,
            "stage2_zero_phase_verification": {
                key: value for key, value in zero_verification.items()
                if key != "checkpoint" and key != "history"
            },
            "artifact_sha256": artifact_inventory(root),
            "completed_at_utc": utc_now(),
        })
    except BaseException as exc:
        manifest.update({
            "status": "FAIL", "error_type": type(exc).__name__,
            "error": str(exc), "artifact_sha256": artifact_inventory(root),
            "completed_at_utc": utc_now(),
        })
        atomic_json(root / "preflight_manifest.json", manifest)
        raise
    manifest["manifest_payload_sha256"] = canonical_hash(manifest)
    atomic_json(root / "preflight_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
