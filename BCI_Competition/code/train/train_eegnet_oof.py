"""在训练 session 内运行 EEGNet 六折 OOF 基线矩阵。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# cuBLAS 要求在首次创建 CUDA 上下文前设置工作区，放在 torch import 前最稳妥。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPROCESSING_DIR = PROJECT_ROOT / "code" / "preprocessing"
MODELS_DIR = PROJECT_ROOT / "code" / "models"
sys.path.insert(0, str(PROJECT_ROOT / "code"))
sys.path.insert(0, str(MODELS_DIR))

from train.oof_training_bundle import (  # noqa: E402
    BUNDLE_ID,
    BundleContext,
    BundleWindowDataset,
    DOMAINS,
    file_hash,
    load_bundle,
    rows_for,
    window_identity_hash,
)
# 直接绑定本仓库的模型目录，避免原仓库 models/models 包在整套测试中抢占导入名。
from model_factory import build_model  # noqa: E402


TRAINING_ID = "bnci2014001_s{subject:02d}_eegnet_oof_native250_v1"
DEFAULT_FOLDS = tuple(range(6))
DEFAULT_STAGES = (1, 2)
DEFAULT_DOMAINS = ("causal", "zero_phase")
DEFAULT_SEEDS = (42, 43, 44)
CLASS_NAMES = {
    1: ("idle", "task"),
    2: ("left_hand", "right_hand", "feet", "tongue"),
}


# ---------- 矩阵身份：一次训练轨迹只由 fold、stage、训练输入域和 seed 决定 ----------
@dataclass(frozen=True)
class JobSpec:
    subject: int
    fold: int
    stage: int
    train_domain: str
    seed: int

    @property
    def name(self) -> str:
        return (
            f"stage{self.stage}_{self.train_domain}_"
            f"fold{self.fold}_seed{self.seed}"
        )

    @property
    def validation_domains(self) -> tuple[str, ...]:
        # 因果训练回答主在线问题；零相位训练额外输出因果输入消融，不重复训练。
        return (("causal",) if self.train_domain == "causal"
                else ("zero_phase", "causal"))


def build_job_specs(subject: int, folds: list[int] | tuple[int, ...],
                    stages: list[int] | tuple[int, ...],
                    domains: list[str] | tuple[str, ...],
                    seeds: list[int] | tuple[int, ...]) -> list[JobSpec]:
    """生成确定顺序的训练矩阵，并在碰 GPU 前拒绝非法或重复配置。"""
    if subject not in range(1, 10):
        raise ValueError("BNCI2014001 subject 必须为 1 至 9")
    if not folds or any(fold not in DEFAULT_FOLDS for fold in folds):
        raise ValueError("fold 必须来自 0 至 5")
    if not stages or any(stage not in DEFAULT_STAGES for stage in stages):
        raise ValueError("stage 必须为 1 或 2")
    if not domains or any(domain not in DOMAINS for domain in domains):
        raise ValueError(f"训练输入域必须来自 {DOMAINS}")
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("seed 必须为非负整数")
    axes = (folds, stages, domains, seeds)
    if any(len(values) != len(set(values)) for values in axes):
        raise ValueError("矩阵各轴不得包含重复值")
    return [
        JobSpec(subject, fold, stage, domain, seed)
        for domain in domains
        for stage in stages
        for fold in folds
        for seed in seeds
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    processed = PROJECT_ROOT / "data" / "processed"
    parser.add_argument(
        "--training-bundle", type=Path, default=None,
        help="自包含 session0-only bundle；缺失时拒绝训练，不自动访问联合上游",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "results" / "checkpoints" / "eegnet_oof_native250_v1",
    )
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--folds", type=int, nargs="+", default=list(DEFAULT_FOLDS))
    parser.add_argument("--stages", type=int, nargs="+", default=list(DEFAULT_STAGES))
    parser.add_argument("--train-domains", nargs="+", choices=DOMAINS,
                        default=list(DEFAULT_DOMAINS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-resume", action="store_true",
                        help="若作业目录已有检查点则报错，不自动覆盖")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--stop-after-epoch", type=int,
        help="只供恢复预检：写完该 epoch 的原子检查点后暂停；仅允许单作业",
    )
    return parser.parse_args()


# ---------- 原子文件与稳定哈希：进程中断不留下半个正式 checkpoint ----------
def canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        # Windows 的 os.fsync 要求可写文件描述符；只读 rb 会报 Bad file descriptor。
        with open(temporary, "rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def tensor_state_hash(state_dict: dict[str, torch.Tensor]) -> str:
    """对模型张量内容做稳定哈希，不依赖 torch checkpoint 容器格式。"""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


# ---------- 上游装载：正式训练只读取自包含 bundle，缺 bundle 时绝不自动回访联合索引 ----------
TrainingContext = BundleContext


def load_context(bundle_path: Path, subject: int) -> TrainingContext:
    context = load_bundle(bundle_path, verify_hashes=True)
    if context.manifest["subject"] != subject:
        raise RuntimeError("OOF bundle subject 与请求不匹配")
    if np.any(context.rows["session"] != 0):
        raise RuntimeError("OOF bundle 意外包含测试 session")
    return context


# ---------- 窗口物化：训练与验证均从同一共同池取行，验证 run 永不参与拟合 ----------
@dataclass
class JobArrays:
    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: dict[str, np.ndarray]
    validation_y: np.ndarray
    validation_rows: np.ndarray


def materialize_dataset(dataset) -> tuple[np.ndarray, np.ndarray]:
    if len(dataset) == 0:
        raise RuntimeError("不允许物化空数据集")
    first_x, first_y = dataset[0]
    features = np.empty((len(dataset), *first_x.shape), dtype=np.float32)
    labels = np.empty(len(dataset), dtype=np.int64)
    features[0], labels[0] = first_x, first_y
    for index in range(1, len(dataset)):
        features[index], labels[index] = dataset[index]
    if not np.isfinite(features).all():
        raise RuntimeError("物化数据含非有限值")
    return np.ascontiguousarray(features), labels


def prepare_job_arrays(context: TrainingContext, spec: JobSpec) -> JobArrays:
    training_rows = rows_for(
        context.manifest, context.rows, spec.fold, spec.stage, "train"
    )
    training = BundleWindowDataset(
        context, training_rows, spec.train_domain, spec.fold, spec.stage
    )
    train_x, train_y = materialize_dataset(training)

    validation_rows = rows_for(
        context.manifest, context.rows, spec.fold, spec.stage, "validation"
    )
    validation_x: dict[str, np.ndarray] = {}
    validation_y: np.ndarray | None = None
    for domain in spec.validation_domains:
        dataset = BundleWindowDataset(
            context, validation_rows, domain, spec.fold, spec.stage
        )
        domain_x, domain_y = materialize_dataset(dataset)
        validation_x[domain] = domain_x
        if validation_y is None:
            validation_y = domain_y
        elif not np.array_equal(validation_y, domain_y):
            raise RuntimeError("不同验证输入域没有使用相同标签顺序")

    train_runs = set(int(value) for value in training_rows["run"])
    if (np.any(training_rows["session"] != 0) or spec.fold in train_runs or
            np.any(validation_rows["session"] != 0) or
            set(int(value) for value in validation_rows["run"]) != {spec.fold}):
        raise RuntimeError("OOF 训练/验证 run 隔离失败")
    expected_classes = set(range(len(CLASS_NAMES[spec.stage])))
    if set(np.unique(train_y).tolist()) != expected_classes:
        raise RuntimeError("训练集类别不完整")
    if validation_y is None or set(np.unique(validation_y).tolist()) != expected_classes:
        raise RuntimeError("验证集类别不完整")
    return JobArrays(train_x, train_y, validation_x, validation_y,
                     validation_rows)


# ---------- 可复现实验配置：数据、代码和超参数共同组成不可静默改变的合同 ----------
def git_provenance() -> dict:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT.parent,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT.parent,
            check=True, capture_output=True, text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = "unavailable", None
    return {"commit": sha, "dirty": dirty}


def execution_fingerprint(device: torch.device) -> dict:
    """冻结会影响逐值恢复的软硬件身份；换设备只能建立新输出根。"""
    gpu = None
    if device.type == "cuda":
        gpu = {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        }
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": gpu,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "allow_tf32": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def runtime_provenance(device: torch.device) -> dict:
    return {
        "timestamp_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "execution_fingerprint": execution_fingerprint(device),
        "git": git_provenance(),
    }


def source_hashes() -> dict[str, str]:
    paths = {
        "trainer": Path(__file__).resolve(),
        "oof_training_bundle_reader": Path(__file__).with_name("oof_training_bundle.py"),
        "model_factory": MODELS_DIR / "model_factory.py",
        "eegnet": MODELS_DIR / "models" / "eegnet.py",
    }
    return {name: file_hash(path) for name, path in paths.items()}


def make_contract(context: TrainingContext, spec: JobSpec, arrays: JobArrays,
                  args: argparse.Namespace, device: torch.device) -> dict:
    entry = context.manifest["folds"][spec.fold]
    return {
        "training_protocol_id": TRAINING_ID.format(subject=spec.subject),
        "job": asdict(spec),
        "model": {
            "name": "eegnet",
            "classes": list(CLASS_NAMES[spec.stage]),
            "channels": int(arrays.train_x.shape[1]),
            "samples": int(arrays.train_x.shape[2]),
        },
        "optimization": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "validation_batch_size": args.validation_batch_size,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "loss": "balanced_cross_entropy",
            "early_stopping": False,
            "deterministic_algorithms": True,
            "execution_fingerprint": execution_fingerprint(device),
        },
        "validation_domains": list(spec.validation_domains),
        "data": {
            "session": 0,
            "train_runs": entry["train_runs"],
            "validation_runs": [spec.fold],
            "train_window_count": len(arrays.train_y),
            "train_window_sha256": entry[f"train_stage{spec.stage}"]["window_sha256"],
            "validation_window_count": len(arrays.validation_y),
            "validation_window_sha256": window_identity_hash(arrays.validation_rows),
            "training_bundle_protocol_id": context.manifest["protocol_id"],
            "training_bundle_manifest_sha256": context.manifest_sha256,
            "normalization_protocol_id": context.manifest["normalization_protocol_id"],
            "source_provenance": context.manifest["source_provenance"],
        },
        "source_sha256": source_hashes(),
        "test_session_access": "forbidden",
    }


# ---------- 随机性与指标：每个 epoch 的洗牌独立定种，恢复后顺序完全相同 ----------
def configure_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def capture_rng(device: torch.device) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        if "torch_cuda" not in state:
            raise RuntimeError("CUDA 作业检查点缺少 CUDA RNG 状态")
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def balanced_class_weights(labels: np.ndarray, classes: int,
                           device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=classes).astype(np.float64)
    if len(counts) != classes or np.any(counts <= 0):
        raise RuntimeError(f"无法从空类别计算权重: {counts.tolist()}")
    values = counts.sum() / (classes * counts)
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def classification_metrics(labels: np.ndarray, logits: np.ndarray,
                           classes: int) -> dict:
    if logits.shape != (len(labels), classes) or not np.isfinite(logits).all():
        raise RuntimeError("验证 logit 形状或数值非法")
    predictions = logits.argmax(axis=1)
    recalls = []
    for label in range(classes):
        mask = labels == label
        if not mask.any():
            raise RuntimeError(f"验证集缺少类别 {label}")
        recalls.append(float((predictions[mask] == label).mean()))
    return {
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_class_recall": recalls,
    }


def move_optimizer_state(optimizer: torch.optim.Optimizer,
                         device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def train_one_epoch(model: nn.Module, optimizer: torch.optim.Optimizer,
                    loss_function: nn.Module, arrays: JobArrays,
                    device: torch.device, batch_size: int,
                    seed: int, epoch: int) -> dict:
    generator = torch.Generator()
    generator.manual_seed(seed * 100_003 + epoch)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(arrays.train_x),
                      torch.from_numpy(arrays.train_y)),
        batch_size=batch_size, shuffle=True, generator=generator,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    model.train()
    total_loss = total_correct = total_count = 0
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = loss_function(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("训练 loss 出现非有限值")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError("训练梯度出现非有限值")
        optimizer.step()
        total_count += len(labels)
        total_loss += float(loss.detach()) * len(labels)
        total_correct += int((logits.argmax(dim=1) == labels).sum())
    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
        "sample_count": total_count,
    }


@torch.inference_mode()
def predict_logits(model: nn.Module, features: np.ndarray,
                   device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)), batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=device.type == "cuda",
    )
    model.eval()
    chunks = [model(batch[0].to(device, non_blocking=True)).cpu().numpy()
              for batch in loader]
    logits = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
    if not np.isfinite(logits).all():
        raise RuntimeError("模型输出非有限 logit")
    return logits


# ---------- 作业持久化：检查点是恢复真值，OOF 文件只保存原始 logit 与窗口身份 ----------
def checkpoint_payload(contract_hash: str, epoch: int, model: nn.Module,
                       optimizer: torch.optim.Optimizer, device: torch.device,
                       history: list[dict], oof_logits: dict[str, list[np.ndarray]]) -> dict:
    return {
        "format_version": 1,
        "contract_sha256": contract_hash,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": capture_rng(device),
        "history": history,
        "oof_logits": {
            domain: np.stack(values).astype(np.float32, copy=False)
            for domain, values in oof_logits.items()
        },
    }


def write_public_job_artifacts(job_dir: Path, contract_hash: str,
                               arrays: JobArrays, history: list[dict],
                               oof_logits: dict[str, list[np.ndarray]],
                               model: nn.Module, status: str) -> None:
    epochs = np.asarray([item["epoch"] for item in history], dtype=np.int16)
    archive: dict[str, np.ndarray] = {
        "epochs": epochs,
        "validation_rows": arrays.validation_rows,
        "validation_labels": arrays.validation_y.astype(np.int64, copy=False),
        "validation_window_sha256": np.asarray(
            window_identity_hash(arrays.validation_rows)
        ),
    }
    for domain, values in oof_logits.items():
        archive[f"{domain}_logits"] = np.stack(values).astype(np.float32, copy=False)
    atomic_npz(job_dir / "oof_predictions.npz", archive)
    atomic_json(job_dir / "history.json", history)
    atomic_json(job_dir / "status.json", {
        "status": status,
        "contract_sha256": contract_hash,
        "completed_epochs": len(history),
        "latest_epoch": int(epochs[-1]) if len(epochs) else 0,
        "model_tensor_sha256": tensor_state_hash(model.state_dict()),
        "updated_at_utc": utc_now(),
    })


def public_artifact_hashes(job_dir: Path) -> dict[str, str]:
    """完成标记绑定可重建产物；checkpoint 另以独立字段绑定。"""
    names = ("job_config.json", "oof_predictions.npz", "history.json", "status.json")
    missing = [name for name in names if not (job_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"作业缺少公开产物: {missing}")
    return {name: file_hash(job_dir / name) for name in names}


def load_checkpoint(path: Path, contract_hash: str, model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> tuple[int, list[dict], dict[str, list[np.ndarray]]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (checkpoint.get("format_version") != 1 or
            checkpoint.get("contract_sha256") != contract_hash):
        raise RuntimeError("检查点版本或实验合同不匹配，禁止续跑")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    move_optimizer_state(optimizer, device)
    history = checkpoint["history"]
    epoch = int(checkpoint["epoch"])
    if len(history) != epoch or [item["epoch"] for item in history] != list(range(1, epoch + 1)):
        raise RuntimeError("检查点 epoch 与 history 不一致")
    oof_logits = {
        domain: [np.asarray(value, dtype=np.float32) for value in stacked]
        for domain, stacked in checkpoint["oof_logits"].items()
    }
    if any(len(values) != epoch for values in oof_logits.values()):
        raise RuntimeError("检查点 OOF logit 数量与 epoch 不一致")
    restore_rng(checkpoint["rng_state"], device)
    return epoch, history, oof_logits


def run_job(context: TrainingContext, spec: JobSpec, arrays: JobArrays,
            args: argparse.Namespace, device: torch.device,
            progress: Callable[[str, int], None] | None = None) -> str:
    job_dir = args.output_root / spec.name
    job_dir.mkdir(parents=True, exist_ok=True)
    contract = make_contract(context, spec, arrays, args, device)
    contract_hash = canonical_hash(contract)
    config_path = job_dir / "job_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("contract_sha256") != contract_hash or existing.get("contract") != contract:
            raise RuntimeError(f"{spec.name}: 已有目录的实验合同不同，禁止覆盖")
    else:
        atomic_json(config_path, {
            "contract_sha256": contract_hash,
            "contract": contract,
            "initial_runtime": runtime_provenance(device),
        })

    configure_determinism(spec.seed)
    classes = len(CLASS_NAMES[spec.stage])
    model = build_model(
        "eegnet", classes, arrays.train_x.shape[1], arrays.train_x.shape[2]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_function = nn.CrossEntropyLoss(
        weight=balanced_class_weights(arrays.train_y, classes, device)
    )
    checkpoint_path = job_dir / "latest.pt"
    checkpoint_exists = checkpoint_path.exists()
    if checkpoint_exists:
        if args.no_resume:
            raise RuntimeError(f"{spec.name}: 已有检查点且指定了 --no-resume")
        completed_epoch, history, oof_logits = load_checkpoint(
            checkpoint_path, contract_hash, model, optimizer, device
        )
    else:
        completed_epoch, history = 0, []
        oof_logits = {domain: [] for domain in spec.validation_domains}

    # completed.json 是最后提交点，但每次重启仍从 checkpoint 重建并核验派生产物。
    completed_path = job_dir / "completed.json"
    if completed_path.exists():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        model_hash = tensor_state_hash(model.state_dict())
        if (completed.get("contract_sha256") != contract_hash or
                completed.get("completed_epochs") != args.epochs or
                completed_epoch != args.epochs or
                completed.get("model_tensor_sha256") != model_hash or
                completed.get("checkpoint_sha256") != file_hash(checkpoint_path)):
            raise RuntimeError(f"{spec.name}: 完成标记与 checkpoint 不一致")
        try:
            artifact_hashes = public_artifact_hashes(job_dir)
        except FileNotFoundError:
            artifact_hashes = {}
        if completed.get("artifact_sha256") != artifact_hashes:
            # 派生产物可由已绑定的 checkpoint 安全重建，随后重新提交完成标记。
            write_public_job_artifacts(
                job_dir, contract_hash, arrays, history, oof_logits, model, "complete"
            )
            completed["artifact_sha256"] = public_artifact_hashes(job_dir)
            completed["repaired_at_utc"] = utc_now()
            atomic_json(completed_path, completed)
        return "already_complete"

    if checkpoint_exists:
        write_public_job_artifacts(
            job_dir, contract_hash, arrays, history, oof_logits, model, "running"
        )

    for epoch in range(completed_epoch + 1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, optimizer, loss_function, arrays, device,
            args.batch_size, spec.seed, epoch,
        )
        validation: dict[str, dict] = {}
        for domain in spec.validation_domains:
            logits = predict_logits(
                model, arrays.validation_x[domain], device,
                args.validation_batch_size,
            )
            oof_logits[domain].append(logits)
            validation[domain] = classification_metrics(
                arrays.validation_y, logits, classes
            )
        history.append({"epoch": epoch, "train": train_metrics,
                        "validation": validation})

        # 先提交包含全部状态的检查点，再派生可读文件；崩溃后以检查点重建即可。
        atomic_torch_save(
            checkpoint_path,
            checkpoint_payload(
                contract_hash, epoch, model, optimizer, device, history, oof_logits
            ),
        )
        job_status = ("paused" if args.stop_after_epoch == epoch else "running")
        write_public_job_artifacts(
            job_dir, contract_hash, arrays, history, oof_logits, model, job_status
        )
        if progress:
            progress(spec.name, epoch)
        latest = validation[spec.validation_domains[0]]["balanced_accuracy"]
        print(
            f"{spec.name} epoch={epoch:02d}/{args.epochs} "
            f"loss={train_metrics['loss']:.5f} "
            f"val_ba={latest:.4f}",
            flush=True,
        )
        if args.stop_after_epoch == epoch:
            return "paused"

    model_hash = tensor_state_hash(model.state_dict())
    # 先完成所有可重建产物，再把 completed.json 作为唯一最终提交点。
    write_public_job_artifacts(
        job_dir, contract_hash, arrays, history, oof_logits, model, "complete"
    )
    atomic_json(completed_path, {
        "status": "complete",
        "contract_sha256": contract_hash,
        "completed_epochs": args.epochs,
        "model_tensor_sha256": model_hash,
        "checkpoint_sha256": file_hash(checkpoint_path),
        "artifact_sha256": public_artifact_hashes(job_dir),
        "completed_at_utc": utc_now(),
    })
    return "complete"


# ---------- 矩阵调度：单 GPU 顺序执行，跨 seed 复用已物化的同一批窗口 ----------
def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 无可用 GPU；禁止静默回退 CPU")
    if device.type not in ("cuda", "cpu"):
        raise ValueError("训练器目前只接受 cuda 或 cpu")
    return device


def validate_hyperparameters(args: argparse.Namespace, job_count: int) -> None:
    if (args.epochs <= 0 or args.batch_size <= 0 or
            args.validation_batch_size <= 0 or args.learning_rate <= 0 or
            args.weight_decay < 0):
        raise ValueError("epoch、batch 和优化器超参数必须为正，weight decay 可为零")
    if args.stop_after_epoch is not None:
        if job_count != 1:
            raise ValueError("--stop-after-epoch 仅允许单作业恢复预检")
        if not 1 <= args.stop_after_epoch <= args.epochs:
            raise ValueError("暂停 epoch 必须落在训练范围内")


def main() -> None:
    args = parse_args()
    specs = build_job_specs(
        args.subject, args.folds, args.stages, args.train_domains, args.seeds
    )
    validate_hyperparameters(args, len(specs))
    if args.training_bundle is None:
        args.training_bundle = (
            PROJECT_ROOT / "data" / "processed" /
            BUNDLE_ID.format(subject=args.subject) / "manifest.json"
        )
    args.output_root = args.output_root.resolve()
    context = load_context(args.training_bundle.resolve(), args.subject)
    device = resolve_device(args.device)

    matrix_contract = {
        "training_protocol_id": TRAINING_ID.format(subject=args.subject),
        "jobs": [asdict(spec) for spec in specs],
        "job_count": len(specs),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "validation_batch_size": args.validation_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "training_bundle_protocol_id": context.manifest["protocol_id"],
        "training_bundle_manifest_sha256": context.manifest_sha256,
        "execution_fingerprint": execution_fingerprint(device),
        "source_sha256": source_hashes(),
        "test_session_access": "forbidden",
    }
    print(json.dumps(matrix_contract, ensure_ascii=False, indent=2), flush=True)
    if args.plan_only:
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    matrix_path = args.output_root / "matrix_manifest.json"
    matrix_hash = canonical_hash(matrix_contract)
    if matrix_path.exists():
        existing = json.loads(matrix_path.read_text(encoding="utf-8"))
        if (existing.get("matrix_contract_sha256") != matrix_hash or
                existing.get("matrix_contract") != matrix_contract):
            raise RuntimeError("输出根目录已有不同训练矩阵，禁止混写")
    else:
        atomic_json(matrix_path, {
            "matrix_contract_sha256": matrix_hash,
            "matrix_contract": matrix_contract,
            "initial_runtime": runtime_provenance(device),
        })

    started = time.monotonic()
    completed_names = [
        spec.name for spec in specs
        if (args.output_root / spec.name / "completed.json").is_file()
    ]

    current_job_name: str | None = None
    current_epoch = 0

    def update_progress(job_name: str, epoch: int) -> None:
        nonlocal current_job_name, current_epoch
        current_job_name, current_epoch = job_name, epoch
        done = [
            spec.name for spec in specs
            if (args.output_root / spec.name / "completed.json").is_file()
        ]
        atomic_json(args.output_root / "matrix_status.json", {
            "status": "running",
            "job_count": len(specs),
            "completed_job_count": len(done),
            "completed_jobs": done,
            "current_job": job_name,
            "current_epoch": epoch,
            "elapsed_seconds_this_invocation": time.monotonic() - started,
            "updated_at_utc": utc_now(),
        })

    cache_key: tuple[int, int, str] | None = None
    cached_arrays: JobArrays | None = None
    try:
        for spec in specs:
            current_job_name, current_epoch = spec.name, 0
            key = (spec.fold, spec.stage, spec.train_domain)
            if key != cache_key:
                cached_arrays = prepare_job_arrays(context, spec)
                cache_key = key
            assert cached_arrays is not None
            update_progress(spec.name, 0)
            result = run_job(
                context, spec, cached_arrays, args, device, update_progress
            )
            if result == "paused":
                atomic_json(args.output_root / "matrix_status.json", {
                    "status": "paused_for_resume_preflight",
                    "current_job": spec.name,
                    "current_epoch": args.stop_after_epoch,
                    "updated_at_utc": utc_now(),
                })
                return
    except BaseException as exc:
        completed_names = [
            spec.name for spec in specs
            if (args.output_root / spec.name / "completed.json").is_file()
        ]
        if current_job_name is not None:
            job_dir = args.output_root / current_job_name
            atomic_json(job_dir / "status.json", {
                "status": "failed_recoverable",
                "current_job": current_job_name,
                "latest_complete_epoch": current_epoch,
                "checkpoint_exists": (job_dir / "latest.pt").is_file(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "updated_at_utc": utc_now(),
            })
        atomic_json(args.output_root / "matrix_status.json", {
            "status": "failed",
            "job_count": len(specs),
            "completed_job_count": len(completed_names),
            "completed_jobs": completed_names,
            "current_job": current_job_name,
            "current_epoch": current_epoch,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "updated_at_utc": utc_now(),
        })
        raise

    completed_names = [
        spec.name for spec in specs
        if (args.output_root / spec.name / "completed.json").is_file()
    ]
    if len(completed_names) != len(specs):
        raise RuntimeError("矩阵循环结束但完成标记数量不足")
    atomic_json(args.output_root / "matrix_status.json", {
        "status": "complete",
        "job_count": len(specs),
        "completed_job_count": len(completed_names),
        "completed_jobs": completed_names,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "updated_at_utc": utc_now(),
    })


if __name__ == "__main__":
    main()
