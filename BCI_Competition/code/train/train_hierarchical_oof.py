# =============================================================================
# Implementation of: Hierarchical OOF Idle/Task and Motor Imagery Training
#
# Reference:
#   Project-specific implementation for BNCI2014001 asynchronous decoding.
#   Stage 1 learns idle-vs-task detection; Stage 2 learns four-class MI only
#   from task windows. OOF folds are leave-one-train-run-out.
#
# Source: No external code copied.
# =============================================================================
"""Train simplified OOF two-stage BNCI2014001 models, then final all-run models."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path("BCI_Competition") if Path("BCI_Competition/code").is_dir() else Path(".")
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from models.model_factory import available_models, build_model, model_source, model_source_id, normalize_model_name


DATA_FILE = PROJECT_ROOT / "data" / "processed" / "bnci2014001_oof_windows.npz"
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints" / "simple_oof"
TABLE_DIR = PROJECT_ROOT / "results" / "tables" / "simple_oof"
BINARY_CLASS_NAMES = ["idle", "task"]
MI_CLASS_NAMES = ["left_hand", "right_hand", "feet", "tongue"]
FINAL_CLASS_NAMES = ["idle", *MI_CLASS_NAMES]
REQUIRED_SCHEMA = "bnci2014001_causal_windows_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DATA_FILE)
    parser.add_argument("--model", default="eegnet", choices=available_models())
    parser.add_argument("--subjects", nargs="+", type=int, default=[1], help="subjects to train; use all available if omitted with --all-subjects")
    parser.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--binary-epochs", type=int, default=30)
    parser.add_argument("--mi-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weight", choices=("none", "balanced"), default="balanced")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        required = [
            "X", "y", "subject", "session", "run", "fold", "split",
            "schema_version", "dataset_id", "dataset_config",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise RuntimeError(f"Missing arrays in {path}: {missing}")
        if str(data["schema_version"].item()) != REQUIRED_SCHEMA:
            raise RuntimeError(f"Expected data schema {REQUIRED_SCHEMA}")
        return {key: data[key] for key in required}


def normalize_by_train(features: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features[train_mask].mean(axis=(0, 2), keepdims=True)
    std = features[train_mask].std(axis=(0, 2), keepdims=True).clip(min=1e-6)
    return ((features - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def class_weights(labels: np.ndarray, num_classes: int, device: torch.device, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    if np.any(counts == 0):
        raise RuntimeError(f"Cannot compute balanced weights with empty classes: {counts.tolist()}")
    weights = counts.sum() / (num_classes * counts)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def train_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    device: torch.device,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weights: torch.Tensor | None,
    name: str,
) -> None:
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = correct = count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name} loss is not finite")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * batch_y.numel()
            correct += int((logits.argmax(dim=1) == batch_y).sum().item())
            count += batch_y.numel()
        print(f"{name} epoch={epoch:03d} loss={loss_sum / count:.4f} acc={correct / count:.3f}")


@torch.no_grad()
def logits_for(model: nn.Module, features: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start : start + batch_size]).to(device)
        output.append(model(batch).cpu().numpy())
    return np.concatenate(output, axis=0)


def hierarchical_from_logits(binary_logits: np.ndarray, mi_logits: np.ndarray) -> np.ndarray:
    binary_pred = binary_logits.argmax(axis=1)
    mi_pred = mi_logits.argmax(axis=1) + 1
    return np.where(binary_pred == 1, mi_pred, 0).astype(np.int64)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> dict:
    labels = list(range(len(names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=names, zero_division=0, output_dict=True
        ),
    }


# 运行身份只由真正影响训练的生效配置生成，同一身份的所有产物共用同一前缀。
def effective_training_config(args: argparse.Namespace, arrays: dict[str, np.ndarray]) -> dict:
    """Return the complete model-affecting configuration stored with every artifact."""
    return {
        "data_schema": REQUIRED_SCHEMA,
        "dataset_id": str(arrays["dataset_id"].item()),
        "dataset_config": json.loads(str(arrays["dataset_config"].item())),
        "data_sha256": file_sha256(args.data_file),
        "source_id": source_fingerprint(args.model),
        "model_source_id": model_source_id(args.model),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "trainer": "hierarchical_oof_v2",
        "model": args.model,
        "binary_epochs": args.binary_epochs,
        "mi_epochs": args.mi_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "class_weight": args.class_weight,
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


def source_fingerprint(model_name: str) -> str:
    digest = hashlib.sha256()
    paths = [Path(__file__).resolve(), PROJECT_ROOT / "code" / "models" / "model_factory.py", model_source(model_name)]
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def artifact_paths(subject: int, model_name: str, experiment_id: str) -> dict[str, Path]:
    prefix = f"s{subject:02d}_{model_name}_{experiment_id}"
    return {
        "checkpoint": CHECKPOINT_DIR / f"{prefix}_final.pt",
        "predictions": TABLE_DIR / f"{prefix}_predictions.npz",
        "metrics": TABLE_DIR / f"{prefix}_metrics.json",
    }


def ensure_writable(paths: list[Path], overwrite: bool) -> None:
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate artifact targets")
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"artifacts already exist; pass --overwrite to replace them:\n{joined}")


def train_stage_pair(
    model_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
    tag: str,
) -> tuple[nn.Module, nn.Module]:
    chans, samples = features.shape[1], features.shape[2]
    binary_model = build_model(model_name, 2, chans, samples).to(device)
    mi_model = build_model(model_name, 4, chans, samples).to(device)

    binary_y = (labels[train_mask] > 0).astype(np.int64)
    task_train_mask = train_mask & (labels > 0)
    mi_y = labels[task_train_mask].astype(np.int64) - 1
    if not task_train_mask.any():
        raise RuntimeError(f"No task windows for {tag}")

    train_model(
        binary_model,
        features[train_mask],
        binary_y,
        device,
        args.batch_size,
        args.binary_epochs,
        args.learning_rate,
        class_weights(binary_y, 2, device, args.class_weight),
        f"{tag}/stage1",
    )
    train_model(
        mi_model,
        features[task_train_mask],
        mi_y,
        device,
        args.batch_size,
        args.mi_epochs,
        args.learning_rate,
        class_weights(mi_y, 4, device, args.class_weight),
        f"{tag}/stage2",
    )
    return binary_model, mi_model


def run_subject(
    subject: int,
    arrays: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    training_config: dict,
    experiment_id: str,
) -> dict:
    # 每个受试者从同一显式 seed 独立开始，结果不依赖本次命令还训练了哪些受试者。
    set_seed(args.seed)
    model_name = normalize_model_name(args.model)
    subject_mask = arrays["subject"] == subject
    train_session = subject_mask & (arrays["split"] == 0)
    if not train_session.any():
        raise RuntimeError(f"No train-session windows for subject {subject}")

    raw_x = arrays["X"].astype(np.float32)
    y = arrays["y"].astype(np.int64)
    folds = sorted(int(fold) for fold in np.unique(arrays["fold"][train_session]) if fold >= 0)
    oof_binary_logits = np.full((train_session.sum(), 2), np.nan, dtype=np.float32)
    oof_mi_logits = np.full((train_session.sum(), 4), np.nan, dtype=np.float32)
    oof_indices = np.where(train_session)[0]
    fold_reports: list[dict] = []

    for fold in folds:
        val_mask = train_session & (arrays["fold"] == fold)
        fold_train_mask = train_session & (arrays["fold"] != fold)
        print(f"\nSubject {subject:02d} fold {fold}: train={fold_train_mask.sum()} val={val_mask.sum()}")
        x_norm, mean, std = normalize_by_train(raw_x, fold_train_mask)
        binary_model, mi_model = train_stage_pair(model_name, x_norm, y, fold_train_mask, device, args, f"s{subject:02d}/fold{fold}")
        val_binary_logits = logits_for(binary_model, x_norm[val_mask], device, args.batch_size)
        val_mi_logits = logits_for(mi_model, x_norm[val_mask], device, args.batch_size)
        local_positions = np.searchsorted(oof_indices, np.where(val_mask)[0])
        oof_binary_logits[local_positions] = val_binary_logits
        oof_mi_logits[local_positions] = val_mi_logits
        val_pred = hierarchical_from_logits(val_binary_logits, val_mi_logits)
        fold_reports.append(
            {
                "fold": fold,
                "train_windows": int(fold_train_mask.sum()),
                "val_windows": int(val_mask.sum()),
                "final_5class": metrics(y[val_mask], val_pred, FINAL_CLASS_NAMES),
            }
        )

    if np.isnan(oof_binary_logits).any() or np.isnan(oof_mi_logits).any():
        raise RuntimeError(f"OOF logits incomplete for subject {subject}")

    oof_y = y[train_session]
    oof_pred = hierarchical_from_logits(oof_binary_logits, oof_mi_logits)
    oof_metrics = {
        "final_5class": metrics(oof_y, oof_pred, FINAL_CLASS_NAMES),
        "stage1_binary": metrics((oof_y > 0).astype(np.int64), oof_binary_logits.argmax(axis=1), BINARY_CLASS_NAMES),
        "stage2_mi_on_true_task_windows": metrics(
            oof_y[oof_y > 0] - 1,
            oof_mi_logits[oof_y > 0].argmax(axis=1),
            MI_CLASS_NAMES,
        ),
    }

    print(f"\nSubject {subject:02d}: final all-run training")
    final_x, final_mean, final_std = normalize_by_train(raw_x, train_session)
    final_binary, final_mi = train_stage_pair(model_name, final_x, y, train_session, device, args, f"s{subject:02d}/final")
    train_binary_logits = logits_for(final_binary, final_x[train_session], device, args.batch_size)
    train_mi_logits = logits_for(final_mi, final_x[train_session], device, args.batch_size)
    train_pred = hierarchical_from_logits(train_binary_logits, train_mi_logits)
    train_metrics = {
        "final_5class": metrics(oof_y, train_pred, FINAL_CLASS_NAMES),
        "stage1_binary": metrics((oof_y > 0).astype(np.int64), train_binary_logits.argmax(axis=1), BINARY_CLASS_NAMES),
        "stage2_mi_on_true_task_windows": metrics(
            oof_y[oof_y > 0] - 1,
            train_mi_logits[oof_y > 0].argmax(axis=1),
            MI_CLASS_NAMES,
        ),
    }

    paths = artifact_paths(subject, model_name, experiment_id)
    checkpoint, prediction_file, metrics_file = paths["checkpoint"], paths["predictions"], paths["metrics"]
    run_id = checkpoint.stem.removesuffix("_final")
    provenance = {"data_file": str(args.data_file.resolve())}

    # checkpoint 先落盘并计算内容哈希，随后预测和指标都绑定这一个精确模型文件。
    torch.save(
        {
            "run_id": run_id,
            "model": model_name,
            "model_source": str(model_source(model_name)),
            "subject": subject,
            "seed": args.seed,
            "training_config": training_config,
            "training_provenance": provenance,
            "binary_state_dict": final_binary.state_dict(),
            "mi_state_dict": final_mi.state_dict(),
            "mean": final_mean,
            "std": final_std,
            "classes": {"binary": BINARY_CLASS_NAMES, "mi": MI_CLASS_NAMES, "final": FINAL_CLASS_NAMES},
        },
        checkpoint,
    )
    checkpoint_sha256 = file_sha256(checkpoint)
    np.savez_compressed(
        prediction_file,
        run_id=np.asarray(run_id),
        subject=np.asarray(subject, dtype=np.int64),
        model=np.asarray(model_name),
        seed=np.asarray(args.seed, dtype=np.int64),
        dataset_id=np.asarray(training_config["dataset_id"]),
        checkpoint_sha256=np.asarray(checkpoint_sha256),
        training_config=np.asarray(json.dumps(training_config, sort_keys=True, ensure_ascii=True)),
        training_provenance=np.asarray(json.dumps(provenance, sort_keys=True, ensure_ascii=True)),
        index=oof_indices,
        y_true=oof_y,
        oof_binary_logits=oof_binary_logits,
        oof_mi_logits=oof_mi_logits,
        oof_pred=oof_pred,
        final_train_binary_logits=train_binary_logits,
        final_train_mi_logits=train_mi_logits,
        final_train_pred=train_pred,
    )
    report = {
        "dataset": "BNCI2014001",
        "run_id": run_id,
        "subject": subject,
        "method": "leave_one_train_run_out_oof_then_final_all_train_runs",
        "model": model_name,
        "seed": args.seed,
        "training_config": training_config,
        "training_provenance": provenance,
        "checkpoint_sha256": checkpoint_sha256,
        "folds": folds,
        "fold_reports": fold_reports,
        "oof_metrics": oof_metrics,
        "final_all_run_train_metrics": train_metrics,
        "checkpoint": checkpoint.as_posix(),
        "prediction_file": prediction_file.as_posix(),
    }
    metrics_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved final model: {checkpoint}")
    print(f"Saved predictions: {prediction_file}")
    print(f"Saved metrics: {metrics_file}")
    return report


def main() -> None:
    args = parse_args()
    args.model = normalize_model_name(args.model)
    arrays = load_arrays(args.data_file)
    subjects = sorted(np.unique(arrays["subject"]).astype(int).tolist()) if args.all_subjects else args.subjects
    if len(set(subjects)) != len(subjects):
        raise ValueError("subjects must not contain duplicates")
    subjects = sorted(subjects)
    training_config = effective_training_config(args, arrays)
    experiment_id = f"seed{args.seed}_{config_fingerprint(training_config)}"
    subject_tag = "-".join(f"s{subject:02d}" for subject in subjects)
    summary_file = TABLE_DIR / f"{args.model}_{experiment_id}_{subject_tag}_oof_summary.json"
    # 在启动耗时训练前一次性检查全部目标，避免中途才发现覆盖冲突。
    planned = [path for subject in subjects for path in artifact_paths(subject, args.model, experiment_id).values()]
    ensure_writable([*planned, summary_file], args.overwrite)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}, model={args.model}")
    reports = [run_subject(subject, arrays, args, device, training_config, experiment_id) for subject in subjects]
    summary = {
        "experiment_id": experiment_id, "subjects": subjects,
        "training_config": training_config,
        "training_provenance": {"data_file": str(args.data_file.resolve())}, "reports": reports,
    }
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary: {summary_file}")


if __name__ == "__main__":
    main()
