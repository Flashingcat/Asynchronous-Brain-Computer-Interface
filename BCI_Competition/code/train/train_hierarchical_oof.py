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
import json
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

from models.model_factory import available_models, build_model, model_source, normalize_model_name
from experiment_identity import build_experiment


DATA_FILE = PROJECT_ROOT / "data" / "processed" / "bnci2014001_oof_windows.npz"
CHECKPOINT_ROOT = PROJECT_ROOT / "results" / "checkpoints"
TABLE_ROOT = PROJECT_ROOT / "results" / "tables"
BINARY_CLASS_NAMES = ["idle", "task"]
MI_CLASS_NAMES = ["left_hand", "right_hand", "feet", "tongue"]
FINAL_CLASS_NAMES = ["idle", *MI_CLASS_NAMES]


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
    parser.add_argument("--run-mode", choices=("final", "validation", "both"), default="final")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        required = ["X", "y", "subject", "session", "run", "fold", "split", "is_pure"]
        missing = [key for key in required if key not in data]
        if missing:
            raise RuntimeError(f"Missing arrays in {path}: {missing}")
        return {key: data[key] for key in required}


def normalize_by_train(features: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features[train_mask].mean(axis=(0, 2), keepdims=True)
    std = features[train_mask].std(axis=(0, 2), keepdims=True).clip(min=1e-6)
    return ((features - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def validation_masks(arrays: dict[str, np.ndarray], subject: int, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """返回无泄漏训练掩码和完整连续留出运行掩码。"""
    session = (arrays["subject"] == subject) & (arrays["split"] == 0)
    train = session & arrays["is_pure"].astype(bool) & (arrays["fold"] != fold)
    validation = session & (arrays["fold"] == fold)
    return train, validation


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


# 同一实验身份的检查点和报告写入同一目录，避免不同配置互相覆盖。
def artifact_paths(subject: int, args: argparse.Namespace) -> dict[str, Path]:
    checkpoint_dir = CHECKPOINT_ROOT / args.experiment_id / f"subject_{subject:02d}"
    table_dir = TABLE_ROOT / args.experiment_id / f"subject_{subject:02d}"
    return {
        "checkpoint": checkpoint_dir / "final.pt",
        "predictions": table_dir / "final_predictions.npz",
        "metrics": table_dir / "final_metrics.json",
    }


def fold_checkpoint_path(subject: int, fold: int, args: argparse.Namespace) -> Path:
    return CHECKPOINT_ROOT / args.experiment_id / f"subject_{subject:02d}" / f"fold_{fold}.pt"

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
) -> dict:
    """训练并保存逐运行留一模型，评估由统一 evaluator 完成。"""
    set_seed(args.seed)
    model_name = normalize_model_name(args.model)
    raw_x = arrays["X"].astype(np.float32)
    y = arrays["y"].astype(np.int64)
    train_session = (arrays["subject"] == subject) & (arrays["split"] == 0)
    folds = sorted(int(fold) for fold in np.unique(arrays["fold"][train_session]) if fold >= 0)
    reports = []
    for fold in folds:
        # 训练只用其他运行的纯窗口；验证保留留出运行的完整连续流。
        fold_train_mask, val_mask = validation_masks(arrays, subject, fold)
        print(f"\nSubject {subject:02d} fold {fold}: train={fold_train_mask.sum()} val={val_mask.sum()}")
        x_norm, mean, std = normalize_by_train(raw_x, fold_train_mask)
        binary_model, mi_model = train_stage_pair(model_name, x_norm, y, fold_train_mask, device, args, f"s{subject:02d}/fold{fold}")
        fold_checkpoint = fold_checkpoint_path(subject, fold, args)
        fold_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model_name,
                "model_source": str(model_source(model_name)),
                "subject": subject,
                "seed": args.seed,
                "augmentation": ["none"],
                "experiment_id": args.experiment_id,
                "experiment": args.experiment,
                "checkpoint_role": "validation_fold",
                "held_out_run": fold,
                "binary_state_dict": binary_model.state_dict(),
                "mi_state_dict": mi_model.state_dict(),
                "mean": mean,
                "std": std,
                "classes": {"binary": BINARY_CLASS_NAMES, "mi": MI_CLASS_NAMES, "final": FINAL_CLASS_NAMES},
            },
            fold_checkpoint,
        )
        reports.append({"fold": fold, "train_windows": int(fold_train_mask.sum()),
                        "val_windows": int(val_mask.sum()), "checkpoint": fold_checkpoint.as_posix()})
    return {"subject": subject, "model": model_name, "seed": args.seed, "folds": reports}


def run_final_subject(
    subject: int,
    arrays: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    """Train one final two-stage model from all labelled train-session windows."""
    set_seed(args.seed)
    model_name = normalize_model_name(args.model)
    # 最终模型同样只学习无边界歧义的纯窗口。
    train_mask = (arrays["subject"] == subject) & (arrays["split"] == 0) & arrays["is_pure"].astype(bool)
    if not train_mask.any():
        raise RuntimeError(f"No train-session windows for subject {subject}")

    raw_x = arrays["X"].astype(np.float32)
    labels = arrays["y"].astype(np.int64)
    x_norm, mean, std = normalize_by_train(raw_x, train_mask)
    print(f"\nSubject {subject:02d}: final all-train-session model ({train_mask.sum()} windows)")
    binary_model, mi_model = train_stage_pair(model_name, x_norm, labels, train_mask, device, args, f"s{subject:02d}/final")
    binary_logits = logits_for(binary_model, x_norm[train_mask], device, args.batch_size)
    mi_logits = logits_for(mi_model, x_norm[train_mask], device, args.batch_size)
    prediction = hierarchical_from_logits(binary_logits, mi_logits)
    train_metrics = {
        "final_5class": metrics(labels[train_mask], prediction, FINAL_CLASS_NAMES),
        "stage1_binary": metrics((labels[train_mask] > 0).astype(np.int64), binary_logits.argmax(axis=1), BINARY_CLASS_NAMES),
        "stage2_mi_on_true_task_windows": metrics(
            labels[train_mask][labels[train_mask] > 0] - 1,
            mi_logits[labels[train_mask] > 0].argmax(axis=1),
            MI_CLASS_NAMES,
        ),
    }

    paths = artifact_paths(subject, args)
    checkpoint, prediction_file, metrics_file = paths["checkpoint"], paths["predictions"], paths["metrics"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model_name,
            "model_source": str(model_source(model_name)),
            "subject": subject,
            "seed": args.seed,
            "augmentation": ["none"],
            "experiment_id": args.experiment_id,
            "experiment": args.experiment,
            "checkpoint_role": "final",
            "binary_state_dict": binary_model.state_dict(),
            "mi_state_dict": mi_model.state_dict(),
            "mean": mean,
            "std": std,
            "classes": {"binary": BINARY_CLASS_NAMES, "mi": MI_CLASS_NAMES, "final": FINAL_CLASS_NAMES},
        },
        checkpoint,
    )
    np.savez_compressed(
        prediction_file,
        index=np.where(train_mask)[0],
        y_true=labels[train_mask],
        binary_logits=binary_logits,
        mi_logits=mi_logits,
        prediction=prediction,
    )
    report = {
        "dataset": "BNCI2014001",
        "subject": subject,
        "method": "final_two_stage_all_train_session",
        "model": model_name,
        "seed": args.seed,
        "experiment_id": args.experiment_id,
        "experiment": args.experiment,
        "final_all_train_metrics": train_metrics,
        "checkpoint": checkpoint.as_posix(),
        "prediction_file": prediction_file.as_posix(),
    }
    metrics_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved final model: {checkpoint}")
    return report


def main() -> None:
    args = parse_args()
    args.model = normalize_model_name(args.model)
    arrays = load_arrays(args.data_file)
    subjects = sorted(np.unique(arrays["subject"]).astype(int).tolist()) if args.all_subjects else args.subjects
    if len(set(subjects)) != len(subjects):
        raise ValueError("subjects must not contain duplicates")
    subjects = sorted(subjects)
    args.experiment_id, args.experiment = build_experiment(
        "hierarchical_oof", args.model, args.seed, ["none"], args, args.data_file, PROJECT_ROOT
    )
    summary_dir = TABLE_ROOT / args.experiment_id
    summary_file = summary_dir / f"{args.run_mode}_summary.json"
    # 在启动耗时训练前一次性锁定实验身份和全部目标。
    (CHECKPOINT_ROOT / args.experiment_id).mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}, model={args.model}, experiment_id={args.experiment_id}")
    if args.run_mode == "final":
        reports = [run_final_subject(subject, arrays, args, device) for subject in subjects]
    elif args.run_mode == "validation":
        reports = [run_subject(subject, arrays, args, device) for subject in subjects]
    else:
        reports = []
        for subject in subjects:
            reports.append({
                "validation": run_subject(subject, arrays, args, device),
                "final": run_final_subject(subject, arrays, args, device),
            })
    summary = {
        "experiment_id": args.experiment_id,
        "experiment": args.experiment,
        "subjects": subjects,
        "reports": reports,
    }
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary: {summary_file}")


if __name__ == "__main__":
    main()
