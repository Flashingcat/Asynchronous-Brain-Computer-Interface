"""Standalone training script with data augmentation for BNCI2014001.

Run this INSTEAD of train_hierarchical_oof.py when you want augmentation.
The original train script remains unchanged.

Usage:
    conda run -n medical_img python code/augmentation/train_augmented.py --model eegnet --all-subjects --augment all
    conda run -n medical_img python code/augmentation/train_augmented.py --model eegnet --subjects 1 --augment noise_trial mult_trial neg
"""

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
from augmentation import AugmentationPipeline, AugmentedDataset
from augmentation.augmentations import _BUILTIN, _PAIR_BUILTIN

# ---- shared constants (mirror train_hierarchical_oof.py) ----
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "bnci2014001_oof_windows.npz"
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints" / "augmented"
TABLE_DIR = PROJECT_ROOT / "results" / "tables" / "augmented"
BINARY_CLASS_NAMES = ["idle", "task"]
MI_CLASS_NAMES = ["left_hand", "right_hand", "feet", "tongue"]
FINAL_CLASS_NAMES = ["idle", *MI_CLASS_NAMES]

# All augmentation keys: single-sample + pair-based + none + all
_SINGLE_KEYS = tuple(_BUILTIN)   # noise_trial, mult_trial, neg, freq_shift, noise_ch, flip_ch, scale_ch, time_reverse, mirror, noise_gaussian, noise_sp, noise_poisson, noise_pink
_PAIR_KEYS  = tuple(_PAIR_BUILTIN)  # mixup, ch_mixure, trial_mixup, dwta
AUGMENT_CHOICES = ("none", "all") + _SINGLE_KEYS + _PAIR_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DATA_FILE)
    parser.add_argument("--model", default="eegnet", choices=available_models())
    parser.add_argument("--subjects", nargs="+", type=int, default=[1])
    parser.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--binary-epochs", type=int, default=30)
    parser.add_argument("--mi-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weight", choices=("none", "balanced"), default="balanced")
    parser.add_argument("--augment", nargs="+", default=["all"], choices=AUGMENT_CHOICES,
                        help="Methods: all, none, noise_trial, mult_trial, neg, freq_shift, noise_ch, flip_ch, scale_ch, time_reverse, mirror, noise_gaussian/sp/poisson/pink, mixup, ch_mixure, trial_mixup, dwta")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def class_weights(labels: np.ndarray, num_classes: int, device: torch.device, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    if np.any(counts == 0):
        raise RuntimeError(f"Cannot compute balanced weights with empty classes: {counts.tolist()}")
    weights = counts.sum() / (num_classes * counts)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


# ---- augmented training loop (the only difference from the original) ----

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
    augment_spec: list[str] | None = None,
) -> None:
    feat = torch.from_numpy(train_x)
    lab = torch.from_numpy(train_y)
    if augment_spec and augment_spec != ["none"]:
        # Split keys into single-sample and pair-based families
        if augment_spec == ["all"]:
            single_keys = list(_BUILTIN)
            pair_keys = list(_PAIR_BUILTIN)
        else:
            single_keys = [k for k in augment_spec if k in _BUILTIN]
            pair_keys   = [k for k in augment_spec if k in _PAIR_BUILTIN]
        # Build single-sample pipeline
        pipeline = AugmentationPipeline(single_keys) if single_keys else None
        # Build pair-based augmentor (take first pair key — reference applies one at a time)
        pair_aug = None
        if pair_keys:
            pair_cls = _PAIR_BUILTIN[pair_keys[0]]
            pair_aug = pair_cls()  # uses class defaults
        dataset = AugmentedDataset(feat, lab, pipeline, pair_aug=pair_aug)
    else:
        dataset = TensorDataset(feat, lab)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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


def train_stage_pair(
    model_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
    tag: str,
    augment_spec: list[str] | None = None,
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
        binary_model, features[train_mask], binary_y, device,
        args.batch_size, args.binary_epochs, args.learning_rate,
        class_weights(binary_y, 2, device, args.class_weight),
        f"{tag}/stage1", augment_spec=augment_spec,
    )
    train_model(
        mi_model, features[task_train_mask], mi_y, device,
        args.batch_size, args.mi_epochs, args.learning_rate,
        class_weights(mi_y, 4, device, args.class_weight),
        f"{tag}/stage2", augment_spec=augment_spec,
    )
    return binary_model, mi_model


def run_subject(subject: int, arrays: dict[str, np.ndarray], args: argparse.Namespace, device: torch.device) -> dict:
    model_name = normalize_model_name(args.model)
    subject_mask = arrays["subject"] == subject
    train_session = subject_mask & (arrays["split"] == 0)
    trainable = train_session & arrays["is_pure"].astype(bool)
    if not trainable.any():
        raise RuntimeError(f"No train-session windows for subject {subject}")

    raw_x = arrays["X"].astype(np.float32)
    y = arrays["y"].astype(np.int64)
    folds = sorted(int(fold) for fold in np.unique(arrays["fold"][train_session]) if fold >= 0)
    oof_binary_logits = np.full((train_session.sum(), 2), np.nan, dtype=np.float32)
    oof_mi_logits = np.full((train_session.sum(), 4), np.nan, dtype=np.float32)
    oof_indices = np.where(train_session)[0]
    fold_reports: list[dict] = []

    augment_spec = None if args.augment == ["none"] else args.augment
    aug_tag = "base" if augment_spec is None else "+".join(augment_spec)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        # 增强训练也只用其他运行的纯窗口，留出运行仍保留完整连续流。
        val_mask = train_session & (arrays["fold"] == fold)
        fold_train_mask = trainable & (arrays["fold"] != fold)
        print(f"\nSubject {subject:02d} fold {fold}: train={fold_train_mask.sum()} val={val_mask.sum()}")
        x_norm, mean, std = normalize_by_train(raw_x, fold_train_mask)
        binary_model, mi_model = train_stage_pair(model_name, x_norm, y, fold_train_mask, device, args,
                                                   f"s{subject:02d}/fold{fold}", augment_spec=augment_spec)
        val_binary_logits = logits_for(binary_model, x_norm[val_mask], device, args.batch_size)
        val_mi_logits = logits_for(mi_model, x_norm[val_mask], device, args.batch_size)
        local_positions = np.searchsorted(oof_indices, np.where(val_mask)[0])
        oof_binary_logits[local_positions] = val_binary_logits
        oof_mi_logits[local_positions] = val_mi_logits
        val_pred = hierarchical_from_logits(val_binary_logits, val_mi_logits)
        fold_checkpoint = CHECKPOINT_DIR / f"s{subject:02d}_{model_name}_{aug_tag}_fold{fold}_validation.pt"
        torch.save({
            "model": model_name, "model_source": str(model_source(model_name)),
            "subject": subject, "seed": args.seed, "augmentation": aug_tag, "held_out_run": fold,
            "binary_state_dict": binary_model.state_dict(), "mi_state_dict": mi_model.state_dict(),
            "mean": mean, "std": std,
            "classes": {"binary": BINARY_CLASS_NAMES, "mi": MI_CLASS_NAMES, "final": FINAL_CLASS_NAMES},
        }, fold_checkpoint)
        fold_reports.append({
            "fold": fold, "train_windows": int(fold_train_mask.sum()),
            "val_windows": int(val_mask.sum()),
            "final_5class": metrics(y[val_mask], val_pred, FINAL_CLASS_NAMES),
            "checkpoint": fold_checkpoint.as_posix(),
        })

    if np.isnan(oof_binary_logits).any() or np.isnan(oof_mi_logits).any():
        raise RuntimeError(f"OOF logits incomplete for subject {subject}")

    oof_y = y[train_session]
    oof_pred = hierarchical_from_logits(oof_binary_logits, oof_mi_logits)
    oof_metrics = {
        "final_5class": metrics(oof_y, oof_pred, FINAL_CLASS_NAMES),
        "stage1_binary": metrics((oof_y > 0).astype(np.int64), oof_binary_logits.argmax(axis=1), BINARY_CLASS_NAMES),
        "stage2_mi_on_true_task_windows": metrics(
            oof_y[oof_y > 0] - 1, oof_mi_logits[oof_y > 0].argmax(axis=1), MI_CLASS_NAMES,
        ),
    }

    print(f"\nSubject {subject:02d}: final all-run training")
    final_x, final_mean, final_std = normalize_by_train(raw_x, trainable)
    final_binary, final_mi = train_stage_pair(model_name, final_x, y, trainable, device, args,
                                               f"s{subject:02d}/final", augment_spec=augment_spec)
    train_binary_logits = logits_for(final_binary, final_x[train_session], device, args.batch_size)
    train_mi_logits = logits_for(final_mi, final_x[train_session], device, args.batch_size)
    train_pred = hierarchical_from_logits(train_binary_logits, train_mi_logits)
    train_metrics = {
        "final_5class": metrics(oof_y, train_pred, FINAL_CLASS_NAMES),
        "stage1_binary": metrics((oof_y > 0).astype(np.int64), train_binary_logits.argmax(axis=1), BINARY_CLASS_NAMES),
        "stage2_mi_on_true_task_windows": metrics(
            oof_y[oof_y > 0] - 1, train_mi_logits[oof_y > 0].argmax(axis=1), MI_CLASS_NAMES,
        ),
    }

    prefix = f"s{subject:02d}_{model_name}_{aug_tag}"
    checkpoint = CHECKPOINT_DIR / f"{prefix}_final.pt"
    prediction_file = TABLE_DIR / f"{prefix}_predictions.npz"
    metrics_file = TABLE_DIR / f"{prefix}_metrics.json"

    torch.save({
        "model": model_name, "model_source": str(model_source(model_name)),
        "subject": subject, "seed": args.seed, "augmentation": aug_tag,
        "binary_state_dict": final_binary.state_dict(),
        "mi_state_dict": final_mi.state_dict(),
        "mean": final_mean, "std": final_std,
        "classes": {"binary": BINARY_CLASS_NAMES, "mi": MI_CLASS_NAMES, "final": FINAL_CLASS_NAMES},
    }, checkpoint)
    np.savez_compressed(
        prediction_file, index=oof_indices, y_true=oof_y,
        oof_binary_logits=oof_binary_logits, oof_mi_logits=oof_mi_logits,
        oof_pred=oof_pred,
        final_train_binary_logits=train_binary_logits,
        final_train_mi_logits=train_mi_logits,
        final_train_pred=train_pred,
    )
    report = {
        "dataset": "BNCI2014001", "subject": subject,
        "augmentation": aug_tag,
        "method": "leave_one_train_run_out_oof_then_final_all_train_runs_with_augmentation",
        "data_file": args.data_file.as_posix(),
        "model": model_name, "seed": args.seed, "folds": folds,
        "fold_reports": fold_reports, "oof_metrics": oof_metrics,
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
    set_seed(args.seed)
    arrays = load_arrays(args.data_file)
    subjects = sorted(np.unique(arrays["subject"]).astype(int).tolist()) if args.all_subjects else args.subjects
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}, model={args.model}, augment={args.augment}")
    reports = [run_subject(subject, arrays, args, device) for subject in subjects]
    aug_tag = "base" if args.augment == ["none"] else "+".join(args.augment)
    summary_file = TABLE_DIR / f"{args.model}_{aug_tag}_oof_summary.json"
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary: {summary_file}")


if __name__ == "__main__":
    main()
