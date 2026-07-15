# =============================================================================
# Implementation of: Hierarchical Idle/Task Detection and Motor Imagery Decoding
#
# Reference:
#   Project-specific implementation for BNCI2014001 asynchronous decoding.
#   Stage 1 learns idle-vs-task detection; Stage 2 decodes four motor imagery
#   classes only for windows predicted as task.
#
# Source: No external code copied.
# =============================================================================
"""Train a two-stage idle/task then MI classifier on BNCI2014001 windows."""

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


DATA_FILE = PROJECT_ROOT / "data" / "processed" / "bnci2014001_subject01_async.npz"
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
BINARY_CLASS_NAMES = ["idle", "task"]
MI_CLASS_NAMES = ["left_hand", "right_hand", "feet", "tongue"]
FINAL_CLASS_NAMES = ["idle", *MI_CLASS_NAMES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="eegnet", choices=available_models(), help="network backbone")
    parser.add_argument("--binary-epochs", type=int, default=30, help="epochs for idle-vs-task training")
    parser.add_argument("--mi-epochs", type=int, default=30, help="epochs for four-class MI training")
    parser.add_argument("--batch-size", type=int, default=32, help="training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--class-weight",
        choices=("none", "balanced"),
        default="balanced",
        help="loss class weighting for each stage",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_session_split(data_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(data_file) as data:
        features = data["X"].astype(np.float32)
        labels = data["y"].astype(np.int64)
        split = data["split"].astype(np.int64)

    train_mask = split == 0
    val_mask = split == 1
    test_mask = split == 2 if np.any(split == 2) else split == 1
    if not train_mask.any() or not test_mask.any():
        raise RuntimeError("Expected train (0) and held-out test/validation windows.")

    mean = features[train_mask].mean(axis=(0, 2), keepdims=True)
    std = features[train_mask].std(axis=(0, 2), keepdims=True).clip(min=1e-6)
    normalized = (features - mean) / std
    return normalized, labels, train_mask, val_mask, test_mask, np.stack((mean, std))


def class_weights(labels: np.ndarray, num_classes: int, device: torch.device, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None

    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    if np.any(counts == 0):
        raise RuntimeError(f"Cannot use balanced class weights with empty classes: counts={counts.tolist()}")
    weights = counts.sum() / (num_classes * counts)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def train_model(
    stage_name: str,
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    device: torch.device,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    loss_weights: torch.Tensor | None,
) -> None:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=loss_weights)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = correct = sample_count = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite {stage_name} loss; aborting the measurement.")
            loss.backward()
            optimizer.step()

            sample_count += batch_y.size(0)
            loss_sum += loss.item() * batch_y.size(0)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()

        print(
            f"{stage_name} epoch={epoch:02d} "
            f"train_loss={loss_sum / sample_count:.4f} train_acc={correct / sample_count:.3f}"
        )


@torch.no_grad()
def predict(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    return model(torch.from_numpy(features).to(device)).argmax(dim=1).cpu().numpy()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
    }


def hierarchical_predict(
    binary_model: nn.Module,
    mi_model: nn.Module,
    features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    binary_pred = predict(binary_model, features, device)
    final_pred = np.zeros(features.shape[0], dtype=np.int64)
    task_mask = binary_pred == 1
    if task_mask.any():
        final_pred[task_mask] = predict(mi_model, features[task_mask], device) + 1
    return binary_pred, final_pred


def save_artifacts(
    binary_model: nn.Module,
    mi_model: nn.Module,
    mean_std: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    binary_true: np.ndarray,
    binary_pred: np.ndarray,
    mi_true: np.ndarray,
    mi_pred: np.ndarray,
    metrics: dict,
    args: argparse.Namespace,
    device: torch.device,
    train_count: int,
    val_count: int,
    test_count: int,
) -> None:
    model_name = normalize_model_name(args.model)
    run_name = f"hierarchical_{model_name}"
    checkpoint_file = CHECKPOINT_DIR / f"{run_name}_bnci2014001_async_subject01.pt"
    prediction_file = TABLE_DIR / f"{run_name}_async_predictions.npz"
    metrics_file = TABLE_DIR / f"{run_name}_async_metrics.json"
    manifest_file = TABLE_DIR / f"{run_name}_run_manifest.json"

    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    prediction_file.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "binary_state_dict": binary_model.state_dict(),
            "mi_state_dict": mi_model.state_dict(),
            "mean": mean_std[0],
            "std": mean_std[1],
            "model": model_name,
            "binary_classes": BINARY_CLASS_NAMES,
            "mi_classes": MI_CLASS_NAMES,
            "final_classes": FINAL_CLASS_NAMES,
        },
        checkpoint_file,
    )
    np.savez_compressed(
        prediction_file,
        y_true=y_true,
        y_pred=y_pred,
        binary_true=binary_true,
        binary_pred=binary_pred,
        mi_true=mi_true,
        mi_pred=mi_pred,
    )
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    manifest = {
        "dataset": "BNCI2014001",
        "subject": 1,
        "method": "hierarchical_idle_task_then_mi",
        "model": model_name,
        "model_file": model_source(model_name).as_posix(),
        "data_file": DATA_FILE.as_posix(),
        "checkpoint_file": checkpoint_file.as_posix(),
        "prediction_file": prediction_file.as_posix(),
        "metrics_file": metrics_file.as_posix(),
        "split": "0 train, 1 validation, 2 test; if split 2 is absent, split 1 is used as held-out evaluation",
        "label_mapping": {
            "stage1": {"0": "idle", "1": "task"},
            "stage2": {"0": "left_hand", "1": "right_hand", "2": "feet", "3": "tongue"},
            "final": {"0": "idle", "1": "left_hand", "2": "right_hand", "3": "feet", "4": "tongue"},
        },
        "seed": args.seed,
        "binary_epochs": args.binary_epochs,
        "mi_epochs": args.mi_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "class_weight": args.class_weight,
        "device": str(device),
        "n_train": train_count,
        "n_val": val_count,
        "n_test": test_count,
    }
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved checkpoint: {checkpoint_file}")
    print(f"Saved predictions: {prediction_file}")
    print(f"Saved metrics: {metrics_file}")


def main() -> None:
    args = parse_args()
    args.model = normalize_model_name(args.model)
    set_seed(args.seed)

    features, labels, train_mask, val_mask, test_mask, mean_std = load_session_split(DATA_FILE)
    train_task_mask = train_mask & (labels > 0)
    val_task_mask = val_mask & (labels > 0)
    test_task_mask = test_mask & (labels > 0)
    if not train_task_mask.any() or not test_task_mask.any():
        raise RuntimeError("Expected task windows in both train and test splits.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chans, samples = features.shape[1], features.shape[2]
    binary_model = build_model(args.model, 2, chans, samples).to(device)
    mi_model = build_model(args.model, 4, chans, samples).to(device)

    binary_train_y = (labels[train_mask] > 0).astype(np.int64)
    mi_train_y = labels[train_task_mask] - 1
    print(f"Training stage1 idle-vs-task model={args.model} source={model_source(args.model)}")
    train_model(
        "binary",
        binary_model,
        features[train_mask],
        binary_train_y,
        device,
        args.batch_size,
        args.binary_epochs,
        args.learning_rate,
        class_weights(binary_train_y, 2, device, args.class_weight),
    )

    print(f"Training stage2 four-class MI model={args.model} source={model_source(args.model)}")
    train_model(
        "mi",
        mi_model,
        features[train_task_mask],
        mi_train_y,
        device,
        args.batch_size,
        args.mi_epochs,
        args.learning_rate,
        class_weights(mi_train_y, 4, device, args.class_weight),
    )

    if val_mask.any():
        val_features = features[val_mask]
        val_labels = labels[val_mask]
        val_binary_true = (val_labels > 0).astype(np.int64)
        val_binary_pred, val_final_pred = hierarchical_predict(binary_model, mi_model, val_features, device)
        val_mi_true = labels[val_task_mask] - 1
        val_mi_pred = predict(mi_model, features[val_task_mask], device) if val_task_mask.any() else np.asarray([], dtype=np.int64)
        val_metrics = {
            "final_5class": compute_metrics(val_labels, val_final_pred, FINAL_CLASS_NAMES),
            "stage1_binary": compute_metrics(val_binary_true, val_binary_pred, BINARY_CLASS_NAMES),
        }
        if val_task_mask.any():
            val_metrics["stage2_mi_on_true_task_windows"] = compute_metrics(val_mi_true, val_mi_pred, MI_CLASS_NAMES)
        print(
            "Validation final 5-class "
            f"accuracy={val_metrics['final_5class']['accuracy']:.3f}; "
            f"balanced_accuracy={val_metrics['final_5class']['balanced_accuracy']:.3f}"
        )
        print(
            "Validation stage1 binary "
            f"accuracy={val_metrics['stage1_binary']['accuracy']:.3f}; "
            f"balanced_accuracy={val_metrics['stage1_binary']['balanced_accuracy']:.3f}"
        )
        if "stage2_mi_on_true_task_windows" in val_metrics:
            print(
                "Validation stage2 MI "
                f"accuracy={val_metrics['stage2_mi_on_true_task_windows']['accuracy']:.3f}; "
                f"balanced_accuracy={val_metrics['stage2_mi_on_true_task_windows']['balanced_accuracy']:.3f}"
            )

    test_features = features[test_mask]
    test_labels = labels[test_mask]
    binary_true = (test_labels > 0).astype(np.int64)
    binary_pred, final_pred = hierarchical_predict(binary_model, mi_model, test_features, device)
    mi_true = labels[test_task_mask] - 1
    mi_pred = predict(mi_model, features[test_task_mask], device)

    metrics = {
        "final_5class": compute_metrics(test_labels, final_pred, FINAL_CLASS_NAMES),
        "stage1_binary": compute_metrics(binary_true, binary_pred, BINARY_CLASS_NAMES),
        "stage2_mi_on_true_task_windows": compute_metrics(mi_true, mi_pred, MI_CLASS_NAMES),
    }
    print(
        "Final 5-class "
        f"accuracy={metrics['final_5class']['accuracy']:.3f}; "
        f"balanced_accuracy={metrics['final_5class']['balanced_accuracy']:.3f}"
    )
    print(
        "Stage1 binary "
        f"accuracy={metrics['stage1_binary']['accuracy']:.3f}; "
        f"balanced_accuracy={metrics['stage1_binary']['balanced_accuracy']:.3f}"
    )
    print(
        "Stage2 MI "
        f"accuracy={metrics['stage2_mi_on_true_task_windows']['accuracy']:.3f}; "
        f"balanced_accuracy={metrics['stage2_mi_on_true_task_windows']['balanced_accuracy']:.3f}"
    )

    save_artifacts(
        binary_model,
        mi_model,
        mean_std,
        test_labels,
        final_pred,
        binary_true,
        binary_pred,
        mi_true,
        mi_pred,
        metrics,
        args,
        device,
        int(train_mask.sum()),
        int(val_mask.sum()),
        int(test_mask.sum()),
    )


if __name__ == "__main__":
    main()
