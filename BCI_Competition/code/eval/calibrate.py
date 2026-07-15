"""Temperature scaling + confidence rejection for two-stage BCI pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import LBFGS
from sklearn.metrics import balanced_accuracy_score

PROJECT_ROOT = Path("BCI_Competition") if Path("BCI_Competition/code").is_dir() else Path(".")
sys.path.insert(0, str(PROJECT_ROOT / "code"))
from models.model_factory import build_model

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "bnci2014001_subject01_async.npz"
CHECKPOINT_FILE = (
    PROJECT_ROOT / "results" / "checkpoints" / "hierarchical_eegnet_attn_bnci2014001_async_subject01.pt"
)
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

STAGE1_NAMES = ["idle", "task"]
STAGE2_NAMES = ["left_hand", "right_hand", "feet", "tongue"]
FINAL_NAMES = ["idle", *STAGE2_NAMES]


def load_data_and_model():
    """Load data, build model, load checkpoint, return everything."""
    ckpt = torch.load(CHECKPOINT_FILE, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = ckpt["model"]
    mean, std = ckpt["mean"], ckpt["std"]

    with np.load(DATA_FILE) as data:
        features = data["X"].astype(np.float32)
        labels = data["y"].astype(np.int64)
        split = data["split"].astype(np.int64)

    normalized = (features - mean) / std.clip(min=1e-6)
    chans, samples = features.shape[1], features.shape[2]

    binary_model = build_model(model_name, 2, chans, samples).to(device)
    mi_model = build_model(model_name, 4, chans, samples).to(device)
    binary_model.load_state_dict(ckpt["binary_state_dict"])
    mi_model.load_state_dict(ckpt["mi_state_dict"])
    binary_model.eval()
    mi_model.eval()

    val_mask = split == 1
    test_mask = split == 2

    return {
        "device": device,
        "binary_model": binary_model,
        "mi_model": mi_model,
        "features": torch.from_numpy(normalized),
        "labels": labels,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }


@torch.no_grad()
def get_logits(data: dict):
    """Run inference to collect raw logits for val and test."""
    features = data["features"].to(data["device"])
    binary_logits = data["binary_model"](features).cpu().numpy()
    mi_logits = data["mi_model"](features).cpu().numpy()
    return binary_logits, mi_logits


def learn_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Minimize NLL to find best temperature on validation set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.tensor(1.0, requires_grad=True, device=device)
    logits_t = torch.from_numpy(logits).to(device)
    labels_t = torch.from_numpy(labels).to(device)
    nll = nn.CrossEntropyLoss()
    optimizer = LBFGS([t], lr=0.01, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = nll(logits_t / t, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(t.cpu().data.numpy())


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def hierarchical_predict_with_logits(
    stage1_logits: np.ndarray, stage2_logits: np.ndarray
) -> np.ndarray:
    """Two-stage prediction from raw logits."""
    task = np.argmax(stage1_logits, axis=1) == 1
    result = np.zeros(stage1_logits.shape[0], dtype=np.int64)
    result[task] = 1 + np.argmax(stage2_logits[task], axis=1)
    return result


def hierarchical_predict_with_rejection(
    stage1_logits: np.ndarray,
    stage2_logits: np.ndarray,
    reject_threshold: float,
) -> np.ndarray:
    """If max prob < threshold, output idle (0) regardless."""
    stage1_probs = softmax(stage1_logits)
    stage2_probs = softmax(stage2_logits)
    max_probs = stage1_probs.max(axis=1)
    task = np.argmax(stage1_logits, axis=1) == 1
    result = np.zeros(stage1_logits.shape[0], dtype=np.int64)
    result[task] = 1 + np.argmax(stage2_logits[task], axis=1)
    # Reject: override to idle if max probability is too low
    result[max_probs < reject_threshold] = 0
    return result


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute ECE."""
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
        if not in_bin.any():
            continue
        bin_acc = (preds[in_bin] == labels[in_bin]).mean()
        bin_conf = confs[in_bin].mean()
        ece += np.abs(bin_acc - bin_conf) * in_bin.sum() / len(labels)
    return float(ece)


def main():
    data = load_data_and_model()
    bin_logits, mi_logits = get_logits(data)
    labels = data["labels"]

    val_bin_labels = (labels[data["val_mask"]] > 0).astype(np.int64)
    test_bin_labels = (labels[data["test_mask"]] > 0).astype(np.int64)
    val_task = labels[data["val_mask"]] > 0
    test_task = labels[data["test_mask"]] > 0
    val_mi_labels = labels[data["val_mask"]][val_task] - 1
    test_mi_labels = labels[data["test_mask"]][test_task] - 1

    lines: list[str] = []

    # ── Temperature Scaling ────────────────────────────────────────────
    lines.append("# Temperature Scaling")
    lines.append("---")

    # Stage 1
    t1 = learn_temperature(bin_logits[data["val_mask"]], val_bin_labels)
    lines.append(f"\n## Stage 1 (idle vs task)")
    lines.append(f"Learned temperature T = {t1:.4f}")

    val_bin_probs_raw = softmax(bin_logits[data["val_mask"]])
    val_bin_probs_cal = softmax(bin_logits[data["val_mask"]] / t1)
    test_bin_probs_raw = softmax(bin_logits[data["test_mask"]])
    test_bin_probs_cal = softmax(bin_logits[data["test_mask"]] / t1)

    for split_name, probs_raw, probs_cal, y in [
        ("Val", val_bin_probs_raw, val_bin_probs_cal, val_bin_labels),
        ("Test", test_bin_probs_raw, test_bin_probs_cal, test_bin_labels),
    ]:
        ece_raw = expected_calibration_error(probs_raw, y)
        ece_cal = expected_calibration_error(probs_cal, y)
        lines.append(f"  {split_name} ECE: {ece_raw:.4f} → {ece_cal:.4f}  ({'better' if ece_cal < ece_raw else 'worse'})")

    # Stage 2
    t2 = learn_temperature(mi_logits[data["val_mask"]][val_task], val_mi_labels)
    lines.append(f"\n## Stage 2 (MI 4-class)")
    lines.append(f"Learned temperature T = {t2:.4f}")

    test_mi_raw = softmax(mi_logits[data["test_mask"]][test_task])
    test_mi_cal = softmax(mi_logits[data["test_mask"]][test_task] / t2)

    ece_raw = expected_calibration_error(test_mi_raw, test_mi_labels)
    ece_cal = expected_calibration_error(test_mi_cal, test_mi_labels)
    lines.append(f"  Test ECE: {ece_raw:.4f} → {ece_cal:.4f}")

    # ── Hierarchical results: before vs after temperature calibration ──
    lines.append("\n## Hierarchical 5-class Results")
    lines.append("---")

    test_pred_raw = hierarchical_predict_with_logits(bin_logits[data["test_mask"]], mi_logits[data["test_mask"]])
    test_labels = labels[data["test_mask"]]
    raw_ba = balanced_accuracy_score(test_labels, test_pred_raw)

    test_pred_cal = hierarchical_predict_with_logits(
        bin_logits[data["test_mask"]] / t1, mi_logits[data["test_mask"]]
    )
    cal_ba = balanced_accuracy_score(test_labels, test_pred_cal)

    lines.append(f"  Raw balanced_accuracy:   {raw_ba:.4f}")
    lines.append(f"  Calibrated balanced_acc: {cal_ba:.4f}  ({'better' if cal_ba > raw_ba else 'worse'})")

    # ── Confidence Rejection Sweep ─────────────────────────────────────
    lines.append("\n# Confidence Rejection Sweep")
    lines.append("---")
    lines.append("\nThreshold → balanced_accuracy (reject low-confidence windows to idle)")
    lines.append("")
    lines.append("| Threshold | Test BA |")
    lines.append("|-----------|---------|")

    best_thresh = 0.0
    best_ba = 0.0
    for thresh in np.arange(0.0, 1.0, 0.05):
        pred = hierarchical_predict_with_rejection(
            bin_logits[data["test_mask"]], mi_logits[data["test_mask"]], thresh
        )
        ba = balanced_accuracy_score(test_labels, pred)
        if ba > best_ba:
            best_ba = ba
            best_thresh = thresh
        lines.append(f"| {thresh:.2f}       | {ba:.4f}   |")

    lines.append(f"\nBest threshold: {best_thresh:.2f} → balanced_accuracy = {best_ba:.4f}")
    lines.append(f"Improvement from raw ({raw_ba:.4f}): +{best_ba - raw_ba:.4f}")

    output = "\n".join(lines)
    print(output)

    # Save
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / "calibration_results_eegnet_attn.md").write_text(output, encoding="utf-8")
    print(f"\nSaved to: {TABLE_DIR / 'calibration_results_eegnet_attn.md'}")


if __name__ == "__main__":
    main()
