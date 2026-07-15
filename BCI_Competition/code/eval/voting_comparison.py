"""Calibrated soft voting vs hard voting for two-stage BCI."""

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

STAGE2_NAMES = ["left_hand", "right_hand", "feet", "tongue"]
FINAL_NAMES = ["idle", *STAGE2_NAMES]


def load_data_and_model():
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
        "split": split,
    }


@torch.no_grad()
def get_logits(data):
    features = data["features"].to(data["device"])
    bin_logits = data["binary_model"](features).cpu().numpy()
    mi_logits = data["mi_model"](features).cpu().numpy()
    return bin_logits, mi_logits


def learn_temperature(logits, labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.tensor(1.0, requires_grad=True, device=device)
    logits_t = torch.from_numpy(logits).to(device)
    labels_t = torch.from_numpy(labels).to(device)
    optimizer = LBFGS([t], lr=0.01, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(logits_t / t, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(t.cpu().data.numpy())


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def two_stage_predict(bin_logits, mi_logits):
    """Standard two-stage pipeline: Stage 1 gates Stage 2."""
    task = np.argmax(bin_logits, axis=1) == 1
    result = np.zeros(bin_logits.shape[0], dtype=np.int64)
    result[task] = 1 + np.argmax(mi_logits[task], axis=1)
    return result


def soft_vote_predict(bin_probs, mi_probs, n_windows):
    """Average probabilities over N windows, then decide."""
    n = len(bin_probs) - n_windows + 1
    result = np.zeros(n, dtype=np.int64)
    for i in range(n):
        bp = bin_probs[i:i + n_windows].mean(axis=0)
        task = np.argmax(bp) == 1
        if task:
            mp = mi_probs[i:i + n_windows].mean(axis=0)
            result[i] = 1 + int(np.argmax(mp))
    return result


def hard_vote_predict(bin_logits, mi_logits, n_windows):
    """Take argmax of each window, then majority vote over N windows."""
    n = len(bin_logits) - n_windows + 1
    result = np.zeros(n, dtype=np.int64)
    for i in range(n):
        votes = []
        for j in range(i, i + n_windows):
            task = np.argmax(bin_logits[j]) == 1
            if task:
                votes.append(1 + int(np.argmax(mi_logits[j])))
            else:
                votes.append(0)
        counts = np.bincount(votes, minlength=5)
        if counts[1:].max() > n_windows // 2:
            result[i] = int(np.argmax(counts[1:]) + 1)
    return result


def main():
    data = load_data_and_model()
    bin_logits, mi_logits = get_logits(data)
    labels = data["labels"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]
    test_labels = labels[test_mask]

    # Learn temperature on validation
    val_bin_labels = (labels[val_mask] > 0).astype(np.int64)
    val_task = labels[val_mask] > 0
    val_mi_labels = labels[val_mask][val_task] - 1
    t1 = learn_temperature(bin_logits[val_mask], val_bin_labels)
    t2 = learn_temperature(mi_logits[val_mask][val_task], val_mi_labels)

    # Prepare calibrated probabilities
    cal_bin_probs = softmax(bin_logits[test_mask] / t1)
    cal_mi_probs = softmax(mi_logits[test_mask] / t2)
    raw_bin_probs = softmax(bin_logits[test_mask])
    raw_mi_probs = softmax(mi_logits[test_mask])

    results = {}

    # 1. Raw single-window baseline
    raw_pred = two_stage_predict(bin_logits[test_mask], mi_logits[test_mask])
    results["raw (single window)"] = balanced_accuracy_score(test_labels, raw_pred)

    # 2. Calibrated single-window
    cal_pred = two_stage_predict(
        bin_logits[test_mask] / t1, mi_logits[test_mask]
    )
    results["calibrated (single window)"] = balanced_accuracy_score(test_labels, cal_pred)

    # 3. Hard voting (majority)
    for n in [3, 5, 7]:
        pred = hard_vote_predict(bin_logits[test_mask], mi_logits[test_mask], n)
        results[f"hard_vote_n{n}"] = balanced_accuracy_score(test_labels[n - 1:], pred)

    # 4. Soft voting (probability average)
    for n in [3, 5, 7]:
        pred = soft_vote_predict(raw_bin_probs, raw_mi_probs, n)
        results[f"soft_vote_n{n}"] = balanced_accuracy_score(test_labels[n - 1:], pred)

    # 5. Calibrated soft voting
    for n in [3, 5, 7]:
        pred = soft_vote_predict(cal_bin_probs, cal_mi_probs, n)
        results[f"cal_soft_vote_n{n}"] = balanced_accuracy_score(test_labels[n - 1:], pred)

    # Print table
    lines = ["# Voting Strategy Comparison on EEGNet_Attn", "---", ""]
    lines.append(f"Stage 1 temperature: T={t1:.4f}")
    lines.append(f"Stage 2 temperature: T={t2:.4f}")
    lines.append("")
    lines.append("| Strategy | Balanced Accuracy | vs Raw |")
    lines.append("|----------|------------------|--------|")
    raw_ba = results["raw (single window)"]
    for name, val in results.items():
        diff = val - raw_ba
        sign = "+" if diff > 0 else ""
        marker = " ← best" if val == max(results.values()) else ""
        lines.append(f"| {name} | {val:.4f} | {sign}{diff:.4f}{marker} |")

    lines.append("")
    lines.append("### Notes")
    lines.append("- `hard_vote_n5` = majority vote over 5 windows' argmax")
    lines.append("- `soft_vote_n5` = average 5 windows' probabilities then argmax")
    lines.append("- `cal_*` = with temperature scaling before voting")

    output = "\n".join(lines)
    print(output)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / "voting_comparison.md").write_text(output, encoding="utf-8")
    print(f"\nSaved to {TABLE_DIR / 'voting_comparison.md'}")


if __name__ == "__main__":
    main()
