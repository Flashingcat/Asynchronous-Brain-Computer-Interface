"""Event-level evaluation metrics for asynchronous BCI.

Usage:
    python BCI_Competition/code/eval/run_evaluation.py --model eegnet_attn

Outputs a JSON report with event-level metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path("BCI_Competition") if Path("BCI_Competition/code").is_dir() else Path(".")
sys.path.insert(0, str(PROJECT_ROOT / "code"))
from models.model_factory import available_models, build_model, normalize_model_name

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "bnci2014001_subject01_async.npz"
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

STAGE2_NAMES = ["left_hand", "right_hand", "feet", "tongue"]
FINAL_NAMES = ["idle", *STAGE2_NAMES]
WINDOW_SECONDS = 2.0
STRIDE_SECONDS = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="eegnet_attn", choices=available_models())
    parser.add_argument("--threshold", type=float, default=0.0, help="confidence threshold (reject below)")
    return parser.parse_args()


def load_model_and_data(model_name: str):
    """Load checkpoint and test data."""
    model_name = normalize_model_name(model_name)
    ckpt_file = CHECKPOINT_DIR / f"hierarchical_{model_name}_bnci2014001_async_subject01.pt"

    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    test_mask = split == 2
    return {
        "device": device,
        "binary_model": binary_model,
        "mi_model": mi_model,
        "features": torch.from_numpy(normalized[test_mask]),
        "labels": labels[test_mask],
    }


@torch.no_grad()
def run_inference(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Run two-stage inference, return (binary_probs, mi_probs)."""
    features = data["features"].to(data["device"])
    bin_logits = data["binary_model"](features).cpu().numpy()
    mi_logits = data["mi_model"](features).cpu().numpy()

    bin_probs = _softmax(bin_logits)
    mi_probs = _softmax(mi_logits)
    return bin_probs, mi_probs


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def two_stage_predict(bin_probs: np.ndarray, mi_probs: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Two-stage prediction with optional confidence rejection."""
    max_probs = bin_probs.max(axis=1)
    task = (np.argmax(bin_probs, axis=1) == 1) & (max_probs >= threshold)
    result = np.zeros(bin_probs.shape[0], dtype=np.int64)
    result[task] = 1 + np.argmax(mi_probs[task], axis=1)
    return result


def group_events(labels: np.ndarray) -> list[dict]:
    """Group consecutive non-zero windows with the same label into events."""
    events = []
    i = 0
    while i < len(labels):
        if labels[i] > 0:
            label = int(labels[i])
            start = i
            while i < len(labels) and labels[i] == label:
                i += 1
            events.append({
                "start": start,
                "stop": i,
                "true_class": label,
                "n_windows": i - start,
            })
        else:
            i += 1
    return events


def find_idle_runs(labels: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous idle segments."""
    runs = []
    i = 0
    while i < len(labels):
        if labels[i] == 0:
            start = i
            while i < len(labels) and labels[i] == 0:
                i += 1
            runs.append((start, i))
        else:
            i += 1
    return runs


def evaluate_events(labels: np.ndarray, predictions: np.ndarray) -> dict:
    """Event-level evaluation of two-stage asynchronous BCI."""
    events = group_events(labels)
    idle_runs = find_idle_runs(labels)

    per_event = []
    correct_count = 0
    wrong_count = 0
    miss_count = 0
    correct_latencies = []

    for ev in events:
        pred_slice = predictions[ev["start"]:ev["stop"]]
        unique_preds = set(pred_slice)
        task_preds = [p for p in pred_slice if p > 0]

        if ev["true_class"] in unique_preds:
            outcome = "correct"
            correct_count += 1
            # Latency: first window with correct prediction
            first_correct = int(np.argmax(pred_slice == ev["true_class"]))
            latency_seconds = first_correct * STRIDE_SECONDS
            correct_latencies.append(latency_seconds)
        elif task_preds:
            outcome = "wrong_class"
            wrong_count += 1
        else:
            outcome = "miss"
            miss_count += 1

        per_event.append({
            "true_class": ev["true_class"],
            "true_class_name": FINAL_NAMES[ev["true_class"]],
            "n_windows": ev["n_windows"],
            "n_task_predictions": len(task_preds),
            "predictions": [int(p) for p in pred_slice],
            "outcome": outcome,
        })

    # Idle false positives
    false_positive_count = 0
    false_positive_classes = []
    for start, stop in idle_runs:
        pred_slice = predictions[start:stop]
        fps = pred_slice[pred_slice > 0]
        false_positive_count += len(fps)
        for fp in fps:
            false_positive_classes.append(int(fp))

    # Per-class stats
    per_class = {}
    for cls in [1, 2, 3, 4]:
        cls_events = [e for e in per_event if e["true_class"] == cls]
        n = len(cls_events)
        correct_cls = sum(1 for e in cls_events if e["outcome"] == "correct")
        per_class[FINAL_NAMES[cls]] = {
            "event_count": n,
            "correct_count": correct_cls,
            "correct_rate": correct_cls / n if n else None,
        }

    # Idle FP rate per minute
    idle_window_count = sum(stop - start for start, stop in idle_runs)
    idle_seconds = idle_window_count * STRIDE_SECONDS
    false_positive_per_minute = false_positive_count / (idle_seconds / 60.0) if idle_seconds > 0 else None

    total = len(events)
    latency = {
        "count": len(correct_latencies),
        "mean": float(np.mean(correct_latencies)) if correct_latencies else None,
        "median": float(np.median(correct_latencies)) if correct_latencies else None,
        "min": float(min(correct_latencies)) if correct_latencies else None,
        "max": float(max(correct_latencies)) if correct_latencies else None,
    }

    return {
        "total_events": total,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "miss_count": miss_count,
        "correct_rate": correct_count / total if total else None,
        "wrong_rate": wrong_count / total if total else None,
        "miss_rate": miss_count / total if total else None,
        "per_class_correct_rate": per_class,
        "false_positive_count": false_positive_count,
        "false_positive_classes": [int(c) for c in false_positive_classes],
        "false_positive_per_minute": false_positive_per_minute,
        "idle_evaluated_seconds": idle_seconds,
        "correct_detection_latency_seconds": latency,
        "n_test_windows": len(labels),
        "idle_windows": idle_window_count,
        "event_details": per_event,
    }


def main():
    args = parse_args()
    print(f"Loading model: {args.model}, confidence threshold: {args.threshold}")

    data = load_model_and_data(args.model)
    bin_probs, mi_probs = run_inference(data)
    predictions = two_stage_predict(bin_probs, mi_probs, args.threshold)
    report = evaluate_events(data["labels"], predictions)

    # Print summary
    print("\n" + "=" * 50)
    print(f"Event-Level Evaluation: {args.model}")
    print("=" * 50)
    print(f"  Total events:     {report['total_events']}")
    print(f"  Correct:          {report['correct_count']}  ({report['correct_rate']*100:.1f}%)")
    print(f"  Wrong class:      {report['wrong_count']}  ({report['wrong_rate']*100:.1f}%)")
    print(f"  Miss:             {report['miss_count']}  ({report['miss_rate']*100:.1f}%)")
    print(f"  False positives:  {report['false_positive_count']}  ({report['false_positive_per_minute']:.2f}/min)")
    print(f"  Latency (median): {report['correct_detection_latency_seconds']['median']:.2f}s")
    print(f"  Latency (mean):   {report['correct_detection_latency_seconds']['mean']:.2f}s")
    print("\nPer-class correct rate:")
    for name, stats in report["per_class_correct_rate"].items():
        print(f"  {name:15s}: {stats['correct_count']:3d}/{stats['event_count']:3d} ({stats['correct_rate']*100:.1f}%)")

    # Save
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    model_name = normalize_model_name(args.model)
    out_file = TABLE_DIR / f"event_eval_{model_name}_thresh{args.threshold:.2f}.json"
    # Strip verbose event_details for the JSON (keep for debugging)
    summary = {k: v for k, v in report.items() if k != "event_details"}
    summary["event_details_path"] = str(out_file.with_suffix(".details.json"))
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Save details separately
    details_file = out_file.with_suffix(".details.json")
    details_file.write_text(json.dumps({"event_details": report["event_details"]}, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
