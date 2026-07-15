"""Event-level + window-level evaluation metrics for asynchronous BCI.

Usage:
    python BCI_Competition/code/eval/run_evaluation.py --model eegnet_attn

Outputs a JSON report with event-level and window-level metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix, roc_auc_score

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
    parser.add_argument("--model", default="eegnet", choices=available_models())
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
            "start": int(ev["start"]),
            "stop": int(ev["stop"]),
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

    # ===== Window-level metrics =====

    # Cohen's Kappa (chance-corrected agreement, official BCI competition metric)
    kappa = float(cohen_kappa_score(labels, predictions))

    # Macro F1: unweighted mean of per-class F1
    macro_f1 = float(f1_score(labels, predictions, average="macro", zero_division=0))

    # Per-class precision, recall, F1
    cls_report = {}
    cm = confusion_matrix(labels, predictions, labels=range(5))
    for i, name in enumerate(FINAL_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        cls_report[name] = {
            "true_count": int(cm[i, :].sum()),
            "pred_count": int(cm[:, i].sum()),
            "tp": int(tp),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    # Confusion matrix (raw counts)
    confusion = {}
    for true_idx, true_name in enumerate(FINAL_NAMES):
        row = {}
        for pred_idx, pred_name in enumerate(FINAL_NAMES):
            count = int(cm[true_idx, pred_idx])
            if count > 0:
                row[pred_name] = count
        confusion[true_name] = row

    # ===== Decision stability =====
    # Count prediction switches between consecutive windows
    switches = int((predictions[1:] != predictions[:-1]).sum())
    # Proportion of windows where prediction differs from previous
    switch_rate = switches / (len(predictions) - 1) if len(predictions) > 1 else 0.0
    # Idle → task transition count (how many times model exits idle)
    idle_task_transitions = 0
    for i in range(1, len(predictions)):
        if predictions[i - 1] == 0 and predictions[i] > 0:
            idle_task_transitions += 1

    # ===== Composite score =====
    # Normalised composite: higher = better overall system
    # Weights chosen to balance correctness, false-positive control, and latency
    w_correct = 0.25
    w_fp = 0.20
    w_kappa = 0.20
    w_f1 = 0.15
    w_stability = 0.10
    w_latency = 0.10

    correct_score = correct_count / total if total else 0.0
    # FP penalty: 1.0 (no FPs) → 0.0 (≥10 FP/min, clip)
    fp_score = max(0.0, 1.0 - (false_positive_per_minute or 0.0) / 10.0)
    # Stability: fewer switches = better (normalise by max possible switches)
    stability_score = 1.0 - switch_rate
    # Latency: shorter = better (normalise by window length, 0.5s → 1.0, 5s → 0.0)
    mean_lat = float(np.mean(correct_latencies)) if correct_latencies else 2.0
    latency_score = max(0.0, 1.0 - mean_lat / 5.0)

    composite = (
        w_correct * correct_score
        + w_fp * fp_score
        + w_kappa * (max(0.0, kappa) if kappa is not None else 0.0)
        + w_f1 * macro_f1
        + w_stability * stability_score
        + w_latency * latency_score
    )

    return {
        # Event-level
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
        # Window-level
        "n_test_windows": len(labels),
        "idle_windows": idle_window_count,
        "cohen_kappa": kappa,
        "macro_f1": macro_f1,
        "per_class_report": cls_report,
        "confusion_matrix": confusion,
        # Decision stability
        "prediction_switches": switches,
        "switch_rate": switch_rate,
        "idle_task_transitions": idle_task_transitions,
        # Composite
        "composite_score": round(composite, 4),
        "composite_weights": {
            "correct_rate": w_correct,
            "fp_penalty": w_fp,
            "kappa": w_kappa,
            "macro_f1": w_f1,
            "stability": w_stability,
            "latency": w_latency,
        },
        # Raw data
        "event_details": per_event,
    }


def compute_itr(accuracy: float, n_classes: int, seconds_per_selection: float) -> dict:
    """Information Transfer Rate (bits/symbol, bits/min) — standard BCI metric."""
    P = np.clip(accuracy, 1e-10, 1 - 1e-10)
    bits_per_symbol = np.log2(n_classes) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (n_classes - 1))
    bits_per_symbol = float(max(0.0, bits_per_symbol))
    bits_per_min = bits_per_symbol * (60.0 / seconds_per_selection)
    return {
        "bits_per_symbol": round(bits_per_symbol, 4),
        "bits_per_minute": round(bits_per_min, 4),
        "n_classes": n_classes,
        "seconds_per_selection": seconds_per_selection,
    }


def compute_binary_auc(labels: np.ndarray, bin_probs: np.ndarray) -> float:
    """Stage 1 AUC: how well the model separates idle vs task at the window level."""
    binary_labels = (labels > 0).astype(np.int32)
    return float(roc_auc_score(binary_labels, bin_probs[:, 1]))


def bootstrap_event_rates(labels: np.ndarray, predictions: np.ndarray,
                          n_iterations: int = 2000) -> dict:
    """Bootstrap confidence intervals for event-level correct/wrong/miss rates."""
    rng = np.random.RandomState(42)
    events = group_events(labels)
    n = len(events)

    outcomes = []
    for ev in events:
        pred_slice = predictions[ev["start"]:ev["stop"]]
        unique_preds = set(pred_slice)
        task_preds = [p for p in pred_slice if p > 0]
        if ev["true_class"] in unique_preds:
            outcomes.append(0)  # correct
        elif task_preds:
            outcomes.append(1)  # wrong
        else:
            outcomes.append(2)  # miss

    outcomes = np.array(outcomes)
    correct_rates, wrong_rates, miss_rates = [], [], []
    for _ in range(n_iterations):
        idx = rng.randint(0, n, n)
        sampled = outcomes[idx]
        correct_rates.append((sampled == 0).mean())
        wrong_rates.append((sampled == 1).mean())
        miss_rates.append((sampled == 2).mean())

    def ci(arr):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return [float(lo), float(hi)]

    return {
        "correct_rate_95ci": ci(correct_rates),
        "wrong_rate_95ci": ci(wrong_rates),
        "miss_rate_95ci": ci(miss_rates),
        "n_iterations": n_iterations,
    }


def compute_model_stats(data: dict) -> dict:
    """Count parameters and measure inference speed."""
    binary_params = sum(p.numel() for p in data["binary_model"].parameters())
    mi_params = sum(p.numel() for p in data["mi_model"].parameters())
    total_params = binary_params + mi_params

    # Inference speed: time for the whole test set
    features = data["features"].to(data["device"])
    n_windows = features.shape[0]

    # Warmup
    _ = data["binary_model"](features[:32])
    _ = data["mi_model"](features[:32])

    torch.cuda.synchronize() if data["device"].type == "cuda" else None
    start = time.perf_counter()
    _ = data["binary_model"](features)
    _ = data["mi_model"](features)
    torch.cuda.synchronize() if data["device"].type == "cuda" else None
    elapsed = time.perf_counter() - start

    windows_per_second = n_windows / elapsed
    seconds_per_100_windows = 100.0 / windows_per_second

    return {
        "params_binary_stage": binary_params,
        "params_mi_stage": mi_params,
        "params_total": total_params,
        "inference_windows_per_second": round(windows_per_second, 1),
        "inference_seconds_per_100_windows": round(seconds_per_100_windows, 3),
        "inference_total_seconds": round(elapsed, 3),
        "n_test_windows": n_windows,
    }


def threshold_sweep(labels: np.ndarray, bin_probs: np.ndarray, mi_probs: np.ndarray,
                    n_thresholds: int = 20) -> dict:
    """Sweep confidence thresholds and compute TPR/FPR trade-off curve."""
    thresholds = np.linspace(0.0, 0.95, n_thresholds)
    points = []
    for thresh in thresholds:
        preds = two_stage_predict(bin_probs, mi_probs, float(thresh))
        task_true = labels > 0
        task_pred = preds > 0

        tp = (task_true & task_pred).sum()
        fn = (task_true & ~task_pred).sum()
        fp = (~task_true & task_pred).sum()
        tn = (~task_true & ~task_pred).sum()

        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        points.append({
            "threshold": float(thresh),
            "tpr": float(round(tpr, 4)),
            "fpr": float(round(fpr, 4)),
            "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
        })

    # Best threshold: max TPR - FPR (Youden's J statistic)
    best = max(points, key=lambda x: x["tpr"] - x["fpr"])

    # Threshold where FPR drops below 0.1 (10%) while keeping highest TPR
    low_fpr = [p for p in points if p["fpr"] <= 0.1]
    best_low_fpr = max(low_fpr, key=lambda x: x["tpr"]) if low_fpr else points[0]

    return {
        "youden_threshold": best["threshold"],
        "youden_tpr": best["tpr"],
        "youden_fpr": best["fpr"],
        "low_fpr_threshold": best_low_fpr["threshold"],
        "low_fpr_tpr": best_low_fpr["tpr"],
        "low_fpr_fpr": best_low_fpr["fpr"],
        "n_thresholds": n_thresholds,
        "curve": points,
    }


def main():
    args = parse_args()
    print(f"Loading model: {args.model}, confidence threshold: {args.threshold}")

    data = load_model_and_data(args.model)
    bin_probs, mi_probs = run_inference(data)
    predictions = two_stage_predict(bin_probs, mi_probs, args.threshold)
    report = evaluate_events(data["labels"], predictions)

    # === Additional metrics ===
    # ITR (event-level, MI-only)
    event_accuracy = report["correct_rate"] or 0.0
    avg_event_seconds = np.mean([
        (ev["stop"] - ev["start"]) * STRIDE_SECONDS
        for ev in report.get("event_details", [])
    ]) if report.get("event_details") else 4.0
    report["itr_mi"] = compute_itr(event_accuracy, 4, avg_event_seconds)
    report["itr_mi_plus_idle"] = compute_itr(event_accuracy, 5, avg_event_seconds)

    # Stage 1 binary AUC
    report["binary_auc"] = compute_binary_auc(data["labels"], bin_probs)

    # Bootstrap confidence intervals
    report["bootstrap"] = bootstrap_event_rates(data["labels"], predictions)

    # Threshold sweep / ROC analysis
    report["threshold_analysis"] = threshold_sweep(data["labels"], bin_probs, mi_probs)

    # Model efficiency
    report["model_stats"] = compute_model_stats(data)

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
    print("\nPer-class correct rate (event-level):")
    for name, stats in report["per_class_correct_rate"].items():
        print(f"  {name:15s}: {stats['correct_count']:3d}/{stats['event_count']:3d} ({stats['correct_rate']*100:.1f}%)")

    # === New metrics ===
    print("\n" + "-" * 50)
    print("Window-Level Metrics:")
    print("-" * 50)
    print(f"  Cohen's Kappa:    {report['cohen_kappa']:.4f}")
    print(f"  Macro F1:         {report['macro_f1']:.4f}")
    print(f"  Prediction switches: {report['prediction_switches']}  ({report['switch_rate']*100:.1f}%)")
    print(f"  Idle->Task transitions: {report['idle_task_transitions']}")
    print("\nPer-class precision / recall / F1:")
    for name, s in report["per_class_report"].items():
        print(f"  {name:15s}: P={s['precision']:.3f}  R={s['recall']:.3f}  F1={s['f1']:.3f}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    header = "          " + "".join(f"{n:>10s}" for n in FINAL_NAMES)
    print(header)
    for true_name in FINAL_NAMES:
        row_vals = [report["confusion_matrix"][true_name].get(pn, 0) for pn in FINAL_NAMES]
        row_str = "".join(f"{v:10d}" for v in row_vals)
        print(f"  {true_name:8s}{row_str}")

    print(f"\n  Composite Score:  {report['composite_score']:.4f}")
    print(f"  (Weights: correct={report['composite_weights']['correct_rate']}, "
          f"FP={report['composite_weights']['fp_penalty']}, "
          f"kappa={report['composite_weights']['kappa']}, "
          f"F1={report['composite_weights']['macro_f1']}, "
          f"stability={report['composite_weights']['stability']}, "
          f"latency={report['composite_weights']['latency']})")

    # === ITR & AUC & Bootstrap ===
    print("\n" + "-" * 50)
    print("Information Transfer Rate & Detection:")
    print("-" * 50)
    print(f"  ITR (4-class MI):      {report['itr_mi']['bits_per_minute']:.2f} bits/min")
    print(f"  ITR (5-class +idle):   {report['itr_mi_plus_idle']['bits_per_minute']:.2f} bits/min")
    print(f"  Stage 1 Binary AUC:    {report['binary_auc']:.4f}")
    print(f"\n  Bootstrap 95% CI (correct rate): "
          f"[{report['bootstrap']['correct_rate_95ci'][0]*100:.1f}%, "
          f"{report['bootstrap']['correct_rate_95ci'][1]*100:.1f}%]")

    # === Threshold Sweep / ROC ===
    print("\n" + "-" * 50)
    print("Threshold Sweep & ROC Analysis:")
    print("-" * 50)
    ta = report["threshold_analysis"]
    print(f"  Youden's best threshold: {ta['youden_threshold']:.2f}  "
          f"(TPR={ta['youden_tpr']:.3f}, FPR={ta['youden_fpr']:.3f})")
    print(f"  Low-FPR operating point: {ta['low_fpr_threshold']:.2f}  "
          f"(TPR={ta['low_fpr_tpr']:.3f}, FPR={ta['low_fpr_fpr']:.3f})")
    print("\n  Threshold curve (threshold -> TPR / FPR):")
    for p in ta["curve"]:
        marker = " (best)" if p["threshold"] == ta["youden_threshold"] else ""
        print(f"    thresh={p['threshold']:.2f}  ->  TPR={p['tpr']:.3f}  FPR={p['fpr']:.3f}{marker}")

    # === Model Efficiency ===
    print("\n" + "-" * 50)
    print("Model Efficiency:")
    print("-" * 50)
    ms = report["model_stats"]
    print(f"  Parameters (total):     {ms['params_total']:,}")
    print(f"    Stage 1 binary:       {ms['params_binary_stage']:,}")
    print(f"    Stage 2 MI:           {ms['params_mi_stage']:,}")
    print(f"  Inference speed:        {ms['inference_windows_per_second']:.0f} windows/sec")
    print(f"                          ({ms['inference_seconds_per_100_windows']:.3f}s per 100 windows)")

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
