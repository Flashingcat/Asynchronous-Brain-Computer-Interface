"""Window and event metrics for labelled test-session predictions."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix


CLASS_NAMES = ("idle", "left_hand", "right_hand", "feet", "tongue")


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    truth, prediction = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    if truth.ndim != 1 or prediction.shape != truth.shape or truth.size == 0:
        raise ValueError("labels must be non-empty aligned vectors")
    if np.any((truth < 0) | (truth > 4) | (prediction < 0) | (prediction > 4)):
        raise ValueError("labels must be in [0,4]")
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=np.arange(5)).tolist(),
        "class_names": list(CLASS_NAMES), "sample_count": int(truth.size),
    }


def event_metrics(y_true: np.ndarray, commands: np.ndarray, run_ids: np.ndarray, *, step_seconds: float = 0.5) -> dict:
    """Score first command per contiguous labelled MI event; no-command is -1."""
    truth, emitted, runs = map(lambda x: np.asarray(x, dtype=np.int64), (y_true, commands, run_ids))
    if truth.ndim != 1 or emitted.shape != truth.shape or runs.shape != truth.shape:
        raise ValueError("truth, commands, and run_ids must be aligned vectors")
    events: list[tuple[int, int, int]] = []
    start = 0
    while start < len(truth):
        label, run, end = int(truth[start]), int(runs[start]), start + 1
        while end < len(truth) and int(truth[end]) == label and int(runs[end]) == run:
            end += 1
        if label != 0:
            events.append((start, end, label))
        start = end
    used: set[int] = set()
    correct = wrong = miss = 0
    latencies: list[float] = []
    for start, end, label in events:
        indices = [i for i in range(start, end) if emitted[i] != -1]
        if not indices:
            miss += 1
        else:
            first = indices[0]; used.add(first)
            if emitted[first] == label:
                correct += 1
            else:
                wrong += 1
            latencies.append((first - start) * step_seconds)
    extra = [i for i in np.flatnonzero(emitted != -1).tolist() if i not in used]
    idle_false = sum(truth[i] == 0 for i in extra)
    additional = len(extra) - idle_false
    total = len(events)
    return {
        "event_count": total, "event_correct": correct, "event_wrong_class": wrong, "event_miss": miss,
        "event_hit_rate": None if not total else correct / total,
        "idle_false_commands": int(idle_false), "additional_event_commands": int(additional),
        "command_count": int(np.count_nonzero(emitted != -1)),
        "mean_latency_seconds": None if not latencies else float(np.mean(latencies)),
        "median_latency_seconds": None if not latencies else float(np.median(latencies)),
        "event_definition": "contiguous same-class test windows within a run",
    }


def policy_diagnostics(reasons: tuple[str | None, ...]) -> dict:
    names = ("candidate_open", "candidate_abort", "candidate_timeout", "candidate_commit", "fast0_commit", "fast1_commit", "idle_reset")
    return {name: sum(reason == name for reason in reasons) for name in names}


def seed_summary(reports: list[dict]) -> dict:
    keys = ("accuracy", "balanced_accuracy", "event_hit_rate", "mean_latency_seconds")
    summary: dict[str, dict | int] = {"seed_count": len(reports)}
    for key in keys:
        values = [item[key] for item in reports if item.get(key) is not None]
        summary[key] = {"mean": None if not values else float(np.mean(values)), "std": None if not values else float(np.std(values))}
    return summary
