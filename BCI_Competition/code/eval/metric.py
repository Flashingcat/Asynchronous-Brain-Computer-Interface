"""Simple window and command metrics for test-session evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix


CLASS_NAMES = ("idle", "left_hand", "right_hand", "feet", "tongue")


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    truth, prediction = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    if truth.ndim != 1 or prediction.shape != truth.shape or not truth.size:
        raise ValueError("labels must be aligned non-empty vectors")
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=np.arange(5)).tolist(),
        "class_names": list(CLASS_NAMES), "sample_count": int(truth.size),
    }


def continuous_event_metrics(
    y_true: np.ndarray,
    commands: np.ndarray,
    runs: np.ndarray,
    decision_samples: np.ndarray,
    events: dict,
    sampling_rate: int,
) -> dict:
    """Score commands against original events and decision sample times."""
    truth, emitted, run_ids, decisions = (
        np.asarray(value, dtype=np.int64) for value in (y_true, commands, runs, decision_samples)
    )
    if emitted.ndim != 1 or any(value.shape != emitted.shape for value in (truth, run_ids, decisions)):
        raise ValueError("commands and window coordinates must align")

    used: set[int] = set()
    correct = wrong = miss = 0
    latency: list[float] = []

    # 原始事件表决定分母，窗口只负责提供事件内首次指令及其真实决策时刻。
    event_rows = zip(events["run"], events["label"], events["start"], events["stop"])
    for run, label, start, stop in event_rows:
        inside = (run_ids == run) & (decisions >= start) & (decisions < stop)
        indices = np.flatnonzero(inside & (emitted != -1))
        if not len(indices):
            miss += 1
            continue

        first = int(indices[0])
        used.add(first)
        if emitted[first] == label:
            correct += 1
        else:
            wrong += 1
        latency.append((int(decisions[first]) - int(start)) / sampling_rate)

    extras = [index for index in np.flatnonzero(emitted != -1) if int(index) not in used]
    idle_false = sum(truth[index] == 0 for index in extras)
    total = len(events["run"])
    return {
        "event_count": total,
        "event_correct": correct,
        "event_wrong_class": wrong,
        "event_miss": miss,
        "event_hit_rate": None if not total else correct / total,
        "idle_false_commands": int(idle_false),
        "additional_event_commands": int(len(extras) - idle_false),
        "command_count": int(np.count_nonzero(emitted != -1)),
        "mean_latency_seconds": None if not latency else float(np.mean(latency)),
        "median_latency_seconds": None if not latency else float(np.median(latency)),
    }


def legacy_event_metrics(
    y_true: np.ndarray,
    commands: np.ndarray,
    streams: np.ndarray,
    step_seconds: float = 0.5,
) -> dict:
    """Reproduce the old pure-window event and compressed-latency metrics."""
    truth, emitted, stream = (
        np.asarray(value, dtype=np.int64) for value in (y_true, commands, streams)
    )
    if truth.ndim != 1 or emitted.shape != truth.shape or stream.shape != truth.shape:
        raise ValueError("truth, commands, and streams must align")

    events: list[tuple[int, int, int]] = []
    start = 0
    while start < len(truth):
        end = start + 1
        while end < len(truth) and truth[end] == truth[start] and stream[end] == stream[start]:
            end += 1
        if truth[start] != 0:
            events.append((start, end, int(truth[start])))
        start = end

    used: set[int] = set()
    correct = wrong = miss = 0
    latency: list[float] = []
    for start, end, label in events:
        indices = np.flatnonzero(emitted[start:end] != -1) + start
        if not len(indices):
            miss += 1
            continue

        first = int(indices[0])
        used.add(first)
        if emitted[first] == label:
            correct += 1
        else:
            wrong += 1
        latency.append((first - start) * step_seconds)

    extras = [index for index in np.flatnonzero(emitted != -1) if int(index) not in used]
    idle_false = sum(truth[index] == 0 for index in extras)
    total = len(events)
    return {
        "event_count": total,
        "event_correct": correct,
        "event_wrong_class": wrong,
        "event_miss": miss,
        "event_hit_rate": None if not total else correct / total,
        "idle_false_commands": int(idle_false),
        "additional_event_commands": int(len(extras) - idle_false),
        "command_count": int(np.count_nonzero(emitted != -1)),
        "mean_latency_seconds": None if not latency else float(np.mean(latency)),
        "median_latency_seconds": None if not latency else float(np.median(latency)),
    }


def policy_diagnostics(reasons: tuple[str | None, ...]) -> dict:
    names = ("candidate_open", "candidate_abort", "candidate_timeout", "candidate_commit", "fast0_commit", "fast1_commit", "idle_reset")
    return {name: sum(reason == name for reason in reasons) for name in names}


def report_summary(reports: list[dict]) -> dict:
    keys = ("accuracy", "balanced_accuracy", "event_hit_rate", "mean_latency_seconds")
    result: dict[str, object] = {"report_count": len(reports)}
    for key in keys:
        values = [float(report[key]) for report in reports if report.get(key) is not None]
        result[key] = {"mean": None if not values else float(np.mean(values)), "std": None if not values else float(np.std(values))}
    return result
