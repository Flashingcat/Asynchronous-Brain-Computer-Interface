"""Window and event metrics for labelled test-session predictions."""

from __future__ import annotations

import json

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


def event_metrics(
    y_true: np.ndarray,
    commands: np.ndarray,
    run_ids: np.ndarray,
    event_ids: np.ndarray,
    decision_samples: np.ndarray,
    events: dict[str, np.ndarray],
    *,
    sampling_rate: float,
) -> dict:
    """Score first commands against original annotation events and real sample times."""
    truth, emitted, runs, window_events, decisions = map(
        lambda x: np.asarray(x, dtype=np.int64),
        (y_true, commands, run_ids, event_ids, decision_samples),
    )
    if truth.ndim != 1 or any(value.shape != truth.shape for value in (emitted, runs, window_events, decisions)):
        raise ValueError("window labels, commands, run/event ids, and decision samples must align")
    if np.any(~np.isin(emitted, (-1, 1, 2, 3, 4))):
        raise ValueError("commands must be -1 or an MI class in [1,4]")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")

    # 事件表来自原始 annotation；与坏试次重叠的事件已在预处理阶段整体排除，
    # 其余事件即使没有可用决策窗口，也必须保留在分母中并记为漏检。
    required = {"run", "event", "label", "start"}
    if required.difference(events):
        raise ValueError(f"event table is missing {sorted(required.difference(events))}")
    event_runs, event_numbers, event_labels, event_starts = (
        np.asarray(events[key], dtype=np.int64) for key in ("run", "event", "label", "start")
    )
    if any(value.shape != event_runs.shape for value in (event_numbers, event_labels, event_starts)):
        raise ValueError("event table columns must align")
    pairs = list(zip(event_runs.tolist(), event_numbers.tolist()))
    if len(set(pairs)) != len(pairs):
        raise ValueError("event table contains duplicate run/event ids")
    catalog = set(pairs)
    referenced = set(zip(runs[window_events >= 0].tolist(), window_events[window_events >= 0].tolist()))
    if referenced.difference(catalog):
        raise ValueError(f"windows reference unknown events: {sorted(referenced.difference(catalog))}")

    used: set[int] = set()
    correct = wrong = miss = 0
    correct_latencies: list[float] = []
    wrong_latencies: list[float] = []
    for run, event, label, onset in zip(event_runs, event_numbers, event_labels, event_starts):
        indices = np.flatnonzero((runs == run) & (window_events == event))
        if indices.size and np.any(truth[indices] != label):
            raise ValueError(f"window labels disagree with event {int(run)}/{int(event)}")
        command_indices = indices[emitted[indices] != -1]
        if not command_indices.size:
            miss += 1
        else:
            first = int(command_indices[0])
            used.add(first)
            latency = (int(decisions[first]) - int(onset)) / sampling_rate
            if emitted[first] == label:
                correct += 1
                correct_latencies.append(latency)
            else:
                wrong += 1
                wrong_latencies.append(latency)
    extra = [i for i in np.flatnonzero(emitted != -1).tolist() if i not in used]
    idle_false = sum(window_events[i] < 0 for i in extra)
    additional = len(extra) - idle_false
    total = len(event_runs)
    return {
        "event_count": total, "event_correct": correct, "event_wrong_class": wrong, "event_miss": miss,
        "event_hit_rate": None if not total else correct / total,
        "event_wrong_class_rate": None if not total else wrong / total,
        "event_miss_rate": None if not total else miss / total,
        "idle_false_commands": int(idle_false), "additional_event_commands": int(additional),
        "command_count": int(np.count_nonzero(emitted != -1)),
        "mean_correct_latency_seconds": None if not correct_latencies else float(np.mean(correct_latencies)),
        "median_correct_latency_seconds": None if not correct_latencies else float(np.median(correct_latencies)),
        "mean_wrong_command_latency_seconds": None if not wrong_latencies else float(np.mean(wrong_latencies)),
        "event_definition": "original annotation event_id within a run",
        "latency_definition": "decision_sample minus annotation onset; correct first commands only",
    }


def policy_diagnostics(reasons: tuple[str | None, ...]) -> dict:
    names = ("candidate_open", "candidate_abort", "candidate_timeout", "candidate_commit", "fast0_commit", "fast1_commit", "idle_reset")
    return {name: sum(reason == name for reason in reasons) for name in names}


def grouped_summary(reports: list[dict]) -> dict:
    """Aggregate seeds only inside matching subject/model/training configurations."""
    metrics = {
        "window_classification": ("accuracy", "balanced_accuracy"),
        "command_policy": (
            "event_hit_rate", "event_wrong_class_rate", "event_miss_rate",
            "mean_correct_latency_seconds", "idle_false_commands", "additional_event_commands",
        ),
    }
    # 只移除 seed，其余数据、训练器和超参必须完全一致才能汇总。
    groups: dict[str, tuple[dict, list[dict]]] = {}
    for report in reports:
        config = dict(report.get("training_config") or {})
        seed = config.pop("seed", report.get("seed"))
        if seed is None:
            raise ValueError(f"checkpoint report has no seed: {report.get('checkpoint')}")
        identity = {"subject": report["subject"], "model": report["model"], "training_config": config}
        key = json.dumps(identity, sort_keys=True, ensure_ascii=True)
        groups.setdefault(key, (identity, []))[1].append({**report, "seed": int(seed)})

    output = []
    for key in sorted(groups):
        identity, items = groups[key]
        seeds = [item["seed"] for item in items]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate seeds for comparable checkpoint group: {seeds}")
        summary = {
            section: {name: _mean_std([item.get(section, {}).get(name) for item in items]) for name in names}
            for section, names in metrics.items()
        }
        output.append({**identity, "seed_count": len(items), "seeds": sorted(seeds), "metrics": summary})
    return {"group_count": len(output), "groups": output}


def _mean_std(values: list[float | None]) -> dict:
    present = [float(value) for value in values if value is not None]
    return {
        "mean": None if not present else float(np.mean(present)),
        "std": None if not present else float(np.std(present)),
        "valid_count": len(present),
        "missing_count": len(values) - len(present),
    }
