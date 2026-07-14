"""在推理完成后按提交路径归因命令，并与原慢通道逐事件配对。"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from protocol_metrics import (
    COMMAND_COMMIT,
    FAST0_COMMAND_COMMIT,
    FAST1_COMMAND_COMMIT,
    NO_COMMAND,
    DecisionRecord,
    MIEvent,
)


PATH_BY_REASON = {
    FAST0_COMMAND_COMMIT: "fast0",
    FAST1_COMMAND_COMMIT: "fast1",
    COMMAND_COMMIT: "slow",
}
PATH_NAMES = ("fast0", "fast1", "slow")


def _distribution(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "q25": None,
            "q75": None,
            "p90": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _overlap(record: DecisionRecord, event: MIEvent) -> int:
    return max(
        0,
        min(record.window_stop_sample, event.offset_sample)
        - max(record.window_start_sample, event.onset_sample),
    )


# ---------- 路径归因：完全复制正式评估器的事件/空闲判定边界 ----------
def diagnose_fast_paths(
    events: Sequence[MIEvent],
    decisions: Sequence[DecisionRecord],
    evaluated: dict,
) -> dict:
    """真值只用于推理后归因；返回每条命令的路径和正式结局。"""
    decision_map = {item.identity: item for item in decisions}
    if len(decision_map) != len(decisions):
        raise ValueError("决策身份不得重复")
    events_by_stream: dict[tuple[int, int, int, int], list[MIEvent]] = {}
    for event in events:
        events_by_stream.setdefault(event.key, []).append(event)
    margin_samples = int(evaluated["min_overlap_samples"])

    match_by_decision: dict[tuple[tuple[int, int, int, int], int], dict] = {}
    for match in evaluated["event_matches"]:
        if match["predicted_class"] is None:
            continue
        identity = (
            (
                match["subject_id"],
                match["session_id"],
                match["run_id"],
                match["segment_id"],
            ),
            match["window_index"],
        )
        if identity in match_by_decision:
            raise RuntimeError("一条命令不得匹配多个真值事件")
        match_by_decision[identity] = match

    counters = {
        path: {
            "command_count": 0,
            "correct_event_count": 0,
            "wrong_class_event_count": 0,
            "idle_false_count": 0,
            "too_early_count": 0,
            "additional_event_count": 0,
        }
        for path in PATH_NAMES
    }
    correct_latencies = {path: [] for path in PATH_NAMES}
    command_rows: list[dict] = []
    for record in decisions:
        if record.emitted_class == NO_COMMAND:
            continue
        path = PATH_BY_REASON.get(record.transition_reason)
        if path is None:
            raise RuntimeError("已输出命令缺少可识别的快/慢提交原因")
        counters[path]["command_count"] += 1
        match = match_by_decision.get(record.identity)
        event_id, latency = None, None
        if match is not None:
            outcome = "correct" if match["outcome"] == "correct" else "wrong_class"
            event_id = match["event_id"]
            latency = match["latency_seconds"]
            counters[path][f"{outcome}_event_count"] += 1
            if outcome == "correct":
                correct_latencies[path].append(float(latency))
        else:
            covering = [
                event
                for event in events_by_stream.get(record.key, [])
                if event.onset_sample <= record.window_stop_sample <= event.offset_sample
            ]
            if not covering:
                outcome = "idle_false"
            else:
                event = max(covering, key=lambda item: _overlap(record, item))
                event_id = event.event_id
                outcome = "too_early" if _overlap(record, event) < margin_samples else "additional_event"
            counters[path][f"{outcome}_count"] += 1
        command_rows.append({
            "subject_id": record.subject_id,
            "session_id": record.session_id,
            "run_id": record.run_id,
            "segment_id": record.segment_id,
            "window_index": record.window_index,
            "decision_sample": record.window_stop_sample,
            "emitted_class": record.emitted_class,
            "path": path,
            "outcome": outcome,
            "event_id": event_id,
            "latency_seconds": latency,
        })

    # 归因合计必须与正式评估器独立计算的总数完全闭合。
    total = lambda name: sum(counters[path][name] for path in PATH_NAMES)
    triggered = sum(
        item["scorable"] and item["predicted_class"] is not None
        for item in evaluated["event_matches"]
    )
    if (
        total("command_count") != evaluated["emitted_command_count"]
        or total("correct_event_count") != evaluated["correct_event_count"]
        or total("wrong_class_event_count") != triggered - evaluated["correct_event_count"]
        or total("idle_false_count") != evaluated["idle_false_command_count"]
        or total("too_early_count") != evaluated["too_early_command_count"]
        or total("additional_event_count") != evaluated["additional_event_command_count"]
    ):
        raise RuntimeError("快/慢路径归因与正式评估器计数不闭合")

    valid_idle_minutes = float(evaluated["valid_idle_seconds"]) / 60.0
    scorable_events = int(evaluated["scorable_event_count"])
    path_summary = {}
    for path in PATH_NAMES:
        row = counters[path]
        commands = row["command_count"]
        triggered_path = row["correct_event_count"] + row["wrong_class_event_count"]
        path_summary[path] = {
            **row,
            "command_share": None if not evaluated["emitted_command_count"] else commands / evaluated["emitted_command_count"],
            "correct_among_all_commands": None if not commands else row["correct_event_count"] / commands,
            "triggered_class_accuracy": None if not triggered_path else row["correct_event_count"] / triggered_path,
            "correct_event_rate": None if not scorable_events else row["correct_event_count"] / scorable_events,
            "idle_false_per_valid_idle_minute": (
                None if valid_idle_minutes <= 0 else row["idle_false_count"] / valid_idle_minutes
            ),
            "correct_latency_seconds": _distribution(correct_latencies[path]),
        }
    return {
        "truth_usage": "post_inference_diagnostics_only",
        "paths": path_summary,
        "command_rows": command_rows,
    }


# ---------- 配对锚点：直接数每个事件被救回、伤害或提前了多少 ----------
def diagnose_against_anchor(anchor: dict, current: dict) -> dict:
    def event_map(payload: dict) -> dict[tuple, dict]:
        return {
            (
                item["subject_id"],
                item["session_id"],
                item["run_id"],
                item["segment_id"],
                item["event_id"],
            ): item
            for item in payload["event_matches"]
        }

    base, cell = event_map(anchor), event_map(current)
    if set(base) != set(cell):
        raise RuntimeError("快速通道与锚点的事件库不一致")
    for key in base:
        if (
            base[key]["true_class"] != cell[key]["true_class"]
            or base[key]["scorable"] != cell[key]["scorable"]
        ):
            raise RuntimeError("快速通道与锚点的真值语义不一致")

    scorable = [key for key in base if base[key]["scorable"]]
    anchor_correct = [key for key in scorable if base[key]["outcome"] == "correct"]
    anchor_miss = [key for key in scorable if base[key]["outcome"] == "miss"]
    anchor_noncorrect = [key for key in scorable if base[key]["outcome"] != "correct"]
    harmed = [key for key in anchor_correct if cell[key]["outcome"] != "correct"]
    rescued_miss = [key for key in anchor_miss if cell[key]["outcome"] == "correct"]
    rescued_noncorrect = [key for key in anchor_noncorrect if cell[key]["outcome"] == "correct"]
    both_correct = [
        key for key in anchor_correct if cell[key]["outcome"] == "correct"
    ]
    headroom = [
        float(base[key]["latency_seconds"] - cell[key]["latency_seconds"])
        for key in both_correct
    ]
    earlier = sum(value > 0 for value in headroom)
    return {
        "anchor_correct_event_count": len(anchor_correct),
        "anchor_correct_harmed_count": len(harmed),
        "anchor_correct_harmed_rate": None if not anchor_correct else len(harmed) / len(anchor_correct),
        "anchor_miss_event_count": len(anchor_miss),
        "anchor_miss_rescued_correct_count": len(rescued_miss),
        "anchor_miss_rescued_correct_rate": None if not anchor_miss else len(rescued_miss) / len(anchor_miss),
        "anchor_noncorrect_event_count": len(anchor_noncorrect),
        "anchor_noncorrect_rescued_correct_count": len(rescued_noncorrect),
        "anchor_noncorrect_rescued_correct_rate": (
            None if not anchor_noncorrect else len(rescued_noncorrect) / len(anchor_noncorrect)
        ),
        "both_correct_event_count": len(both_correct),
        "both_correct_earlier_count": earlier,
        "both_correct_earlier_rate": None if not both_correct else earlier / len(both_correct),
        "anchor_minus_current_correct_latency_seconds": _distribution(headroom),
    }
