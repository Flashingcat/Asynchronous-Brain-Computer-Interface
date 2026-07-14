from __future__ import annotations

from typing import Sequence

import numpy as np

from protocol_metrics import (
    COMMAND_COMMIT,
    IDLE_RESET,
    NO_COMMAND,
    WAIT_IDLE,
    DecisionRecord,
    ExpectedWindow,
    MIEvent,
    ScoringSegment,
)


def _distribution(values: list[float]) -> dict:
    """统一保存时长分布；空集合不伪造零延迟。"""
    if not values:
        return {
            "count": 0, "mean": None, "median": None,
            "q25": None, "q75": None, "p90": None,
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


def _overlap(window: ExpectedWindow | DecisionRecord, event: MIEvent) -> int:
    return max(
        0,
        min(window.window_stop_sample, event.offset_sample)
        - max(window.window_start_sample, event.onset_sample),
    )


def diagnose_commit_reset(
    segments: Sequence[ScoringSegment],
    events: Sequence[MIEvent],
    windows: Sequence[ExpectedWindow],
    decisions: Sequence[DecisionRecord],
    evaluated: dict,
) -> dict:
    """在推理结束后用真值解释提交与复位时机，不把诊断反向送入决策器。"""
    if len(windows) != len(decisions):
        raise ValueError("窗口与决策数量必须完全一致")
    sampling_rate = float(evaluated["sampling_rate"])
    margin_samples = int(evaluated["min_overlap_samples"])
    decision_map = {item.identity: item for item in decisions}
    if len(decision_map) != len(decisions):
        raise ValueError("决策身份不得重复")
    segment_map = {item.key: item for item in segments}
    events_by_stream: dict[tuple[int, int, int, int], list[MIEvent]] = {}
    windows_by_stream: dict[tuple[int, int, int, int], list[ExpectedWindow]] = {}
    decisions_by_stream: dict[tuple[int, int, int, int], list[DecisionRecord]] = {}
    for event in events:
        events_by_stream.setdefault(event.key, []).append(event)
    for window in windows:
        windows_by_stream.setdefault(window.key, []).append(window)
    for decision in decisions:
        decisions_by_stream.setdefault(decision.key, []).append(decision)

    # ---------- WAIT_IDLE 区间：从命令提交时刻计到复位时刻或 segment 结束 ----------
    completed_wait: list[float] = []
    censored_wait: list[float] = []
    reset_after_commit: dict[tuple[tuple[int, int, int, int], int], DecisionRecord] = {}
    unresolved_count = 0
    for key, stream in decisions_by_stream.items():
        pending: DecisionRecord | None = None
        for record in stream:
            if record.transition_reason == COMMAND_COMMIT:
                if pending is not None:
                    raise RuntimeError("WAIT_IDLE 尚未结束时出现第二次命令提交")
                pending = record
            elif record.transition_reason == IDLE_RESET:
                if pending is None:
                    raise RuntimeError("IDLE 复位缺少对应命令提交")
                completed_wait.append(
                    (record.window_stop_sample - pending.window_stop_sample) / sampling_rate
                )
                reset_after_commit[pending.identity] = record
                pending = None
        if pending is not None:
            unresolved_count += 1
            censored_wait.append(
                (segment_map[key].stop_sample - pending.window_stop_sample) / sampling_rate
            )

    # ---------- 复位相对真实 MI 结束时刻：仅匹配到事件的命令参与 ----------
    event_lookup = {(*item.key, item.event_id): item for item in events}
    matched_command_ids: set[tuple[tuple[int, int, int, int], int]] = set()
    reset_relative_offset: list[float] = []
    matched_without_reset = 0
    for match in evaluated["event_matches"]:
        if match["predicted_class"] is None:
            continue
        identity = (
            (
                match["subject_id"], match["session_id"],
                match["run_id"], match["segment_id"],
            ),
            match["window_index"],
        )
        matched_command_ids.add(identity)
        reset = reset_after_commit.get(identity)
        if reset is None:
            matched_without_reset += 1
            continue
        event = event_lookup[(
            match["subject_id"], match["session_id"], match["run_id"],
            match["segment_id"], match["event_id"],
        )]
        reset_relative_offset.append(
            (reset.window_stop_sample - event.offset_sample) / sampling_rate
        )

    # ---------- 事件锁定：判断每个可计分窗口是否都没有提交资格 ----------
    first_window_locked = 0
    fully_locked = 0
    starts_locked_then_rearms = 0
    scorable_count = 0
    match_by_event = {
        (item["subject_id"], item["session_id"], item["run_id"], item["segment_id"], item["event_id"]): item
        for item in evaluated["event_matches"]
    }
    for event in events:
        eligible = [
            window for window in windows_by_stream.get(event.key, [])
            if window.window_stop_sample <= event.offset_sample
            and _overlap(window, event) >= margin_samples
        ]
        if not eligible:
            continue
        scorable_count += 1
        states = [decision_map[item.identity].decision_state_before for item in eligible]
        if states[0] == WAIT_IDLE:
            first_window_locked += 1
            if any(value != WAIT_IDLE for value in states[1:]):
                starts_locked_then_rearms += 1
        if all(value == WAIT_IDLE for value in states):
            fully_locked += 1
            match = match_by_event[(*event.key, event.event_id)]
            if match["outcome"] != "miss":
                raise RuntimeError("整个事件均处于 WAIT_IDLE 时不应产生匹配命令")

    # ---------- FAR 归因：区分 MI 内开候选、MI 后才提交与其他 IDLE 误指令 ----------
    commit_intervals = {
        (
            (
                item["subject_id"], item["session_id"],
                item["run_id"], item["segment_id"],
            ),
            item["exit_window_index"],
        ): item
        for item in evaluated["candidate_diagnostics"]["candidate_intervals"]
        if item["outcome"] == COMMAND_COMMIT
    }
    idle_false_count = 0
    spillover_count = 0
    for record in decisions:
        if record.emitted_class == NO_COMMAND or record.identity in matched_command_ids:
            continue
        covering = [
            event for event in events_by_stream.get(record.key, [])
            if event.onset_sample <= record.window_stop_sample <= event.offset_sample
        ]
        if covering:
            continue  # 该命令由正式评估器归入过早输出或事件内额外输出。
        idle_false_count += 1
        interval = commit_intervals.get(record.identity)
        if interval is None:
            raise RuntimeError("IDLE 误指令缺少候选提交区间")
        opened_inside_event = any(
            event.onset_sample <= interval["open_decision_sample"] <= event.offset_sample
            and record.window_stop_sample > event.offset_sample
            for event in events_by_stream.get(record.key, [])
        )
        spillover_count += int(opened_inside_event)
    if idle_false_count != evaluated["idle_false_command_count"]:
        raise RuntimeError("独立重算的 IDLE 误指令数与正式评估器不一致")

    miss_count = sum(
        item["scorable"] and item["outcome"] == "miss"
        for item in evaluated["event_matches"]
    )
    valid_idle_minutes = evaluated["valid_idle_seconds"] / 60.0
    premature_count = sum(value < 0 for value in reset_relative_offset)
    matched_count = len(matched_command_ids)
    return {
        "truth_usage": "post_inference_diagnostics_only",
        "wait_idle": {
            "completed_duration_seconds": _distribution(completed_wait),
            "segment_end_unresolved_count": unresolved_count,
            "segment_end_censored_duration_seconds": _distribution(censored_wait),
        },
        "reset_relative_to_matched_event_offset": {
            "seconds": _distribution(reset_relative_offset),
            "matched_command_count": matched_count,
            "matched_command_without_reset_count": matched_without_reset,
            "premature_reset_count": premature_count,
            "premature_reset_rate": None if not matched_count else premature_count / matched_count,
        },
        "event_lock": {
            "scorable_event_count": scorable_count,
            "first_eligible_window_wait_idle_count": first_window_locked,
            "first_eligible_window_wait_idle_rate": (
                None if not scorable_count else first_window_locked / scorable_count
            ),
            "fully_wait_idle_event_count": fully_locked,
            "fully_wait_idle_event_rate": (
                None if not scorable_count else fully_locked / scorable_count
            ),
            "starts_locked_then_rearms_within_event_count": starts_locked_then_rearms,
            "fully_wait_idle_among_miss_rate": (
                None if not miss_count else fully_locked / miss_count
            ),
        },
        "idle_false_attribution": {
            "idle_false_command_count": idle_false_count,
            "post_mi_spillover_count": spillover_count,
            "other_idle_false_count": idle_false_count - spillover_count,
            "post_mi_spillover_per_valid_idle_minute": (
                None if valid_idle_minutes <= 0 else spillover_count / valid_idle_minutes
            ),
            "other_idle_false_per_valid_idle_minute": (
                None if valid_idle_minutes <= 0
                else (idle_false_count - spillover_count) / valid_idle_minutes
            ),
        },
    }
