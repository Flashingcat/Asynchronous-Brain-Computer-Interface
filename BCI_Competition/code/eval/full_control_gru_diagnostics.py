"""严格两状态轨迹的提交、WAIT_IDLE 与复位时机诊断。"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from protocol_metrics import (
    NO_COMMAND,
    READY,
    WAIT_IDLE,
    DecisionRecord,
    ExpectedWindow,
    MIEvent,
    ScoringSegment,
)


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


def _overlap(window: ExpectedWindow, event: MIEvent) -> int:
    return max(
        0,
        min(window.window_stop_sample, event.offset_sample)
        - max(window.window_start_sample, event.onset_sample),
    )


# ---------- 严格两状态诊断：只解释既有轨迹，绝不向推理返回真值信息 ----------
def diagnose_two_state_reset(
    segments: Sequence[ScoringSegment],
    events: Sequence[MIEvent],
    windows: Sequence[ExpectedWindow],
    decisions: Sequence[DecisionRecord],
    evaluated: dict,
) -> dict:
    if len(windows) != len(decisions):
        raise ValueError("窗口与决策数量必须完全一致")
    sampling_rate = float(evaluated["sampling_rate"])
    margin_samples = int(evaluated["min_overlap_samples"])
    segment_map = {item.key: item for item in segments}
    decision_map = {item.identity: item for item in decisions}
    if len(decision_map) != len(decisions):
        raise ValueError("两状态决策身份不得重复")
    windows_by_stream: dict[tuple[int, int, int, int], list[ExpectedWindow]] = {}
    decisions_by_stream: dict[tuple[int, int, int, int], list[DecisionRecord]] = {}
    for window in windows:
        windows_by_stream.setdefault(window.key, []).append(window)
    for decision in decisions:
        decisions_by_stream.setdefault(decision.key, []).append(decision)

    # ---------- WAIT_IDLE 区间：直接从两状态转换识别，不依赖候选 transition_reason ----------
    completed_wait: list[float] = []
    censored_wait: list[float] = []
    reset_after_commit: dict[tuple[tuple[int, int, int, int], int], DecisionRecord] = {}
    unresolved_count = 0
    command_count = 0
    reset_count = 0
    for key, stream in decisions_by_stream.items():
        pending: DecisionRecord | None = None
        ordered = sorted(stream, key=lambda item: (item.window_stop_sample, item.window_index))
        for record in ordered:
            submit = (
                record.emitted_class != NO_COMMAND
                and record.decision_state_before == READY
                and record.decision_state_after == WAIT_IDLE
            )
            reset = (
                record.emitted_class == NO_COMMAND
                and record.decision_state_before == WAIT_IDLE
                and record.decision_state_after == READY
            )
            if submit:
                if pending is not None:
                    raise RuntimeError("WAIT_IDLE 未结束时出现重复命令")
                pending = record
                command_count += 1
            elif reset:
                if pending is None:
                    raise RuntimeError("两状态复位缺少对应提交")
                completed_wait.append(
                    (record.window_stop_sample - pending.window_stop_sample) / sampling_rate
                )
                reset_after_commit[pending.identity] = record
                pending = None
                reset_count += 1
        if pending is not None:
            unresolved_count += 1
            censored_wait.append(
                (segment_map[key].stop_sample - pending.window_stop_sample) / sampling_rate
            )
    if command_count != evaluated["emitted_command_count"]:
        raise RuntimeError("两状态诊断重算的命令数与正式 evaluator 不一致")

    # ---------- 复位相对真实事件结束：只解释正式匹配到事件的首次命令 ----------
    event_lookup = {(*item.key, item.event_id): item for item in events}
    matched_command_ids: set[tuple[tuple[int, int, int, int], int]] = set()
    reset_relative_offset: list[float] = []
    matched_without_reset = 0
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
        matched_command_ids.add(identity)
        reset = reset_after_commit.get(identity)
        if reset is None:
            matched_without_reset += 1
            continue
        event = event_lookup[(
            match["subject_id"],
            match["session_id"],
            match["run_id"],
            match["segment_id"],
            match["event_id"],
        )]
        reset_relative_offset.append(
            (reset.window_stop_sample - event.offset_sample) / sampling_rate
        )

    # ---------- 事件锁定：量化前一条命令是否让下一事件缺少可提交窗口 ----------
    first_window_locked = 0
    fully_locked = 0
    starts_locked_then_rearms = 0
    scorable_count = 0
    match_by_event = {
        (
            item["subject_id"], item["session_id"], item["run_id"],
            item["segment_id"], item["event_id"],
        ): item
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
            if match_by_event[(*event.key, event.event_id)]["outcome"] != "miss":
                raise RuntimeError("整个事件均锁定时不应产生匹配命令")

    matched_count = len(matched_command_ids)
    premature_count = sum(value < 0 for value in reset_relative_offset)
    return {
        "truth_usage": "post_inference_diagnostics_only",
        "state_machine": "strict_READY_WAIT_IDLE",
        "wait_idle": {
            "command_count": command_count,
            "reset_count": reset_count,
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
        },
        "idle_false_attribution": {
            "status": "not_defined_for_candidate_free_gru_v1",
            "official_idle_false_command_count": evaluated["idle_false_command_count"],
        },
    }
