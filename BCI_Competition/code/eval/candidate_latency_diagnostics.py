from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from logit_candidate_strategies import LogitStrategyResult
from protocol_metrics import (
    COMMAND_COMMIT,
    NO_COMMAND,
    WAIT_IDLE,
    DecisionRecord,
    ExpectedWindow,
    MIEvent,
)


@dataclass(frozen=True)
class _Episode:
    """保存一个已经完成推理的候选区间及其开门窗起算的 Stage 2 分数。"""

    key: tuple[int, int, int, int]
    open_window_index: int
    open_sample: int
    exit_window_index: int | None
    exit_sample: int | None
    outcome: str
    rows: tuple[dict, ...]


def _distribution(values: list[float]) -> dict:
    """空集合保留为 None，避免把“没有机会”误报成零延迟。"""
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


def _probability_summary(logits: np.ndarray) -> tuple[int, float, float]:
    """用稳定 softmax 返回 top-1 类别、概率以及 top1-top2 差值。"""
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Stage 2 logits 必须是四维有限向量")
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    probabilities = weights / float(np.sum(weights))
    order = np.sort(probabilities)
    return (
        int(np.argmax(probabilities)) + 1,
        float(order[-1]),
        float(order[-1] - order[-2]),
    )


def _overlap(window: ExpectedWindow, event: MIEvent) -> int:
    return max(
        0,
        min(window.window_stop_sample, event.offset_sample)
        - max(window.window_start_sample, event.onset_sample),
    )


def _eligible(window: ExpectedWindow, event: MIEvent, margin_samples: int) -> bool:
    return (
        window.key == event.key
        and window.window_stop_sample <= event.offset_sample
        and _overlap(window, event) >= margin_samples
    )


def _build_episodes(
    windows: Sequence[ExpectedWindow],
    stage2_logits: np.ndarray,
    strategy: LogitStrategyResult,
    evaluated: dict,
    *,
    alpha: float,
) -> tuple[tuple[_Episode, ...], dict[tuple[tuple[int, int, int, int], int], _Episode]]:
    """按真实候选边界重建开门窗起算的反事实 EWMA；不改变原决策轨迹。"""
    positions_by_stream: dict[tuple[int, int, int, int], list[int]] = {}
    for position, window in enumerate(windows):
        positions_by_stream.setdefault(window.key, []).append(position)

    episodes: list[_Episode] = []
    by_exit: dict[tuple[tuple[int, int, int, int], int], _Episode] = {}
    for interval in evaluated["candidate_diagnostics"]["candidate_intervals"]:
        key = (
            interval["subject_id"], interval["session_id"],
            interval["run_id"], interval["segment_id"],
        )
        open_index = int(interval["open_window_index"])
        exit_index = interval["exit_window_index"]
        stop_index = windows[positions_by_stream[key][-1]].window_index if exit_index is None else int(exit_index)
        positions = [
            position for position in positions_by_stream[key]
            if open_index <= windows[position].window_index <= stop_index
        ]
        if not positions or windows[positions[0]].window_index != open_index:
            raise RuntimeError("候选区间无法映射到冻结窗口")

        ewma: np.ndarray | None = None
        rows: list[dict] = []
        for age, position in enumerate(positions, start=1):
            centered = np.asarray(stage2_logits[position], dtype=np.float64)
            centered = centered - float(np.mean(centered))
            ewma = centered.copy() if ewma is None else alpha * centered + (1.0 - alpha) * ewma
            raw_class, raw_probability, raw_gap = _probability_summary(centered)
            ewma_class, ewma_probability, ewma_gap = _probability_summary(ewma)
            rows.append({
                "position": position,
                "age_including_open": age,
                "raw_class": raw_class,
                "raw_probability": raw_probability,
                "raw_gap": raw_gap,
                "open_ewma_class": ewma_class,
                "open_ewma_probability": ewma_probability,
                "open_ewma_gap": ewma_gap,
                # 开门窗已满足 task_on；后续窗仍必须尊重原 Stage 1 task_hold 证据。
                "stage1_allows_submit": (
                    True if age == 1 else bool(strategy.trace[position].evidence.task_hold)
                ),
            })
        episode = _Episode(
            key=key,
            open_window_index=open_index,
            open_sample=int(interval["open_decision_sample"]),
            exit_window_index=None if exit_index is None else int(exit_index),
            exit_sample=(
                None if interval["exit_decision_sample"] is None
                else int(interval["exit_decision_sample"])
            ),
            outcome=str(interval["outcome"]),
            rows=tuple(rows),
        )
        episodes.append(episode)
        if episode.exit_window_index is not None:
            identity = (key, episode.exit_window_index)
            if identity in by_exit:
                raise RuntimeError("候选退出身份重复")
            by_exit[identity] = episode
    return tuple(episodes), by_exit


def _earliest_oracles(
    event: MIEvent,
    episodes: Sequence[_Episode],
    windows: Sequence[ExpectedWindow],
    *,
    margin_samples: int,
    top_probability: float,
    probability_gap: float,
) -> dict[str, int | None]:
    """真值只在轨迹完成后选择“正确类别”窗口，因此这些量不是可部署策略成绩。"""
    candidates: dict[str, list[int]] = {
        "raw_correct_top1": [],
        "raw_correct_confident": [],
        "open_ewma_correct_top1": [],
        "open_ewma_correct_confident_min1": [],
        "open_ewma_correct_confident_min2": [],
    }
    for episode in episodes:
        for row in episode.rows:
            window = windows[row["position"]]
            if not _eligible(window, event, margin_samples) or not row["stage1_allows_submit"]:
                continue
            sample = window.window_stop_sample
            if row["raw_class"] == event.true_class:
                candidates["raw_correct_top1"].append(sample)
                if row["raw_probability"] >= top_probability and row["raw_gap"] >= probability_gap:
                    candidates["raw_correct_confident"].append(sample)
            if row["open_ewma_class"] == event.true_class:
                candidates["open_ewma_correct_top1"].append(sample)
                confident = (
                    row["open_ewma_probability"] >= top_probability
                    and row["open_ewma_gap"] >= probability_gap
                )
                if confident:
                    candidates["open_ewma_correct_confident_min1"].append(sample)
                    if row["age_including_open"] >= 2:
                        candidates["open_ewma_correct_confident_min2"].append(sample)
    return {name: None if not values else min(values) for name, values in candidates.items()}


def _first_crossings(
    event: MIEvent,
    episodes: Sequence[_Episode],
    windows: Sequence[ExpectedWindow],
    *,
    margin_samples: int,
    top_probability: float,
    probability_gap: float,
) -> dict[str, dict | None]:
    """在真值事件边界内选择首 crossing；选择类别时不读取 true_class。"""
    candidates: dict[str, list[tuple[int, int]]] = {
        "raw_first_confident": [],
        "open_ewma_first_confident_min1": [],
        "open_ewma_first_confident_min2": [],
    }
    for episode in episodes:
        for row in episode.rows:
            window = windows[row["position"]]
            if not _eligible(window, event, margin_samples) or not row["stage1_allows_submit"]:
                continue
            if row["raw_probability"] >= top_probability and row["raw_gap"] >= probability_gap:
                candidates["raw_first_confident"].append((window.window_stop_sample, row["raw_class"]))
            if (
                row["open_ewma_probability"] >= top_probability
                and row["open_ewma_gap"] >= probability_gap
            ):
                candidates["open_ewma_first_confident_min1"].append(
                    (window.window_stop_sample, row["open_ewma_class"])
                )
                if row["age_including_open"] >= 2:
                    candidates["open_ewma_first_confident_min2"].append(
                        (window.window_stop_sample, row["open_ewma_class"])
                    )
    result: dict[str, dict | None] = {}
    for name, values in candidates.items():
        if not values:
            result[name] = None
            continue
        sample, predicted_class = min(values, key=lambda item: item[0])
        result[name] = {
            "sample": sample,
            "predicted_class": predicted_class,
            "correct": predicted_class == event.true_class,
        }
    return result


def _event_summary(rows: list[dict], sampling_rate: float, fixed_wait_seconds: float) -> dict:
    """同时报告覆盖率和逐事件配对 headroom，避免只留下容易事件后伪造低延迟。"""
    scorable_count = len(rows)
    correct_rows = [row for row in rows if row["baseline_outcome"] == "correct"]
    spillover_rows = [row for row in rows if row["correct_class_spillover_sample"] is not None]
    missed_spillover_rows = [row for row in spillover_rows if row["baseline_outcome"] == "miss"]

    def rate(count: int, denominator: int) -> float | None:
        return None if denominator == 0 else count / denominator

    oracle_summary: dict[str, dict] = {}
    for name in (
        "raw_correct_top1", "raw_correct_confident", "open_ewma_correct_top1",
        "open_ewma_correct_confident_min1", "open_ewma_correct_confident_min2",
    ):
        samples = [row[f"{name}_sample"] for row in rows if row[f"{name}_sample"] is not None]
        latencies = [
            (row[f"{name}_sample"] - row["event_onset_sample"]) / sampling_rate
            for row in rows if row[f"{name}_sample"] is not None
        ]
        oracle_summary[name] = {
            "event_count": len(samples),
            "event_coverage_rate": rate(len(samples), scorable_count),
            "truth_aware_latency_seconds": _distribution(latencies),
        }

    paired: dict[str, dict] = {}
    for name in (
        "open_ewma_correct_confident_min1",
        "open_ewma_correct_confident_min2",
    ):
        headrooms = [
            (row["baseline_decision_sample"] - row[f"{name}_sample"]) / sampling_rate
            for row in correct_rows
            if row[f"{name}_sample"] is not None
            and row[f"{name}_sample"] <= row["baseline_decision_sample"]
        ]
        positive = [value for value in headrooms if value > 0]
        paired[name] = {
            "baseline_correct_event_count": len(correct_rows),
            "oracle_available_not_later_count": len(headrooms),
            "earlier_count": len(positive),
            "earlier_rate_among_baseline_correct": rate(len(positive), len(correct_rows)),
            "positive_headroom_seconds": _distribution(positive),
        }

    spillover_rescue: dict[str, dict] = {}
    for name in (
        "open_ewma_correct_confident_min1",
        "open_ewma_correct_confident_min2",
    ):
        rescued = [row for row in missed_spillover_rows if row[f"{name}_sample"] is not None]
        headrooms = [
            (row["correct_class_spillover_sample"] - row[f"{name}_sample"]) / sampling_rate
            for row in rescued
        ]
        spillover_rescue[name] = {
            "baseline_miss_correct_class_spillover_event_count": len(missed_spillover_rows),
            "truth_aware_rescuable_count": len(rescued),
            "truth_aware_rescuable_rate": rate(len(rescued), len(missed_spillover_rows)),
            "spillover_to_oracle_headroom_seconds": _distribution(headrooms),
        }

    # class-label-free 首 crossing 不跳过错误类别，但事件边界和 margin 仍来自真值；
    # 它也未在全部 IDLE 候选上重放状态改变，所以只能称事后事件局部诊断。
    first_crossing_summary: dict[str, dict] = {}
    crossing_pairs = (
        ("raw_first_confident", "raw_correct_confident"),
        ("open_ewma_first_confident_min1", "open_ewma_correct_confident_min1"),
        ("open_ewma_first_confident_min2", "open_ewma_correct_confident_min2"),
    )
    for crossing_name, oracle_name in crossing_pairs:
        available = [row for row in rows if row[f"{crossing_name}_sample"] is not None]
        correct = [row for row in available if row[f"{crossing_name}_correct"]]
        correct_earlier = [
            row for row in correct_rows
            if row[f"{crossing_name}_correct"] is True
            and row[f"{crossing_name}_sample"] < row["baseline_decision_sample"]
        ]
        wrong_on_baseline_correct = [
            row for row in correct_rows
            if row[f"{crossing_name}_correct"] is False
            and row[f"{crossing_name}_sample"] < row["baseline_decision_sample"]
        ]
        oracle_available = [row for row in rows if row[f"{oracle_name}_sample"] is not None]
        prior_wrong = [
            row for row in oracle_available
            if row[f"{crossing_name}_correct"] is False
            and row[f"{crossing_name}_sample"] < row[f"{oracle_name}_sample"]
        ]
        first_crossing_summary[crossing_name] = {
            "opportunity_count": len(available),
            "opportunity_coverage_rate": rate(len(available), scorable_count),
            "correct_count": len(correct),
            "class_accuracy_at_first_crossing": rate(len(correct), len(available)),
            "correct_event_rate": rate(len(correct), scorable_count),
            "correct_latency_seconds": _distribution([
                (row[f"{crossing_name}_sample"] - row["event_onset_sample"]) / sampling_rate
                for row in correct
            ]),
            "baseline_correct_event_count": len(correct_rows),
            "earlier_and_correct_among_baseline_correct_count": len(correct_earlier),
            "earlier_and_correct_rate_among_baseline_correct": rate(
                len(correct_earlier), len(correct_rows),
            ),
            "earlier_and_correct_headroom_seconds": _distribution([
                (row["baseline_decision_sample"] - row[f"{crossing_name}_sample"])
                / sampling_rate
                for row in correct_earlier
            ]),
            "wrong_crossing_among_baseline_correct_count": len(wrong_on_baseline_correct),
            "wrong_crossing_rate_among_baseline_correct": rate(
                len(wrong_on_baseline_correct), len(correct_rows),
            ),
            "truth_oracle_available_count": len(oracle_available),
            "truth_oracle_preceded_by_wrong_crossing_count": len(prior_wrong),
            "truth_oracle_preceded_by_wrong_crossing_rate": rate(
                len(prior_wrong), len(oracle_available),
            ),
        }

    dwell = [row["baseline_candidate_dwell_seconds"] for row in correct_rows]
    if any(value is None or value + 1e-12 < fixed_wait_seconds for value in dwell):
        raise RuntimeError("基线正确提交违反开门后固定两窗等待下界")
    return {
        "scorable_event_count": scorable_count,
        "baseline_correct_event_count": len(correct_rows),
        "baseline_correct_event_rate": rate(len(correct_rows), scorable_count),
        "first_eligible_window_wait_idle_rate": rate(
            sum(row["first_eligible_window_wait_idle"] for row in rows), scorable_count,
        ),
        "fully_wait_idle_event_rate": rate(
            sum(row["fully_wait_idle_event"] for row in rows), scorable_count,
        ),
        "event_with_candidate_episode_rate": rate(
            sum(row["candidate_episode_count"] > 0 for row in rows), scorable_count,
        ),
        "baseline_correct_latency_seconds": _distribution([
            row["baseline_latency_seconds"] for row in correct_rows
        ]),
        "baseline_correct_stage1_open_latency_seconds": _distribution([
            row["baseline_candidate_open_latency_seconds"] for row in correct_rows
        ]),
        "baseline_correct_candidate_dwell_seconds": _distribution(dwell),
        "fixed_exclude_open_plus_min2_wait_seconds": fixed_wait_seconds,
        "baseline_correct_extra_after_fixed_wait_seconds": _distribution([
            value - fixed_wait_seconds for value in dwell
        ]),
        "truth_aware_oracles": oracle_summary,
        "label_free_first_crossings": first_crossing_summary,
        "paired_baseline_correct": paired,
        "correct_class_post_mi_spillover": {
            "event_count": len(spillover_rows),
            "event_rate": rate(len(spillover_rows), scorable_count),
            "baseline_miss_event_count": len(missed_spillover_rows),
            "already_triggered_event_count": len(spillover_rows) - len(missed_spillover_rows),
            "actual_latency_seconds": _distribution([
                (row["correct_class_spillover_sample"] - row["event_onset_sample"])
                / sampling_rate
                for row in spillover_rows
            ]),
            "rescue": spillover_rescue,
        },
    }


def diagnose_candidate_latency(
    events: Sequence[MIEvent],
    windows: Sequence[ExpectedWindow],
    stage2_logits: np.ndarray,
    strategy: LogitStrategyResult,
    evaluated: dict,
    *,
    stage2_alpha: float,
    stage2_top_probability: float,
    stage2_probability_gap: float,
    baseline_min_candidate_windows: int,
) -> dict:
    """完成基线推理后，按事件诊断正确候选可提前提交的上限。"""
    decisions = strategy.policy.decisions
    scores = np.asarray(stage2_logits, dtype=np.float64)
    if (
        len(windows) != len(decisions)
        or len(windows) != len(strategy.trace)
        or scores.shape != (len(windows), 4)
        or not np.isfinite(scores).all()
    ):
        raise ValueError("延迟诊断输入与冻结窗口不一致")
    if not 0.0 < stage2_alpha <= 1.0:
        raise ValueError("Stage 2 EWMA alpha 必须位于 (0, 1]")
    if baseline_min_candidate_windows != 2:
        raise ValueError("首轮延迟诊断只接受固定的两候选窗基线")

    sampling_rate = float(evaluated["sampling_rate"])
    margin_samples = int(evaluated["min_overlap_samples"])
    step_samples = int(evaluated["step_samples"])
    fixed_wait_seconds = baseline_min_candidate_windows * step_samples / sampling_rate
    window_map = {window.identity: window for window in windows}
    decision_map = {decision.identity: decision for decision in decisions}
    if len(window_map) != len(windows) or len(decision_map) != len(decisions):
        raise ValueError("窗口或决策身份重复")

    episodes, episode_by_exit = _build_episodes(
        windows, scores, strategy, evaluated, alpha=stage2_alpha,
    )
    episodes_by_stream: dict[tuple[int, int, int, int], list[_Episode]] = {}
    events_by_stream: dict[tuple[int, int, int, int], list[MIEvent]] = {}
    for episode in episodes:
        episodes_by_stream.setdefault(episode.key, []).append(episode)
    for event in events:
        events_by_stream.setdefault(event.key, []).append(event)

    match_map = {
        (item["subject_id"], item["session_id"], item["run_id"], item["segment_id"], item["event_id"]): item
        for item in evaluated["event_matches"]
    }
    rows: list[dict] = []
    for event in sorted(events, key=lambda item: (*item.key, item.onset_sample)):
        match = match_map[(*event.key, event.event_id)]
        if not match["scorable"]:
            continue
        eligible_windows = [window for window in windows if _eligible(window, event, margin_samples)]
        event_episodes = [
            episode for episode in episodes_by_stream.get(event.key, [])
            if any(_eligible(windows[row["position"]], event, margin_samples) for row in episode.rows)
        ]
        oracles = _earliest_oracles(
            event, event_episodes, windows,
            margin_samples=margin_samples,
            top_probability=stage2_top_probability,
            probability_gap=stage2_probability_gap,
        )
        first_crossings = _first_crossings(
            event, event_episodes, windows,
            margin_samples=margin_samples,
            top_probability=stage2_top_probability,
            probability_gap=stage2_probability_gap,
        )

        baseline_episode: _Episode | None = None
        if match["window_index"] is not None:
            baseline_episode = episode_by_exit.get((event.key, int(match["window_index"])))
            if baseline_episode is None or baseline_episode.outcome != COMMAND_COMMIT:
                raise RuntimeError("事件匹配命令缺少对应候选提交区间")

        correct_spillovers: list[_Episode] = []
        for episode in episodes_by_stream.get(event.key, []):
            if (
                episode.outcome != COMMAND_COMMIT
                or episode.exit_window_index is None
                or episode.exit_sample is None
                or not (event.onset_sample <= episode.open_sample <= event.offset_sample)
                or episode.exit_sample <= event.offset_sample
            ):
                continue
            exit_decision = decision_map[(episode.key, episode.exit_window_index)]
            inside_any_event = any(
                other.onset_sample <= episode.exit_sample <= other.offset_sample
                for other in events_by_stream.get(event.key, [])
            )
            if not inside_any_event and exit_decision.emitted_class == event.true_class:
                correct_spillovers.append(episode)
        spillover = None if not correct_spillovers else min(
            correct_spillovers, key=lambda item: item.exit_sample,
        )

        states = [decision_map[window.identity].decision_state_before for window in eligible_windows]
        row = {
            "subject_id": event.subject_id,
            "session_id": event.session_id,
            "run_id": event.run_id,
            "segment_id": event.segment_id,
            "event_id": event.event_id,
            "true_class": event.true_class,
            "event_onset_sample": event.onset_sample,
            "event_offset_sample": event.offset_sample,
            "first_eligible_decision_sample": eligible_windows[0].window_stop_sample,
            "baseline_outcome": match["outcome"],
            "baseline_predicted_class": match["predicted_class"],
            "baseline_decision_sample": match["decision_sample"],
            "baseline_latency_seconds": match["latency_seconds"],
            "candidate_episode_count": len(event_episodes),
            "first_eligible_window_wait_idle": states[0] == WAIT_IDLE,
            "fully_wait_idle_event": all(state == WAIT_IDLE for state in states),
            "correct_class_spillover_sample": None if spillover is None else spillover.exit_sample,
        }
        for name, sample in oracles.items():
            row[f"{name}_sample"] = sample
            row[f"{name}_latency_seconds"] = (
                None if sample is None else (sample - event.onset_sample) / sampling_rate
            )
        for name, crossing in first_crossings.items():
            row[f"{name}_sample"] = None if crossing is None else crossing["sample"]
            row[f"{name}_predicted_class"] = (
                None if crossing is None else crossing["predicted_class"]
            )
            row[f"{name}_correct"] = None if crossing is None else crossing["correct"]
        if match["outcome"] == "correct":
            if baseline_episode is None or match["decision_sample"] is None:
                raise RuntimeError("正确事件缺少基线候选区间")
            row["baseline_candidate_open_sample"] = baseline_episode.open_sample
            row["baseline_candidate_open_latency_seconds"] = (
                baseline_episode.open_sample - event.onset_sample
            ) / sampling_rate
            row["baseline_candidate_dwell_seconds"] = (
                int(match["decision_sample"]) - baseline_episode.open_sample
            ) / sampling_rate
        else:
            row["baseline_candidate_open_sample"] = None
            row["baseline_candidate_open_latency_seconds"] = None
            row["baseline_candidate_dwell_seconds"] = None
        rows.append(row)

    summary = _event_summary(rows, sampling_rate, fixed_wait_seconds)
    if summary["baseline_correct_event_count"] != sum(
        item["scorable"] and item["outcome"] == "correct"
        for item in evaluated["event_matches"]
    ):
        raise RuntimeError("延迟诊断的正确事件数与正式评估器不一致")
    return {
        "truth_usage": "post_inference_oracle_diagnostics_only",
        "online_policy_status": "baseline_trajectory_unchanged_oracles_not_deployable",
        "latency_clock": {
            "definition": "decision_sample/250 - MI_onset_sample/250",
            "causal_filter_group_delay": "included_in_end_to_end_score_timing_and_not_subtracted",
            "filter_model_state_machine_wall_clock_compute": "not_measured",
        },
        "counterfactual_semantics": {
            "raw_correct_top1": "earliest formally eligible held Stage1 window whose raw Stage2 top1 equals truth",
            "raw_correct_confident": "raw_correct_top1 plus frozen probability and gap thresholds",
            "open_ewma_correct_top1": "EWMA starts on candidate opening window; earliest correct top1",
            "open_ewma_correct_confident_min1": "opening window may submit when truth-aware correct confidence is sufficient",
            "open_ewma_correct_confident_min2": "opening window counts as window one; at least one later window is required",
            "label_free_first_crossings": (
                "select class without true_class inside post-inference truth-defined event "
                "onset/offset and margin, then annotate correctness; event-local only and "
                "not a full stateful counterfactual replay"
            ),
        },
        "event_rows": rows,
        "summary": summary,
    }
